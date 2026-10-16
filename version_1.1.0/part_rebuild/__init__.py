bl_info = {'name': 'Part Rebuild', 'author': 'Ricardo Lugaresi', 'version': (1, 1, 0),
           'blender': (4, 2, 0), 'location': 'Vista 3D > N > Part Rebuild',
           'description': 'Sustituye zonas fundidas de modelos de IA por piezas limpias: muelles, tubos, '
                          'cilindros, revolución, extrusión, simetría y clonado. Todo en modo Objeto',
           'category': 'Mesh'}

import math

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from . import canvas, common, edit, fit, build

# ---------------------------------------------------------------- ajustes
KIND_NAMES = {'CYLINDER': 'Cilindro', 'SPRING': 'Muelle', 'TUBE': 'Tubo', 'REVOLVE': 'Revolución',
              'EXTRUDE': 'Extrusión', 'CLONE': 'Clon'}


def _regen(self, context):
    obj = self.id_data
    if not self.lock and edit.is_part(obj):
        edit.generate(obj)


class PartProps(bpy.types.PropertyGroup):
    lock: bpy.props.BoolProperty(default=False, options={'HIDDEN'})
    radius: bpy.props.FloatProperty(name='Radio', subtype='DISTANCE', min=1e-5, update=_regen, precision=3)
    wire: bpy.props.FloatProperty(name='Grosor del hilo', subtype='DISTANCE', min=1e-5, update=_regen, precision=3)
    length: bpy.props.FloatProperty(name='Largo', subtype='DISTANCE', min=1e-5, update=_regen, precision=3)
    turns: bpy.props.FloatProperty(name='Espiras', min=1.0, max=200.0, update=_regen, precision=1)
    closed_ends: bpy.props.BoolProperty(name='Extremos cerrados', default=True, update=_regen,
                                        description='La última vuelta de cada lado apretada, como un muelle real')
    rod: bpy.props.BoolProperty(name='Con vástago', default=True, update=_regen,
                                description='Pone dentro el cuerpo del amortiguador (la zona vieja se quita entera)')
    hand: bpy.props.EnumProperty(name='Giro', items=[('RIGHT', 'Derecha', ''), ('LEFT', 'Izquierda', '')],
                                 default='RIGHT', update=_regen)
    phase: bpy.props.FloatProperty(name='Inicio', subtype='ANGLE', update=_regen)
    corner: bpy.props.FloatProperty(name='Curva en esquinas', subtype='DISTANCE', min=0.0, update=_regen, precision=3)
    scale: bpy.props.FloatProperty(name='Escala', default=1.0, min=0.05, max=20.0, update=_regen)
    segments: bpy.props.IntProperty(name='Suavidad', default=48, min=12, max=256, update=_regen)


class SceneProps(bpy.types.PropertyGroup):
    size: bpy.props.IntProperty(name='Tamaño del pincel', default=25, min=3, max=300, subtype='PIXEL')
    tool: bpy.props.EnumProperty(name='Herramienta', default='BRUSH', items=[
        ('BRUSH', 'Pincel', 'Marca lo que tocas al arrastrar (Shift: borrar marca)'),
        ('FILL', 'Cubo', 'Un clic marca la zona hasta las aristas')])
    angle: bpy.props.FloatProperty(name='Ángulo de arista', default=25.0, min=1.0, max=90.0)
    smooth: bpy.props.EnumProperty(name='Fuerza', default='MEDIUM', items=[
        ('LOW', 'Suave', ''), ('MEDIUM', 'Media', ''), ('HIGH', 'Fuerte', '')])
    remove_old: bpy.props.BoolProperty(name='Quitar lo viejo', default=True,
                                       description='Borra la zona marcada de la pieza y tapa el hueco')
    sym_axis: bpy.props.EnumProperty(name='Eje', default='X', items=[('X', 'X', ''), ('Y', 'Y', ''), ('Z', 'Z', '')])
    sym_refine: bpy.props.BoolProperty(name='Afinar plano', default=True,
                                       description='Busca el plano de simetría exacto cerca del eje elegido')
    clone_scale: bpy.props.BoolProperty(name='Ajustar tamaño', default=False,
                                        description='El clon se escala para encajar en la pieza mala')
    join_mode: bpy.props.EnumProperty(name='Modo', default='JOIN', items=[
        ('JOIN', 'Juntar', 'Todo en un objeto con sus colores; el laminador funde los solapes al imprimir. Sin perder detalle'),
        ('FUSE', 'Fundir', 'Sólido cerrado por vóxeles. Ojo: las paredes muy finas pueden desaparecer')])
    detail: bpy.props.EnumProperty(name='Detalle', default='HIGH', items=[
        ('LOW', 'Bajo', 'Rápido, para piezas grandes'),
        ('MEDIUM', 'Medio', 'Equilibrado'),
        ('HIGH', 'Alto', 'Más detalle, más lento')])
    done: bpy.props.StringProperty()


# ---------------------------------------------------------------- ayudantes
def host_of(context):
    obj = context.active_object
    if obj is None or obj.type != 'MESH':
        return None
    if edit.is_part(obj) or obj.get(edit.MOLD):
        h = bpy.data.objects.get(obj.get(edit.HOST, obj.get(edit.MOLD, '')))
        return h
    if common.is_helper(obj):
        return None
    return obj


_cache = {'key': None, 'canvas': None}


def get_canvas(obj):
    key = (obj.name, obj.data.name, len(obj.data.polygons), tuple(round(x, 6) for row in obj.matrix_world for x in row))
    if _cache['key'] != key:
        _cache['canvas'] = canvas.Canvas(obj)
        _cache['key'] = key
    else:
        c = _cache['canvas']
        c.mask = canvas.load_mask(obj.data)
    return _cache['canvas']


def forget_canvas():
    _cache['key'] = _cache['canvas'] = None
    _overlay['key'] = None


def check_mask(host, mask, minimum=40):
    """La marca tiene que ser una zona seguida, no puntos sueltos repartidos por la pieza."""
    n = int(mask.sum())
    if n < minimum:
        raise common.AddonError(f'Solo hay {n} caras marcadas: pinta toda la pieza (arrastrando con el clic apretado) '
                                'hasta que quede entera en rosa.')
    cv = get_canvas(host)
    biggest = 0
    left = mask.copy()
    for _ in range(6):
        if not left.any():
            break
        seed = int(np.nonzero(left)[0][0])
        part = cv.region(seed, allowed=mask, same_color=False)
        biggest = max(biggest, len(part))
        left[part] = False
    if biggest < 0.6 * n:
        raise common.AddonError('La marca está repartida en varios trozos sueltos. Quítala (X) y pinta solo la pieza, '
                                'de una pasada, hasta que quede entera en rosa.')


def to_mm(context, host, value):
    return value * common.factor(context.scene, 'AUTO', host)


def prepare_host(context):
    host = host_of(context)
    if host is None:
        raise common.AddonError('Selecciona la pieza (una malla).')
    work = canvas.working_copy(context, host)
    if work is not host:
        forget_canvas()
    return work


# ---------------------------------------------------------------- dibujo de la zona marcada
_overlay = {'key': None, 'batch': None}
_shader = {'s': None}


def _uniform():
    if _shader['s'] is None:
        _shader['s'] = gpu.shader.from_builtin('UNIFORM_COLOR')
    return _shader['s']


def _draw_mask():
    ctx = bpy.context
    obj = ctx.active_object
    if obj is None or obj.type != 'MESH' or ctx.mode != 'OBJECT':
        return
    att = obj.data.attributes.get(canvas.MASK_ATTR)
    if att is None:
        return
    key = (obj.name, obj.data.name, len(obj.data.polygons), obj.get('pr_ver', 0),
           tuple(round(x, 5) for row in obj.matrix_world for x in row))
    if _overlay['key'] != key:
        mask = canvas.load_mask(obj.data)
        if not mask.any():
            _overlay['key'], _overlay['batch'] = key, None
        else:
            me = obj.data
            n = len(me.polygons)
            ls = np.empty(n, np.int64); me.polygons.foreach_get('loop_start', ls)
            lt = np.empty(n, np.int64); me.polygons.foreach_get('loop_total', lt)
            loops = np.empty(len(me.loops), np.int64); me.loops.foreach_get('vertex_index', loops)
            co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', co)
            co = co.reshape(-1, 3)
            nrm = np.empty(n * 3); me.polygons.foreach_get('normal', nrm)
            nrm = nrm.reshape(-1, 3)
            m = np.array(obj.matrix_world)
            tris = []
            off = (np.ptp(co, axis=0).max() if len(co) else 1) * 0.0008
            for f in np.nonzero(mask)[0]:
                vs = loops[ls[f]:ls[f] + lt[f]]
                for k in range(1, len(vs) - 1):
                    tris.extend([co[vs[0]] + nrm[f] * off, co[vs[k]] + nrm[f] * off, co[vs[k + 1]] + nrm[f] * off])
            pts = np.array(tris) @ m[:3, :3].T + m[:3, 3]
            _overlay['batch'] = batch_for_shader(_uniform(), 'TRIS', {'pos': pts.astype(np.float32)})
            _overlay['key'] = key
    if _overlay['batch'] is None:
        return
    sh = _uniform()
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    sh.bind()
    sh.uniform_float('color', (1.0, 0.1, 0.75, 0.55))
    _overlay['batch'].draw(sh)
    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('NONE')


def bump(obj):
    obj['pr_ver'] = obj.get('pr_ver', 0) + 1


def _circle(x, y, r, segments=40):
    return [(x + r * math.cos(2 * math.pi * i / segments), y + r * math.sin(2 * math.pi * i / segments)) for i in range(segments)]


def _draw_brush(op, context):
    if op.mouse is None:
        return
    p = context.scene.part_rebuild_props
    sh = _uniform()
    x, y = op.mouse
    r = p.size if p.tool == 'BRUSH' else 8
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.0)
    for rr, c in ((r, (1, .1, .75, 1)), (r + 2, (1, 1, 1, .8))):
        batch = batch_for_shader(sh, 'LINE_LOOP', {'pos': _circle(x, y, rr)})
        sh.bind()
        sh.uniform_float('color', c)
        batch.draw(sh)
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


class _ViewMixin:
    """Manejo de la vista dentro de los operadores modales: botón central gira, rueda acerca."""

    def view_event(self, event):
        if event.type == 'MIDDLEMOUSE':
            self.orbit = event.value == 'PRESS'
            return True
        if getattr(self, 'orbit', False) and event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            return True
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} and not event.shift:
            return True
        if event.type in {'TRACKPADPAN', 'TRACKPADZOOM', 'NDOF_MOTION'} or \
                (event.type.startswith('NUMPAD') and event.value == 'PRESS'):
            return True
        return False

    def ray(self, context, x, y):
        from bpy_extras import view3d_utils
        region, rv3d = context.region, context.region_data
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, (x, y))
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, (x, y))
        loc, index = self.canvas.ray(origin, direction)
        return loc, index, direction

    def inside(self, context, event):
        return context.region is not None and 0 <= event.mouse_region_x < context.region.width \
            and 0 <= event.mouse_region_y < context.region.height


# ---------------------------------------------------------------- marcar
class MESH_OT_pr_mark(_ViewMixin, bpy.types.Operator):
    """Marca con el pincel la zona fundida que quieres sustituir"""
    bl_idname = 'mesh.pr_mark'
    bl_label = 'Marcar zona'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return common.poll_object_mode(cls, context) and host_of(context) is not None

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, 'Usa este botón desde la Vista 3D.')
            return {'CANCELLED'}
        try:
            self.obj = prepare_host(context)
            context.window.cursor_set('WAIT')
            self.canvas = get_canvas(self.obj)
        except common.AddonError as exc:
            context.window.cursor_set('DEFAULT')
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        for o in context.selected_objects:
            o.select_set(False)
        self.obj.select_set(True)
        context.view_layer.objects.active = self.obj
        context.window.cursor_set('PAINT_BRUSH')
        self.mouse = None
        self.painting = False
        self.erase = False
        self.orbit = False
        self.handle = bpy.types.SpaceView3D.draw_handler_add(_draw_brush, (self, context), 'WINDOW', 'POST_PIXEL')
        context.window_manager.modal_handler_add(self)
        context.area.header_text_set('Marcar: clic izquierdo pinta · Shift: borrar marca · F cubo / B pincel · '
                                     '[ ] tamaño · Ctrl+Z deshacer · botón central gira · Esc o Intro: terminar')
        return {'RUNNING_MODAL'}

    def _world_radius(self, context, x, y, loc, px):
        from bpy_extras import view3d_utils
        a = view3d_utils.region_2d_to_location_3d(context.region, context.region_data, (x, y), loc)
        b = view3d_utils.region_2d_to_location_3d(context.region, context.region_data, (x + px, y), loc)
        return max((b - a).length, 1e-9)

    def _apply(self, context, x, y, erase):
        p = context.scene.part_rebuild_props
        loc, index, _d = self.ray(context, x, y)
        if loc is None:
            return
        if p.tool == 'FILL':
            faces = self.canvas.region(index, angle=p.angle, same_color=False)
        else:
            faces = self.canvas.faces_near(loc, self._world_radius(context, x, y, loc, p.size))
            faces = np.union1d(faces, [index]).astype(np.int64)
        if canvas.mark(self.canvas, faces, not erase):
            canvas.save_mask(self.obj.data, self.canvas.mask)
            bump(self.obj)

    def _finish(self, context):
        bpy.types.SpaceView3D.draw_handler_remove(self.handle, 'WINDOW')
        context.area.header_text_set(None)
        context.window.cursor_set('DEFAULT')
        context.area.tag_redraw()
        n = int(self.canvas.mask.sum())
        context.scene.part_rebuild_props.done = f'Zona marcada: {n} caras' if n else ''
        return {'FINISHED'}

    def modal(self, context, event):
        p = context.scene.part_rebuild_props
        if self.view_event(event):
            return {'PASS_THROUGH'}
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} and event.shift:
            p.size = int(p.size * (1.15 if event.type == 'WHEELUPMOUSE' else 1 / 1.15))
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type in {'ESC', 'RET', 'RIGHTMOUSE'} and event.value == 'PRESS':
            return self._finish(context)
        if event.value == 'PRESS':
            if event.type == 'Z' and event.ctrl:
                if canvas.unmark_last(self.canvas):
                    canvas.save_mask(self.obj.data, self.canvas.mask)
                    bump(self.obj)
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            if event.type in {'F', 'B'}:
                p.tool = 'FILL' if event.type == 'F' else 'BRUSH'
                return {'RUNNING_MODAL'}
            if event.type in {'LEFT_BRACKET', 'RIGHT_BRACKET'}:
                p.size = int(p.size * (1.2 if event.type == 'RIGHT_BRACKET' else 1 / 1.2))
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
        inside = self.inside(context, event)
        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            self.mouse = (event.mouse_region_x, event.mouse_region_y) if inside else None
            if self.painting and inside and p.tool == 'BRUSH':
                self._apply(context, *self.mouse, self.erase)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE':
            if not inside:
                return {'PASS_THROUGH'}
            if event.value == 'PRESS':
                self.erase = event.shift
                self.painting = p.tool == 'BRUSH'
                self._apply(context, event.mouse_region_x, event.mouse_region_y, self.erase)
                context.area.tag_redraw()
            elif event.value == 'RELEASE':
                self.painting = False
            return {'RUNNING_MODAL'}
        return {'RUNNING_MODAL'}


class MESH_OT_pr_clear_mark(bpy.types.Operator):
    """Quita la marca de la pieza"""
    bl_idname = 'mesh.pr_clear_mark'
    bl_label = 'Borrar marca'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        host = host_of(context)
        if host is not None:
            canvas.clear_mask(host.data)
            bump(host)
            forget_canvas()
        context.scene.part_rebuild_props.done = ''
        return {'FINISHED'}


# ---------------------------------------------------------------- reconstruir desde la zona marcada
class MESH_OT_pr_rebuild(bpy.types.Operator):
    """Mide la zona marcada y pone en su sitio una pieza limpia"""
    bl_idname = 'mesh.pr_rebuild'
    bl_label = 'Reconstruir'
    bl_options = {'REGISTER', 'UNDO'}
    kind: bpy.props.EnumProperty(items=[('AUTO', 'Automático', ''), ('SPRING', 'Muelle', ''), ('CYLINDER', 'Cilindro', ''),
                                        ('REVOLVE', 'Revolución', ''), ('EXTRUDE', 'Extrusión', '')])

    @classmethod
    def description(cls, context, props):
        return {'AUTO': 'Prueba todas las formas y pone la que mejor encaja con la zona marcada',
                'SPRING': 'Muelle limpio: mide eje, radio, hilo y espiras de la zona marcada',
                'CYLINDER': 'Cilindro limpio (varillas, ejes, vástagos)',
                'REVOLVE': 'Pieza redonda alrededor de un eje (ruedas, pomos, tuercas): saca el perfil y lo gira',
                'EXTRUDE': 'Pieza con la misma sección a lo largo (lamas, soportes, perfiles)'}[props.kind]

    @classmethod
    def poll(cls, context):
        return common.poll_object_mode(cls, context) and host_of(context) is not None

    def execute(self, context):
        p = context.scene.part_rebuild_props
        try:
            host = prepare_host(context)
            mask = edit.get_mask(host)
            check_mask(host, mask)
            P, N, A = edit.face_data(host, mask)
            mat = edit.main_material(host, mask)
            kind, why = self.kind, ''
            if kind == 'AUTO':
                kind, why = fit.choose(P, N, A)
                if kind is None:
                    raise common.AddonError('Ninguna forma encaja bien con la zona marcada. Prueba «Suavizar zona», '
                                            '«Copiar del otro lado» o «Clonar».')
                why = f' ({why})'
            name = f'{host.name}_{KIND_NAMES[kind].lower()}'
            if kind == 'SPRING':
                r = fit.spring(P, N, A)
                VF = build.spring(r['radius'], r['wire'], r['length'], r['turns'], hand=r['hand'], phase=r['phase'], seg=12,
                                  rod=r['radius'] * 0.38)
                M = edit.frame_matrix(r['base'], r['axis'], r['u'])
                part = edit.new_part(context, host, 'SPRING', VF, M, mat, name, {
                    'radius': r['radius'], 'wire': r['wire'], 'length': r['length'], 'turns': r['turns'],
                    'hand': 'RIGHT' if r['hand'] > 0 else 'LEFT', 'phase': r['phase'], 'segments': 48})
                extra = f'{r["turns"]:.1f} espiras, hilo {to_mm(context, host, r["wire"]):.2f} mm'
                if not r['confident']:
                    extra += ' (espiras poco claras: ajústalas)'
            elif kind == 'CYLINDER':
                r = fit.cylinder(P, N, A)
                VF = build.cylinder(r['radius'], r['length'])
                M = edit.frame_matrix(r['base'], r['axis'])
                part = edit.new_part(context, host, 'CYLINDER', VF, M, mat, name,
                                     {'radius': r['radius'], 'length': r['length'], 'segments': 48})
                extra = f'Ø {to_mm(context, host, 2 * r["radius"]):.2f} mm'
            elif kind == 'REVOLVE':
                r = fit.revolution(P, N, A)
                VF = build.revolve(r['z'], r['rho'])
                M = edit.frame_matrix(r['base'], r['axis'])
                part = edit.new_part(context, host, 'REVOLVE', VF, M, mat, name,
                                     {'length': float(r['z'][-1]), 'scale': 1.0, 'segments': 64})
                part['pr_perfil_z'] = [float(x) for x in r['z']]
                part['pr_perfil_r'] = [float(x) for x in r['rho']]
                extra = f'perfil de {len(r["z"])} tramos'
            else:
                r = fit.extrusion(P, N, A, tris=edit.face_tris(host, mask))
                VF = build.extrude(r['outline'], r['length'])
                M = edit.frame_matrix(r['origin'], r['axis'], r['u'])
                part = edit.new_part(context, host, 'EXTRUDE', VF, M, mat, name,
                                     {'length': r['length'], 'scale': 1.0})
                part['pr_contorno'] = [float(x) for x in np.asarray(r['outline']).ravel()]
                extra = f'contorno de {len(r["outline"])} puntos'
            removed = ''
            if p.remove_old:
                edit.remove_faces(host, mask)
                forget_canvas()
                removed = ' · zona vieja quitada'
            else:
                canvas.clear_mask(host.data)
            bump(host)
        except common.AddonError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        err = to_mm(context, host, r['error'])
        p.done = f'{KIND_NAMES[kind]}{why}: {extra} · encaje ±{err:.2f} mm{removed}'
        self.report({'INFO'}, p.done)
        return {'FINISHED'}


class MESH_OT_pr_smooth(bpy.types.Operator):
    """Suaviza la zona marcada (grumos, zonas rugosas, restos) sin tocar el resto"""
    bl_idname = 'mesh.pr_smooth'
    bl_label = 'Suavizar zona'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return common.poll_object_mode(cls, context) and host_of(context) is not None

    def execute(self, context):
        p = context.scene.part_rebuild_props
        try:
            host = prepare_host(context)
            mask = edit.get_mask(host, minimum=3)
            n = edit.smooth_region(host, mask, {'LOW': 6, 'MEDIUM': 18, 'HIGH': 45}[p.smooth])
            forget_canvas()
            bump(host)
        except common.AddonError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        p.done = f'Zona suavizada ({n} vértices). Pulsa otra vez para más, o la X para quitar la marca.'
        self.report({'INFO'}, p.done)
        return {'FINISHED'}


# ---------------------------------------------------------------- tubo por clics
def _draw_path(op, context):
    pts = [Vector(p) for p in op.centers]
    if op.preview is not None:
        pts.append(Vector(op.preview))
    if not pts:
        return
    sh = _uniform()
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    gpu.state.line_width_set(3.0)
    if len(pts) > 1:
        b = batch_for_shader(sh, 'LINE_STRIP', {'pos': [tuple(p) for p in pts]})
        sh.bind(); sh.uniform_float('color', (1, .1, .75, 1)); b.draw(sh)
    gpu.state.point_size_set(9.0)
    b = batch_for_shader(sh, 'POINTS', {'pos': [tuple(p) for p in op.centers]})
    sh.bind(); sh.uniform_float('color', (1, 1, 1, 1)); b.draw(sh)
    gpu.state.line_width_set(1.0)
    gpu.state.point_size_set(1.0)
    gpu.state.blend_set('NONE')


class MESH_OT_pr_tube(_ViewMixin, bpy.types.Operator):
    """Tubo limpio: haz clic a lo largo del tubo (en cada esquina) e Intro para terminar"""
    bl_idname = 'mesh.pr_tube'
    bl_label = 'Tubo'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return common.poll_object_mode(cls, context) and host_of(context) is not None

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, 'Usa este botón desde la Vista 3D.')
            return {'CANCELLED'}
        try:
            self.obj = prepare_host(context)
            context.window.cursor_set('WAIT')
            self.canvas = get_canvas(self.obj)
        except common.AddonError as exc:
            context.window.cursor_set('DEFAULT')
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        context.window.cursor_set('CROSSHAIR')
        self.centers, self.radii, self.preview, self.orbit = [], [], None, False
        dims = max(self.obj.dimensions) or 1.0
        self.depth = dims * 0.2
        self.handle = bpy.types.SpaceView3D.draw_handler_add(_draw_path, (self, context), 'WINDOW', 'POST_VIEW')
        context.window_manager.modal_handler_add(self)
        context.area.header_text_set('Tubo: clic en el tubo en cada extremo y esquina · Retroceso: quitar el último · '
                                     'Intro: crear · Esc: cancelar · botón central gira')
        return {'RUNNING_MODAL'}

    def _center(self, context, x, y):
        loc, index, direction = self.ray(context, x, y)
        if loc is None:
            return None, None
        n = Vector(self.canvas.normals()[index])
        if n.dot(direction) > 0:
            n = -n
        return fit.tube_center(self.canvas.tree, loc, n, self.depth)

    def _end(self, context):
        bpy.types.SpaceView3D.draw_handler_remove(self.handle, 'WINDOW')
        context.area.header_text_set(None)
        context.window.cursor_set('DEFAULT')
        context.area.tag_redraw()

    def _create(self, context):
        p = context.scene.part_rebuild_props
        host = self.obj
        if len(self.centers) < 2:
            self.report({'ERROR'}, 'Haz al menos dos clics sobre el tubo.')
            return {'CANCELLED'}
        r = float(np.median(self.radii))
        pts = np.array(self.centers)
        origin = pts[0]
        local = pts - origin
        M = edit.frame_matrix(origin, (0, 0, 1), (1, 0, 0))
        VF = build.tube(local, r, corner=2.5 * r)
        # material: el de las caras bajo los clics
        near = np.unique(np.concatenate([self.canvas.faces_near(Vector(c), r * 1.5) for c in self.centers]))
        mask = np.zeros(len(self.canvas.mat), bool)
        mask[near] = True
        mat = edit.main_material(host, mask)
        part = edit.new_part(context, host, 'TUBE', VF, M, mat, f'{host.name}_tubo',
                             {'radius': r, 'corner': 2.5 * r, 'segments': 48})
        part['pr_puntos'] = [float(x) for x in local.ravel()]
        removed = ''
        if p.remove_old:
            path = build._densify(build.rounded_path(pts, 2.5 * r), r * 0.5)
            faces = np.unique(np.concatenate([self.canvas.faces_near(Vector(c), r * 1.35) for c in path]))
            m = np.zeros(len(self.canvas.mat), bool)
            m[faces] = True
            edit.remove_faces(host, m)
            forget_canvas()
            removed = ' · zona vieja quitada'
        p.done = f'Tubo: Ø {to_mm(context, host, 2 * r):.2f} mm, {len(self.centers)} puntos{removed}'
        self.report({'INFO'}, p.done)
        return {'FINISHED'}

    def modal(self, context, event):
        if self.view_event(event):
            return {'PASS_THROUGH'}
        if event.type == 'ESC' and event.value == 'PRESS':
            self._end(context)
            return {'CANCELLED'}
        if event.type in {'RET', 'NUMPAD_ENTER', 'RIGHTMOUSE'} and event.value == 'PRESS':
            self._end(context)
            return self._create(context)
        if event.type == 'BACK_SPACE' and event.value == 'PRESS':
            if self.centers:
                self.centers.pop(); self.radii.pop()
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            c, _r = self._center(context, event.mouse_region_x, event.mouse_region_y) if self.inside(context, event) else (None, None)
            self.preview = c
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if not self.inside(context, event):
                return {'PASS_THROUGH'}
            c, r = self._center(context, event.mouse_region_x, event.mouse_region_y)
            if c is not None:
                self.centers.append(c)
                self.radii.append(r)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        return {'RUNNING_MODAL'}


# ---------------------------------------------------------------- copiar lo bueno
class MESH_OT_pr_symmetry(bpy.types.Operator):
    """Sustituye la zona marcada (lado malo) por la copia reflejada del lado bueno"""
    bl_idname = 'mesh.pr_symmetry'
    bl_label = 'Copiar del otro lado'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return common.poll_object_mode(cls, context) and host_of(context) is not None

    def execute(self, context):
        p = context.scene.part_rebuild_props
        try:
            host = prepare_host(context)
            mask = edit.get_mask(host)
            n, normal = edit.mirror_replace(host, mask, 'XYZ'.index(p.sym_axis), p.sym_refine)
            forget_canvas()
            bump(host)
        except common.AddonError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        tilt = math.degrees(math.acos(min(1.0, abs(normal['XYZ'.index(p.sym_axis)]))))
        p.done = f'Simetría: {n} caras copiadas del lado bueno (plano a {tilt:.1f}° del eje {p.sym_axis})'
        self.report({'INFO'}, p.done)
        return {'FINISHED'}


class MESH_OT_pr_mold_save(bpy.types.Operator):
    """Guarda la zona marcada (la pieza buena) como molde para copiarla sobre las malas"""
    bl_idname = 'mesh.pr_mold_save'
    bl_label = 'Guardar la buena'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return common.poll_object_mode(cls, context) and host_of(context) is not None

    def execute(self, context):
        p = context.scene.part_rebuild_props
        try:
            host = prepare_host(context)
            mask = edit.get_mask(host)
            edit.save_mold(context, host, mask)
            bump(host)
            for o in context.selected_objects:
                o.select_set(False)
            host.select_set(True)
            context.view_layer.objects.active = host
        except common.AddonError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        p.done = 'Pieza buena guardada. Marca ahora una mala y pulsa «Poner aquí».'
        self.report({'INFO'}, p.done)
        return {'FINISHED'}


class MESH_OT_pr_mold_place(bpy.types.Operator):
    """Pone una copia de la pieza buena encajada sobre la zona marcada"""
    bl_idname = 'mesh.pr_mold_place'
    bl_label = 'Poner aquí'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        h = host_of(context)
        return common.poll_object_mode(cls, context) and h is not None and \
            any(o.get(edit.MOLD) == h.name for o in bpy.data.objects)

    def execute(self, context):
        p = context.scene.part_rebuild_props
        try:
            host = prepare_host(context)
            mask = edit.get_mask(host)
            context.window.cursor_set('WAIT')
            clone, err = edit.place_mold(context, host, mask, p.clone_scale)
            if p.remove_old:
                edit.remove_faces(host, mask)
                forget_canvas()
            else:
                canvas.clear_mask(host.data)
            bump(host)
            for o in context.selected_objects:
                o.select_set(False)
            host.select_set(True)
            context.view_layer.objects.active = host
        except common.AddonError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        finally:
            context.window.cursor_set('DEFAULT')
        p.done = f'Clon colocado · encaje ±{to_mm(context, host, err):.2f} mm. Marca otra mala o termina.'
        self.report({'INFO'}, p.done)
        return {'FINISHED'}


# ---------------------------------------------------------------- unir en sólido
class MESH_OT_pr_join(bpy.types.Operator):
    """Junta la pieza y sus reconstrucciones en un objeto para imprimir (en una copia)"""
    bl_idname = 'mesh.pr_join'
    bl_label = 'Unir en sólido'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return common.poll_object_mode(cls, context) and host_of(context) is not None

    def execute(self, context):
        p = context.scene.part_rebuild_props
        host = host_of(context)
        size = max(host.dimensions) or 1.0
        mm = common.factor(context.scene, 'AUTO', host)
        voxel = None
        if p.join_mode == 'FUSE':
            voxel = max(size / {'LOW': 300, 'MEDIUM': 600, 'HIGH': 1000}[p.detail],
                        {'LOW': 0.15, 'MEDIUM': 0.08, 'HIGH': 0.05}[p.detail] / mm)
        try:
            context.window.cursor_set('WAIT')
            solid = edit.join_solid(context, host, voxel)
        except common.AddonError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        finally:
            context.window.cursor_set('DEFAULT')
        if voxel is None:
            p.done = f'Listo para exportar: {solid.name} ({len(solid.data.polygons)} caras). Lo demás queda oculto.'
        else:
            p.done = f'Sólido listo: {solid.name} ({len(solid.data.polygons)} caras, vóxel {to_mm(context, host, voxel):.2f} mm). Revisa las paredes finas. Lo demás queda oculto.'
        self.report({'INFO'}, p.done)
        return {'FINISHED'}


# ---------------------------------------------------------------- panel
class VIEW3D_PT_part_rebuild(bpy.types.Panel):
    bl_label = 'Part Rebuild'
    bl_idname = 'VIEW3D_PT_part_rebuild'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Part Rebuild'

    def draw(self, context):
        layout = self.layout
        p = context.scene.part_rebuild_props
        if common.draw_mode_warning(layout, context):
            return
        obj = context.active_object
        host = host_of(context)
        if host is None:
            r = layout.row(); r.alert = True
            r.label(text='Selecciona la pieza', icon='ERROR')
            return

        if edit.is_part(obj):
            box = layout.box()
            box.label(text=f'Pieza nueva: {KIND_NAMES.get(obj.get(edit.PART), "")}', icon='MODIFIER')
            kind = obj.get(edit.PART)
            pr = obj.pr
            col = box.column(align=True)
            if kind in ('CYLINDER', 'SPRING', 'TUBE'):
                col.prop(pr, 'radius')
            if kind == 'SPRING':
                col.prop(pr, 'wire')
                col.prop(pr, 'turns')
            if kind in ('CYLINDER', 'SPRING', 'REVOLVE', 'EXTRUDE'):
                col.prop(pr, 'length')
            if kind in ('REVOLVE', 'EXTRUDE'):
                col.prop(pr, 'scale')
            if kind == 'TUBE':
                col.prop(pr, 'corner')
            if kind == 'SPRING':
                r = box.row(); r.prop(pr, 'closed_ends'); r.prop(pr, 'rod')
                box.row().prop(pr, 'hand', expand=True)
                box.prop(pr, 'phase')
            if kind in ('CYLINDER', 'SPRING', 'TUBE', 'REVOLVE'):
                box.prop(pr, 'segments')
            box.label(text='Muévela o gírala como cualquier objeto', icon='INFO')
            layout.separator()

        box = layout.box()
        box.label(text='1 · Marca la zona mala', icon='BRUSH_DATA')
        r = box.row(); r.prop(p, 'tool', expand=True)
        if p.tool == 'BRUSH':
            box.prop(p, 'size', slider=True)
        else:
            box.prop(p, 'angle', slider=True)
        col = box.column(); col.scale_y = 1.4
        col.operator('mesh.pr_mark', icon='BRUSH_DATA')
        n = int(canvas.load_mask(host.data).sum()) if host.data.attributes.get(canvas.MASK_ATTR) else 0
        r = box.row()
        r.label(text=f'Marcadas: {n} caras', icon='CHECKMARK' if n else 'DOT')
        if n:
            r.operator('mesh.pr_clear_mark', text='', icon='X')

        box = layout.box()
        box.label(text='2 · Arréglala', icon='MESH_CYLINDER')
        c = box.column(); c.scale_y = 1.6
        c.operator('mesh.pr_rebuild', text='Automático', icon='SHADERFX').kind = 'AUTO'
        box.label(text='o elige la forma:')
        g = box.grid_flow(columns=2, align=True)
        g.scale_y = 1.3
        g.operator('mesh.pr_rebuild', text='Muelle', icon='MOD_SCREW').kind = 'SPRING'
        g.operator('mesh.pr_rebuild', text='Cilindro', icon='MESH_CYLINDER').kind = 'CYLINDER'
        g.operator('mesh.pr_rebuild', text='Revolución', icon='MESH_TORUS').kind = 'REVOLVE'
        g.operator('mesh.pr_rebuild', text='Extrusión', icon='MOD_SOLIDIFY').kind = 'EXTRUDE'
        c = box.column(); c.scale_y = 1.3
        c.operator('mesh.pr_tube', text='Tubo (clics a lo largo)', icon='CURVE_PATH')
        box.prop(p, 'remove_old')
        sub = box.box()
        sub.label(text='Si no es ninguna forma:', icon='QUESTION')
        r = sub.row(align=True)
        r.prop(p, 'smooth', expand=True)
        sub.operator('mesh.pr_smooth', icon='MOD_SMOOTH')

        box = layout.box()
        box.label(text='3 · Copia lo bueno', icon='DUPLICATE')
        r = box.row(align=True)
        r.prop(p, 'sym_axis', expand=True)
        r.prop(p, 'sym_refine', text='', icon='VIEWZOOM')
        box.operator('mesh.pr_symmetry', icon='MOD_MIRROR')
        r = box.row(align=True)
        r.operator('mesh.pr_mold_save', icon='IMPORT')
        r.operator('mesh.pr_mold_place', icon='EXPORT')
        box.prop(p, 'clone_scale')

        box = layout.box()
        box.label(text='4 · Para imprimir', icon='MOD_REMESH')
        box.row().prop(p, 'join_mode', expand=True)
        if p.join_mode == 'FUSE':
            box.row().prop(p, 'detail', expand=True)
        box.operator('mesh.pr_join', text='Juntar todo' if p.join_mode == 'JOIN' else 'Fundir en sólido', icon='MOD_REMESH')

        if p.done:
            col = layout.column(); col.scale_y = .8
            for line in _wrap(p.done, 42):
                col.label(text=line)
        if not canvas.is_working_copy(host):
            layout.label(text='Se trabaja en una copia: el original se oculta.', icon='INFO')


def _wrap(text, width):
    words, lines, cur = text.split(), [], ''
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = (cur + ' ' + w).strip()
    if cur:
        lines.append(cur)
    return lines


classes = (PartProps, SceneProps, MESH_OT_pr_mark, MESH_OT_pr_clear_mark, MESH_OT_pr_rebuild, MESH_OT_pr_smooth, MESH_OT_pr_tube,
           MESH_OT_pr_symmetry, MESH_OT_pr_mold_save, MESH_OT_pr_mold_place, MESH_OT_pr_join, VIEW3D_PT_part_rebuild)
_handle = {'h': None}


def register():
    common.register_classes(classes, 'part_rebuild_props', SceneProps)
    bpy.types.Object.pr = bpy.props.PointerProperty(type=PartProps)
    if not bpy.app.background:
        _handle['h'] = bpy.types.SpaceView3D.draw_handler_add(_draw_mask, (), 'WINDOW', 'POST_VIEW')


def unregister():
    if _handle['h'] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle['h'], 'WINDOW')
        _handle['h'] = None
    forget_canvas()
    if hasattr(bpy.types.Object, 'pr'):
        del bpy.types.Object.pr
    common.unregister_classes(classes, 'part_rebuild_props')
