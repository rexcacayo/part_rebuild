"""Operaciones sobre la pieza: crear piezas reconstruidas, quitar la zona vieja, simetría,
clonar y unir en sólido. Todo sin bpy.ops y en modo Objeto."""
import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from . import build, canvas, fit
from .common import AddonError

PART = 'pr_tipo'          # en las piezas reconstruidas: tipo
HOST = 'pr_pieza'         # nombre de la pieza a la que pertenecen
MOLD = 'pr_molde'


# --------------------------------------------------------------------------- datos de la malla
def face_data(obj, mask=None):
    """Centros, normales y áreas de las caras en coordenadas de mundo."""
    me = obj.data
    n = len(me.polygons)
    c = np.empty(n * 3); me.polygons.foreach_get('center', c)
    nr = np.empty(n * 3); me.polygons.foreach_get('normal', nr)
    ar = np.empty(n); me.polygons.foreach_get('area', ar)
    m = np.array(obj.matrix_world)
    c = c.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
    ninv = np.array(obj.matrix_world.to_3x3().inverted().transposed())
    nr = nr.reshape(-1, 3) @ ninv.T
    nr /= np.maximum(np.linalg.norm(nr, axis=1, keepdims=True), 1e-12)
    scale = abs(np.linalg.det(m[:3, :3])) ** (2 / 3)
    ar = ar * scale
    if mask is not None:
        return c[mask], nr[mask], ar[mask]
    return c, nr, ar


def face_tris(obj, mask):
    """Triángulos (en mundo) de las caras marcadas."""
    me = obj.data
    n = len(me.polygons)
    ls = np.empty(n, np.int64); me.polygons.foreach_get('loop_start', ls)
    lt = np.empty(n, np.int64); me.polygons.foreach_get('loop_total', lt)
    loops = np.empty(len(me.loops), np.int64); me.loops.foreach_get('vertex_index', loops)
    co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', co)
    m = np.array(obj.matrix_world)
    co = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
    out = []
    for f in np.nonzero(mask)[0]:
        vs = loops[ls[f]:ls[f] + lt[f]]
        for k in range(1, len(vs) - 1):
            out.append((vs[0], vs[k], vs[k + 1]))
    return co[np.array(out)] if out else np.zeros((0, 3, 3))


def main_material(obj, mask):
    me = obj.data
    if not len(me.materials):
        return None
    mi = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get('material_index', mi)
    vals = mi[mask]
    if not len(vals):
        return None
    idx = int(np.bincount(vals).argmax())
    return me.materials[idx] if idx < len(me.materials) else None


def get_mask(host, minimum=12):
    mask = canvas.load_mask(host.data)
    if mask.sum() < minimum:
        raise AddonError('Marca primero la zona con «Marcar zona» (pinta encima de la pieza mala).')
    return mask


# --------------------------------------------------------------------------- piezas reconstruidas
def frame_matrix(origin, axis, u=None):
    a = Vector(axis).normalized()
    if u is None:
        uu, vv, _ = fit.frame_from_axis(np.array(a))
        u = Vector(uu)
    u = Vector(u)
    u = (u - a * u.dot(a)).normalized()
    v = a.cross(u)
    M = Matrix((u, v, a)).transposed().to_4x4()
    M.translation = Vector(origin)
    return M


def set_geometry(obj, VF):
    V, F = VF
    me = obj.data
    mats = list(me.materials)
    new = bpy.data.meshes.new(me.name)
    new.from_pydata(np.asarray(V).tolist(), [], [list(map(int, f)) for f in F])
    new.validate()
    if obj.get(PART) != 'EXTRUDE':           # lados suaves, tapas planas
        lt = np.empty(len(new.polygons), np.int32)
        new.polygons.foreach_get('loop_total', lt)
        new.polygons.foreach_set('use_smooth', lt == 4)
    for m in mats:
        new.materials.append(m)
    obj.data = new
    if me.users == 0:
        bpy.data.meshes.remove(me)


def generate(obj):
    """Vuelve a crear la malla de una pieza a partir de sus ajustes (deslizadores)."""
    p = obj.pr
    kind = obj.get(PART)
    if kind == 'CYLINDER':
        VF = build.cylinder(p.radius, p.length, seg=p.segments)
    elif kind == 'SPRING':
        VF = build.spring(p.radius, p.wire, p.length, p.turns, closed_ends=p.closed_ends,
                          hand=1 if p.hand == 'RIGHT' else -1, phase=p.phase, seg=max(8, p.segments // 4),
                          rod=p.radius * 0.38 if p.rod else 0.0)
    elif kind == 'TUBE':
        pts = np.array(obj.get('pr_puntos', [])).reshape(-1, 3)
        if len(pts) < 2:
            return
        VF = build.tube(pts, p.radius, corner=p.corner, seg=max(12, p.segments // 2))
    elif kind == 'REVOLVE':
        z = np.array(obj.get('pr_perfil_z', []))
        r = np.array(obj.get('pr_perfil_r', []))
        if len(z) < 2:
            return
        VF = build.revolve(z * p.length / max(z[-1], 1e-12), r * p.scale, seg=p.segments)
    elif kind == 'EXTRUDE':
        xy = np.array(obj.get('pr_contorno', [])).reshape(-1, 2)
        if len(xy) < 3:
            return
        VF = build.extrude(xy * p.scale, p.length)
    else:
        return
    set_geometry(obj, VF)


def new_part(context, host, kind, VF, matrix, material, name, settings):
    col = host.users_collection[0] if host.users_collection else context.scene.collection
    me = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, me)
    col.objects.link(obj)
    obj.matrix_world = matrix
    obj[PART] = kind
    obj[HOST] = host.name
    if material is not None:
        me.materials.append(material)
    obj.pr.lock = True
    for k, v in settings.items():
        setattr(obj.pr, k, v)
    obj.pr.lock = False
    set_geometry(obj, VF)
    for o in context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    return obj


def is_part(obj):
    return obj is not None and obj.get(PART) is not None


# --------------------------------------------------------------------------- quitar la zona vieja
def remove_faces(host, mask, fill=True):
    """Borra las caras marcadas de la pieza y tapa los agujeros que quedan."""
    me = host.data
    bm = bmesh.new()
    bm.from_mesh(me)
    try:
        bm.faces.ensure_lookup_table()
        old_boundary = {e.index for e in bm.edges if e.is_boundary}
        doomed = [bm.faces[i] for i in np.nonzero(mask)[0]]
        bmesh.ops.delete(bm, geom=doomed, context='FACES')
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context='VERTS')
        filled = 0
        if fill:
            bm.edges.index_update()
            edges = [e for e in bm.edges if e.is_boundary]
            if edges:
                res = bmesh.ops.holes_fill(bm, edges=edges, sides=0)
                new_faces = res.get('faces', [])
                if new_faces:
                    bmesh.ops.triangulate(bm, faces=new_faces)
                filled = len(new_faces)
        bm.to_mesh(me)
    finally:
        bm.free()
    canvas.clear_mask(me)
    me.update()
    return filled


# --------------------------------------------------------------------------- suavizar zona
def smooth_region(host, mask, iterations=15):
    """Suaviza solo la zona marcada (Taubin: quita grumos sin encoger). El borde de la zona
    no se mueve, así empalma con lo que la rodea."""
    me = host.data
    n = len(me.polygons)
    lt = np.empty(n, np.int64); me.polygons.foreach_get('loop_total', lt)
    loops = np.empty(len(me.loops), np.int64); me.loops.foreach_get('vertex_index', loops)
    face_of_loop = np.repeat(np.arange(n), lt)
    nv = len(me.vertices)
    inside = np.zeros(nv, bool)
    inside[loops[mask[face_of_loop]]] = True
    outside = np.zeros(nv, bool)
    outside[loops[~mask[face_of_loop]]] = True
    move = inside & ~outside
    if not move.any():
        raise AddonError('La zona marcada es demasiado pequeña para suavizar.')
    ev = np.empty(len(me.edges) * 2, np.int64); me.edges.foreach_get('vertices', ev)
    ev = ev.reshape(-1, 2)
    keep = inside[ev[:, 0]] | inside[ev[:, 1]]
    a, b = ev[keep, 0], ev[keep, 1]
    deg = np.bincount(np.concatenate([a, b]), minlength=nv).astype(float)
    co = np.empty(nv * 3); me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    idx = np.nonzero(move)[0]
    for k in range(iterations * 2):
        lam = 0.5 if k % 2 == 0 else -0.53
        s = np.zeros_like(co)
        np.add.at(s, a, co[b])
        np.add.at(s, b, co[a])
        avg = s[idx] / np.maximum(deg[idx], 1)[:, None]
        co[idx] += lam * (avg - co[idx])
    me.vertices.foreach_set('co', co.ravel())
    me.update()
    return int(move.sum())


# --------------------------------------------------------------------------- simetría
def mirror_replace(host, mask, axis_index, refine=True):
    """Sustituye la zona marcada (lado malo) por la copia reflejada del lado bueno."""
    me = host.data
    centers, _n, _a = face_data(host)
    bb = np.array([list(host.matrix_world @ Vector(c)) for c in host.bound_box])
    center = (bb.min(0) + bb.max(0)) / 2
    rng = np.random.default_rng(1)
    sample = centers[rng.choice(len(centers), min(len(centers), 6000), replace=False)]
    point, normal = fit.symmetry_plane(sample, axis_index, center, refine=refine)
    S = np.eye(3) - 2 * np.outer(normal, normal)

    def reflect(P):
        return (S @ (np.asarray(P) - point).T).T + point
    bad = centers[mask]
    side_bad = np.sign(np.median((bad - point) @ normal))
    # candidatas: caras del otro lado cuya imagen cae dentro de la zona mala
    lo, hi = bad.min(0), bad.max(0)
    pad = (hi - lo).max() * 0.05 + 1e-6
    img = reflect(centers)
    cand = np.nonzero(~mask & np.all((img >= lo - pad) & (img <= hi + pad), axis=1)
                      & (np.sign((centers - point) @ normal) == -side_bad))[0]
    if not len(cand):
        raise AddonError('No hay nada al otro lado del plano: prueba con otro eje de simetría.')
    # Para cada cara mala, la cara buena que le corresponde al reflejarla (así no quedan huecos),
    # y luego dos anillos más alrededor para que solape con lo que rodea a la zona.
    kd = KDTree(len(cand))
    for i, p in enumerate(centers[cand]):
        kd.insert(p, i)
    kd.balance()
    rbad = reflect(bad)
    picked = set()
    for p in rbad:
        _co, i, _d = kd.find(Vector(p))
        if i is not None:
            picked.add(int(cand[i]))
    if not picked:
        raise AddonError('El otro lado no coincide con la zona marcada: revisa el eje de simetría.')
    allowed = np.zeros(len(centers), bool)
    allowed[cand] = True
    n = len(me.polygons)
    ls = np.empty(n, np.int64); me.polygons.foreach_get('loop_start', ls)
    lt = np.empty(n, np.int64); me.polygons.foreach_get('loop_total', lt)
    loops = np.empty(len(me.loops), np.int64); me.loops.foreach_get('vertex_index', loops)
    face_of_loop = np.repeat(np.arange(n), lt)
    sel = np.zeros(n, bool)
    sel[list(picked)] = True
    for _ in range(2):
        verts = np.zeros(len(me.vertices), bool)
        verts[loops[sel[face_of_loop]]] = True
        grow = np.zeros(n, bool)
        grow[face_of_loop[verts[loops]]] = True
        sel |= grow & allowed
    take = np.nonzero(sel)[0].tolist()
    ar = np.empty(n); me.polygons.foreach_get('area', ar)
    edge = float(np.sqrt(np.median(ar[mask]) * 2)) * abs(np.linalg.det(np.array(host.matrix_world)[:3, :3])) ** (1 / 3)
    mw = host.matrix_world
    inv = mw.inverted()
    bm = bmesh.new()
    bm.from_mesh(me)
    try:
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        src = [bm.faces[i] for i in take]
        doomed = [bm.faces[i] for i in np.nonzero(mask)[0]]
        vmap = {}
        new_faces = []
        for f in src:
            vs = []
            for v in f.verts:
                if v.index not in vmap:
                    w = np.array(mw @ v.co)
                    co = inv @ Vector(reflect(w[None])[0])
                    vmap[v.index] = bm.verts.new(co)
                vs.append(vmap[v.index])
            try:
                nf = bm.faces.new(list(reversed(vs)))
                nf.material_index = f.material_index
                new_faces.append(nf)
            except ValueError:
                pass
        bmesh.ops.delete(bm, geom=doomed, context='FACES')
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context='VERTS')
        border = list({v for e in bm.edges if e.is_boundary for v in e.verts})
        if border:
            bmesh.ops.remove_doubles(bm, verts=border, dist=edge * 0.6)
        bm.to_mesh(me)
    finally:
        bm.free()
    canvas.clear_mask(me)
    me.update()
    return len(new_faces), normal


# --------------------------------------------------------------------------- clonar
def save_mold(context, host, mask):
    """Copia la zona marcada como molde (pieza cerrada y oculta)."""
    me = host.data
    bm = bmesh.new()
    bm.from_mesh(me)
    try:
        bm.faces.ensure_lookup_table()
        keep = set(np.nonzero(mask)[0].tolist())
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.index not in keep], context='FACES')
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context='VERTS')
        edges = [e for e in bm.edges if e.is_boundary]
        if edges:
            res = bmesh.ops.holes_fill(bm, edges=edges, sides=0)
            if res.get('faces'):
                bmesh.ops.triangulate(bm, faces=res['faces'])
        bm.transform(host.matrix_world)
        data = bpy.data.meshes.new(host.name + '_molde')
        bm.to_mesh(data)
    finally:
        bm.free()
    for m in me.materials:
        data.materials.append(m)
    col = host.users_collection[0] if host.users_collection else context.scene.collection
    old = [o for o in bpy.data.objects if o.get(MOLD) == host.name]
    for o in old:
        bpy.data.objects.remove(o, do_unlink=True)
    mold = bpy.data.objects.new(host.name + '_molde', data)
    col.objects.link(mold)
    mold[MOLD] = host.name
    mold.hide_set(True)
    mold.hide_render = True
    canvas.clear_mask(me)
    return mold


def place_mold(context, host, mask, scale=False):
    mold = next((o for o in bpy.data.objects if o.get(MOLD) == host.name), None)
    if mold is None:
        raise AddonError('Guarda primero la pieza buena como molde.')
    src = np.array([list(v.co) for v in mold.data.vertices])
    target, _n, _a = face_data(host, mask)
    M, err = fit.icp(src, target, scale=scale)
    col = host.users_collection[0] if host.users_collection else context.scene.collection
    data = mold.data.copy()
    obj = bpy.data.objects.new(host.name + '_clon', data)
    col.objects.link(obj)
    obj.matrix_world = Matrix(M.tolist())
    obj[PART] = 'CLONE'
    obj[HOST] = host.name
    return obj, err
