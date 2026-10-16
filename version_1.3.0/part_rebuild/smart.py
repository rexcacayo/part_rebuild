# SPDX-FileCopyrightText: 2026 Ricardo Lugaresi
# SPDX-License-Identifier: GPL-3.0-or-later

"""Marca inteligente: un clic marca la pieza entera, y una marca pintada a ojo se corrige sola.

Idea: alrededor del clic se prueba si hay una pieza "tubular" (muelle, varilla, tubo,
amortiguador): se busca el tamaño de zona en el que mejor encaja un cilindro con el eje
dentro. Después se alarga a lo largo del eje mientras la pieza siga igual y se para donde
cambia (soporte, chasis…).
"""
import numpy as np

from . import fit

DEBUG = False


def _areas(cv, faces):
    W, ls, lt, lp = cv.world, cv.loop_start, cv.loop_total, cv.loops
    out = np.zeros(len(faces))
    for k, f in enumerate(faces):
        vs = W[lp[ls[f]:ls[f] + lt[f]]]
        for i in range(1, len(vs) - 1):
            out[k] += 0.5 * np.linalg.norm(np.cross(vs[i] - vs[0], vs[i + 1] - vs[0]))
    return out


def _coverage(theta, sectors=8):
    """Cuántos sectores alrededor del eje tienen caras (0..1)."""
    s = ((theta + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors
    return len(np.unique(s)) / sectors


def _tube_fit(cv, faces):
    """Cilindro sobre unas caras y su calidad. None si no parece una pieza redonda."""
    if len(faces) < 30:
        return None
    P = cv.center[faces]
    N = cv.normals()[faces]
    try:
        cyl = fit.cylinder(P, N)
    except Exception:
        return None
    a = cyl['axis']
    side = np.abs(N @ a) < 0.35
    if side.sum() < 20:
        return None
    R = float(np.median(cyl['radial'][side]))
    rc = float(np.median(np.abs(cyl['radial'][side] - R)) / max(R, 1e-12))
    q = P - cyl['base']
    u, v, _ = fit.frame_from_axis(a)
    theta = np.arctan2(q @ v, q @ u)
    Rout = float(np.percentile(cyl['radial'][side], 90))
    return {'cyl': cyl, 'R': R, 'Rout': Rout, 'rc': rc, 'cov': _coverage(theta[side]), 'axis': a, 'base': cyl['base']}


def _connected(cv, seed, allowed):
    allowed = allowed.copy()
    allowed[seed] = True
    return cv.region(seed, allowed=allowed, same_color=False)


def select_part(cv, seed, hit, max_size):
    """Marca la pieza tubular bajo el clic. Devuelve (máscara, texto) o (None, motivo)."""
    hit = np.asarray(hit, float)
    nrm = cv.normals()[seed]
    rho = max(max_size / 400, 1e-6)
    best = None
    for _ in range(14):
        # zona centrada medio radio hacia dentro: si la pieza mide ~rho de diámetro, la
        # esfera abarca toda su vuelta y no solo la cara que se ve
        near = np.union1d(cv.faces_near(hit - nrm * (rho / 2), rho), cv.faces_near(hit, rho * 0.6))
        if len(near) > 60000:
            break
        allowed = np.zeros(len(cv.mat), bool)
        allowed[near] = True
        faces = _connected(cv, seed, allowed)
        f = _tube_fit(cv, faces)
        if f is not None:
            dist_axis = np.linalg.norm(np.cross(hit - f['base'], f['axis']))
            ok = (f['rc'] < 0.3 and f['cov'] >= 0.6 and f['Rout'] < 1.1 * rho
                  and abs(dist_axis - f['Rout']) < 0.35 * f['Rout'])
            if DEBUG:
                print(f'  rho {rho:.2f} R {f["R"]:.2f} rc {f["rc"]:.2f} cov {f["cov"]:.2f} dist {dist_axis:.2f} ok {ok}')
            if ok:
                best = (rho, f, faces)
        rho *= 1.45
        if rho > max_size / 8:
            break
    if best is None:
        return None, 'aquí no se ve una pieza redonda (muelle, varilla, tubo). Píntala y la marca se corrige sola.'
    rho, f, faces = best
    mask = grow_along_axis(cv, faces, f)
    return mask, f'pieza redonda de Ø {2 * f["Rout"]:.2f}'


def grow_along_axis(cv, faces, f, max_steps=80):
    """Alarga la marca por el eje mientras la pieza siga teniendo el mismo grosor y la
    rodee entera; se para donde cambia. Rellena todo lo que queda dentro del grosor."""
    a = f['axis']
    base = f['base']
    u, v, _ = fit.frame_from_axis(a)
    C, N = cv.center, cv.normals()
    q = C[faces] - base
    t = q @ a
    radial = np.linalg.norm(q - np.outer(t, a), axis=1)
    side = np.abs(N[faces] @ a) < 0.35
    r_out = float(np.percentile(radial[side] if side.sum() > 10 else radial, 90))
    lim = r_out * 1.18
    t0, t1 = float(t.min()), float(t.max())
    spacing = float(np.sqrt(np.mean(_areas(cv, faces[:2000])) * 2)) if len(faces) else 0.0
    step = max(r_out * 0.3, spacing * 3, 1e-6)
    # candidatas: caras cerca del eje (cálculo directo sobre todas las caras: más rápido que
    # buscar en una esfera enorme)
    qa = C - base
    ta = qa @ a
    ra = np.linalg.norm(qa - np.outer(ta, a), axis=1)
    reach = (t1 - t0) + 2 * step * max_steps
    band = (ra <= lim * 1.8) & (ta >= t0 - reach) & (ta <= t1 + reach)
    idx = np.nonzero(band)[0]
    tt_all, rr_all = ta[idx], ra[idx]
    inside = rr_all <= lim
    near, tt, rr = idx[inside], tt_all[inside], rr_all[inside]
    qn = qa[near]
    th = np.arctan2(qn @ v, qn @ u)
    nside = np.abs(N[near] @ a) < 0.35
    # caras un poco más allá del grosor: si en un tramo hay muchas, algo más ancho se une ahí
    tw = tt_all[~inside]

    def slab_ok(lo, hi):
        s = (tt >= lo) & (tt < hi)
        if s.sum() < 8:
            return False
        ss = s & nside
        if ss.sum() < 5:
            return False
        env = np.percentile(rr[ss], 90)
        outside = ((tw >= lo) & (tw < hi)).sum()
        if outside > 0.25 * s.sum():
            return False
        return abs(env - r_out) < 0.2 * r_out and _coverage(th[ss]) >= 0.6

    for _ in range(max_steps):
        grew = False
        if slab_ok(t1, t1 + step):
            t1 += step
            grew = True
        if slab_ok(t0 - step, t0):
            t0 -= step
            grew = True
        if not grew:
            break
    keep = near[(tt >= t0 - step * 0.25) & (tt <= t1 + step * 0.25)]
    allowed = np.zeros(len(cv.mat), bool)
    allowed[keep] = True
    allowed[faces] = True
    seed = int(faces[0])
    region = _connected(cv, seed, allowed)
    mask = np.zeros(len(cv.mat), bool)
    mask[region] = True
    mask[faces] = True
    return mask


# --------------------------------------------------------------------------- corregir una marca pintada
def _face_vertex_ops(cv):
    n = len(cv.loop_start)
    face_of_loop = np.repeat(np.arange(n), cv.loop_total)
    return face_of_loop, cv.loops


def _dilate(cv, mask, rings=1):
    fol, lp = _face_vertex_ops(cv)
    m = mask.copy()
    for _ in range(rings):
        verts = np.zeros(len(cv.world), bool)
        verts[lp[m[fol]]] = True
        grow = np.zeros(len(m), bool)
        grow[fol[verts[lp]]] = True
        m |= grow
    return m


def _erode(cv, mask, rings=1):
    fol, lp = _face_vertex_ops(cv)
    m = mask.copy()
    for _ in range(rings):
        verts_out = np.zeros(len(cv.world), bool)
        verts_out[lp[~m[fol]]] = True
        touch = np.zeros(len(m), bool)
        touch[fol[verts_out[lp]]] = True
        m &= ~touch
    return m


def refine_mask(cv, mask):
    """Corrige una marca pintada a ojo. Devuelve (máscara nueva, texto con lo que ha hecho)."""
    n0 = int(mask.sum())
    if n0 == 0:
        return mask, ''
    m = mask.copy()
    # 1) quitar trozos sueltos: se queda el trozo grande (y los que sean al menos la mitad)
    left = m.copy()
    parts = []
    for _ in range(50):
        if not left.any():
            break
        seed = int(np.nonzero(left)[0][0])
        reg = cv.region(seed, allowed=m, same_color=False)
        parts.append(reg)
        left[reg] = False
    parts.sort(key=len, reverse=True)
    keep = np.zeros_like(m)
    for reg in parts:
        if len(reg) >= 0.5 * len(parts[0]):
            keep[reg] = True
    loose = int(m.sum() - keep.sum())
    m = keep
    # 2) rellenar huecos pequeños (cerrar)
    rings = 2
    closed = _erode(cv, _dilate(cv, m, rings), rings) | m
    filled = int(closed.sum() - m.sum())
    m = closed
    notes = []
    # 3) si es una pieza redonda: recortar lo que sobresale y alargar hasta los extremos
    faces = np.nonzero(m)[0]
    f = _tube_fit(cv, faces)
    if f is not None and f['rc'] < 0.45 and f['cov'] >= 0.6:
        q = cv.center[faces] - f['base']
        t = q @ f['axis']
        radial = np.linalg.norm(q - np.outer(t, f['axis']), axis=1)
        side = np.abs(cv.normals()[faces] @ f['axis']) < 0.35
        r_out = float(np.percentile(radial[side] if side.sum() > 10 else radial, 90))
        trimmed = faces[radial <= r_out * 1.25]
        cut = len(faces) - len(trimmed)
        grown = grow_along_axis(cv, trimmed, f)
        added = int((grown & ~m).sum())
        m = grown
        if cut:
            notes.append(f'{cut} de más quitadas')
        if added:
            notes.append(f'{added} añadidas hasta los extremos')
    if loose:
        notes.insert(0, f'{loose} sueltas quitadas')
    if filled:
        notes.append(f'{filled} huecos rellenados')
    return m, ', '.join(notes) if notes else 'la marca ya estaba bien'
