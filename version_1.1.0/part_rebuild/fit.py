"""Medir piezas a partir de puntos y normales de la zona marcada (numpy puro).

Todas las funciones trabajan en coordenadas de mundo y devuelven diccionarios con lo
medido y un `error` (desviación típica en unidades de Blender) para que el usuario vea
si la forma encaja.
"""
import math

import numpy as np

from .common import AddonError


# --------------------------------------------------------------------------- utilidades
def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def frame_from_axis(axis):
    """Dos vectores perpendiculares al eje (base ortonormal u, v, eje)."""
    a = _unit(axis)
    helper = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = _unit(np.cross(a, helper))
    v = np.cross(a, u)
    return u, v, a


def circle_fit_2d(xy, w=None, iterations=4):
    """Círculo por mínimos cuadrados (Kåsa) con descarte iterativo de puntos raros."""
    xy = np.asarray(xy, float)
    w = np.ones(len(xy)) if w is None else np.asarray(w, float)
    keep = np.ones(len(xy), bool)
    cx = cy = r = 0.0
    for _ in range(iterations):
        p, ww = xy[keep], w[keep]
        if len(p) < 3:
            break
        A = np.column_stack([p[:, 0], p[:, 1], np.ones(len(p))])
        b = (p ** 2).sum(1)
        sw = np.sqrt(ww)
        sol, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
        cx, cy = sol[0] / 2, sol[1] / 2
        r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-18))
        res = np.abs(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r)
        mad = np.median(res[keep]) + 1e-12
        keep = res < 3.0 * mad * 1.4826 + 1e-9
    return np.array([cx, cy]), r


def axis_from_normals(normals, weights=None):
    """Eje de una pieza alargada (cilindro, muelle, prisma): la dirección a la que las
    normales de los lados son perpendiculares = vector propio de menor valor."""
    n = np.asarray(normals, float)
    w = np.ones(len(n)) if weights is None else np.asarray(weights, float)
    M = (n * w[:, None]).T @ n
    vals, vecs = np.linalg.eigh(M)
    return vecs[:, 0], vals


# --------------------------------------------------------------------------- cilindro
def cylinder(points, normals, weights=None):
    """Cilindro que mejor encaja. Devuelve base (punto del eje en el extremo inicial),
    eje, radio, largo y error."""
    P = np.asarray(points, float)
    if len(P) < 12:
        raise AddonError('Marca más superficie: hacen falta al menos una docena de caras.')
    o = P.mean(0)
    q = P - o
    # Candidatos de eje: por las normales (sirve en cilindros cortos) y por la forma de la
    # nube (sirve en muelles, cuyas espiras tienen normales también a lo largo del eje).
    _vals, nvecs = np.linalg.eigh((np.asarray(normals, float) * (np.ones(len(P)) if weights is None else np.asarray(weights, float))[:, None]).T @ np.asarray(normals, float))
    _pv, pvecs = np.linalg.eigh(np.cov(q.T))
    best = None
    for axis in (nvecs[:, 0], pvecs[:, 2], nvecs[:, 1], nvecs[:, 2], pvecs[:, 0], pvecs[:, 1]):
        u, v, a = frame_from_axis(axis)
        xy = np.column_stack([q @ u, q @ v])
        c2, r = circle_fit_2d(xy, weights)
        res = np.abs(np.hypot(xy[:, 0] - c2[0], xy[:, 1] - c2[1]) - r)
        score = np.median(res) / max(r, 1e-12)
        if r > 0 and (best is None or score < best[0]):
            best = (score, u, v, a, c2)
    _s, u, v, a, c2 = best
    # Centro afinado con las normales: en un cilindro (y en todo lo concéntrico: muelle,
    # vástago...) las rectas normales pasan por el eje. No se sesga si solo se ve medio arco.
    N = np.asarray(normals, float)
    w = np.ones(len(P)) if weights is None else np.asarray(weights, float)
    nxy = np.column_stack([N @ u, N @ v])
    ln = np.linalg.norm(nxy, axis=1)
    radialish = ln > 0.7
    if radialish.sum() >= 12:
        d = nxy[radialish] / ln[radialish, None]
        pxy = np.column_stack([q @ u, q @ v])[radialish]
        ww = w[radialish]
        c = c2.copy()
        for _ in range(6):
            A = np.zeros((2, 2)); b = np.zeros(2)
            Pr = np.eye(2)[None] - d[:, :, None] * d[:, None, :]
            A = (Pr * ww[:, None, None]).sum(0)
            b = np.einsum('nij,nj,n->i', Pr, pxy, ww)
            c = np.linalg.solve(A + 1e-12 * np.eye(2), b)
            off = pxy - c
            dist = np.abs(off[:, 0] * d[:, 1] - off[:, 1] * d[:, 0])
            mad = np.median(dist) * 1.4826 + 1e-12
            ww = w[radialish] * (dist < 3 * mad)
        c2 = c
    center = o + c2[0] * u + c2[1] * v
    t = (P - center) @ a
    t0, t1 = np.percentile(t, 0.5), np.percentile(t, 99.5)
    radial = np.linalg.norm((P - center) - np.outer(t, a), axis=1)
    r_med = float(np.median(radial))
    err = float(np.sqrt(np.mean((radial - r_med) ** 2)))
    return {'base': center + t0 * a, 'axis': a, 'radius': r_med, 'length': float(t1 - t0),
            'error': err, 'radial': radial, 't': t - t0}


# --------------------------------------------------------------------------- muelle
def spring(points, normals, weights=None):
    """Muelle: eje y radio como un cilindro; grosor del hilo por el reparto radial y paso
    de espira buscando la periodicidad a lo largo del eje (fase de hélice o anillos)."""
    cyl = cylinder(points, normals, weights)
    P = np.asarray(points, float)
    a = cyl['axis']
    base = cyl['base']
    radial, t = cyl['radial'], cyl['t']
    L = cyl['length']
    u, v, _a = frame_from_axis(a)
    q = P - base
    N = np.asarray(normals, float)
    rdir = q - np.outer(q @ a, a)
    rdir /= np.maximum(np.linalg.norm(rdir, axis=1, keepdims=True), 1e-12)
    facing = np.einsum('ij,ij->i', N, rdir)
    out_face = facing > 0.6                 # caras que miran hacia fuera del eje
    in_face = facing < -0.6                 # caras que miran hacia el eje (interior del hilo)
    r_out = float(np.percentile(radial[out_face], 90)) if out_face.sum() > 20 else float(np.percentile(radial, 95))
    inner = in_face & (radial > 0.3 * r_out)
    if inner.sum() > 20:
        r_in = float(np.percentile(radial[inner], 50))
        wire = r_out - r_in
    else:
        wire = 0.25 * r_out
    wire = float(np.clip(wire, 0.08 * r_out, 0.5 * r_out))
    theta = np.arctan2(q @ v, q @ u)
    pitch, conf = _pitch_from_envelope(t, theta, radial, facing > 0.3, L, wire)
    hand, phase = 1, 0.0
    band = out_face & (radial > r_out - wire)
    if band.sum() < 50:
        band = radial > r_out - wire
    tb, thb = t[band], theta[band]
    if pitch is not None and len(tb) > 50:
        best = (-1.0, pitch, 1, 0.0)
        for hd in (1, -1, 0):
            for p in np.linspace(pitch * 0.96, pitch * 1.04, 21):
                z = np.exp(1j * (2 * np.pi * tb / p - hd * thb)).mean()
                if abs(z) > best[0]:
                    best = (abs(z), p, hd, float(np.angle(z)))
        _sc, pitch, hand, phase = best
    if pitch is None:
        pitch = 1.8 * wire
    else:
        wire = min(wire, 0.75 * pitch)      # que se vean las espiras separadas
    score = conf
    confident = conf >= 0.12
    turns = max(1.0, L / pitch)
    return {'base': base, 'axis': a, 'u': u, 'radius': r_out, 'wire': wire, 'length': L,
            'turns': float(turns), 'pitch': float(pitch), 'hand': -1 if hand == -1 else 1,
            'rings': hand == 0, 'phase': phase, 'confidence': float(score), 'confident': confident,
            'error': cyl['error']}


def _pitch_from_envelope(t, theta, radial, outward, L, wire, bins=400, sectors=8):
    """Paso de espira: el borde exterior del muelle sube y baja con cada espira. Se mira
    ese borde en 8 sectores alrededor del eje y se busca su periodo (autocorrelación).
    Funciona con espiras en hélice y con espiras fundidas en anillos."""
    if L <= 0 or outward.sum() < 50:
        return None, 0.0
    dt = L / bins
    bi = np.clip((t / L * bins).astype(int), 0, bins - 1)
    si = ((theta + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors
    env = np.full(sectors * bins, -np.inf)
    sel = np.nonzero(outward)[0]
    np.maximum.at(env, si[sel] * bins + bi[sel], radial[sel])
    env = env.reshape(sectors, bins)
    ac = np.zeros(bins)
    used = 0
    x = np.arange(bins)
    win = max(3, int(3 * wire / dt))
    for e in env:
        ok = np.isfinite(e)
        if ok.sum() < bins * 0.3:
            continue
        e = np.interp(x, x[ok], e[ok])
        ma = np.convolve(np.pad(e, win, mode='edge'), np.ones(2 * win + 1) / (2 * win + 1), 'valid')
        e = e - ma
        e -= e.mean()
        f = np.fft.rfft(e, 2 * bins)
        a = np.fft.irfft(np.abs(f) ** 2)[:bins]
        if a[0] > 0:
            ac += a / a[0]
            used += 1
    if not used:
        return None, 0.0
    ac /= used
    lo, hi = max(1, int(0.6 * wire / dt)), int(L / 3 / dt)
    if hi <= lo + 2:
        return None, 0.0
    peaks = [i for i in range(lo + 1, hi - 1) if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1]]
    if not peaks:
        return None, 0.0
    i = max(peaks, key=lambda k: ac[k])
    # afinado parabólico del pico
    y0, y1, y2 = ac[i - 1], ac[i], ac[i + 1]
    den = y0 - 2 * y1 + y2
    off = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    return float((i + off) * dt), float(ac[i])


# --------------------------------------------------------------------------- tubo (centro bajo un clic)
def tube_center(tree, hit, normal, max_depth):
    """Centro del tubo bajo el punto tocado: rayo hacia dentro hasta la pared opuesta."""
    from mathutils import Vector
    n = Vector(normal).normalized()
    start = Vector(hit) - n * (max_depth * 1e-4 + 1e-6)
    loc, _nn, _i, dist = tree.ray_cast(start, -n, max_depth)
    if loc is None:
        return None, None
    d = dist + max_depth * 1e-4
    return np.array(Vector(hit) - n * (d / 2)), d / 2


# --------------------------------------------------------------------------- revolución
def revolution_axis(points, normals, weights=None):
    """Eje de una superficie de revolución: todas las rectas normales lo cortan
    (complejo lineal de rectas, Pottmann & Randrup)."""
    P = np.asarray(points, float)
    N = np.array([_unit(n) for n in normals]) if len(normals) < 10 else \
        np.asarray(normals, float) / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    o = P.mean(0)
    scale = np.linalg.norm(P - o, axis=1).mean() + 1e-12
    Q = (P - o) / scale
    w = np.ones(len(P)) if weights is None else np.asarray(weights, float)
    V = np.hstack([np.cross(Q, N), N])              # coeficientes de (a, a_barra)
    M = (V * w[:, None]).T @ V
    vals, vecs = np.linalg.eigh(M)
    x = vecs[:, 0]
    a, abar = x[:3], x[3:]
    na = np.linalg.norm(a)
    if na < 1e-6:
        raise AddonError('No parece una pieza de revolución (no hay un eje claro).')
    point = np.cross(a, abar) / (na * na)
    a = a / na
    return o + point * scale, a


def revolution(points, normals, weights=None, bins=96):
    """Perfil exterior alrededor del eje: radio máximo (percentil 97) por tramo de altura."""
    P = np.asarray(points, float)
    o, a = revolution_axis(points, normals, weights)
    t = (P - o) @ a
    rho = np.linalg.norm((P - o) - np.outer(t, a), axis=1)
    t0, t1 = float(t.min()), float(t.max())
    if t1 - t0 < 1e-9:
        raise AddonError('La zona marcada es plana: no se puede hacer una pieza de revolución.')
    edges = np.linspace(t0, t1, bins + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, bins - 1)
    prof = np.full(bins, np.nan)
    for i in range(bins):
        sel = rho[idx == i]
        if len(sel):
            prof[i] = np.percentile(sel, 97)
    ok = ~np.isnan(prof)
    if ok.sum() < 3:
        raise AddonError('Hay muy pocas caras marcadas para sacar el perfil.')
    centers = (edges[:-1] + edges[1:]) / 2
    prof = np.interp(centers, centers[ok], prof[ok])
    k = np.array([1, 2, 3, 2, 1], float); k /= k.sum()
    smooth = np.convolve(np.pad(prof, 2, mode='edge'), k, mode='valid')
    # error: distancia de los puntos de la "piel" exterior al perfil
    model = np.interp(t, centers, smooth)
    skin = rho > model * 0.8
    err = float(np.sqrt(np.mean((rho[skin] - model[skin]) ** 2))) if skin.any() else 0.0
    return {'base': o + t0 * a, 'axis': a, 'z': centers - t0, 'rho': smooth, 'length': t1 - t0, 'error': err}


# --------------------------------------------------------------------------- extrusión
def _extrusion_direction(normals, weights):
    n = np.asarray(normals, float)
    w = np.ones(len(n)) if weights is None else np.asarray(weights, float)
    _vals, vecs = np.linalg.eigh((n * w[:, None]).T @ n)
    best, best_score = vecs[:, 0], -1
    for k in range(3):
        e = vecs[:, k]
        d = np.abs(n @ e)
        score = (w * ((d < 0.2) | (d > 0.9))).sum() / w.sum() + 0.5 * (w * (d < 0.2)).sum() / w.sum()
        if score > best_score:
            best, best_score = e, score
    return best


def _trace(img):
    """Contorno exterior del mayor grupo de píxeles (vecindad de Moore). Devuelve [(fila, col)]."""
    H, W = img.shape
    rows, cols = np.nonzero(img)
    if not len(rows):
        return []
    start = (rows[0], cols[np.argmin(cols[rows == rows[0]]) if False else 0])
    r0 = rows.min()
    c0 = cols[rows == r0].min()
    start = (r0, c0)
    dirs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    contour = [start]
    cur, back = start, 6            # venimos de la izquierda
    for _ in range(4 * H * W):
        found = False
        for k in range(8):
            d = (back + 1 + k) % 8
            nr, nc = cur[0] + dirs[d][0], cur[1] + dirs[d][1]
            if 0 <= nr < H and 0 <= nc < W and img[nr, nc]:
                back = (d + 4) % 8
                cur = (nr, nc)
                found = True
                break
        if not found or cur == start:
            break
        contour.append(cur)
    return contour


def _simplify(pts, tol):
    """Douglas-Peucker en un polígono cerrado."""
    pts = np.asarray(pts, float)
    if len(pts) < 4:
        return pts

    def rdp(seg):
        if len(seg) < 3:
            return seg
        a, b = seg[0], seg[-1]
        ab = b - a
        L = np.linalg.norm(ab)
        if L < 1e-12:
            d = np.linalg.norm(seg - a, axis=1)
        else:
            d = np.abs(np.cross(ab, seg - a)) / L
        i = int(np.argmax(d))
        if d[i] > tol:
            return np.vstack([rdp(seg[:i + 1])[:-1], rdp(seg[i:])])
        return np.vstack([a, b])
    far = int(np.argmax(np.linalg.norm(pts - pts[0], axis=1)))
    first = rdp(pts[:far + 1])
    second = rdp(np.vstack([pts[far:], pts[:1]]))
    out = np.vstack([first[:-1], second[:-1]])
    return out


def sample_triangles(tris, step, limit=400000, seed=0, return_index=False):
    """Puntos repartidos por la superficie de los triángulos (unos por cada `step`²)."""
    T = np.asarray(tris, float)
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    count = np.clip((area / max(step * step, 1e-18) * 2).astype(int), 1, None)
    if count.sum() > limit:
        count = np.maximum(1, (count * limit / count.sum()).astype(int))
    rng = np.random.default_rng(seed)
    idx = np.repeat(np.arange(len(T)), count)
    r1, r2 = rng.random(len(idx)), rng.random(len(idx))
    s = np.sqrt(r1)
    pts = (1 - s)[:, None] * a[idx] + (s * (1 - r2))[:, None] * b[idx] + (s * r2)[:, None] * c[idx]
    return (pts, idx) if return_index else pts


def extrusion(points, normals, weights=None, resolution=220, tris=None):
    """Prisma: dirección de extrusión, contorno 2D de la sección y largo.
    Con `tris` (triángulos de la zona) la sección se rellena mejor que solo con centros."""
    P = np.asarray(points, float)
    d = _extrusion_direction(normals, weights)
    u, v, a = frame_from_axis(d)
    o = P.mean(0)
    q = P - o
    x, y, t = q @ u, q @ v, q @ a
    lo = np.array([x.min(), y.min()])
    span = max(x.max() - x.min(), y.max() - y.min())
    if span < 1e-9:
        raise AddonError('La zona marcada es demasiado pequeña.')
    px = span / resolution
    if tris is not None and len(tris):
        S = sample_triangles(tris, px * 0.7) - o
        x = np.concatenate([x, S @ u])
        y = np.concatenate([y, S @ v])
    W = int((x.max() - lo[0]) / px) + 7
    H = int((y.max() - lo[1]) / px) + 7
    img = np.zeros((H, W), bool)
    ci = ((x - lo[0]) / px).astype(int) + 3
    ri = ((y - lo[1]) / px).astype(int) + 3
    img[ri, ci] = True

    def dilate(m, k=1):
        out = m.copy()
        for dr in range(-k, k + 1):
            for dc in range(-k, k + 1):
                out |= np.roll(np.roll(m, dr, 0), dc, 1)
        return out

    def erode(m, k=1):
        return ~dilate(~m, k)
    img = erode(dilate(img, 3), 3)
    # rellenar agujeros: lo que no se alcanza desde el borde es interior
    outside = np.zeros_like(img)
    outside[0, :] = outside[-1, :] = outside[:, 0] = outside[:, -1] = True
    outside &= ~img
    for _ in range(H + W):
        grown = dilate(outside) & ~img
        if (grown == outside).all():
            break
        outside = grown
    solid = ~outside
    contour = _trace(solid)
    if len(contour) < 4:
        raise AddonError('No se ha podido sacar el contorno de la sección.')
    c = np.array(contour, float)
    xy = np.column_stack([(c[:, 1] - 3 + 0.5) * px + lo[0], (c[:, 0] - 3 + 0.5) * px + lo[1]])
    xy = _simplify(xy, px * 2.0)
    # sentido antihorario
    area = 0.5 * np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
    if area < 0:
        xy = xy[::-1]
    t0, t1 = float(np.percentile(t, 0.5)), float(np.percentile(t, 99.5))
    return {'origin': o + t0 * a, 'u': u, 'v': v, 'axis': a, 'outline': xy, 'length': t1 - t0,
            'error': px}


# --------------------------------------------------------------------------- alineado (clonar)
def _kabsch(A, B, scale=False):
    """Transformación (s, R, t) que lleva A sobre B (Umeyama)."""
    ca, cb = A.mean(0), B.mean(0)
    X, Y = A - ca, B - cb
    U, S, Vt = np.linalg.svd(X.T @ Y)
    D = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        D[2, 2] = -1
    R = Vt.T @ D @ U.T
    s = (S * np.diag(D)).sum() / (X ** 2).sum() if scale else 1.0
    t = cb - s * R @ ca
    return s, R, t


def icp(source, target, scale=False, iterations=30, sample=3000, seed=0):
    """Alinea `source` sobre `target` probando las orientaciones de sus ejes principales."""
    from mathutils import Vector
    from mathutils.kdtree import KDTree
    rng = np.random.default_rng(seed)
    S = np.asarray(source, float)
    T = np.asarray(target, float)
    if len(S) > sample:
        S = S[rng.choice(len(S), sample, replace=False)]
    if len(T) > sample * 3:
        T = T[rng.choice(len(T), sample * 3, replace=False)]
    kd = KDTree(len(T))
    for i, p in enumerate(T):
        kd.insert(p, i)
    kd.balance()

    def nearest(P):
        idx = np.empty(len(P), int)
        dist = np.empty(len(P))
        for i, p in enumerate(P):
            _co, j, d = kd.find(Vector(p))
            idx[i], dist[i] = j, d
        return idx, dist

    def axes(P):
        c = P.mean(0)
        _w, V = np.linalg.eigh(np.cov((P - c).T))
        return c, V[:, ::-1]
    cs, Vs = axes(S)
    ct, Vt = axes(T)
    ratio = math.sqrt(np.trace(np.cov((T - ct).T)) / max(np.trace(np.cov((S - cs).T)), 1e-18)) if scale else 1.0
    best = None
    for sx in (1, -1):
        for sy in (1, -1):
            F = Vt @ np.diag([sx, sy, sx * sy * np.linalg.det(Vt) * np.linalg.det(Vs)]) @ Vs.T
            s, R, t = ratio, F, ct - ratio * F @ cs
            for _ in range(iterations):
                P = (s * (R @ S.T)).T + t
                idx, dist = nearest(P)
                keep = dist <= np.percentile(dist, 90)
                ds, dR, dt = _kabsch(P[keep], T[idx[keep]], scale)
                s, R, t = ds * s, dR @ R, ds * dR @ t + dt
            P = (s * (R @ S.T)).T + t
            _idx, dist = nearest(P)
            err = float(np.sqrt(np.mean(dist ** 2)))
            if best is None or err < best[0]:
                best = (err, s, R, t)
    err, s, R, t = best
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = t
    return M, err


def symmetry_plane(points, axis_index, center, refine=True):
    """Plano de simetría: empieza en el plano del eje elegido por el centro y se afina
    alineando la nube reflejada con la original."""
    P = np.asarray(points, float)
    n0 = np.zeros(3)
    n0[axis_index] = 1.0
    c = np.asarray(center, float)
    if not refine:
        return c, n0
    Sref = np.eye(3) - 2 * np.outer(n0, n0)
    mirrored = (Sref @ (P - c).T).T + c
    M, _err = icp(mirrored, P, iterations=20, sample=2500)
    R, t = M[:3, :3], M[:3, 3]
    # x -> R (S (x - c)) + c + t  es la reflexión afinada
    img = (R @ (Sref @ (P - c).T)).T + c + t
    diff = img - P
    big = np.linalg.norm(diff, axis=1) > 1e-9
    if big.sum() < 10:
        return c, n0
    dn = diff[big] / np.linalg.norm(diff[big], axis=1, keepdims=True)
    dn *= np.sign(dn @ n0)[:, None]
    normal = _unit(dn.mean(0))
    point = ((img + P) / 2).mean(0)
    if abs(normal @ n0) < 0.9:          # el afinado se fue lejos: mejor el plano inicial
        return c, n0
    return point, normal


# --------------------------------------------------------------------------- elegir la forma sola
def choose(points, normals, weights=None):
    """Prueba las formas y devuelve (tipo, explicación). Tipo None si ninguna encaja bien.
    Orden: cilindro (lo más simple) → muelle → revolución → extrusión."""
    P = np.asarray(points, float)
    N = np.asarray(normals, float)
    w = np.ones(len(P)) if weights is None else np.asarray(weights, float)
    notes = []
    rc = 1e9
    try:
        cyl = cylinder(P, N, w)
        a = cyl['axis']
        side = np.abs(N @ a) < 0.3
        if (w[side].sum() / w.sum()) > 0.4:
            R = float(np.median(cyl['radial'][side]))
            rc = float(np.median(np.abs(cyl['radial'][side] - R)) / max(R, 1e-12))
        if rc < 1e8:
            notes.append(f'cilindro {rc * 100:.0f}%')
    except AddonError:
        pass
    if rc < 0.05:
        return 'CYLINDER', 'encaja como cilindro'
    try:
        sp = spring(P, N, w)
        steady = False
        if sp['length'] >= 1.2 * sp['radius']:
            # el borde exterior de un muelle tiene el mismo radio a lo largo de todo el eje
            q = P - sp['base']
            t = q @ sp['axis']
            rad = np.linalg.norm(q - np.outer(t, sp['axis']), axis=1)
            edges = np.linspace(0, sp['length'], 9)
            env = [np.percentile(rad[(t >= a) & (t < b)], 60) for a, b in zip(edges[:-1], edges[1:])
                   if ((t >= a) & (t < b)).sum() > 10]
            if len(env) >= 5:
                env = np.array(env)
                cv = np.std(env) / max(np.mean(env), 1e-12)
                notes.append(f'muelle cv {cv:.2f}')
                steady = cv < 0.2
        if steady and sp['confident'] and sp['turns'] >= 3 and sp['confidence'] >= 0.15:
            return 'SPRING', f'se ven {sp["turns"]:.0f} espiras'
    except AddonError:
        pass
    rr = 1e9
    try:
        rv = revolution(P, N, w)
        rr = rv['error'] / max(float(np.max(rv['rho'])), 1e-12)
        notes.append(f'revolución {rr * 100:.0f}%')
    except AddonError:
        pass
    if rr < 0.06:
        return 'REVOLVE', 'es redonda alrededor de un eje'
    d = _extrusion_direction(N, w)
    dots = np.abs(N @ d)
    ps = float((w * ((dots < 0.2) | (dots > 0.9))).sum() / w.sum())
    notes.append(f'prisma {ps * 100:.0f}%')
    if ps > 0.85:
        return 'EXTRUDE', 'tiene la misma sección a lo largo'
    return None, ', '.join(notes)
