# SPDX-FileCopyrightText: 2026 Ricardo Lugaresi
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generadores de piezas limpias (numpy). Devuelven (vértices n×3, caras [listas]) en
coordenadas locales: el eje de la pieza es +Z y empieza en z = 0. Todas son sólidos
cerrados (manifold) listos para imprimir."""
import math

import numpy as np


def _ring(center, n1, n2, radius, seg):
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    return center + radius * (np.outer(np.cos(a), n1) + np.outer(np.sin(a), n2))


def _sweep(centers, n1s, n2s, radius, seg, caps=True):
    """Tubo de sección circular a lo largo de una serie de centros con sus marcos."""
    m = len(centers)
    verts = [_ring(centers[i], n1s[i], n2s[i], radius if np.isscalar(radius) else radius[i], seg) for i in range(m)]
    V = np.vstack(verts)
    F = []
    for i in range(m - 1):
        a0, b0 = i * seg, (i + 1) * seg
        for k in range(seg):
            k1 = (k + 1) % seg
            F.append([a0 + k, a0 + k1, b0 + k1, b0 + k])
    if caps:
        c0 = len(V)
        c1 = c0 + 1
        V = np.vstack([V, centers[0], centers[-1]])
        last = (m - 1) * seg
        for k in range(seg):
            k1 = (k + 1) % seg
            F.append([c0, k1, k])
            F.append([c1, last + k, last + k1])
    return V, F


def _parallel_frames(points):
    """Marcos sin giros bruscos a lo largo de una curva (transporte paralelo)."""
    P = np.asarray(points, float)
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    helper = np.array([0, 0, 1.0]) if abs(T[0][2]) < 0.9 else np.array([1.0, 0, 0])
    n = np.cross(T[0], helper)
    n /= np.linalg.norm(n)
    N = [n]
    for i in range(1, len(P)):
        v = N[-1] - T[i] * (N[-1] @ T[i])
        nv = np.linalg.norm(v)
        N.append(v / nv if nv > 1e-12 else N[-1])
    N = np.array(N)
    B = np.cross(T, N)
    return T, N, B


# --------------------------------------------------------------------------- cilindro
def cylinder(radius, length, seg=48):
    centers = np.array([[0, 0, 0.0], [0, 0, length]])
    n1 = np.array([[1.0, 0, 0]] * 2)
    n2 = np.array([[0, 1.0, 0]] * 2)
    return _sweep(centers, n1, n2, radius, seg)


# --------------------------------------------------------------------------- muelle
def spring(radius, wire, length, turns, closed_ends=True, hand=1, phase=0.0, rings=False,
           seg=12, per_turn=40, rod=0.0):
    """Muelle helicoidal. `radius` = radio exterior; `wire` = grosor del hilo.
    Con extremos cerrados, la última vuelta de cada lado está apretada (asienta plano).
    Si el original eran anillos fundidos, igualmente se hace en hélice (es lo real)."""
    wire = max(1e-6, min(wire, radius * 0.95))
    rc = radius - wire / 2
    N = max(1.0, float(turns))
    ne = min(0.75, N / 4) if closed_ends else 0.0
    z0, z1 = wire / 2, max(wire / 2 + 1e-6, length - wire / 2)
    active = N - 2 * ne
    pa = (z1 - z0 - 2 * ne * wire) / active if active > 1e-9 else 0.0
    if pa < wire * 0.5:                      # no caben: se reparte todo igual
        ne, pa = 0.0, (z1 - z0) / N
    steps = max(8, int(per_turn * N))
    u = np.linspace(0, N, steps + 1)

    def zf(x):
        if x <= ne:
            return z0 + wire * x
        if x >= N - ne:
            return z1 - wire * (N - x)
        return z0 + wire * ne + pa * (x - ne)
    z = np.array([zf(x) for x in u])
    sgn = -1 if hand < 0 else 1
    ang = sgn * 2 * np.pi * u + phase
    C = np.column_stack([rc * np.cos(ang), rc * np.sin(ang), z])
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    R = np.column_stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)])
    n1 = R - T * np.einsum('ij,ij->i', R, T)[:, None]
    n1 /= np.linalg.norm(n1, axis=1, keepdims=True)
    n2 = np.cross(T, n1)
    V, F = _sweep(C, n1, n2, wire / 2, seg)
    if rod > 0:                              # vástago del amortiguador por dentro
        rv, rf = cylinder(min(rod, rc - wire / 2) , length, seg=32)
        off = len(V)
        V = np.vstack([V, rv])
        F = F + [[i + off for i in f] for f in rf]
    return V, F


# --------------------------------------------------------------------------- tubo por puntos
def rounded_path(points, corner, samples=10):
    """Polilínea con las esquinas redondeadas (radio `corner`, limitado a media arista)."""
    P = [np.asarray(p, float) for p in points]
    if len(P) < 3 or corner <= 0:
        return np.array(P)
    out = [P[0]]
    for i in range(1, len(P) - 1):
        a, b, c = P[i - 1], P[i], P[i + 1]
        d1, d2 = a - b, c - b
        l1, l2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if l1 < 1e-9 or l2 < 1e-9:
            continue
        d1, d2 = d1 / l1, d2 / l2
        cosang = np.clip(d1 @ d2, -1, 1)
        theta = math.acos(cosang)
        if theta > math.pi - 1e-3:           # casi recto: sin redondeo
            out.append(b)
            continue
        tan_len = corner / math.tan(theta / 2)
        tan_len = min(tan_len, 0.5 * l1, 0.5 * l2)
        r = tan_len * math.tan(theta / 2)
        p1, p2 = b + d1 * tan_len, b + d2 * tan_len
        bis = d1 + d2
        bis /= np.linalg.norm(bis)
        center = b + bis * (r / math.sin(theta / 2))
        v1, v2 = p1 - center, p2 - center
        phi = math.acos(np.clip((v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
        axis = np.cross(v1, v2)
        axis /= max(np.linalg.norm(axis), 1e-12)
        for k in range(samples + 1):
            ang = phi * k / samples
            rot = v1 * math.cos(ang) + np.cross(axis, v1) * math.sin(ang) + axis * (axis @ v1) * (1 - math.cos(ang))
            out.append(center + rot)
    out.append(P[-1])
    # quitar puntos repetidos
    clean = [out[0]]
    for p in out[1:]:
        if np.linalg.norm(p - clean[-1]) > 1e-9:
            clean.append(p)
    return np.array(clean)


def _densify(path, step):
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        L = np.linalg.norm(b - a)
        n = max(1, int(math.ceil(L / step)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)


def tube(points, radius, corner=None, seg=24):
    """Tubo macizo que pasa por los puntos, con curvas suaves en las esquinas."""
    corner = radius * 2.5 if corner is None else corner
    path = rounded_path(points, corner)
    path = _densify(path, max(radius * 0.6, 1e-6))
    T, N, B = _parallel_frames(path)
    return _sweep(path, N, B, radius, seg)


# --------------------------------------------------------------------------- revolución
def revolve(z, rho, seg=64):
    """Sólido de revolución a partir del perfil (z creciente, radio ≥ 0), cerrado en el eje."""
    z = np.asarray(z, float)
    rho = np.maximum(np.asarray(rho, float), 0.0)
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ca, sa = np.cos(a), np.sin(a)
    V = [np.array([0, 0, z[0]])]
    for zi, ri in zip(z, rho):
        V.extend(np.column_stack([ri * ca, ri * sa, np.full(seg, zi)]))
    V.append(np.array([0, 0, z[-1]]))
    V = np.array(V)
    F = []
    m = len(z)
    top = len(V) - 1
    for k in range(seg):
        k1 = (k + 1) % seg
        F.append([0, 1 + k1, 1 + k])
    for i in range(m - 1):
        a0, b0 = 1 + i * seg, 1 + (i + 1) * seg
        for k in range(seg):
            k1 = (k + 1) % seg
            F.append([a0 + k, a0 + k1, b0 + k1, b0 + k])
    last = 1 + (m - 1) * seg
    for k in range(seg):
        k1 = (k + 1) % seg
        F.append([top, last + k, last + k1])
    return V, F


# --------------------------------------------------------------------------- extrusión
def extrude(outline, length):
    """Prisma a partir de un contorno 2D (antihorario) estirado `length` en +Z.
    Las tapas son polígonos (n-gonos); Blender las triangula al exportar."""
    xy = np.asarray(outline, float)
    n = len(xy)
    bottom = np.column_stack([xy, np.zeros(n)])
    top = np.column_stack([xy, np.full(n, length)])
    V = np.vstack([bottom, top])
    F = [list(range(n - 1, -1, -1)), list(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        F.append([i, j, n + j, n + i])
    return V, F
