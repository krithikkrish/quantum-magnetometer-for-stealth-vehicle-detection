# ============================================================================
# Phase 1C - LTDW light-trapping (RAY OPTICS, 2D Monte Carlo)
# ----------------------------------------------------------------------------
# Pure Python (numpy + matplotlib). Runs in normal Anaconda/Python on Windows -
# NO Meep / WSL needed.
#
# Shoots many light rays from an NV point inside diamond and traces them with
# Total Internal Reflection (TIR) + Fresnel transmission until they escape the
# top (collected) or are lost/absorbed. Compares:
#   BARE : flat diamond  -> light leaks out the sides, little collected
#   LTDW : 45-deg facets -> facets recycle trapped light up to the top
# Prints the collection efficiencies and saves a ray-path picture.
# (2D over-estimates absolute %; the device 2%->30%, x3.87 is the 3D value.)
# ============================================================================
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
n_dia = 2.4                        # diamond refractive index
thetac = np.arcsin(1.0/n_dia)      # critical angle for TIR

def fresnelT(ci):
    # unpolarized transmittance, diamond->air; ci = cos(incidence angle)
    si = np.sqrt(max(0.0, 1 - ci*ci)); st = n_dia*si
    if st >= 1: return 0.0          # total internal reflection
    ct = np.sqrt(1 - st*st)
    rs = ((n_dia*ci - ct)/(n_dia*ci + ct))**2
    rp = ((n_dia*ct - ci)/(n_dia*ct + ci))**2
    return 1 - 0.5*(rs + rp)

def cross(a, b): return a[0]*b[1] - a[1]*b[0]

def edge_normals(verts):
    c = np.mean(verts, axis=0); E = []
    for i in range(len(verts)):
        a = np.array(verts[i], float); b = np.array(verts[(i+1) % len(verts)], float)
        v = b - a; n = np.array([v[1], -v[0]]); n = n/np.linalg.norm(n)
        if np.dot(n, (a+b)/2 - c) < 0: n = -n     # make normal point outward
        E.append((a, b, n))
    return E

def trace(verts, types, src, alpha, maxb=400, record=False):
    E = edge_normals(verts)
    pos = np.array(src, float)
    ang = rng.uniform(0, 2*np.pi); d = np.array([np.cos(ang), np.sin(ang)])
    path = [pos.copy()]
    for _ in range(maxb):
        best = None
        for k, (a, b, n) in enumerate(E):
            v = b - a; den = cross(d, v)
            if abs(den) < 1e-12: continue
            t = cross(a - pos, v)/den            # distance along the ray
            s = cross(a - pos, d)/den            # position along the edge (0..1)
            if t > 1e-7 and -1e-9 <= s <= 1+1e-9:
                if best is None or t < best[0]: best = (t, k, n)
        if best is None: return ("lost", path)
        t, k, n = best; pos = pos + t*d
        if record: path.append(pos.copy())
        if rng.random() > np.exp(-alpha*t): return ("absorbed", path)   # bulk loss
        ci = abs(np.dot(d, n))
        if types[k] == "mirror":
            d = d - 2*np.dot(d, n)*n              # perfect reflector (recycle)
        else:                                     # diamond->air interface
            thi = np.arccos(min(1.0, ci))
            if thi < thetac and rng.random() < fresnelT(ci):
                return (("collected" if types[k] == "collect" else "lost"), path)
            d = d - 2*np.dot(d, n)*n              # TIR / Fresnel reflection
        pos = pos + 1e-6*d                        # nudge off the surface
    return ("lost", path)

# ---- geometries (diamond cross-section) ----
W, H = 2.0, 1.0
bare_v = [(-W/2, 0), (W/2, 0), (W/2, -H), (-W/2, -H)]
bare_t = ["collect", "loss", "loss", "loss"]      # only top collects; sides/bottom lose
c = 0.32
ltdw_v = [(-W/2+c, 0), (W/2-c, 0), (W/2, -c), (W/2, -H+c),
          (W/2-c, -H), (-W/2+c, -H), (-W/2, -H+c), (-W/2, -c)]
ltdw_t = ["collect"] + ["mirror"]*7               # 45-deg facets recycle to top

src = (0, -0.15); alpha = 0.09                     # source position + bulk absorption

def efficiency(v, t, N=30000):
    col = sum(trace(v, t, src, alpha)[0] == "collected" for _ in range(N))
    return col/N

eb = efficiency(bare_v, bare_t)
el = efficiency(ltdw_v, ltdw_t)
print(f"BARE collection = {100*eb:.1f}%   LTDW collection = {100*el:.1f}%")
print(f"photon gain = x{el/eb:.1f}   sensitivity gain = x{(el/eb)**0.5:.2f}")
print("(device value: 2% -> 30% = x15 -> sqrt(15) = 3.87x, 3D, Clevenson 2015)")

# ---- ray-path picture ----
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
for ax, (v, t, title, eff) in zip(axes,
        [(bare_v, bare_t, "BARE diamond (flat)", eb),
         (ltdw_v, ltdw_t, "LTDW (45 deg facets)", el)]):
    vv = v + [v[0]]
    ax.fill([p[0] for p in v], [p[1] for p in v], color="#cfe8ff", alpha=0.5, zorder=0)
    ax.plot([p[0] for p in vv], [p[1] for p in vv], 'k-', lw=2)
    for _ in range(140):
        o, path = trace(v, t, src, alpha, record=True)
        P = np.array(path)
        if o == "collected":
            ax.plot(P[:, 0], P[:, 1], color="#1a9850", lw=0.7, alpha=0.8, zorder=2)
        else:
            ax.plot(P[:, 0], P[:, 1], color="#bbbbbb", lw=0.4, alpha=0.5, zorder=1)
    ax.plot(src[0], src[1], 'r*', ms=14, zorder=5)
    ax.set_title(f"{title}\ncollection = {100*eff:.1f}%")
    ax.set_aspect('equal'); ax.set_xlabel("x"); ax.set_ylim(-1.15, 0.35)
axes[0].set_ylabel("z  (green = escaped to detector, grey = lost)")
fig.suptitle("Phase 1C - LTDW light-trapping mechanism (2D ray optics)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("LTDW_ray_collection.png", dpi=130)
print("saved LTDW_ray_collection.png")
plt.show()