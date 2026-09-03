# ============================================================================
# Phase 1D - Metasurface extraction, FAST 2D grating SWEEP  (Meep FDTD, 2D)
# ----------------------------------------------------------------------------
# WHY THIS VERSION:
#   The 3D pillar run gave 0.91x (worse than bare) because the period (0.28 um)
#   was SUB-WAVELENGTH in diamond and cannot DIFFRACT.  Extraction of TIR-
#   trapped light REQUIRES a diffraction grating with period
#         Lambda ~ lambda/(n-1) = 0.637/(2.4-1) ~ 0.45 um   (and nearby).
#   This script runs in 2D (seconds per case, not hours) and SWEEPS the period
#   across the diffractive range to FIND the optimum.  The optimal period and
#   the gain ratio transfer to 3D; do ONE 3D run at the winner to confirm.
#
#   extraction = (upward power through a plane above the grating)
#                / (total power emitted by the dipole)
#
# RUN (WSL Ubuntu, conda env with pymeep):
#     conda activate meep
#     python meep_1d_metasurface_2d_sweep.py
# ============================================================================
import meep as mp
import numpy as np

# ---- units: a = 1 micron ----
n_dia = 2.4
wvl   = 0.637
fcen  = 1.0/wvl
df    = 0.2*fcen
res   = 40                      # 2D -> cheap; 40 is plenty for a sweep

# ---- domain (microns), 2D in the x-z plane (invariant in y) ----
t_dia = 1.0
dpml  = 1.0
t_air = 2.0
Lx    = 12.0                    # wide so trapped light can reach the grating
sx    = Lx + 2*dpml
sz    = t_dia + t_air + 2*dpml
cell  = mp.Vector3(sx, 0, sz)   # the 0 in y makes it a 2D simulation
pml   = [mp.PML(dpml)]
src_z = -0.05                   # NV just under the diamond top surface (z=0)

# ---- grating geometry ----
h_ridge = 0.25                  # grating tooth height (microns)
duty    = 0.5                   # fraction of period filled by diamond tooth
z_coll  = h_ridge + 0.6         # collection plane height (above the teeth)

def geometry(period):
    # diamond slab z in [-t_dia, 0]
    geom = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, t_dia),
                     center=mp.Vector3(0, 0, -t_dia/2),
                     material=mp.Medium(index=n_dia))]
    if period is not None:
        w = duty*period
        n = int(Lx/period) + 4
        x0 = -n*period/2.0
        for i in range(n):
            xc = x0 + (i+0.5)*period
            geom.append(mp.Block(size=mp.Vector3(w, mp.inf, h_ridge),
                                 center=mp.Vector3(xc, 0, h_ridge/2),
                                 material=mp.Medium(index=n_dia)))
    return geom

def run(period):
    sim = mp.Simulation(cell_size=cell, resolution=res, boundary_layers=pml,
                        geometry=geometry(period),
                        sources=[mp.Source(mp.GaussianSource(fcen, fwidth=df),
                                           component=mp.Ex,          # in-plane NV dipole
                                           center=mp.Vector3(0, 0, src_z))],
                        default_material=mp.Medium(index=1.0))
    # TOTAL emitted power: closed box (4 sides in 2D) around the dipole
    b = 0.4
    total = sim.add_flux(fcen, 0, 1,
        mp.FluxRegion(center=mp.Vector3(+b,0,src_z), size=mp.Vector3(0,0,2*b), direction=mp.X, weight=+1),
        mp.FluxRegion(center=mp.Vector3(-b,0,src_z), size=mp.Vector3(0,0,2*b), direction=mp.X, weight=-1),
        mp.FluxRegion(center=mp.Vector3(0,0,src_z+b), size=mp.Vector3(2*b,0,0), direction=mp.Z, weight=+1),
        mp.FluxRegion(center=mp.Vector3(0,0,src_z-b), size=mp.Vector3(2*b,0,0), direction=mp.Z, weight=-1))
    # COLLECTED power: full-width plane in the air above the grating
    coll = sim.add_flux(fcen, 0, 1,
        mp.FluxRegion(center=mp.Vector3(0,0,z_coll), size=mp.Vector3(Lx,0,0), direction=mp.Z, weight=+1))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        20, mp.Ex, mp.Vector3(0,0,z_coll), 1e-3))
    Ptot = mp.get_fluxes(total)[0]
    Pcol = mp.get_fluxes(coll)[0]
    return Pcol/Ptot

if __name__ == "__main__":
    # ---- bare baseline ----
    eff_bare = run(None)
    print("\n================ RESULTS (2D) ================")
    print("Bare flat diamond        : extraction = %.2f %%" % (100*eff_bare))

    # ---- sweep the grating period across the diffractive range ----
    periods = [0.30, 0.38, 0.45, 0.52, 0.58, 0.64, 0.70]
    best = (None, 0.0)
    print("\n period(um)   extraction   gain vs bare   sensitivity(sqrt)")
    for p in periods:
        eff = run(p)
        g = eff/eff_bare
        print("   %.2f         %5.2f %%       %5.2fx         %5.2fx"
              % (p, 100*eff, g, g**0.5))
        if eff > best[1]:
            best = (p, eff)

    gbest = best[1]/eff_bare
    print("\n---------------------------------------------")
    print("BEST: period = %.2f um -> extraction = %.2f %%" % (best[0], 100*best[1]))
    print("      photon gain = %.2fx   sensitivity gain = %.2fx" % (gbest, gbest**0.5))
    print("Diffraction expected to peak near lambda/(n-1) = %.2f um." % (wvl/(n_dia-1)))
    print("Next: run ONE 3D confirmation at period = %.2f um." % best[0])
    print("=============================================")