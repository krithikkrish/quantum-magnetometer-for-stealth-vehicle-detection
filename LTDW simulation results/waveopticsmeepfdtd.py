# ============================================================================
# Phase 1C - LTDW photon-collection efficiency (Meep FDTD)
# ----------------------------------------------------------------------------
# Models an NV centre (point electric dipole at the 637 nm zero-phonon line)
# inside diamond and measures the EXTRACTION / COLLECTION EFFICIENCY:
#     efficiency = (power escaping through the top into air) / (total emitted power)
#
# Bare diamond is limited by Total Internal Reflection (TIR) to only a few %.
# The LTDW (angled 45-deg facets) recycles trapped light so far more escapes.
# Sensitivity is photon-shot-noise limited (dB ~ 1/sqrt(N)), so a photon gain G
# gives a sensitivity gain sqrt(G).  Target: 2% -> 30% = x15 -> sqrt(15) = 3.87x.
#
# RUN (inside WSL Ubuntu, conda env with pymeep):
#     conda activate meep
#     python meep_1c_ltdw.py
# ============================================================================
import meep as mp
import numpy as np

# ---- units: a = 1 micron, c = 1 ----
n_dia = 2.4                 # diamond refractive index
wvl   = 0.637              # NV ZPL wavelength (microns)
fcen  = 1.0 / wvl          # center frequency in Meep units
df    = 0.2 * fcen         # source bandwidth
res   = 50                 # resolution (pixels/micron). RAISE this for convergence.

# ---- geometry (microns) ----
t_dia = 1.0                # diamond micro-slab thickness (representative)
Lxy   = 4.0                # lateral size
dpml  = 1.0                # PML thickness (~1 wavelength)
t_air = 2.0                # air gap above

sx = Lxy + 2*dpml
sy = Lxy + 2*dpml
sz = t_dia + t_air + 2*dpml
cell = mp.Vector3(sx, sy, sz)
pml  = [mp.PML(dpml)]

src_z = -0.05              # NV just below the diamond top surface (z=0)

def make_geometry(kind):
    # diamond slab: z in [-t_dia, 0]; air above z=0
    geom = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, t_dia),
                     center=mp.Vector3(0, 0, -t_dia/2),
                     material=mp.Medium(index=n_dia))]
    if kind == "textured":
        # demonstrator anti-TIR texturing on the top surface: a row of diamond
        # ridges (triangular grating) that break TIR and boost extraction.
        # Replace/extend with your real LTDW 45-deg facet geometry.
        period = 0.5
        nrid = int(Lxy/period)
        for i in range(nrid):
            x0 = -Lxy/2 + (i+0.5)*period
            geom.append(mp.Prism(
                vertices=[mp.Vector3(x0-period/2, 0, 0),
                          mp.Vector3(x0+period/2, 0, 0),
                          mp.Vector3(x0, 0, 0.25)],
                height=Lxy, axis=mp.Vector3(0,1,0),
                center=mp.Vector3(x0, 0, 0.08),
                material=mp.Medium(index=n_dia)))
    return geom

def run(kind):
    sim = mp.Simulation(cell_size=cell, resolution=res, boundary_layers=pml,
                        geometry=make_geometry(kind),
                        sources=[mp.Source(mp.GaussianSource(fcen, fwidth=df),
                                           component=mp.Ex,
                                           center=mp.Vector3(0,0,src_z))],
                        default_material=mp.Medium(index=1.0))   # air background
    # TOTAL emitted power: closed flux box (6 faces) around the dipole
    b = 0.4
    total = sim.add_flux(fcen, 0, 1,
        mp.FluxRegion(center=mp.Vector3(+b,0,src_z), size=mp.Vector3(0,2*b,2*b), direction=mp.X, weight=+1),
        mp.FluxRegion(center=mp.Vector3(-b,0,src_z), size=mp.Vector3(0,2*b,2*b), direction=mp.X, weight=-1),
        mp.FluxRegion(center=mp.Vector3(0,+b,src_z), size=mp.Vector3(2*b,0,2*b), direction=mp.Y, weight=+1),
        mp.FluxRegion(center=mp.Vector3(0,-b,src_z), size=mp.Vector3(2*b,0,2*b), direction=mp.Y, weight=-1),
        mp.FluxRegion(center=mp.Vector3(0,0,src_z+b), size=mp.Vector3(2*b,2*b,0), direction=mp.Z, weight=+1),
        mp.FluxRegion(center=mp.Vector3(0,0,src_z-b), size=mp.Vector3(2*b,2*b,0), direction=mp.Z, weight=-1))
    # COLLECTED power: a plane in the air above the diamond
    coll = sim.add_flux(fcen, 0, 1,
        mp.FluxRegion(center=mp.Vector3(0,0,0.8), size=mp.Vector3(Lxy,Lxy,0), direction=mp.Z, weight=+1))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        20, mp.Ex, mp.Vector3(0,0,0.8), 1e-4))
    Ptot = mp.get_fluxes(total)[0]
    Pcol = mp.get_fluxes(coll)[0]
    return Pcol/Ptot

if __name__ == "__main__":
    eff_bare = run("bare")
    print("Bare flat diamond  : extraction efficiency = %.2f %%" % (100*eff_bare))
    eff_tex = run("textured")
    print("Textured/LTDW top  : extraction efficiency = %.2f %%" % (100*eff_tex))
    G = eff_tex/eff_bare
    print("photon gain        = %.1fx  ->  sensitivity gain = sqrt = %.2fx" % (G, G**0.5))
    print("(Device target: 2% -> 30% = x15 photons -> sqrt(15) = 3.87x, Clevenson 2015)")