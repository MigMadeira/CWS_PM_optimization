import os
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from simsopt.geo import SurfaceRZFourier, curves_to_vtk, PermanentMagnetGrid
from simsopt.field import Current, Coil, BiotSavart, DipoleField
import simsoptpp as sopp
from simsopt.mhd.vmec import Vmec
from simsopt import load
from simsopt.util.permanent_magnet_helper_functions import *
from simsopt.solve import GPMO
from simsopt.objectives import SquaredFlux

# Set some parameters
nphi = 64 # need to set this to 64 for a real run
ntheta = 64 # same as above
dr = 0.06  #dr is used when using cylindrical coordinates

input_name = './inputs/equilibria/wout_W7-X_standard_configuration.nc'
algorithm = "baseline"

# Make the output directory
OUT_DIR = "./W7-X/PM_opt/N=4_k=2/"   
os.makedirs(OUT_DIR, exist_ok=True)

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent).resolve()
surface_filename = str(TEST_DIR/input_name)

s = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s.to_vtk(OUT_DIR + "surf_plot")

#Loading the coils
coilfile = str(TEST_DIR/"./W7-X/coil_output/N=4_k=2/biot_savart_opt.json")
bs = load(coilfile)
coils = bs.coils
ncoils = len(coils)

# Set up BiotSavart fields
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# Make higher resolution surface for plotting Bnormal
qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_vmec_input(
    surface_filename, range="full torus",
    quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta
)

# check average on-axis magnetic field strength
calculate_on_axis_B(bs, s)

#create inside surface
s_in = SurfaceRZFourier.from_vmec_input(surface_filename, range="full torus", nphi=2*nphi, ntheta=ntheta)
s_in.extend_via_projected_normal(0.25)
s_in.to_vtk(OUT_DIR + "surface_in")

#create outside surface
s_out = SurfaceRZFourier(ntor=nphi*2, mpol=ntheta, nfp = s.nfp)
s_out.extend_via_projected_normal(0.52 - 0.1)
s_out.to_vtk(OUT_DIR + "surface_out")


#initialize the permanent magnet class
kwargs_geo = {"dr": dr, "coordinate_flag": "cylindrical"}  
pm_opt = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
    s, Bnormal, s_in, s_out, **kwargs_geo
)

print('Number of available dipoles = ', pm_opt.ndipoles)
exit()
# Set some hyperparameters for the optimization
kwargs = initialize_default_kwargs('GPMO')
kwargs['K'] = 29500
kwargs['nhistory'] = 500

if algorithm == 'backtracking':
    kwargs['backtracking'] = 100  # How often to perform the backtrackinig
    kwargs['Nadjacent'] = 1
    kwargs['dipole_grid_xyz'] = np.ascontiguousarray(pm_opt.dipole_grid_xyz)
    kwargs['max_nMagnets'] = 5600

# Optimize the permanent magnets greedily
t1 = time.time()
R2_history, Bn_history, m_history = GPMO(pm_opt, algorithm = algorithm, **kwargs)
t2 = time.time()
print('GPMO took t = ', t2 - t1, ' s')

# plot the MSE history
iterations = np.linspace(0, kwargs['K'], len(R2_history), endpoint=False)
plt.figure()
plt.semilogy(iterations, R2_history, label=r'$f_B$')
plt.semilogy(iterations, Bn_history, label=r'$<|Bn|>$')
plt.grid(True)
plt.xlabel('Number of Magnets')
plt.ylabel('Metric values')
plt.legend()
plt.savefig(OUT_DIR + 'GPMO_MSE_history.png')

# Set final m to the minimum achieved during the optimization
min_ind = np.argmin(R2_history)
pm_opt.m = np.ravel(m_history[:, :, min_ind])

print("best result = ", 0.5 * np.sum((pm_opt.A_obj @ pm_opt.m - pm_opt.b_obj) ** 2))
np.savetxt(OUT_DIR + 'best_result_m=' + str(int(kwargs['K'] / (kwargs['nhistory']) * min_ind )) + '.txt', m_history[:, :, min_ind ].reshape(pm_opt.ndipoles * 3))
b_dipole = DipoleField(pm_opt.dipole_grid_xyz, m_history[:, :, min_ind ].reshape(pm_opt.ndipoles * 3),
                       nfp=s.nfp, coordinate_flag=pm_opt.coordinate_flag, m_maxima=pm_opt.m_maxima,)
b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
b_dipole._toVTK(OUT_DIR + "Dipole_Fields_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * min_ind)))
bs.set_points(s_plot.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
Bnormal_total = Bnormal + Bnormal_dipoles
# For plotting Bn on the full torus surface at the end with just the dipole fields
make_Bnormal_plots(b_dipole, s_plot, OUT_DIR, "only_m_optimized_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * min_ind)))
pointData = {"B_N": Bnormal_total[:, :, None]}
s_plot.to_vtk(OUT_DIR + "m_optimized_K" + str(int(kwargs['K'] / (kwargs['nhistory']) * min_ind)), extra_data=pointData)

# Print effective permanent magnet volume
M_max = 1.465 / (4 * np.pi * 1e-7)
dipoles = pm_opt.m.reshape(pm_opt.ndipoles, 3)
print('Volume of permanent magnets is = ', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))) / M_max)
print('sum(|m_i|)', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))))

save_plots = True
if save_plots:
    # Save the MSE history and history of the m vectors
    #np.savetxt(OUT_DIR + 'mhistory_K' + str(kwargs['K']) + '_nphi' + str(nphi) + '_ntheta' + str(ntheta) + '.txt', m_history.reshape(pm_opt.ndipoles * 3, kwargs['nhistory'] + 1)) #this file occupies alot of space ~2gb, use with care
    np.savetxt(OUT_DIR + 'R2history_K' + str(kwargs['K']) + '_nphi' + str(nphi) + '_ntheta' + str(ntheta) + '.txt', R2_history)
    # Plot the SIMSOPT GPMO solution
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
    make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_optimized")

    # Look through the solutions as function of K and make plots
    for k in range(0, kwargs["nhistory"] + 1, 50):
        mk = m_history[:, :, k].reshape(pm_opt.ndipoles * 3)
        np.savetxt(OUT_DIR + 'result_m=' + str(int(kwargs['max_nMagnets'] / (kwargs['nhistory']) * k)) + '.txt', m_history[:, :, k].reshape(pm_opt.ndipoles * 3))
        b_dipole = DipoleField(
            pm_opt.dipole_grid_xyz,
            mk,
            nfp=s.nfp,
            coordinate_flag=pm_opt.coordinate_flag,
            m_maxima=pm_opt.m_maxima,
        )
        b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
        b_dipole._toVTK(OUT_DIR + "Dipole_Fields_K" + str(int(kwargs['K'] / kwargs['nhistory'] * k)))
        print("Total fB = ",
              0.5 * np.sum((pm_opt.A_obj @ mk - pm_opt.b_obj) ** 2))
        Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
        Bnormal_total = Bnormal + Bnormal_dipoles

        # For plotting Bn on the full torus surface at the end with just the dipole fields
        make_Bnormal_plots(b_dipole, s_plot, OUT_DIR, "only_m_optimized_K" + str(int(kwargs['K'] / kwargs['nhistory'] * k)))
        pointData = {"B_N": Bnormal_total[:, :, None]}
        s_plot.to_vtk(OUT_DIR + "m_optimized_K" + str(int(kwargs['K'] / kwargs['nhistory'] * k)), extra_data=pointData)

    # write solution to FAMUS-type file
pm_opt.write_to_famus(Path(OUT_DIR))

# Compute metrics with permanent magnet results
dipoles_m = pm_opt.m.reshape(pm_opt.ndipoles, 3)
num_nonzero = np.count_nonzero(np.sum(dipoles_m ** 2, axis=-1)) / pm_opt.ndipoles * 100
print("Number of possible dipoles = ", pm_opt.ndipoles)
print("% of dipoles that are nonzero = ", num_nonzero)

# Print optimized f_B and other metrics
### Note this will only agree with the optimization in the high-resolution
### limit where nphi ~ ntheta >= 64!
b_dipole = DipoleField(
    pm_opt.dipole_grid_xyz,
    pm_opt.m,
    nfp=s.nfp,
    coordinate_flag=pm_opt.coordinate_flag,
    m_maxima=pm_opt.m_maxima,
)
b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
bs.set_points(s_plot.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
f_B_sf = SquaredFlux(s_plot, b_dipole, -Bnormal).J()
print('f_B = ', f_B_sf)
B_max = 1.465
mu0 = 4 * np.pi * 1e-7
total_volume = np.sum(np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * 2 * mu0 / B_max
print('Total volume = ', total_volume)

# Optionally make a QFM and pass it to VMEC
# This is worthless unless plasma
# surface is at least 64 x 64 resolution.
vmec_flag = False
if vmec_flag:
    from mpi4py import MPI
    from simsopt.mhd.vmec import Vmec
    from simsopt.util.mpi import MpiPartition
    mpi = MpiPartition(ngroups=1)
    comm = MPI.COMM_WORLD

    # Make the QFM surfaces
    t1 = time.time()
    Bfield = bs + b_dipole
    Bfield.set_points(s_plot.gamma().reshape((-1, 3)))
    qfm_surf = make_qfm(s_plot, Bfield)
    qfm_surf = qfm_surf.surface
    t2 = time.time()
    print("Making the QFM surface took ", t2 - t1, " s")

    # Run VMEC with new QFM surface
    t1 = time.time()

    ### Always use the QA VMEC file and just change the boundary
    vmec_input = "../../tests/test_files/input.LandremanPaul2021_QA"
    equil = Vmec(vmec_input, mpi)
    equil.boundary = qfm_surf
    equil.run()
