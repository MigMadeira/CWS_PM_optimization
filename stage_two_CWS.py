import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from simsopt.geo import curves_to_vtk
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt._core.derivative import Derivative
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (
    CurveLength, CurveCurveDistance,
    MeanSquaredCurvature, LpCurveCurvature, ArclengthVariation, CurveCWSFourier, plot
)

class Parameters:
    def __init__(self, nphi, ntheta, ncoils, order, R0, R1, LENGTH_WEIGHT, LENGTH_THRESHOLD, CC_THRESHOLD, 
                 CC_WEIGHT, CURVATURE_THRESHOLD, CURVATURE_WEIGHT, 
                 MSC_THRESHOLD, MSC_WEIGHT,ARCLENGTH_WEIGHT,  tolerance):
        
        self.nphi = nphi
        self.ntheta = ntheta
        self.ncoils = ncoils
        self.order = order
        self.R0 = R0
        self.R1 = R1
        self.LENGTH_WEIGHT = LENGTH_WEIGHT
        self.LENGTH_THRESHOLD = LENGTH_THRESHOLD
        self.CC_THRESHOLD = CC_THRESHOLD
        self.CC_WEIGHT = CC_WEIGHT
        self.CURVATURE_THRESHOLD = CURVATURE_THRESHOLD
        self.CURVATURE_WEIGHT = CURVATURE_WEIGHT
        self.MSC_THRESHOLD = MSC_THRESHOLD
        self.MSC_WEIGHT = MSC_WEIGHT
        self.ARCLENGTH_WEIGHT = ARCLENGTH_WEIGHT
        self.tolerance = tolerance
        
    def save(self, filename="parameters.txt"):
        with open(filename, "w") as file:
            for key, value in vars(self).items():
                file.write(f"{key}: {value}\n")
                
class Results:
    def __init__(self, f, f_B, max_Bn, avg_Bn, f_L, L_values, f_CC, 
                 min_CC, f_K, K_values, f_MSC, MSC_values, f_Arc, Arc_values):
        
        self.f = f
        self.f_B = f_B
        self.max_Bn = max_Bn
        self.avg_Bn = avg_Bn
        self.f_L = f_L
        self.L_values = L_values
        self.f_CC = f_CC
        self.min_CC = min_CC
        self.f_K = f_K
        self.K_values = K_values
        self.f_MSC = f_MSC
        self.MSC_values = MSC_values
        self.f_Arc = f_Arc
        self.Arc_values = Arc_values
        
    def save(self, filename="results.txt"):
        with open(filename, "w") as file:
            for key, value in vars(self).items():
                file.write(f"{key}: {value}\n")

#OUT_DIR = "./coil_output/nfp2_rescaled_Aries_CWS_1.307/"                
OUT_DIR = "./coil_output/nfp3_rescaled_Aries_CWS_2/"
os.makedirs(OUT_DIR, exist_ok=True)

# Read in the plasma equilibrium file
#wout_name = './inputs/equilibria/scaled_equilibria/wout_maxmode3_nfp2_scaled_AriesCS_PHIEDGE=51.61979227917805.nc'
wout_name = './inputs/equilibria/scaled_equilibria/wout_maxmode4_nfp3_scaled_AriesCS_PHIEDGE=52.57297761698136.nc'
TEST_DIR = (Path(__file__).parent).resolve()
surface_filename = str(TEST_DIR/wout_name)


MAXITER = 100000
#minor_radius_factor_cws = 1.9



ntheta = 64
nphi = 32


mpol = 1
ntor = 0

JACOBIAN_THRESHOLD = 100

# CREATE SURFACES
s = SurfaceRZFourier.from_wout(surface_filename, range="half period", ntheta=ntheta, nphi=nphi)
s_full = SurfaceRZFourier.from_wout(surface_filename, range="full torus", ntheta=ntheta, nphi=int(nphi*2*s.nfp))
#cws = SurfaceRZFourier.from_nphi_ntheta(nphi, ntheta, "half period", s.nfp, mpol=mpol, ntor=ntor)
#cws_full = SurfaceRZFourier.from_nphi_ntheta(int(nphi*2*s.nfp), ntheta, "full torus", s.nfp, mpol=mpol, ntor=ntor)
cws  = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
cws.extend_via_projected_normal(2)
cws_full = SurfaceRZFourier.from_wout(surface_filename, range="full torus", nphi=nphi, ntheta=ntheta)
cws_full.extend_via_projected_normal(2)

if s.nfp==2:
    ncoils = 3
    
    # Threshold and weight for the maximum length of each individual coil:
    LENGTH_THRESHOLD = 80
    LENGTH_WEIGHT = 1

    # Threshold and weight for the coil-to-coil distance penalty in the objective function:
    CC_THRESHOLD = 0.5
    CC_WEIGHT = 10

    # Threshold and weight for the curvature penalty in the objective function:
    CURVATURE_THRESHOLD = 2.6
    CURVATURE_WEIGHT = 10
    
    # Threshold and weight for the mean squared curvature penalty in the objective function:
    MSC_THRESHOLD = 0.4
    MSC_WEIGHT = 10

    ARCLENGTH_WEIGHT = 1e-1
    
    order = 16  # order of dofs of cws curves
    quadpoints = 260*2
    
    tolerance = 1e-20
elif s.nfp ==3:
    ncoils = 4
    # Threshold and weight for the maximum length of each individual coil:
    LENGTH_THRESHOLD = 36
    LENGTH_WEIGHT = 1

    # Threshold and weight for the coil-to-coil distance penalty in the objective function:
    CC_THRESHOLD = 1.0
    CC_WEIGHT = 100

    # Threshold and weight for the curvature penalty in the objective function:
    CURVATURE_THRESHOLD = 33
    CURVATURE_WEIGHT = 0.1

    # Threshold and weight for the mean squared curvature penalty in the objective function:
    MSC_THRESHOLD = 3.5
    MSC_WEIGHT = 0.1

    ARCLENGTH_WEIGHT = 3e-5
    
    order = 12  # order of dofs of cws curves
    quadpoints = 100
    
    tolerance = 1e-20

R0 = s.get_rc(0, 0) - 1
R1 = s.get_zs(1, 0) + 3.5
#cws.rc[0, 0] = R0
#cws.rc[1, 0] = R1
#cws.zs[1, 0] = R1
cws.local_full_x = cws.get_dofs()
cws_full.x = cws.x


def create_cws_from_dofs(dofs_cws=cws.x, dofs_coils=None):
    base_curves = []
    for i in range(ncoils):
        curve_cws = CurveCWSFourier(
            mpol=cws.mpol,
            ntor=cws.ntor,
            idofs=dofs_cws,
            quadpoints=quadpoints,
            order=order,
            nfp=cws.nfp,
            stellsym=cws.stellsym,
        )
        angle = (i+0.5)*(2*np.pi)/((2)*s.nfp*ncoils)
        curve_dofs = np.zeros(len(curve_cws.get_dofs()),)
        curve_dofs[0] = 1
        curve_dofs[2*order+2] = 0
        curve_dofs[2*order+3] = angle
        curve_cws.set_dofs(curve_dofs)
        curve_cws.fix(0)
        curve_cws.fix(2*order+2)
        base_curves.append(curve_cws)
    base_currents = [Current(1)*1e5 for i in range(ncoils)]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    curves = [c.curve for c in coils]
    bs = BiotSavart(coils)
    bs.set_points(s.gamma().reshape((-1, 3)))
    Jf = SquaredFlux(s, bs,definition='local')
    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, CC_THRESHOLD)
    Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
    Jals = [ArclengthVariation(c) for c in base_curves]
    JF = (
        Jf
        + LENGTH_WEIGHT * sum(QuadraticPenalty(J, LENGTH_THRESHOLD) for J in Jls)
        + CC_WEIGHT * Jccdist
        + CURVATURE_WEIGHT * sum(Jcs)
        + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs)
        + ARCLENGTH_WEIGHT * sum(Jals)
    )
    if dofs_coils is not None:
        JF.x = dofs_coils
    return JF, bs, base_curves, curves, coils, Jf, Jls, Jccdist, Jcs, Jmscs, Jals


JF, bs, base_curves, curves, coils, Jf, Jls, Jccdist, Jcs, Jmscs, Jals = create_cws_from_dofs()

bs.set_points(s_full.gamma().reshape((-1, 3)))
curves_to_vtk(curves, OUT_DIR + "curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((int(nphi*2*s_full.nfp), ntheta, 3)) * s_full.unitnormal(), axis=2)[:, :, None]}
s_full.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)
cws_full.to_vtk(OUT_DIR + "cws_init")

bs.set_points(s.gamma().reshape((-1, 3)))

dofs_coils_length = len(JF.x)


def fun(dofs):
    JF.x = dofs
    J = JF.J()
    if np.isnan(J):
        J = JACOBIAN_THRESHOLD
        grad = [0]*len(dofs)
        print(f'J is nan, outputting J = {JACOBIAN_THRESHOLD}')
    else:
        coils_dJ = JF.dJ()
        grad = coils_dJ
        jf = Jf.J()
        BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
        outstr = f"J={J:.3e}, Jf={jf:.3e}, ⟨B·n⟩={BdotN:.1e}"
        cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
        outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
        print(outstr)
    return J, grad

dofs_coils = np.copy(JF.x)
dofs = dofs_coils

res = minimize(
    fun,
    dofs,
    jac=True,
    method="BFGS",
    #options={"maxiter": MAXITER},
    tol=tolerance,
)

bs.set_points(s_full.gamma().reshape((-1, 3)))
curves_to_vtk(curves, OUT_DIR + "curves_opt")
curves_to_vtk(base_curves, OUT_DIR + "base_curves_opt")
pointData = {"B_N": np.sum(bs.B().reshape((int(nphi*2*s_full.nfp), ntheta, 3)) * s_full.unitnormal(), axis=2)[:, :, None]}
s_full.to_vtk(OUT_DIR + "surf_opt", extra_data=pointData)
cws_full.to_vtk(OUT_DIR + "cws_opt")
bs.set_points(s.gamma().reshape((-1, 3)))

B_dot_n = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
BdotN = np.mean(np.abs(B_dot_n))
print('Final max|B dot n|:', np.max(np.abs(B_dot_n)))
#plot(coils + [s], engine="plotly", close=True)

cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
arc_string = ", ".join([f"{np.max(c.incremental_arclength())/np.min(c.incremental_arclength())-1:.2e}" for c in base_curves])

result = Results(JF.J(),Jf.J(), np.max(np.abs(B_dot_n)),  BdotN, 
                 LENGTH_WEIGHT * sum(Jls).J(), cl_string,  CC_WEIGHT * Jccdist.J(), 
                 Jccdist.shortest_distance(),
                 CURVATURE_WEIGHT * sum(Jcs).J(), kap_string, 
                 MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs).J(), msc_string,
                 ARCLENGTH_WEIGHT * sum(Jals).J(), arc_string)

#result.save(OUT_DIR + "results_short.txt")
result.save(OUT_DIR + "results.txt")

# Save the used parameters
param = Parameters(nphi, ntheta, ncoils, order, R0, R1, LENGTH_WEIGHT, LENGTH_THRESHOLD, CC_THRESHOLD, 
                 CC_WEIGHT, CURVATURE_THRESHOLD, CURVATURE_WEIGHT, 
                 MSC_THRESHOLD, MSC_WEIGHT,ARCLENGTH_WEIGHT, tolerance)

param.save(OUT_DIR + "parameters.txt")