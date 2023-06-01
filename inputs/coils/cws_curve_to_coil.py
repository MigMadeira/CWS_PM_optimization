from simsopt import load
from simsopt.geo import CurveXYZFourier, curves_to_vtk
from simsopt.field import Coil, BiotSavart
import numpy as np
from scipy.fft import rfft


coilfile = "biot_savart_opt_maxmode4_nfp3.json"
bs = load(coilfile)
coils = bs.coils
ncoils = len(coils)
base_curves = [coils[i].curve for i in range(ncoils)]
base_currents = [coils[i].current for i in range(ncoils)]

coil_data = []

order = 100

for curve in base_curves:
    xArr, yArr, zArr = np.transpose(curve.gamma())

    curves_Fourier = []

        # Compute the Fourier coefficients
    for x in [xArr, yArr, zArr]:
        assert len(x) >= 2*order  # the order of the fft is limited by the number of samples
        xf = rfft(x) / len(x)

        fft_0 = [xf[0].real]  # find the 0 order coefficient
        fft_cos = 2 * xf[1:order + 1].real  # find the cosine coefficients
        fft_sin = -2 * xf[:order + 1].imag  # find the sine coefficients

        combined_fft = np.concatenate([fft_sin, fft_0, fft_cos])
        curves_Fourier.append(combined_fft)

    coil_data.append(np.concatenate(curves_Fourier))
    
coil_data = np.asarray(coil_data)
coil_data = coil_data.reshape(6 * ncoils, order + 1)  # There are 6 * order coefficients per coil
coil_data = np.transpose(coil_data)

assert coil_data.shape[1] % 6 == 0
assert order <= coil_data.shape[0]-1

num_coils = coil_data.shape[1] // 6
ppp = 20

curves = [CurveXYZFourier(order*ppp, order) for i in range(num_coils)]
for ic in range(num_coils):
    dofs = curves[ic].dofs_matrix
    dofs[0][0] = coil_data[0, 6*ic + 1]
    dofs[1][0] = coil_data[0, 6*ic + 3]
    dofs[2][0] = coil_data[0, 6*ic + 5]
    for io in range(0, min(order, coil_data.shape[0] - 1)):
        dofs[0][2*io+1] = coil_data[io+1, 6*ic + 0]
        dofs[0][2*io+2] = coil_data[io+1, 6*ic + 1]
        dofs[1][2*io+1] = coil_data[io+1, 6*ic + 2]
        dofs[1][2*io+2] = coil_data[io+1, 6*ic + 3]
        dofs[2][2*io+1] = coil_data[io+1, 6*ic + 4]
        dofs[2][2*io+2] = coil_data[io+1, 6*ic + 5]
    curves[ic].local_x = np.concatenate(dofs)

converted_coils = []
for i in range(len(base_curves)):
    converted_coils.append(Coil(curves[i],base_currents[i])) 

bs = BiotSavart(converted_coils)
curves_to_vtk(curves,"coils_" + coilfile)
bs.save("coils_" + coilfile)