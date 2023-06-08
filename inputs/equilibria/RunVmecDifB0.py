import os
import glob
import shutil
import numpy as np
from pathlib import Path
from simsopt.mhd import Vmec
from simsopt.util import MpiPartition

def append_to_line(f, i, text):
    lines = f.readlines()  # Read all lines into a list
    if i > len(lines):
        raise ValueError("Line number out of range")  # Check if line number is out of range
    splitted = lines[i-1].split(sep = " =")
    lines[i-1] = splitted[0] + " = " + str(text) + '\n'  # Append the text to the end of the line
    f.seek(0)  # Move the file pointer to the beginning of the file
    f.writelines(lines)  # Write the modified lines back to the file
    
mpi = MpiPartition()
ntheta_VMEC = 91
nphi_VMEC = 91
nfp = 2


PHIEDGE = [51.4468222437525*(5.86461221551616/1.01405906647965)*(5.86461221551616/33.8031872281897)] #for nfp = 2
#PHIEDGE = [(5.86461221551616/0.999619663142818)*(5.86461221551616/0.654457178533868)] #for nfp = 3

results_path = os.path.join(os.path.dirname(__file__), 'scaled_equilibria')
Path(results_path).mkdir(parents=True, exist_ok=True)


vmec = Vmec('input.maxmode3_nfp2', mpi=mpi, verbose=True, ntheta=ntheta_VMEC, nphi=nphi_VMEC*2*nfp)

rescale = (1.7/0.10780687303734) #Aries has a 1.7 minor radius and our equilibrium has 0.10780687303734 for nfp=2
#rescale = (1.7/0.108681855858562) #0.108681855858562 for nfp=3

print(vmec.boundary.rc.shape[0])
     
for m in range(vmec.boundary.mpol):
    for n in range(-vmec.boundary.ntor,vmec.boundary.ntor+1):
        print(n,m)
        vmec.boundary.set_rc(m,n,rescale*vmec.boundary.get_rc(m,n)) 
        vmec.boundary.set_zs(m,n,rescale*vmec.boundary.get_zs(m,n)) 

filename = 'input.maxmode3_nfp2_scaled_AriesCS' 

os.chdir(results_path)

vmec.write_input(filename)

for phi_edge in PHIEDGE:
    shutil.copyfile(filename, filename + '_copy')
    with open(filename + '_copy','r+') as f:
        append_to_line(f,16, phi_edge)
        
    vmec = Vmec(filename + '_copy', mpi=mpi, verbose=True, ntheta=ntheta_VMEC, nphi=nphi_VMEC*2*nfp)
    vmec.run()

    shutil.move("wout_" + filename.split(sep=".")[1] + "_copy_000_000000.nc", "wout_" + filename.split(sep=".")[1] + f"_PHIEDGE=" + str(phi_edge) + ".nc")
    os.remove(filename + '_copy')

  
for objective_file in glob.glob(f"jac_*"): os.remove(objective_file)
for objective_file in glob.glob(f"jac_*"): os.remove(objective_file)
for objective_file in glob.glob(f"objective_*"): os.remove(objective_file)
for objective_file in glob.glob(f"residuals_*"): os.remove(objective_file)
for objective_file in glob.glob(f"*000_*"): os.remove(objective_file)
for objective_file in glob.glob(f"parvmec*"): os.remove(objective_file)
for objective_file in glob.glob(f"threed*"): os.remove(objective_file)