from pyscf import gto, scf, ao2mo, fci
import UGA

# (Trivial) 2-electron example: STO-3G H2

bohr = 0.529177249

H2sep = 1.605 * bohr

mol = gto.Mole()
mol.verbose = 1
mol.atom = 'H 0 0 0; H 0 0 ' + str(H2sep)
mol.basis = 'sto-3g'
mol.spin = 0
mol.build()

Enuc = mol.energy_nuc()

mf = scf.ROHF(mol)
mf.kernel()

h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
eri = ao2mo.kernel(mol, mf.mo_coeff)

cisolver = fci.FCI(mol, mf.mo_coeff)

CSF = UGA.gen_CSF(0, 0, 2, [0,3])
print(CSF)
print(cisolver.energy(h1, eri, CSF, 2, (1,1)) + Enuc)
print(fci.spin_square(CSF, 2, (1,1)))

CSF = UGA.gen_CSF(0, 0, 2, [1,2])
print(CSF)
print(cisolver.energy(h1, eri, CSF, 2, (1,1)) + Enuc)
print(fci.spin_square(CSF, 2, (1,1)))

CSF = UGA.gen_CSF(0, 0, 2, [3,0])
print(CSF)
print(cisolver.energy(h1, eri, CSF, 2, (1,1)) + Enuc)
print(fci.spin_square(CSF, 2, (1,1)))

# [[1. 0.]
#  [0. 0.]]
# -1.1026388111917234
# (0.0, 1.0)

print(mf.e_tot)
#-1.1026388111917238

CSF = UGA.gen_CSF(1, 1, 2, [1,1])
print(CSF)
print(cisolver.energy(h1, eri, CSF, 2, (2,0)) + Enuc)
print(fci.spin_square(CSF, 2, (2,0)))

CSF = UGA.gen_CSF(1, 0, 2, [1,1])
print(CSF)
print(cisolver.energy(h1, eri, CSF, 2, (1,1)) + Enuc)
print(fci.spin_square(CSF, 2, (1,1)))

CSF = UGA.gen_CSF(1, -1, 2, [1,1])
print(CSF)
print(cisolver.energy(h1, eri, CSF, 2, (0,2)) + Enuc)
print(fci.spin_square(CSF, 2, (0,2)))



# Dissociation
H2sep = 10 * bohr

Enuc = mol.energy_nuc()

mf = scf.ROHF(mol)
mf.kernel()

h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
eri = ao2mo.kernel(mol, mf.mo_coeff)

cisolver = fci.FCI(mol, mf.mo_coeff)

# RHF determinant
CSF = UGA.gen_CSF(0, 0, 2, [3,0])
print(CSF)
print(cisolver.energy(h1, eri, CSF, 2, (1,1)) + Enuc)
print(fci.spin_square(CSF, 2, (1,1)))
# [[1. 0.]
#  [0. 0.]]
# -0.5959706311340207
# (0.0, 1.0)

print(mf.e_tot)
#-0.5959706311340206

# UHF
uhf = scf.UHF(mol)
uhf.kernel()
print(uhf.e_tot)
#-0.5959706311340205
# Has not broken symmetry

cisolver.kernel()
#(-0.9331637120042693,
# array([[-7.07221980e-01, -7.81563703e-12],
#        [-7.81540122e-12,  7.06991563e-01]]))

# Broken-symmetry FCI solution

#Now same again, but with atomic orbitals, ignoring MOs
h1at = mf.get_hcore()

import numpy as np

eriat = ao2mo.kernel(mol, np.array([[1.,0.],[0.,1.]]))
#array([[7.74605944e-01, 8.68188091e-06, 9.99999928e-02],
#       [8.68188091e-06, 8.58357615e-10, 8.68188091e-06],
#       [9.99999928e-02, 8.68188091e-06, 7.74605944e-01]])

cisolverat = fci.FCI(mol, np.array([[1.,0.],[0.,1.]]))

cisolverat.kernel()
#(-0.9331637200129319,
# array([[-1.26915301e-04, -7.07106776e-01],
#        [-7.07106764e-01, -1.26915301e-04]]))
#FCI energy same (to 7sf)

#Open-shell singlet
CSF = UGA.gen_CSF(0, 0, 2, [1,2])
#array([[0.        , 0.70710678],
#       [0.70710678, 0.        ]])

print(cisolverat.energy(h1at, eriat, CSF, 2, (1,1)) + Enuc)
#-0.9331636982805085
#Close to FCI energy (6sf)