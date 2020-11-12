from pyscf import gto, scf, ao2mo, fci
import UGA

# 3-electron example: linear H3

bohr = 0.529177249

H2sep = 1.605 * bohr

mol = gto.Mole()
mol.verbose = 1
mol.atom = 'H 0 0 0; H 0 0 ' + str(H2sep) + '; H 0 0 ' + str(2 * H2sep)
mol.basis = 'sto-3g'
mol.spin = 1
mol.build()

Enuc = mol.energy_nuc()

mf = scf.ROHF(mol)
mf.kernel()

h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
eri = ao2mo.kernel(mol, mf.mo_coeff)

cisolver = fci.FCI(mol, mf.mo_coeff)

CSF = UGA.gen_CSF(1.5, 1.5, 3, [1,1,1])
print(CSF)
print(cisolver.energy(h1, eri, CSF, 3, (3,0)) + Enuc)
print(fci.spin_square(CSF, 3, (3,0)))

CSF = UGA.gen_CSF(1.5, 0.5, 3, [1,1,1])
print(CSF)
print(cisolver.energy(h1, eri, CSF, 3, (2,1)) + Enuc)
print(fci.spin_square(CSF, 3, (2,1)))

CSF = UGA.gen_CSF(0.5, 0.5, 3, [0,1,3])
print(CSF)
print(cisolver.energy(h1, eri, CSF, 3, (2,1)) + Enuc)
print(fci.spin_square(CSF, 3, (2,1)))

CSF = UGA.gen_CSF(0.5, 0.5, 3, [1,0,3])
print(CSF)
print(cisolver.energy(h1, eri, CSF, 3, (2,1)) + Enuc)
print(fci.spin_square(CSF, 3, (2,1)))

CSF = UGA.gen_CSF(0.5, 0.5, 3, [1,2,1])
print(CSF)
print(cisolver.energy(h1, eri, CSF, 3, (2,1)) + Enuc)
print(fci.spin_square(CSF, 3, (2,1)))

CSF = UGA.gen_CSF(0.5, 0.5, 3, [1,1,2])
print(CSF)
print(cisolver.energy(h1, eri, CSF, 3, (2,1)) + Enuc)
print(fci.spin_square(CSF, 3, (2,1)))