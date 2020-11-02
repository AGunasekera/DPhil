from pyscf import gto, scf, ao2mo, fci
import CSF as CSFmod

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

CSF = CSFmod.gen_CSF(0, 0, [0,3])
print(CSF)
print(fci.spin_square(CSF, 2, (1,1)))

CSF = CSFmod.gen_CSF(0, 0, [1,2])
print(CSF)
print(fci.spin_square(CSF, 2, (1,1)))

CSF = CSFmod.gen_CSF(0, 0, [3,0])
print(CSF)
print(fci.spin_square(CSF, 2, (1,1)))

CSF = CSFmod.gen_CSF(1, 1, [1,1])
print(CSF)
print(fci.spin_square(CSF, 2, (2,0)))

CSF = CSFmod.gen_CSF(1, 0, [1,1])
print(CSF)
print(fci.spin_square(CSF, 2, (1,1)))

CSF = CSFmod.gen_CSF(1, -1, [1,1])
print(CSF)
print(fci.spin_square(CSF, 2, (0,2)))