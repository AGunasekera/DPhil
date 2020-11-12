from pyscf import gto, scf, ao2mo, fci
import UGA


#bohr = 0.529177249

N2sep = 1.09
O2sep = 1.21

mol = gto.Mole()
mol.verbose = 1
mol.atom = 'O 0 0 0; O 0 0 ' + str(O2sep)
mol.basis = 'sto-3g'
mol.spin = 0
mol.build()

Enuc = mol.energy_nuc()

mf = scf.ROHF(mol)
mf.kernel()

print mf.e_tot

h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
eri = ao2mo.kernel(mol, mf.mo_coeff)

cisolver = fci.FCI(mol, mf.mo_coeff)

CSF = UGA.gen_CSF(0, 0, 10, [3,3,3,3,3,3,3,1,2,0])
print(CSF)
print(cisolver.energy(h1, eri, CSF, 10, (8,8)) + Enuc)
print(fci.spin_square(CSF, 10, (8,8)))

CSF = UGA.gen_CSF(0, 0, 10, [3,3,3,3,3,3,0,3,0,0])
print(CSF)
print(cisolver.energy(h1, eri, CSF, 10, (7,7)) + Enuc)
print(fci.spin_square(CSF, 10, (7,7)))

CSF = UGA.gen_CSF(0, 0, 10, [3,3,3,3,3,1,1,2,2,0])
print(CSF)
print(cisolver.energy(h1, eri, CSF, 10, (7,7)) + Enuc)
print(fci.spin_square(CSF, 10, (7,7)))