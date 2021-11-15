from time import time
from GeneralisedWick import *
import CC, texify
from pyscf import cc, mp

fockTensor = Tensor("f", ['g'], ['g'])
h2Tensor = Tensor("v", ['g', 'g'], ['g', 'g'])

fockTensor.getAllDiagramsGeneral()
h2Tensor.getAllDiagramsGeneral()

t1Tensor = Tensor("{t_{1}}", ['p'], ['h'])
t2Tensor = Tensor("{t_{2}}", ['p', 'p'], ['h', 'h'])

normalOrderedHamiltonian = sum(fockTensor.diagrams) + (1. / 2.) * sum(h2Tensor.diagrams)
BCH = CC.BCHSimilarityTransform(normalOrderedHamiltonian, t1Tensor + 0.25 * t2Tensor, 4)

t0 = time()
energyEquation = CC.getEnergyEquation(BCH)
t1 = time()
print("Time to find energy equation:", t1 - t0)
print("number of terms:", len(energyEquation.summandList))
singlesAmplitudeEquation = CC.getAmplitudeEquation(BCH, 1)
t2 = time()
print("Time to find singles amplitude equation:", t2 - t1)
print("number of terms:", len(singlesAmplitudeEquation.summandList))
doublesAmplitudeEquation = CC.getAmplitudeEquation(BCH, 2)
t3 = time()
print("Time to find doubles amplitude equation:", t3 - t2)
print("number of terms:", len(doublesAmplitudeEquation.summandList))

#########

bohr = 0.529177249

H2sep = 1.605 * bohr

mol = gto.Mole()
mol.verbose = 1
#mol.atom = 'Be 0 0 0'
#mol.basis = 'sto-3g'
mol.atom = 'H 0 0 0; H 0 0 ' + str(H2sep)
mol.basis = '6-31g'
mol.spin = 0
mol.build()

Enuc = mol.energy_nuc()

mf = scf.ROHF(mol)
mf.kernel()

cisolver = fci.FCI(mol, mf.mo_coeff)

Norbs = mol.nao
Nocc = mf.nelectron_alpha
vacuum = [1] * Nocc + [0] * (Norbs - Nocc)

h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
eri = ao2mo.kernel(mol, mf.mo_coeff, compact=False)

h2Tensor.array = eri.reshape((Norbs, Norbs, Norbs, Norbs)).swapaxes(2,3).swapaxes(1,2)

fock = h1
for p in range(Norbs):
    for q in range(Norbs):
        fock[p,q] += sum([2 * h2Tensor.array[p,i,q,i] - h2Tensor.array[p,i,i,q] for i in range(Nocc)])
fockTensor.array = fock

fockTensor.assignDiagramArrays(vacuum)
h2Tensor.assignDiagramArrays(vacuum)
t1Tensor.getShape(vacuum)
t2Tensor.getShape(vacuum)

CC.convergeCCSDAmplitudes(t1Tensor, t2Tensor, energyEquation, singlesAmplitudeEquation, doublesAmplitudeEquation, fockTensor)
t4 = time()
print("Time for CCSD calculation:", t4 - t3)

print("\n")
print("Comparison with true answers")
print("MP2")
trueMP2 = mp.MP2(mf)
print(trueMP2.kernel())
t5 = time()
print("MP2 time:", t5-t4)

print("\n")
print("CCSD")
trueCCSD = cc.CCSD(mf)
print(trueCCSD.kernel())
t6 = time()
print("CCSD time:", t6-t5)

texify.texify([energyEquation, singlesAmplitudeEquation, doublesAmplitudeEquation], "CCSDEquations")