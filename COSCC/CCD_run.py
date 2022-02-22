from time import time
from GeneralisedWick import *
import CC
from pyscf import cc, mp
import pickle

#with open("CCDEquations.pkl", 'rb') as f:
with open("collectedCCDEquations.pkl", 'rb') as f:
    up = pickle.Unpickler(f)
    d = up.load()
    energyEquation = d["energyEquation"]
    doublesAmplitudeEquation = d["doublesAmplitudeEquation"]
    fockTensor = d["fockTensor"]
    h2Tensor = d["h2Tensor"]
    t2Tensor = d["t2Tensor"]

t0 = time()

bohr = 0.529177249

H2sep = 1.605 * bohr

mol = gto.Mole()
mol.verbose = 1
mol.atom = 'Ne 0 0 0'
#mol.basis = 'sto-3g'
#mol.atom = 'H 0 0 0; H 0 0 ' + str(H2sep)
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
t2Tensor.getShape(vacuum)

t1 = time()

print("\n")
print("Comparison with true answers")
print("MP2")
trueMP2 = mp.MP2(mf)
print(trueMP2.kernel())
t6 = time()
print("MP2 time:", t6-t1)

print("\n")
print("CCD")
trueCCD = cc.CCSD(mf)
old_update_amps = trueCCD.update_amps
def update_amps(t1, t2, eris):
    t1, t2 = old_update_amps(t1, t2, eris)
#    print(t1)
    return (np.zeros_like(t1[0]), np.zeros_like(t1[1])), t2
trueCCD.update_amps = update_amps
print(trueCCD.kernel())
t7 = time()
print("CCD time:", t7-t6)

residual = Tensor("R", ['p', 'p'], ['h', 'h'])
residual.array = contractTensorSum(doublesAmplitudeEquation)
print(residual.array)
print(h2Tensor.diagrams[12].array)
CC.iterateDoublesAmplitudes(t2Tensor, residual, fockTensor.array)
t5 = time()
print("Time for MP2 calculation:", t5 - t7)
print(contractTensorSum(energyEquation))
print(t2Tensor.array)
t8 = time()
CC.convergeDoublesAmplitudes(t2Tensor, energyEquation, doublesAmplitudeEquation, fockTensor, biorthogonal=True)
t2 = time()
print("Time for CCD calculation:", t2 - t8)

#print("CCD with collected equations")
#collectedEnergyEquation = energyEquation.collectIsomorphicTerms()
#collectedDoublesAmplitudeEquation = doublesAmplitudeEquation.collectIsomorphicTerms()
#print(collectedEnergyEquation)
#print(collectedDoublesAmplitudeEquation)
#t3 = time()
#CC.convergeCollectedDoublesAmplitudes(t2Tensor, collectedEnergyEquation, collectedDoublesAmplitudeEquation, fockTensor)
#t4 = time()
#print("Time for CCD calculation:", t4 - t3)
