import numpy as np
from pyscf import gto, scf, ao2mo, fci
from Wick import *
from math import factorial, sqrt
from scipy.optimize import fsolve
import itertools

def fermiVacuum(mf_):
    Norbs = mf_.mol.nao
    Nocc = mf_.nelectron_alpha
    return [1 for i in range(Nocc)] + [0 for i in range (Norbs - Nocc)]

def spinFreeSingleExcitation(p, q):
    summandList = []
    for spin in range(2):
        summandList.append(operatorProduct([basicOperator(p,1,spin), basicOperator(q,0,spin)]))
    return operatorSum(summandList)

def spinFreeDoubleExcitation(p, q, r, s):
    summandList = []
    for spin1 in range(2):
        for spin2 in range(2):
            summandList.append(operatorProduct([basicOperator(p,1,spin2), basicOperator(q,1,spin1), basicOperator(s,0,spin1), basicOperator(r,0,spin2)]))
    return operatorSum(summandList)

def spinFreeNtupleExcitation(creTuple, annTuple):
    N = len(creTuple)
    if len(annTuple) != N:
        print("particle number not conserved")
        return 0
    summandList = []
    spinCombs = itertools.product(range(2), repeat=N)
    for spinComb in spinCombs:
        productList = []
        for o in reversed(range(N)):
            productList = [basicOperator(creTuple[o], 1, spinComb[o])] + productList + [basicOperator(annTuple[o], 0, spinComb[o])]
        summandList.append(operatorProduct(productList))
    return operatorSum(summandList)

def get1bodyHamiltonian(mf_):
    h1 = mf_.mo_coeff.T.dot(mf_.get_hcore()).dot(mf_.mo_coeff)
    Norbs_ = mf_.mol.nao
    hamiltonian1Body = operatorSum([])
    for p in range(Norbs_):
        for q in range(Norbs_):
            hamiltonian1Body = hamiltonian1Body + h1[p, q] * spinFreeSingleExcitation(p, q)
    return hamiltonian1Body

def get2bodyHamiltonian(mf_):
    eri = ao2mo.kernel(mf_.mol, mf_.mo_coeff)
    Norbs_ = mf_.mol.nao
    hamiltonian2Body = operatorSum([])
    for p in range(Norbs_):
        for q in range(Norbs_):
            for r in range(p + 1):
                for s in range(q + 1):
                    x = int(p + Norbs_ * r - 0.5 * r * (r + 1))
                    y = int(q + Norbs_ * s - 0.5 * s * (s + 1))
                    if p == r and q == s:
                        hamiltonian2Body = hamiltonian2Body + 0.5 * eri[x, y] * spinFreeDoubleExcitation(p, q, r, s)
                    else:
                        hamiltonian2Body = hamiltonian2Body + 0.5 * eri[x, y] * spinFreeDoubleExcitation(p, q, r, s) + 0.5 * np.conjugate(eri[x, y]) * spinFreeDoubleExcitation(r, s, p, q)
    return hamiltonian2Body

def getClusterSingles(singlesAmplitudes_):
    singlesAmplitudeShape = singlesAmplitudes_.shape
    clusterSingles = operatorSum([])
    for a in range(singlesAmplitudeShape[0]):
        for k in range(singlesAmplitudeShape[1]):
            r = k + singlesAmplitudeShape[0]
            clusterSingles = clusterSingles + singlesAmplitudes_[a,k] * spinFreeSingleExcitation(r, a)
    clusterSingles.checkNilpotency()
    return clusterSingles

def getClusterDoubles(doublesAmplitudes_):
    doublesAmplitudeShape = doublesAmplitudes_.shape
    clusterDoubles = operatorSum([])
    for a in range(doublesAmplitudeShape[0]):
        for b in range(doublesAmplitudeShape[1]):
            for k in range(doublesAmplitudeShape[2]):
                r = k + doublesAmplitudeShape[0]
                for l in range(doublesAmplitudeShape[3]):
                    s = l + doublesAmplitudeShape[1]
                    clusterDoubles = clusterDoubles + doublesAmplitudes_[a,b,k,l] * spinFreeDoubleExcitation(r, s, a, b)
    clusterDoubles.checkNilpotency()
    return clusterDoubles

def exponentialOperator(operator, maxOrder):
    exponential = operatorSum([operatorProduct([], 1.)])
    for k in range(maxOrder):
        exponential += (1 / factorial(k + 1)) * pow(operator, k + 1)
    return exponential

def commutator(operator1, operator2):
    return operator1 * operator2 + (-1) * operator2 * operator1

def BCHSimilarityTransform(H, T, order):
    result = H
    for k in range(order):
        nestedCommutator = H
        for i in range(k + 1):
            nestedCommutator = commutator(nestedCommutator, T)
        result += (1 / factorial(k + 1)) * nestedCommutator
    result.checkNilpotency()
    return result

def genNthOrderSpinFreeExcitations(excitationsList, Nocc_, Norbs_, order):
    occupiedCombinations = itertools.product(range(Nocc_), repeat=order)
    for o in occupiedCombinations:
        virtualCombinations = itertools.combinations_with_replacement(range(Nocc_, Norbs_), order)
        for v in virtualCombinations:
            spinFreeExcitation = spinFreeNtupleExcitation(v, o)
            spinFreeExcitation.checkNilpotency()
            excitationsList.append(spinFreeExcitation)

def genExcitationList(Nocc_, Norbs_, excitationOrders):
    excitations = []
    for order in excitationOrders:
        genNthOrderSpinFreeExcitations(excitations, Nocc_, Norbs_, order)
    return excitations

def getClusterOperator(excitationList, amplitudeList):
    if len(excitationList) != len(amplitudeList):
        print("different number of amplitudes and excitations")
        return 0
    clusterOperator = operatorSum([])
    for i in range(len(excitationList)):
        clusterOperator = clusterOperator + amplitudeList[i] * excitationList[i]
    return clusterOperator

def testDoublesAmplitudes(hamiltonian_, doublesAmplitudes_, vacuum):
    doublesAmplitudeShape = doublesAmplitudes_.shape
    newbch = BCHSimilarityTransform(hamiltonian_, getClusterDoubles(doublesAmplitudes_), 4)
    newbch.checkNilpotency()
    excitations = []
    for a in range(doublesAmplitudeShape[0]):
        for b in range(doublesAmplitudeShape[1]):
            for k in range(doublesAmplitudeShape[2]):
                r = k + doublesAmplitudeShape[0]
                for l in range(doublesAmplitudeShape[3]):
                    s = l + doublesAmplitudeShape[1]
                    excitations.append(spinFreeDoubleExcitation(r, s, a, b))
    projectedVEVs = []
    for excitation in excitations:
        newprojected = excitation.conjugate() * newbch
        projectedVEVs.append(vacuumExpectationValue(newprojected, vacuum))
    return projectedVEVs

def testFunc(flattenedDoublesAmplitudes):
#    doublesAmplitudes = flattenedDoublesAmplitudes.reshape()
    test = testDoublesAmplitudes(flattenedDoublesAmplitudes)
    print(flattenedDoublesAmplitudes, test)
    return test

def testAmplitudes(amplitudes_, excitations_, hamiltonian_, vacuum):
    newbch = BCHSimilarityTransform(hamiltonian_, getClusterOperator(excitations_, amplitudes_), 4)
    newbch.checkNilpotency()
    projectedVEVs = []
    for excitation in excitations_:
        newprojected = excitation.conjugate() * newbch
        projectedVEVs.append(vacuumExpectationValue(newprojected, vacuum))
    return projectedVEVs

def coupledCluster(mf_, excitationOrders):
    Norbs = mf_.mol.nao
    Nocc = mf_.nelectron_alpha
    vacuum = [1 for i in range(Nocc)] + [0 for i in range (Norbs - Nocc)]
    hamiltonian = get1bodyHamiltonian(mf_) + get2bodyHamiltonian(mf_)

    excitations = genExcitationList(Nocc, Norbs, excitationOrders)
    print(len(excitations))
    initialAmplitudes = np.zeros(len(excitations))
    finalAmplitudes = fsolve(testAmplitudes, initialAmplitudes, args=(excitations, hamiltonian, vacuum))
    return getClusterOperator(excitations, finalAmplitudes)

def CCD(mf_):
    Norbs = mf_.mol.nao
    Nocc = mf_.nelectron_alpha
    doublesAmplitudes = np.zeros((Nocc, Nocc, Norbs-Nocc, Norbs-Nocc))
