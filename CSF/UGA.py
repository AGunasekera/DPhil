import numpy as np
from math import sqrt, factorial
from pyscf import fci#, gto, scf, ao2mo

# class CSF:
#     def __init__(self, mol_, mf_, spin, projSpin, d):
#         self.mol = mol
#         self.mo_coeff = mf_.mo_coeff
# #        self.mo_occ = mf_.mo_occ
#         self.S = spin
#         self.M = projSpin
#         self.dvec = d

#    def couple_new_orbital(case):
#    if case == 0:
#    elif case == 1:
#        self.S = self.S + 0.5
#    elif case == 2:
#        self.S = self.S - 0.5
#    elif case == 3:

#    def get_CIExpansion(self):



# E001 = cisolver.energy(h1[0,0], eri[0,0], np.array([1.]),1,0) + Enuc
# E010 = cisolver.energy(h1[0,0], eri[0,0], np.array([1.]),1,1) + Enuc
# E100 = cisolver.energy(h1[0,0], eri[0,0], np.array([1.]),1,2) + Enuc
# E101 = cisolver.energy(h1, eri, np.array([1.,0.,0.,0.]),2,2) + Enuc
# TripE020 = cisolver.energy(h1, eri, np.array([0.,1/sqrt(2),-1/sqrt(2),0.]),2,2) + Enuc
# SingE020 = cisolver.energy(h1, eri, np.array([0.,1/sqrt(2),1/sqrt(2),0.]),2,2) + Enuc

def PartialCoreH(h1, norb):
    """
    Restrict a given 1-electron Hamiltonian to only the first norb orbitals
    """
    return h1[:norb,:norb]

def PartialERI(eri, norb):
    """
    Restrict a given 2-electron Hamiltonian to only the first norb orbitals
    """
    n = int(norb * (norb + 1) / 2)
    return eri[:n,:n]

def PartialEnergy(cisolver_, h1ematrix, erimatrix, CIExpansion, norb, nelec, Enuc_):
    """
    Energy of a given CIExpansion, restricted to only the first norb orbitals
    """
    return cisolver_.energy(PartialCoreH(h1ematrix, norb), PartialERI(erimatrix, norb), CIExpansion, norb, nelec) + Enuc_

def CGCoeff(Sold, Mold, s, m, S, M):
    '''
    General Clebsh--Gordan coefficient for addition of |s,m> orbital to |Sold, Mold> state to generate |S,M> state.
    '''
    if (Mold + m == M):
        A = sqrt((2 * S + 1) * factorial(int(S + Sold - s)) * factorial(int(S - Sold + s)) * factorial(int(Sold + s - S)) / factorial(int(Sold + s + S + 1)))
        B = sqrt(factorial(int(S + M)) * factorial(int(S - M)) * factorial(int(Sold + Mold)) * factorial(int(Sold - Mold)) * factorial(int(s + m)) * factorial(int(s - m)))
        C = 0
        for k in range(int(max(Sold + s - S, Sold - Mold, s + m)) + 1):
            if (Sold + s - S < k) or (Sold - Mold < k) or (s + m < k) or (S - s + Mold < -k) or (S - Sold - m < -k):
                C = C
            else:
                C += pow(-1, k) / (factorial(k) * factorial(int(Sold + s - S - k)) * factorial(int(Sold - Mold - k)) * factorial(int(s + m - k)) * factorial(int(S - s + Mold + k)) * factorial(int(S - Sold - m + k)))
        return A * B * C
    return 0

#cisolver.kernel()

# class Determinant:
#     def __init__(self, mol, mf):
#         self.mol = mol
#         self.mo_coeff = mf.mo_coeff
#         self.mo_occ = mf.mo_occ
    
#     def get_1e_ints(self):
#         gto.

#     def get_energy(self):


# #Energy of a linear combination of orthonormal Slater determinants
# def LCSD_energy(coeffs, dets):
#     E = 0
#     for d in range(len(coeffs)):
#         c = coeffs[d]
#         E += np.conjugate(c) * c * dets[d].get_energy()
#     return E

# def determinant_to_cibasis(stringa, stringb, neleca, nelecb, norb):
#     addra = fci.cistring.str2addr(norb, neleca, stringa)
#     addrb = fci.cistring.str2addr(norb, nelecb, stringb)

#     nstrsa = fci.cistring.num_strings(norb, neleca)
#     nstrsb = fci.cistring.num_strings(norb, nelecb)

#     cibasis = np.zeros((nstrsa, nstrsb))
#     cibasis[addra, addrb] = 1

#     return cibasis

def gen_CSF_summand(S, M, norbs, dvec, mvec):
    """
    Generate one term (corresponding to a given mvec)
    in the summation of a CSF of given S, M, and dvec
    by the Yamanouchi--Kotani scheme
    """
# Set up vacuum state
    neleca, nelecb = 0, 0
    state = np.array([[1.]])

    i = len(dvec)
    Si = S
    Mi = M
    CGProd = 1

# Populate orbitals using second-quantized algebra in fci.addons
    while i > 0:
        i = i - 1
        mi = mvec[i]
        case = dvec[i]
        if case == 0:
            state = state
            CGProd = CGProd * CGCoeff(Si, Mi, 0, 0, Si, Mi)
        elif case == 1:
            if mi == 0.5:
                state = fci.addons.cre_a(state, norbs, (neleca, nelecb), i)
                CGProd = CGProd * CGCoeff(Si - 0.5, Mi - 0.5, 0.5, 0.5, Si, Mi)
                neleca += 1
                Mi = Mi - 0.5
            elif mi == -0.5:
                state = fci.addons.cre_b(state, norbs, (neleca, nelecb), i)
                CGProd = CGProd * CGCoeff(Si - 0.5, Mi + 0.5, 0.5, -0.5, Si, Mi)
                nelecb += 1
                Mi = Mi + 0.5
            Si = Si - 0.5
        elif case == 2:
            if mi == 0.5:
                state = fci.addons.cre_a(state, norbs, (neleca, nelecb), i)
                CGProd = CGProd * CGCoeff(Si + 0.5, Mi - 0.5, 0.5, 0.5, Si, Mi)
                neleca += 1
                Mi = Mi - 0.5
            elif mi == -0.5:
                state = fci.addons.cre_b(state, norbs, (neleca, nelecb), i)
                CGProd = CGProd * CGCoeff(Si + 0.5, Mi + 0.5, 0.5, -0.5, Si, Mi)
                nelecb += 1
                Mi = Mi + 0.5
            Si = Si + 0.5
        elif case == 3:
            state = fci.addons.cre_b(state, norbs, (neleca, nelecb), i)
            nelecb += 1
            state = fci.addons.cre_a(state, norbs, (neleca, nelecb), i)
            neleca += 1
            state = state
            CGProd = CGProd * CGCoeff(Si, Mi, 0, 0, Si, Mi)

    return state * CGProd

def gen_mvec_list(M, dvec):
    """
    Generate list of mvecs commensurate with the given dvec and the total M
    """
    nocc = len(dvec)
    mvecs = np.reshape(np.zeros(len(dvec)), (1, -1))

    for i in range(nocc):
        if dvec[i] == 1 or dvec[i] == 2:
            newmvecs = mvecs.copy()
            for v in range(len(mvecs)):
                mvecs[v,i] = 0.5
                newmvecs[v,i] = -0.5
            mvecs = np.concatenate((mvecs, newmvecs))
        elif dvec[i] == 0 or dvec[i] == 3:
            mvecs = mvecs
    vecs_to_delete = []
    for v in range(len(mvecs)):
        sum_mi = 0
        si = 0
        for i in range(len(mvecs[v])):
            sum_mi = sum_mi + mvecs[v,i]
            if dvec[i] == 1:
                si = si + 0.5
            elif dvec[i] == 2:
                si = si - 0.5
            if abs(sum_mi) > si:
                vecs_to_delete.append(v)
        if np.sum(mvecs[v]) != M and v not in vecs_to_delete:
            vecs_to_delete.append(v)
    return np.delete(mvecs, vecs_to_delete, axis = 0)

def gen_CSF(S, M, norbs, dvec):
    '''
    Generate the CI wavevector for Yamanouchi--Kotani state
    equivalent (up to phase) to Gel'fand--Tsetlin state corresponding to step vector dvec,
    with remaining (norbs - len(dvec)) orbitals empty

    Uses PySCF framework for CI wavevectors:
    Each determinant for alpha or beta spins is referred to by an occupation number bitstring, fci.cistring.
    Each of these bitstrings is assigned an address, which is an integer starting from 0.
    A CI wavevector represents the coefficients of slater determinants as a matrix,
    with rows representing the same alpha-spin determinant, and columns representing the same beta-spin determinant.
    '''
    mvecList = gen_mvec_list(M, dvec)

    CSF = 0

    for i in range(len(mvecList)):
        mvec = mvecList[i]
        CSF = CSF + gen_CSF_summand(S, M, norbs, dvec, mvec)

    return CSF

# Unitary group generators and products
def UG_generator_pq_alpha(state, norbs, nelec, p, q):
    return fci.addons.cre_a(fci.addons.des_a(state, norbs, nelec, q), norbs, (nelec[0]-1, nelec[1]), p)

def UG_generator_pq_beta(state, norbs, nelec, p, q):
    return fci.addons.cre_b(fci.addons.des_b(state, norbs, nelec, q), norbs, (nelec[0], nelec[1]-1), p)

def UG_generator_pq(state, norbs, nelec, p, q):
    return UG_generator_pq_alpha(state, norbs, nelec, p, q) + UG_generator_pq_beta(state, norbs, nelec, p, q)

def UG_2particle_replacement(state, norbs, nelec, p1, p2, q1, q2):
    if p1 == q2:
        return UG_generator_pq(UG_generator_pq(state, norbs, nelec, p2, q2), norbs, nelec, p1, q1) - UG_generator_pq(state, norbs, nelec, p2, q1)
    return UG_generator_pq(UG_generator_pq(state, norbs, nelec, p2, q2), norbs, nelec, p1, q1)

def UG_triple_generator_product(state, norbs, nelec, p1, p2, p3, q1, q2, q3):
    return UG_generator_pq(UG_generator_pq(UG_generator_pq(state, norbs, nelec, p3, q3), norbs, nelec, p2, q2), norbs, nelec, p1, q1)

# Spin-adapted excitation operators for closed shell states
def closedshell_singleexcitation(state, norbs, nelec, a, r):
    return UG_generator_pq(state, norbs, nelec, a, r)

def closedshell_1_doubleexcitation(state, norbs, nelec, a, b, r, s):
    return (1 / 2) * (UG_2particle_replacement(state, norbs, nelec, a, b, r, s) + UG_2particle_replacement(state, norbs, nelec, a, b, s, r))

def closedshell_3_doubleexcitation(state, norbs, nelec, a, b, r, s):
    return (1 / 2) * (UG_2particle_replacement(state, norbs, nelec, a, b, r, s) - UG_2particle_replacement(state, norbs, nelec, a, b, s, r))

def closedshell_2_11_tripleexcitation(state, norbs, nelec, a, b, c, r, s, t):
    return (1 / 6) * (2 * UG_triple_generator_product(state, norbs, nelec, a, b, c, r, s, t) + 2 * UG_triple_generator_product(state, norbs, nelec, a, b, c, s, r, t) - UG_triple_generator_product(state, norbs, nelec, a, b, c, t, s, r) - UG_triple_generator_product(state, norbs, nelec, a, b, c, r, t, s) - UG_triple_generator_product(state, norbs, nelec, a, b, c, t, r, s) - UG_triple_generator_product(state, norbs, nelec, a, b, c, s, t, r))

def closedshell_2_12_tripleexcitation(state, norbs, nelec, a, b, c, r, s, t):
    return (sqrt(3) / 6) * (UG_triple_generator_product(state, norbs, nelec, a, b, c, r, t, s) - UG_triple_generator_product(state, norbs, nelec, a, b, c, t, s, r) + UG_triple_generator_product(state, norbs, nelec, a, b, c, s, t, r) - UG_triple_generator_product(state, norbs, nelec, a, b, c, t, r, s))

def closedshell_2_21_tripleexcitation(state, norbs, nelec, a, b, c, r, s, t):
    return (sqrt(3) / 6) * (UG_triple_generator_product(state, norbs, nelec, a, b, c, r, t, s) - UG_triple_generator_product(state, norbs, nelec, a, b, c, t, s, r) - UG_triple_generator_product(state, norbs, nelec, a, b, c, s, t, r) + UG_triple_generator_product(state, norbs, nelec, a, b, c, t, r, s))

def closedshell_2_22_tripleexcitation(state, norbs, nelec, a, b, c, r, s, t):
    return (1 / 6) * (2 * UG_triple_generator_product(state, norbs, nelec, a, b, c, r, s, t) - 2 * UG_triple_generator_product(state, norbs, nelec, a, b, c, s, r, t) + UG_triple_generator_product(state, norbs, nelec, a, b, c, t, s, r) + UG_triple_generator_product(state, norbs, nelec, a, b, c, r, t, s) - UG_triple_generator_product(state, norbs, nelec, a, b, c, t, r, s) - UG_triple_generator_product(state, norbs, nelec, a, b, c, s, t, r))

def closedshell_4_tripleexcitation(state, norbs, nelec, a, b, c, r, s, t):
    return (1 / 6) * (UG_triple_generator_product(state, norbs, nelec, a, b, c, r, s, t) - UG_triple_generator_product(state, norbs, nelec, a, b, c, s, r, t) - UG_triple_generator_product(state, norbs, nelec, a, b, c, t, s, r) - UG_triple_generator_product(state, norbs, nelec, a, b, c, r, t, s) + UG_triple_generator_product(state, norbs, nelec, a, b, c, t, r, s) + UG_triple_generator_product(state, norbs, nelec, a, b, c, s, t, r))

# Spin-adapted excitation operators for doublet states
def doubletstate_S_0_1(state, norbs, nelec, a, k):
    return UG_generator_pq(state, norbs, nelec, a, k)

def doubletstate_S_0_2(state, norbs, nelec, k, r):
    return UG_generator_pq(state, norbs, nelec, k, r)

def doubletstate_D_1_1(state, norbs, nelec, a, b, r, k):
    return sqrt(2 / (1 + (if a == b))) * (1 / 2) * (UG_2particle_replacement(state, norbs, nelec, a, b, r, k) + UG_2particle_replacement(state, norbs, nelec, a, b, k, r))

def doubletstate_D_1_1prime(state, norbs, nelec, a, b, r, k):
    return sqrt(2 / 3) * (1 / 2) * (UG_2particle_replacement(state, norbs, nelec, a, b, r, k) - UG_2particle_replacement(state, norbs, nelec, a, b, k, r))

def doubletstate_D_1_2(state, norbs, nelec, a, k, r, s):
    return sqrt(2 / (1 + (if r == s))) * (1 / 2) * (UG_2particle_replacement(state, norbs, nelec, a, k, r, s) + UG_2particle_replacement(state, norbs, nelec, a, k, r, s))

def doubletstate_D_1_2prime(state, norbs, nelec, a, k, r, s):
    return sqrt(2 / 3) * (1 / 2) * (UG_2particle_replacement(state, norbs, nelec, a, k, r, s) - UG_2particle_replacement(state, norbs, nelec, a, k, s, r))

def doubletstate_D_1_3(state, norbs, nelec, a, b, r, s):
    return (1 / sqrt((1 + (if a == b)) * (1 + (if a == b)))) * (1 / 2) * (UG_2particle_replacement(state, norbs, nelec, a, b, r, s) + UG_2particle_replacement(state, norbs, nelec, a, b, s, r))

def doubletstate_D_2_3(state, norbs, nelec, a, b, r, s, k):
    return (2 / sqrt(3 * (1 + (if r == s)))) * (1 / 2) * (1 / 2) * ((UG_triple_generator_product(state, norbs, nelec, k, a, b, r, s, k) - UG_triple_generator_product(state, norbs, nelec, k, b, a, r, s, k)) + (UG_triple_generator_product(state, norbs, nelec, k, a, b, s, r, k) - UG_triple_generator_product(state, norbs, nelec, k, b, a, s, r, k)))

def doubletstate_D_2_3prime(state, norbs, nelec, a, b, r, s, k):
    return (2 / sqrt(3 * (1 + (if a == b)))) * (1 / 2) * (1 / 2) * ((UG_triple_generator_product(state, norbs, nelec, k, a, b, r, s, k) - UG_triple_generator_product(state, norbs, nelec, k, a, b, s, r, k)) + (UG_triple_generator_product(state, norbs, nelec, k, b, a, r, s, k) - UG_triple_generator_product(state, norbs, nelec, k, b, a, s, r, k)))

def doubletstate_S_0_3(state, norbs, nelec, a, r):
    return (1 / sqrt(2)) * UG_generator_pq(state, norbs, nelec, a, r)

def doubletstate_S_1_3(state, norbs, nelec, a, r, k):
    return sqrt(2 / 3) * (UG_2particle_replacement(state, norbs, nelec, a, k, k, r) + (1 / 2) * UG_generator_pq(state, norbs, nelec, a, r))

def doubletstate_D_1_3prime(state, norbs, nelec, a, b, r, s):
    return (1 / sqrt(3)) * (UG_2particle_replacement(state, norbs, nelec, a, b, r, s) - UG_2particle_replacement(state, norbs, nelec, a, b, s, r))

def doubletstate_D_2_3pprime(state, norbs, nelec, a, b, r, s, k):
    return sqrt(2 / 3) * ((UG_triple_generator_product(state, norbs, nelec, a, b, k, k, r, s) - (1 / 2) * UG_2particle_replacement(state, norbs, nelec, a, b, r, s)) - (UG_triple_generator_product(state, norbs, nelec, a, b, k, k, s, r) - (1 / 2) * UG_2particle_replacement(state, norbs, nelec, a, b, s, r)) - (UG_triple_generator_product(state, norbs, nelec, b, a, k, k, r, s) - (1 / 2) * UG_2particle_replacement(state, norbs, nelec, b, a, r, s)) + (UG_triple_generator_product(state, norbs, nelec, b, a, k, k, s, r) - (1 / 2) * UG_2particle_replacement(state, norbs, nelec, b, a, s, r)))
