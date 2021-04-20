import numpy as np
from pyscf import fci
from copy import deepcopy, copy

#define vacuum
#take in operator string
#format: string of indices, bitstring of creation vs annihilation
#return normal-ordered operator string plus contracted terms, according to Wick's theorem

#Class for a single second-quantized operator
class basicOperator:
    def __init__(self, orbital_, creation_annihilation_, spin_):
        self.orbital = orbital_ #orbital index
        self.spin = spin_ #1 for alpha, 0 for beta
        self.creation_annihilation = creation_annihilation_ #1 for creation, 0 for annihilation
#        self.quasi_cre_ann = None

    def apply(self, state):
        neleca, nelecb = state.nelec[0], state.nelec[1]
        norb = state.norb
        if bool(self.creation_annihilation):
            if bool(self.spin):
#                if neleca == norb:
#                    state.array = np.array([[0.]])
#                    state.nelec = (0, 0)
#                else:
                state.array = fci.addons.cre_a(state.array, state.norb, state.nelec, self.orbital)
                state.nelec = (neleca + 1, nelecb)
            else:
#                if nelecb == norb:
#                    state.array = np.array([[0.]])
#                    state.nelec = (0, 0)
#                else:
                state.array = fci.addons.cre_b(state.array, state.norb, state.nelec, self.orbital)
                state.nelec = (neleca, nelecb + 1)
        else:
            if bool(self.spin):
#                if neleca == 0:
#                    state.array = np.array([[0.]])
#                    state.nelec = (0, 0)
#                else:
                state.array = fci.addons.des_a(state.array, state.norb, state.nelec, self.orbital)
                state.nelec = (neleca - 1, nelecb)
            else:
#                if nelecb == 0:
#                    state.array = np.array([[0.]])
#                    state.nelec = (0, 0)
#                else:
                state.array = fci.addons.des_b(state.array, state.norb, state.nelec, self.orbital)
                state.nelec = (neleca, nelecb - 1)