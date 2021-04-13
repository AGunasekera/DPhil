from pyscf import fci

#define vacuum
#take in operator string
#format: string of indices, bitstring of creation vs annihilation
#return normal-ordered operator string plus contracted terms, according to Wick's theorem

class operatorString:
    def __init__(self, indexString, creationOrAnnihilation, alphaOrBeta):
        #list of indices in the string of second-quantised operators
        self.iS = indexString
        #bitstring of same length, with 1 corresponding to creation and 0 for annihilation
        self.cOA = creationOrAnnihilation
        #bitstring of same length, with 1 corresponding to alpha and 0 for beta
        self.aOB = alphaOrBeta

    #Apply the string of second-quantised operators to an arbitrary state
    def applyOperator(self, state, norb, (nelec, nelecb)):
        i = len(self.iS)
        while(i > 0):
            i = i - 1
            print(i)
            print(self.iS[i])
            print(self.cOA[i])
            print(self.aOB[i])
            if self.cOA[i]:
                if self.aOB[i]:
                    state = fci.addons.cre_a(state, norb, (neleca, nelecb)), self.iS[i])
                    neleca = neleca + 1
                else:
                    state = fci.addons.cre_b(state, norb, (neleca, nelecb), self.iS[i])
                    nelecb = nelecb + 1
            else:
                if self.aOB[i]:
                    state = fci.addons.des_a(state, norb, (neleca, nelecb), self.iS[i])
                    neleca = neleca - 1
                else:
                    state = fci.addons.des_b(state, norb, (neleca, nelecb), self.iS[i])
                    nelecb = nelecb - 1
#        return state

    def printstuff(self):
        i = len(self.iS)
        while(i > 0):
            i = i - 1
            print(i)
            print(self.iS[i])
            print(self.cOA[i])
            print(self.aOB[i])
        return 0