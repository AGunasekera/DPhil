import numpy as np
from numbers import Number
from pyscf import fci
from copy import deepcopy
from itertools import combinations

class basicOperator:
    def __init__(self, orbital_, creation_annihilation_, spin_):
        self.orbital = orbital_ #orbital index
        self.spin = spin_ #1 for alpha, 0 for beta
        self.creation_annihilation = creation_annihilation_ #1 for creation, 0 for annihilation
        self.quasi_cre_ann = self.creation_annihilation

    def applyFermiVacuum(self, vacuum):
        self.quasi_cre_ann = (vacuum[self.orbital] and (not self.creation_annihilation)) or (not vacuum[self.orbital] and self.creation_annihilation)

    def __str__(self):
        string = "a"
        if bool(self.creation_annihilation):
            string = string + "^"
        else:
            string = string + "_"
        if bool(self.spin):
            string = string + "{" + str(self.orbital) + "\\alpha}"
        else:
            string = string + "{" + str(self.orbital) + "\\beta}"
        return string

    def __eq__(self, other):
        if isinstance(other, basicOperator):
            return self.orbital == other.orbital and self.spin == other.spin and self.creation_annihilation == other.creation_annihilation
        return False

    def conjugate(self):
        return basicOperator(self.orbital, not self.creation_annihilation, self.spin)

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

class operatorProduct:
    def __init__(self, operatorList_, prefactor_=1.):
        self.operatorList = operatorList_
        self.prefactor = prefactor_

    def __str__(self):
        string = str(self.prefactor)
        if(len(self.operatorList) > 0):
            string = string + " * "
        for o in self.operatorList:
            string = string + o.__str__()
        return string

    def __mul__(self, other):
        if isinstance(other, operatorProduct):
            return operatorProduct(self.operatorList + other.operatorList, self.prefactor * other.prefactor)
        elif isinstance(other, operatorSum):
            newSummandList = []
            for s in other.summandList:
                newSummandList.append(self * s)
            return operatorSum(newSummandList)
        elif isinstance(other, Number):
            return operatorProduct(self.operatorList, self.prefactor * other)

    def __rmul__(self, other):
        if isinstance(other, Number):
            return operatorProduct(self.operatorList, other * self.prefactor)

    def __add__(self, other):
        if isinstance(other, operatorProduct):
            return operatorSum([self, other])
        elif isinstance(other, operatorSum):
            return operatorSum([self, other.summandList])

    def __eq__(self, other):
        if isinstance(other, operatorProduct):
            return self.operatorList == other.operatorList and self.prefactor == other.prefactor

    def checkNilpotency(self):
        nonZero = True
        i = 0
        while i < len(self.operatorList):
            j = i + 1
            while j < len(self.operatorList):
                if self.operatorList[j] == self.operatorList[i]:
                    nonZero = False
                elif self.operatorList[j] == self.operatorList[i].conjugate():
                    break
                j = j + 1
            i = i + 1
        return int(nonZero)

    def conjugate(self):
        return operatorProduct([o.conjugate() for o in self.operatorList], np.conjugate(self.prefactor))

    def apply(self, state):
        i = len(self.operatorList)
        while i>0:
            i = i - 1
            self.operatorList[i].apply(state)

class operatorSum:
    def __init__(self, summandList_):
        self.summandList = summandList_

    def __str__(self):
        string = self.summandList[0].__str__()
        s = 1
        while s < len(self.summandList):
            string = string + " + " + self.summandList[s].__str__()
            s = s + 1
        return string

    def __add__(self, other):
        if isinstance(other, operatorSum):
            return operatorSum(self.summandList + other.summandList)
        elif isinstance(other, operatorProduct):
            if other.prefactor == 0:
                return self
            return operatorSum(self.summandList + [other])
            
    def __radd__(self, other):
        if isinstance(other, operatorProduct):
            if other.prefactor == 0:
                return self
            return operatorSum([other] + self.summandList)

    def __mul__(self, other):
        if isinstance(other, operatorProduct):
            newSummandList = []
            for s in self.summandList:
                newSummandList.append(s * other)
            return operatorSum(newSummandList)
        elif isinstance(other, operatorSum):
            newSummandList = []
            for o in other.summandList:
                partialSum = self * o
                newSummandList = newSummandList + partialSum.summandList
            return operatorSum(newSummandList)
        elif isinstance(other, Number):
            return operatorSum([self.summandList[s] * other for s in range(len(self.summandList))])

    def __rmul__(self, other):
        if isinstance(other, operatorProduct):
            newSummandList = []
            for s in self.summandList:
                newSummandList.append(other * s)
            return operatorSum(newSummandList)
        elif isinstance(other, Number):
            return operatorSum([other * self.summandList[s] for s in range(len(self.summandList))])

    def apply(self, state_):
        result = state(np.array([[0.]]), state_.norb, state_.nelec)
        for o in self.summandList:
            statecopy = deepcopy(state_)
            o.apply(statecopy)
            print(statecopy.array)
            result = statecopy + result
            print(result.array)
#        state1, state2 = copy(state), copy(state)
#        self.operator1.apply(state1)
#        self.operator2.apply(state2)
#        state = state1 + state2
        state_ = deepcopy(result)
        print(state_.array)

def normalOrder(operator, vacuum):
    '''
    Input: an operatorProduct or operatorSum and a list corresponding to which orbitals are occupied in the Fermi vacuum
    Output: normal ordered form of input, with respect to vacuum
    '''
    if isinstance(operator, operatorSum):
        return operatorSum([normalOrder(product, vacuum) for product in operator.summandList])
    quasiCreationList, quasiAnnihilationList = [], []
    quasiCreationCount = 0
    sign = 1
    for o in range(len(operator.operatorList)):
        op = operator.operatorList[o]
#        print(operator.orbital)
#        print(operator.creation_annihilation)
#        print(operator.spin)
        op.applyFermiVacuum(vacuum)
        if bool(op.quasi_cre_ann):
            quasiCreationList.append(op)
            if (o - quasiCreationCount) % 2 == 1:
                sign = -sign
            quasiCreationCount += 1
        else:
            quasiAnnihilationList.append(op)
    return operatorProduct(quasiCreationList + quasiAnnihilationList, sign * operator.prefactor)

def anticommute(operatorProduct_, first):
    '''
    Apply fermionic anticommutation relation to two adjacent second-quantized operators in an operatorProduct
    '''
    operatorList_ = operatorProduct_.operatorList
    return operatorProduct(operatorList_[:first] + operatorList_[first + 1] + operatorList_[first] + operatorList_[first + 1:], operatorProduct_.prefactor * (-1))

def anticommuteInPlace(operatorProduct_, first):
    '''
    Apply fermionic anticommutation relation in place to two adjacent second-quantized operators in an operatorProduct
    '''
    operatorList_ = operatorProduct_.operatorList
    if operatorProduct_ == []:
        return
    firstOperatorCopy = deepcopy(operatorList_[first])
    operatorList_[first], operatorList_[first + 1] = operatorList_[first + 1], firstOperatorCopy
    operatorProduct_.prefactor = -operatorProduct_.prefactor
    return

def contract(operatorProduct_, first, second):
    '''
    Make a contraction between the indices of operators at the positions given by first and second in operatorProduct_
    '''
    operatorList_ = operatorProduct_.operatorList
    #Case that operatorProduct_ is a scalar
    if operatorList_ == []:
        return operatorProduct_
    firstIndex = operatorList_[first].orbital
    secondIndex = operatorList_[second].orbital
    if firstIndex == secondIndex and operatorList_[first].spin == operatorList_[second].spin:
        if not bool(operatorList_[first].quasi_cre_ann) and bool(operatorList_[second].quasi_cre_ann):
            return operatorProduct(operatorList_[:first] + operatorList_[first+1:second] + operatorList_[second + 1:], operatorProduct_.prefactor * ((-1) ** (1 + second - first)))
    return operatorProduct([],0)

def reorderForContraction(operatorProduct_, first, second):
    '''
    Reorder operators using anticommutation relation so that two operators to be contracted are in positions 0 and 1
    '''
    operatorProductCopy = deepcopy(operatorProduct_)
    i = first
    j = second
    while i > 0:
        anticommuteInPlace(operatorProductCopy, i-1)
        i = i - 1
    while j > 1:
        anticommuteInPlace(operatorProductCopy, j-1)
        j = j - 1
    return operatorProductCopy

def reorderForMultipleContraction(operatorProduct_, pairsList):
    '''
    Reorder operators using anticommutation relation so that each pair of operators to be contracted are adjacent
    '''
    operatorProductCopy = deepcopy(operatorProduct_)
    finalPosition = 0
    for positionPair in pairsList:
        i = positionPair[0]
        j = positionPair[1]
        while i > finalPosition:
            anticommuteInPlace(operatorProductCopy, i-1)
            i = i - 1
        while j > finalPosition + 1:
            anticommuteInPlace(operatorProductCopy, j-1)
            j = j - 1
        finalPosition = finalPosition + 2
    return operatorProductCopy

def updateFlattenedPositionList(flattenedPositions):
    for i in range(len(flattenedPositions)):
        if flattenedPositions[i] > flattenedPositions[1]:
            flattenedPositions[i] -= 2
        elif flattenedPositions[i] > flattenedPositions[0]:
            flattenedPositions[i] -= 1
    return flattenedPositions[2:]

def multipleContraction(operatorProduct_, positionPairsList):
    '''
    Contractions between multiple pairs of operators in a product, at the positions given by entries 2i and 2i+1 in positionPairsList
    '''
    product = deepcopy(operatorProduct_)
    while len(positionPairsList) > 0:
#    for positionPair in pairsList:
        reorderedProduct = reorderForContraction(product, positionPairsList[0], positionPairsList[1])
        product = contract(reorderedProduct, 0, 1)
        positionPairsList = updateFlattenedPositionList(positionPairsList) #NOTE: reduces length of positionPairsList by 2
    return product

def genPairOrderedLists(list_):
    '''
    Given an ordered list with an even number of elements, returns list of all permutations of that list where each adjacent pair is ordered, and where the first elements of pairs are ordered
    '''
    pairOrderedLists = []
    if len(list_) == 2:
        pairOrderedLists.append(list_)
    else:
        for i in range(1,len(list_)):
            swappedList = [list_[0]] + [list_[i]] + list_[1:i]
            if i < len(list_):
                swappedList = swappedList + list_[i+1:]
            subLists = genPairOrderedLists(swappedList[2:])
            for l in range(len(subLists)):
                pairOrderedLists.append(swappedList[:2] + subLists[l])
    return pairOrderedLists

def getPositionsForMultipleContraction(operatorProduct_, n):
    nOperators = len(operatorProduct_.operatorList)
    operatorPositions = range(nOperators)
    nToChoose = 2 * n
    if nToChoose <= nOperators:
        return combinations(operatorPositions, nToChoose)

def sumNFoldContractions(operatorProduct_, n):
    chosenPositions = getPositionsForMultipleContraction(operatorProduct_, n)
    if n == 0:
        return operatorProduct_
    operatorSum_ = operatorSum([])
    for c in chosenPositions:
        pairOrderedList = genPairOrderedLists(list(c))
        for l in pairOrderedList:
            operatorSum_ = operatorSum_ + multipleContraction(operatorProduct_, l)
    return operatorSum_

def wickExpand(operator, vacuum):
    if isinstance(operator, operatorSum):
        wickExpansion = operatorSum([])
        for product in operator.summandList:
            wickExpansion = wickExpansion + wickExpand(product, vacuum)
        return wickExpansion
    wickExpansion = operatorSum([normalOrder(operator, vacuum)])
    highestOrder = len(operator.operatorList) // 2
    for n in range(highestOrder):
        wickExpansion = wickExpansion + sumNFoldContractions(operator, n + 1)
    return normalOrder(wickExpansion, vacuum)

def vacuumExpectationValue(operator, vacuum):
    wickExpansion = wickExpand(operator, vacuum)
    vEV = 0.
    for summand in wickExpansion.summandList:
        if summand.operatorList == []:
            vEV += summand.prefactor
    return vEV