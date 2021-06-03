import numpy as np
from numbers import Number
from pyscf import fci
from copy import deepcopy
from itertools import combinations

class basicOperator:
    '''
    Class for a single second-quantized creation or annihilation operator for a particle in a given orbital with a given spin
    '''
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
    '''
    Class for a product of basicOperators with a given prefactor
    '''
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
            result = operatorProduct(self.operatorList + other.operatorList, self.prefactor * other.prefactor)
            result.checkNilpotency()
            return result
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
        if self.prefactor == 0:
            return other
        if isinstance(other, operatorProduct):
            if other.prefactor == 0:
                return self
            elif other.operatorList == self.operatorList:
                return operatorProduct(self.operatorList, self.prefactor + other.prefactor)
            return operatorSum([self, other])
        elif isinstance(other, operatorSum):
            return operatorSum([self] + other.summandList)
        elif isinstance(other, Number):
            if other == 0:
                return self
            return operatorSum([self, operatorProduct([], other)])

    def __radd__(self, other):
        if isinstance(other, Number):
            if other == 0:
                return self
            return operatorSum([operatorProduct([], other), self])

    def __eq__(self, other):
        if isinstance(other, operatorProduct):
            return self.operatorList == other.operatorList and self.prefactor == other.prefactor

    def __pow__(self, other):
        if isinstance(other, int):
            result = 1.
            for i in range(other):
                result = result * self
            return result
        return

    def applyFermiVacuum(self, vacuum):
        for o in self.operatorList:
            o.applyFermiVacuum(vacuum)

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
        if not nonZero:
            self.operatorList = []
            self.prefactor = 0.

    def conjugate(self):
        return operatorProduct([o.conjugate() for o in self.operatorList][::-1], np.conjugate(self.prefactor))

    def compareProductPriority(self, other):
        '''
        Compare priority of operatorProducts for sorting in a list for operatorSum
        Returns 1 (-1) if self is of higher (lower) priority than other, or 0 if they have the same sequence
        '''
        selfList = self.operatorList
        otherList = other.operatorList

        priorityComparison = 0

        if len(selfList) < len(otherList):
            priorityComparison = 1

        elif len(selfList) > len(otherList):
            priorityComparison = -1
        else:
            i = 0
            while i < len(selfList):
                if selfList[i].quasi_cre_ann > otherList[i].quasi_cre_ann:
                    priorityComparison = 1
                    break
                elif selfList[i].quasi_cre_ann < otherList[i].quasi_cre_ann:
                    priorityComparison = -1
                    break
                else:
                    if selfList[i].orbital < otherList[i].orbital:
                        priorityComparison = 1
                        break
                    elif selfList[i].orbital > otherList[i].orbital:
                        priorityComparison = -1
                        break
                    else:
                        if selfList[i].spin > otherList[i].spin:
                            priorityComparison = 1
                            break
                        elif selfList[i].spin < otherList[i].spin:
                            priorityComparison = -1
                            break
                i += 1
        return priorityComparison

    def apply(self, state):
        i = len(self.operatorList)
        while i>0:
            i = i - 1
            self.operatorList[i].apply(state)

class operatorSum:
    '''
    Class for a general second-quantized operator, as a sum of operatorProducts
    '''
    def __init__(self, summandList_):
        self.summandList = summandList_
        self.sortSummandList()

    def __str__(self):
        if len(self.summandList) == 0:
            return "operatorSum([])"
        string = self.summandList[0].__str__()
        s = 1
        while s < len(self.summandList):
            string = string + "\n + " + self.summandList[s].__str__()
            s = s + 1
        return string

    def __add__(self, other):
        result = self
        if isinstance(other, operatorSum):
            result = operatorSum(self.summandList + other.summandList)
        elif isinstance(other, operatorProduct):
            if other.prefactor == 0:
                return self
            result = operatorSum(self.summandList + [other])
        elif isinstance(other, Number):
            if other == 0:
                return self
            result = operatorSum(self.summandList + [operatorProduct([], other)])
        result.sortSummandList()
        return result
            
    def __radd__(self, other):
        result = self
        if isinstance(other, operatorProduct):
            if other.prefactor == 0:
                return self
            result = operatorSum([other] + self.summandList)
        elif isinstance(other, Number):
            if other == 0:
                return self
            result = operatorSum(self.summandList + [operatorProduct([], other)])
        result.sortSummandList()
        return result

    def __mul__(self, other):
        result = self
        if isinstance(other, operatorProduct):
            newSummandList = []
            for s in self.summandList:
                newSummandList.append(s * other)
            result = operatorSum(newSummandList)
        elif isinstance(other, operatorSum):
            newSummandList = []
            for o in other.summandList:
                partialSum = self * o
                newSummandList = newSummandList + partialSum.summandList
            result = operatorSum(newSummandList)
        elif isinstance(other, Number):
            result = operatorSum([self.summandList[s] * other for s in range(len(self.summandList))])
        result.sortSummandList()
        return result

    def __rmul__(self, other):
        result = self
        if isinstance(other, operatorProduct):
            newSummandList = []
            for s in self.summandList:
                newSummandList.append(other * s)
            result = operatorSum(newSummandList)
        elif isinstance(other, Number):
            result = operatorSum([other * self.summandList[s] for s in range(len(self.summandList))])
        result.sortSummandList()
        return result

    def __pow__(self, other):
        if isinstance(other, int):
            result = 1.
            for i in range(other):
                result = result * self
            return result
        return

    def applyFermiVacuum(self, vacuum):
        for p in self.summandList:
            p.applyFermiVacuum(vacuum)

    def sortSummandList(self):
        already_sorted = True
        l = len(self.summandList)
        i = 0
        while i < l:
            j = 0
            while j < (l - i - 1):
                priorityComparison = self.summandList[j].compareProductPriority(self.summandList[j+1])
                if priorityComparison == -1:
                    priorCopy = deepcopy(self.summandList[j+1])
                    self.summandList[j+1] = self.summandList[j]
                    self.summandList[j] = priorCopy
                    already_sorted = False
                elif priorityComparison == 0:
                    self.summandList[j] = self.summandList[j] + self.summandList[j+1]
                    self.summandList = self.summandList[:j+1] + self.summandList[j+2:]
                    l -= 1
                    already_sorted = False
                j += 1
            if already_sorted:
                break
            i += 1

    def checkNilpotency(self):
        for product in self.summandList:
            product.checkNilpotency()

    def conjugate(self):
        return operatorSum([p.conjugate() for p in self.summandList])

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
    if operatorList_ == []:
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
        return operatorSum([operatorProduct_])
    operatorSum_ = operatorSum([])
    for c in chosenPositions:
        pairOrderedList = genPairOrderedLists(list(c))
        for l in pairOrderedList:
            term = multipleContraction(operatorProduct_, l)
            print(term)
            operatorSum_ = operatorSum_ + term
    return operatorSum_

def wickExpand(operator, vacuum):
    if isinstance(operator, operatorSum):
        wickExpansion = operatorSum([])
        for product in operator.summandList:
            wickExpansion = wickExpansion + wickExpand(product, vacuum)
        return wickExpansion
    operator.checkNilpotency()
    wickExpansion = operatorSum([normalOrder(operator, vacuum)])
    highestOrder = len(operator.operatorList) // 2
    for n in range(highestOrder):
        wickExpansion = wickExpansion + sumNFoldContractions(operator, n + 1)
    return normalOrder(wickExpansion, vacuum)

def fullContraction(operator, vacuum):
    if isinstance(operator, operatorSum):
        fullyContracted = operatorSum([])
        for product in operator.summandList:
            fullyContracted = fullyContracted + fullContraction(product, vacuum)
        return fullyContracted
    operator.checkNilpotency()
    for o in operator.operatorList:
        o.applyFermiVacuum(vacuum)
    fullyContracted = sumNFoldContractions(operator, len(operator.operatorList)//2)
    return fullyContracted

#def vacuumExpectationValue(operator, vacuum):
#    wickExpansion = wickExpand(operator, vacuum)
#    vEV = 0.
#    for summand in wickExpansion.summandList:
#        if summand.operatorList == []:
#            vEV += summand.prefactor
#    return vEV

#def vacuumExpectationValue2(operator, vacuum):
#    fullyContracted = fullContraction(operator, vacuum)
#    if len(fullyContracted.summandList) == 1:
#        return fullyContracted.summandList[0].prefactor
#    else:
#        return 0.

def contract2operators(o1, o2):
    if o1.quasi_cre_ann:
        return 0
    elif o2.quasi_cre_ann and (o1.spin == o2.spin):
        return int(o1.orbital == o2.orbital)
    else:
        return 0

def recursiveFullContraction(operatorProduct_, vacuum):
    if isinstance(operatorProduct_, Number):
        return operatorProduct_
    operatorProduct_.applyFermiVacuum(vacuum)
    operatorList_ = operatorProduct_.operatorList
    if len(operatorList_) == 0:
        return operatorProduct_.prefactor
    elif len(operatorList_) == 2:
        return operatorProduct_.prefactor * contract2operators(operatorList_[0], operatorList_[1])
    elif len(operatorList_) % 2 == 0:
        result = 0
        for i in range(1, len(operatorList_) - 1):
            if contract2operators(operatorList_[0], operatorList_[i]):
                result += pow(-1, i-1) * recursiveFullContraction(operatorProduct(operatorList_[1:i] + operatorList_[i+1:], operatorProduct_.prefactor), vacuum)
#            else:
#                result += 0
        if contract2operators(operatorList_[0], operatorList_[-1]):
            result += recursiveFullContraction(operatorProduct(operatorList_[1:-1], operatorProduct_.prefactor), vacuum)
#        else:
#            result += 0
        return result

def vacuumExpectationValue(operator, vacuum, printing=False):
    if isinstance(operator, operatorProduct):
        return recursiveFullContraction(operator, vacuum)
    elif isinstance(operator, operatorSum):
        result = 0
        for product in operator.summandList:
            term = recursiveFullContraction(product, vacuum)
            if printing and term != 0.:
                print(product)
            result += term
        return result
    elif isinstance(operator, Number):
        return operator
    else:
        return 0