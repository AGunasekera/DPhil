import numpy as np
from numbers import Number
from pyscf import gto, scf, ao2mo, fci
from copy import deepcopy, copy
from math import factorial
import itertools
import string

class Index:
    '''
    Class for an orbital index
    name (str): name given to index
    occupiedInVaccum (bool): whether the index refers to orbitals that are occupied in the vacuum (i.e. hole orbitals) or not (particle orbitals)
    '''
    def __init__(self, name, occupiedInVacuum):
        self.name = name
        self.occupiedInVacuum = occupiedInVacuum
        self.tuple = (self.name, self.occupiedInVacuum)

    def __hash__(self):
        return hash(self.tuple)

    def __eq__(self, other):
        return self.tuple == other

    def __str__(self):
        return self.name

class basicOperator:
    '''
    Class for a basic fermionic creation or annihilation operator, with arbitrary index
    index (Index): the index for the orbitals on which this operator acts
    creation_annihilation (bool): True for creation, False for annihilation
    spin (bool): True for alpha, False for beta
    quasi_cre_ann (bool): creation or annihilation with respect to Fermi vacuum
    '''
    def __init__(self, index_, creation_annihilation_, spin_):
        self.index = index_
        self.spin = spin_
        self.creation_annihilation = creation_annihilation_
        self.quasi_cre_ann = not (self.creation_annihilation == self.index.occupiedInVacuum)

    def __copy__(self):
        return basicOperator(self.index, self.creation_annihilation, self.spin)

    def __str__(self):
        string = "a"
        if bool(self.creation_annihilation):
            string = string + "^"
        else:
            string = string + "_"
        if bool(self.spin):
            string = string + "{" + self.index.__str__() + "\\alpha}"
        else:
            string = string + "{" + self.index.__str__() + "\\beta}"
        return string

    def __eq__(self, other):
        if isinstance(other, basicOperator):
            return self.index == other.index and self.spin == other.spin and self.creation_annihilation == other.creation_annihilation
        return False

    def conjugate(self):
        return basicOperator(self.index, not self.creation_annihilation, self.spin)

class operatorProduct:
    def __init__(self, operatorList_=[], prefactor_=1., contractionsList_=[]):
        self.operatorList = operatorList_
        self.prefactor = prefactor_
        self.contractionsList = contractionsList_

    def isProportional(self, other):
        if isinstance(other, operatorProduct):
            return self.operatorList == other.operatorList and set(self.contractionsList) == set(other.contractionsList)
        else:
            return NotImplemented

    def __copy__(self):
        return operatorProduct(copy(self.operatorList), self.prefactor, copy(self.contractionsList))

    def __str__(self):
        string = str(self.prefactor)
        if(len(self.operatorList) + len(self.contractionsList) > 0):
            string = string + " * "
        for o in self.operatorList:
            string = string + o.__str__()
        for contraction in self.contractionsList:
            string = string + "\delta^{" + contraction[0].__str__() + "}_{" + contraction[1].__str__() + "}"
        return string

    def __mul__(self, other):
        if isinstance(other, operatorProduct):
            return operatorProduct(self.operatorList + other.operatorList, self.prefactor * other.prefactor, self.contractionsList + other.contractionsList)
        elif isinstance(other, operatorSum):
            newSummandList = []
            for s in other.summandList:
                newSummandList.append(self * s)
            return operatorSum(newSummandList)
        elif isinstance(other, Number):
            return operatorProduct(self.operatorList, self.prefactor * other, self.contractionsList)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Number):
            return operatorProduct(self.operatorList, other * self.prefactor, self.contractionsList)
        else:
            return NotImplemented

    def __add__(self, other):
        if isinstance(other, operatorProduct):
            if self.isProportional(other):
                return operatorProduct(self.operatorList, self.prefactor + other.prefactor, self.contractionsList)
            else:
                return operatorSum([self, other])
        elif other == 0:
            return self
        else:
            return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, operatorProduct):
            return self.operatorList == other.operatorList and self.prefactor == other.prefactor and self.contractionsList == other.contractionsList
        else:
            return NotImplemented
            
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
        return nonZero

    def conjugate(self):
        return operatorProduct([o.conjugate() for o in self.operatorList], np.conjugate(self.prefactor), self.contractionsList)

class operatorSum:
    def __init__(self, summandList_=[]):
        self.summandList = self.collectSummandList(summandList_)

    def collectSummandList(self, summandList):
        oldSummandList = copy(summandList)
        newSummandList = []
        while len(oldSummandList) > 0:
            newSummand = oldSummandList[0]
            i = 1
            while i < len(oldSummandList):
                if newSummand.isProportional(oldSummandList[i]):
                    newSummand += oldSummandList[i]
                    oldSummandList.pop(i)
                else:
                    i += 1
            newSummandList.append(newSummand)
            oldSummandList.pop(0)
        return newSummandList

    def __copy__(self):
        return operatorSum([copy(summand) for summand in self.summandList])

    def __str__(self):
        if len(self.summandList) == 0:
            return str(0)
        string = self.summandList[0].__str__()
        s = 1
        while s < len(self.summandList):
            string = string + "\n + " + self.summandList[s].__str__()
            s = s + 1
        return string

    def __add__(self, other):
        if isinstance(other, operatorSum):
            return operatorSum(self.summandList + other.summandList)
        elif isinstance(other, operatorProduct):
            if other.prefactor == 0:
                return self
            newSummandList = copy(self.summandList)
            alreadyInSum = False
            for summand in newSummandList:
                if summand.isProportional(other):
                    alreadyInSum = True
                    summand.prefactor += other.prefactor
            if not alreadyInSum:
                newSummandList.append(other)
            return operatorSum(newSummandList)
        elif other == 0:
            return self
        else:
            return NotImplemented
            
    def __radd__(self, other):
        if isinstance(other, operatorProduct):
            if other.prefactor == 0:
                return self
            newSummandList = copy(self.summandList)
            alreadyInSum = False
            for summand in newSummandList:
                if summand.isProportional(other):
                    alreadyInSum = True
                    summand.prefactor += other.prefactor
            if not alreadyInSum:
                newSummandList.append(other)
            return operatorSum(newSummandList)
        elif other == 0:
            return self
        else:
            return NotImplemented

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
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, operatorProduct):
            newSummandList = []
            for s in self.summandList:
                newSummandList.append(other * s)
            return operatorSum(newSummandList)
        elif isinstance(other, Number):
            return operatorSum([other * self.summandList[s] for s in range(len(self.summandList))])
        else:
            return NotImplemented

    def conjugate(self):
        return operatorSum([s.conjugate() for s in self.summandList])

class excitation(operatorProduct):
    def __init__(self, creationIndicesList_, annihilationIndicesList_, spinList_):
        super(excitation, self).__init__()
        self.operatorList = []
        for i in range(len(creationIndicesList_)):
            self.operatorList.append(basicOperator(creationIndicesList_[i], True, spinList_[i]))
        for i in range(len(annihilationIndicesList_)):
            self.operatorList.append(basicOperator(annihilationIndicesList_[i], False, spinList_[i+len(creationIndicesList_)]))

class spinFreeExcitation(operatorSum):
    def __init__(self, creationList_, annihilationList_):
        self.summandList = []
        spinLists = np.reshape(np.zeros(len(creationList_)), (1, -1))
        for i in range(len(creationList_)):
            newspinLists = copy(spinLists)
            for s in range(len(spinLists)):
                newspinLists[s,i] = 1
            spinLists = np.concatenate((spinLists, newspinLists))
        for l in range(len(spinLists)):
            spinList = spinLists[l]
            self.summandList.append(excitation(creationList_, annihilationList_, [*spinList, *spinList[::-1]]))

class Tensor:
    '''
    Class for amplitude tensors of spin-free excitation operators
    '''
    def __init__(self, name, lowerIndexTypesList, upperIndexTypesList):
        self.name = name
        self.lowerIndexTypes = lowerIndexTypesList
        self.upperIndexTypes = upperIndexTypesList
        self.excitationRank = len(self.lowerIndexTypes)

    def getShape(self, vacuum):
        Norbs = len(vacuum)
        Nocc = sum(vacuum)
        shapeList = []
        for iType in self.lowerIndexTypes:
            if iType == 'g':
                shapeList.append(Norbs)
            elif iType == 'p':
                shapeList.append(Norbs - Nocc)
            elif iType == 'h':
                shapeList.append(Nocc)
            else:
                print('Orbital index type Error')
        for iType in self.upperIndexTypes:
            if iType == 'g':
                shapeList.append(Norbs)
            elif iType == 'p':
                shapeList.append(Norbs - Nocc)
            elif iType == 'h':
                shapeList.append(Nocc)
            else:
                print('Orbital index type Error')
        self.array = np.zeros(tuple(shapeList))

    def setArray(self, array):
        if array.shape == self.array.shape:
            self.array = array
        else:
            print("Array is of wrong shape")

    def getOperator(self, spinFree):
        return TensorProduct([self]).getOperator(spinFree)

    def getDiagrams(self, vacuum):
        Nocc = sum(vacuum)
        Norbs = len(vacuum)
        diagrams = []
        lowerGeneralIndexCount = sum(i == 'g' for i in self.lowerIndexTypes)
        lowerSplits = list(itertools.combinations_with_replacement(['h', 'p'], lowerGeneralIndexCount))
        upperGeneralIndexCount = sum(i == 'g' for i in self.upperIndexTypes)
        upperSplits = list(itertools.combinations_with_replacement(['h', 'p'], upperGeneralIndexCount))
        for lowerSplit in lowerSplits:
            lowerSlices = [slice(None)] * self.excitationRank
            lowerSplitIndexTypes = list(lowerSplit)
            lGI = 0
            newLowerIndexTypes = copy.copy(self.lowerIndexTypes)
            for lI in range(len(newLowerIndexTypes)):
                if newLowerIndexTypes[lI] == 'g':
                    newLI = lowerSplitIndexTypes[lGI]
                    if newLI == 'h':
                        lowerSlices[lI] = slice(None,Nocc)
                    elif newLI == 'p':
                        lowerSlices[lI] = slice(Nocc, None)
                    newLowerIndexTypes[lI] = newLI
                    lGI += 1
            for upperSplit in upperSplits:
                upperSlices = [slice(None)] * self.excitationRank
                upperSplitIndexTypes = list(upperSplit)
                uGI = 0
                newUpperIndexTypes = copy.copy(self.upperIndexTypes)
                for uI in range(len(newUpperIndexTypes)):
                    if newUpperIndexTypes[uI] == 'g':
                        newUI = upperSplitIndexTypes[uGI]
                        if newUI == 'h':
                            upperSlices[uI] = slice(None,Nocc)
                        elif newUI == 'p':
                            upperSlices[uI] = slice(Nocc, None)
                        newUpperIndexTypes[uI] = newUI
                        uGI += 1
                slices = lowerSlices + upperSlices
                print(lowerSplitIndexTypes)
                print(upperSplitIndexTypes)
                print(newLowerIndexTypes)
                print(newUpperIndexTypes)
                print(slices)
                diagram = Tensor(self.name, newLowerIndexTypes, newUpperIndexTypes, self.array[slices])
                diagrams.append(diagram)
        return diagrams

    def __add__(self, other):
        if isinstance(other, Tensor):
            return TensorSum([TensorProduct([self]), TensorProduct([other])])
        elif other == 0:
            return self
        else:
            return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return TensorProduct([self, other])
        elif isinstance(other, Number):
            return TensorProduct([self], other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Number):
            return TensorProduct([self], other)
        else:
            return NotImplemented

    def __str__(self):
        string = self.name + "_{"
        for p in self.lowerIndexTypes:
            string += p.__str__()
        string += "}^{"
        for q in self.upperIndexTypes:
            string += q.__str__()
        string += "}"
        return string

class Vertex:
    '''
    Class for amplitude tensors of spin-free excitation operators
    '''
    def __init__(self, tensor, lowerIndicesList, upperIndicesList):
        self.name = tensor.name
        self.tensor = tensor
        self.lowerIndices = lowerIndicesList
        self.upperIndices = upperIndicesList
        self.excitationRank = len(self.lowerIndices)

    def applyContraction(self, contraction):
        for lowerIndex in self.lowerIndices:
            if lowerIndex == contraction[0]:
                lowerIndex = contraction[1]

    def getOperator(self, spinFree):
        if spinFree:
            return normalOrder(spinFreeExcitation(self.lowerIndices, self.upperIndices))
        else:
            return normalOrder(operatorSum([excitation(self.lowerIndices, self.upperIndices, [True] * (2 * self.excitationRank))]))

    def __copy__(self):
        return Vertex(self.tensor, copy(self.lowerIndices), copy(self.upperIndices))

    def __str__(self):
        string = self.name + "_{"
        for p in self.lowerIndices:
            string += p.__str__()
        string += "}^{"
        for q in self.upperIndices:
            string += q.__str__()
        string += "}"
        return string

class TensorProduct:
    def __init__(self, tensorList, prefactor=1., vertexList=None):
        self.tensorList = tensorList
        self.lowerIndices = {'g':[], 'p':[], 'h':[], 'a':[]}
        self.upperIndices = {'g':[], 'p':[], 'h':[], 'a':[]}
        self.prefactor = prefactor
        self.vertexList = vertexList
        if self.vertexList is None:
            self.vertexList = self.getVertexList(tensorList)

    def applyContraction(self, contraction):
        for vertex in self.vertexList:
            vertex.applyContraction(contraction)
        for orbitalType in self.lowerIndices:
            if contraction[0] in self.lowerIndices[orbitalType]:
                self.lowerIndices[orbitalType].remove(contraction[0])

    def addNewIndex(self, orbitalType, lowerBool):
        count = len(self.lowerIndices[orbitalType]) + len(self.upperIndices[orbitalType])
        newIndexName = orbitalType + "_{" + str(count) + "}"
        occupiedInVacuum = None
        if orbitalType == "h" or orbitalType == "a":
            occupiedInVacuum = True
        elif orbitalType == "p":
            occupiedInVacuum = False
        newIndex = Index(newIndexName, occupiedInVacuum)
        if lowerBool:
            self.lowerIndices[orbitalType].append(newIndex)
        else:
            self.upperIndices[orbitalType].append(newIndex)
        return newIndex

    def getVertexList(self, tensorList_):
        vertexList = []
        for t in tensorList_:
            lowerIndexList = []
            for i in t.lowerIndexTypes:
                lowerIndexList.append(self.addNewIndex(i, True))
            upperIndexList = []
            for i in t.upperIndexTypes:
                upperIndexList.append(self.addNewIndex(i, False))
            vertexList.append(Vertex(t, lowerIndexList, upperIndexList))
        return vertexList

    def getOperator(self, spinFree):
        operator = 1
        for vertex in self.vertexList:
            operator = operator * vertex.getOperator(spinFree)
        return operator

    def getVacuumExpectationValue(self, spinFree):
        return vacuumExpectationValue(self.getOperator(spinFree))

    def __copy__(self):
        return TensorProduct(copy(self.tensorList), copy(self.prefactor), [copy(vertex) for vertex in self.vertexList])

    def __add__(self, other):
        if isinstance(other, TensorProduct):
            return TensorSum([self, other])
        elif isinstance(other, Tensor):
            return TensorSum([self, TensorProduct([other])])
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Tensor):
            return TensorSum([TensorProduct([other]), self])
        elif other == 0:
            return self
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return TensorProduct(self.tensorList + [other], self.prefactor)
        elif isinstance(other, TensorProduct):
            return TensorProduct(self.tensorList + other.tensorList, self.prefactor * other.prefactor)
        elif isinstance(other, Number):
            return TensorProduct(self.tensorList, self.prefactor * other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Tensor):
            return TensorProduct([other] + self.tensorList, self.prefactor)
        elif isinstance(other, Number):
            return TensorProduct(self.tensorList, other * self.prefactor)
        else:
            return NotImplemented

    def __str__(self):
        string = str(self.prefactor)
        if(len(self.vertexList) > 0):
            string = string + " * "
        for v in self.vertexList:
            string += v.__str__()
        return string

class TensorSum:
    def __init__(self, summandList):
        self.summandList = summandList

    def getOperator(self, spinFree):
        operator = 0
        for summand in self.summandList:
            operator = operator + summand.getOperator(spinFree)
        return operator

    def __copy__(self):
        return TensorSum([copy(summand) for summand in self.summandList])

    def __add__(self, other):
        if isinstance(other, TensorSum):
            return TensorSum(self.summandList + other.summandList)
        elif isinstance(other, TensorProduct):
            return TensorSum(self.summandList + [other])
        elif isinstance(other, Tensor):
            return TensorSum(self.summandList + [TensorProduct([other])])
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, TensorProduct):
            return TensorSum([other] + self.summandList)
        elif isinstance(other, Tensor):
            return TensorSum(self.summandList + [TensorProduct([other])])
        elif other == 0:
            return self
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Tensor) or isinstance(other, TensorProduct) or isinstance(other, TensorSum) or isinstance(other, Number):
            return TensorSum([summand * other for summand in self.summandList])

    def __rmul__(self, other):
        if isinstance(other, Tensor) or isinstance(other, TensorProduct) or isinstance(other, TensorSum) or isinstance(other, Number):
            return TensorSum([other * summand for summand in self.summandList])

    def __str__(self):
        if len(self.summandList) == 0:
            return ""
        string = self.summandList[0].__str__()
        for summand in self.summandList[1:]:
            string += "\n + "
            string += summand.__str__()
        return string

def normalOrder(operator):
    '''
    Input: an operatorProduct or operatorSum and a list corresponding to which orbitals are occupied in the Fermi vacuum
    Output: normal ordered form of input, with respect to vacuum
    '''
    if isinstance(operator, operatorSum):
        return operatorSum([normalOrder(product) for product in operator.summandList])
    quasiCreationList, quasiAnnihilationList = [], []
    quasiCreationCount = 0
    sign = 1
    for o in range(len(operator.operatorList)):
        op = operator.operatorList[o]
        if bool(op.quasi_cre_ann):
            quasiCreationList.append(op)
            if (o - quasiCreationCount) % 2 == 1:
                sign = -sign
            quasiCreationCount += 1
        else:
            quasiAnnihilationList.append(op)
    return operatorProduct(quasiCreationList + quasiAnnihilationList, sign * operator.prefactor, operator.contractionsList)

def canContract(o1, o2):
    if o1.quasi_cre_ann:
        return 0
    elif o2.quasi_cre_ann and (o1.spin == o2.spin):
        return int(o1.index.occupiedInVacuum == o2.index.occupiedInVacuum)
    else:
        return 0

def recursiveFullContraction(operatorProduct_, speedup=False):
    operatorList_ = operatorProduct_.operatorList
    if speedup:
        if not sum(o.quasi_cre_ann for o in operatorList_) == sum(not o.quasi_cre_ann for o in operatorList_):
            return 0
    existingContractions = operatorProduct_.contractionsList
    if len(operatorList_) == 0:
        return operatorProduct_
    elif len(operatorList_) == 2:
        if canContract(operatorList_[0], operatorList_[1]):
            contractionTuple = tuple()
            if operatorList_[0].creation_annihilation:
                contractionTuple = (operatorList_[0].index, operatorList_[1].index)
            else:
                contractionTuple = (operatorList_[1].index, operatorList_[0].index)
            return operatorProduct([], operatorProduct_.prefactor, existingContractions + [contractionTuple])
        else:
            return 0
    elif len(operatorList_) % 2 == 0:
        result = 0
        for i in range(1, len(operatorList_) - 1):
            if canContract(operatorList_[0], operatorList_[i]):
                contractionTuple = tuple()
                if operatorList_[0].creation_annihilation:
                    contractionTuple = (operatorList_[0].index, operatorList_[i].index)
                else:
                    contractionTuple = (operatorList_[i].index, operatorList_[0].index)
                result += pow(-1, i-1) * recursiveFullContraction(operatorProduct(operatorList_[1:i] + operatorList_[i+1:], operatorProduct_.prefactor, existingContractions + [contractionTuple]))
        if canContract(operatorList_[0], operatorList_[-1]):
            contractionTuple = tuple()
            if operatorList_[0].creation_annihilation:
                contractionTuple = (operatorList_[0].index, operatorList_[-1].index)
            else:
                contractionTuple = (operatorList_[-1].index, operatorList_[0].index)
            result += recursiveFullContraction(operatorProduct(operatorList_[1:-1], operatorProduct_.prefactor, existingContractions + [contractionTuple]))
        return result
    else:
        return 0

def vacuumExpectationValue(operator, speedup=False, printing=False):
    if isinstance(operator, operatorProduct):
        return recursiveFullContraction(operator, speedup)
    elif isinstance(operator, operatorSum):
        result = operatorSum([])
        for product in operator.summandList:
            term = recursiveFullContraction(product, speedup)
            if printing and term != 0.:
                print(product)
            result += term
        return result
    elif isinstance(operator, Number):
        return operatorSum([operatorProduct([], operator)])
    else:
        return operatorSum([])

def evaluateWick(term, spinFree):
    '''
    Wick's theorem applied to a term

    input: term (TensorProduct)
    output: sum of fully contracted terms (TensorSum)
    '''
    if isinstance(term, TensorSum):
        return sum([evaluateWick(summand, spinFree) for summand in term.summandList])
    summandList = []
    fullContractions = vacuumExpectationValue(term.getOperator(spinFree), speedup=True)
    for topology in fullContractions.summandList:
        contractionsList = topology.contractionsList
        prefactor = topology.prefactor
        contractedTerm = copy(term)
        contractedTerm.prefactor *= topology.prefactor
        for c, contraction in enumerate(contractionsList):
            for v, vertex in reversed(list(enumerate(term.vertexList))):
                if contraction[0] in vertex.lowerIndices:
                    contractedTerm.vertexList[v].lowerIndices[vertex.lowerIndices.index(contraction[0])] = contraction[1]
                    break
                elif contraction[1] in vertex.upperIndices:
                    contractedTerm.vertexList[v].upperIndices[vertex.upperIndices.index(contraction[1])] = contraction[0]
                    break
        summandList.append(contractedTerm)
    return TensorSum(summandList)

def getAxis(vertex, index):
    for a in range(vertex.excitationRank):
        if vertex.lowerIndices[a] == index:
            return a
        elif vertex.upperIndices[a] == index:
            return vertex.excitationRank + a

def getContractedArray(tensorProduct_, targetLowerIndexList=None, targetUpperIndexList=None):
    lowerIndexList = list(itertools.chain.from_iterable([vertex.lowerIndices for vertex in tensorProduct_.vertexList]))
    upperIndexList = list(itertools.chain.from_iterable([vertex.upperIndices for vertex in tensorProduct_.vertexList]))
    lowerIndexLetters = string.ascii_lowercase[:len(lowerIndexList)]
    upperIndexLetters = ''
    freeLowerIndexMask = np.ones(len(lowerIndexList))
    freeUpperIndexMask = np.ones(len(upperIndexList))
    nFreeUpperIndices = 0
    for uI, upperIndex in enumerate(upperIndexList):
        free = True
        for lI, lowerIndex in enumerate(lowerIndexList):
            if upperIndex == lowerIndex:
                upperIndexLetters += lowerIndexLetters[lI]
                freeLowerIndexMask[lI] = 0
                freeUpperIndexMask[uI] = 0
                free = False
        if free:
            upperIndexLetters += string.ascii_lowercase[len(lowerIndexList) + nFreeUpperIndices]
            nFreeUpperIndices += 1
    newLowerIndexList = [lowerIndex for lI, lowerIndex in enumerate(lowerIndexList) if freeLowerIndexMask[lI]]
    newUpperIndexList = [upperIndex for uI, upperIndex in enumerate(upperIndexList) if freeUpperIndexMask[uI]]
    summandZero = False
    if targetLowerIndexList == None and targetUpperIndexList == None:
        targetLowerIndexList = newLowerIndexList
        targetUpperIndexList = newUpperIndexList
        summandZero = True
    freeLowerIndexLetters = "".join([lowerIndexLetters[lowerIndexList.index(lowerIndex)] for lowerIndex in targetLowerIndexList])
    freeUpperIndexLetters = "".join([upperIndexLetters[upperIndexList.index(upperIndex)] for upperIndex in targetUpperIndexList])
    einsumSubstrings = []
    start = 0
    for vertex in tensorProduct_.vertexList:
        end = start + vertex.excitationRank
        einsumSubstring = lowerIndexLetters[start:end] + upperIndexLetters[start:end]
        einsumSubstrings.append(einsumSubstring)
        start = end
    einsumString = ",".join(einsumSubstrings)
    einsumString += '->' + freeLowerIndexLetters + freeUpperIndexLetters
    contribution = tensorProduct_.prefactor * np.einsum(einsumString, *[vertex.tensor.array for vertex in tensorProduct_.vertexList])
    if summandZero:
        return contribution, newLowerIndexList, newUpperIndexList
    return contribution

def contractTensorSum(tensorSum_):
    contractedArray, lowerIndexList, upperIndexList = getContractedArray(tensorSum_.summandList[0])
    i = 1
    while i < len(tensorSum_.summandList):
        contractedArray += getContractedArray(tensorSum_.summandList[i], lowerIndexList, upperIndexList)
        i += 1
    return contractedArray