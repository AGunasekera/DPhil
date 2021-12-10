from GeneralisedWick import *

def commutator(operator1, operator2):
    return operator1 * operator2 + (-1) * operator2 * operator1

def BCHSimilarityTransform(H, T, order):
    result = H
    for k in range(order):
        nestedCommutator = H
        for i in range(k + 1):
            nestedCommutator = commutator(nestedCommutator, T)
        result += (1 / factorial(k + 1)) * nestedCommutator
    return result

def projectionManifold(excitationLevel):
    return Tensor('\Phi', ['h'] * excitationLevel, ['p'] * excitationLevel)

def getEnergyEquation(similarityTransformedHamiltonian, spinFree=True):
    return evaluateWick(similarityTransformedHamiltonian, spinFree)

def getAmplitudeEquation(similarityTransformedHamiltonian, excitationLevel, spinFree=True):
    projection = projectionManifold(excitationLevel)
    projectedAmplitudeEquation = evaluateWick(projection * similarityTransformedHamiltonian, spinFree)
    amplitudeEquation = copy(projectedAmplitudeEquation)
    for summand in amplitudeEquation.summandList:
        summand.tensorList.pop(0)
        summand.vertexList.pop(0)
    return amplitudeEquation

def iterateDoublesAmplitudes(doublesTensor, residual, fockMatrix, spinFree=True):
    amplitudes = doublesTensor.array
    for i in range(amplitudes.shape[0]):
        for j in range(amplitudes.shape[1]):
            for k in range(amplitudes.shape[2]):
                for l in range(amplitudes.shape[3]):
                    amplitudes[i,j,k,l] -= residual.array[i,j,k,l] / (fockMatrix[i + amplitudes.shape[2], i + amplitudes.shape[2]] + fockMatrix[j + amplitudes.shape[3], j + amplitudes.shape[3]] - fockMatrix[k, k] - fockMatrix[l, l])
#    if spinFree:
#        amplitudes = (1./3.) * amplitudes + (1./6.) * amplitudes.swapaxes(0,1)
    return amplitudes

def iterateSinglesAmplitudes(singlesTensor, residual, fockMatrix):
    amplitudes = singlesTensor.array
    for i in range(amplitudes.shape[0]):
        for j in range(amplitudes.shape[1]):
#            amplitudes[i,j] -= (2 - (i==j)) * residual.array[i,j] / (fockMatrix[i + amplitudes.shape[1], i + amplitudes.shape[1]] - fockMatrix[j, j])
            amplitudes[i,j] -= residual.array[i,j] / (fockMatrix[i + amplitudes.shape[1], i + amplitudes.shape[1]] - fockMatrix[j, j])
    return amplitudes

def iterateTriplesAmplitudes(triplesTensor, residual, fockMatrix):
    amplitudes = triplesTensor.array
    for i in range(amplitudes.shape[0]):
        for j in range(amplitudes.shape[1]):
            for k in range(amplitudes.shape[2]):
                for l in range(amplitudes.shape[3]):
                    for m in range(amplitudes.shape[4]):
                        for n in range(amplitudes.shape[5]):
                            amplitudes[i,j,k,l,m,n] -= residual.array[i,j,k,l,m,n] / (fockMatrix[i + amplitudes.shape[3], i + amplitudes.shape[3]] + fockMatrix[j + amplitudes.shape[4], j + amplitudes.shape[4]] + fockMatrix[k + amplitudes.shape[5], k + amplitudes.shape[5]] - fockMatrix[l, l] - fockMatrix[m, m] - fockMatrix[n, n])
    return amplitudes

def convergeDoublesAmplitudes(doublesTensor, CCDEnergyEquation, CCDAmplitudeEquation, fockTensor):
    residualTensor = Tensor("R", ['p', 'p'], ['h', 'h'])
    doublesTensor.array = np.zeros_like(doublesTensor.array)
    Energy = contractTensorSum(CCDEnergyEquation)
    residualTensor.array = contractTensorSum(CCDAmplitudeEquation)
    residualTensor.array = (1./3.) * residualTensor.array + (1./6.) * residualTensor.array.swapaxes(0,1)
    while True:
        print(Energy)
        doublesTensor.array = iterateDoublesAmplitudes(doublesTensor, residualTensor, fockTensor.array)
        residualTensor.array = contractTensorSum(CCDAmplitudeEquation)
        residualTensor.array = (1./3.) * residualTensor.array + (1./6.) * residualTensor.array.swapaxes(0,1)
        Energy = contractTensorSum(CCDEnergyEquation)
        if np.all(abs(residualTensor.array) < 0.00000000000001):
            break
    print(Energy)
    print(doublesTensor.array)

def convergeCCSDAmplitudes(singlesTensor, doublesTensor, CCSDEnergyEquation, singlesCCSDAmplitudeEquation, doublesCCSDAmplitudeEquation, fockTensor):
    singlesResidualTensor = Tensor("R", ['p'], ['h'])
    doublesResidualTensor = Tensor("R", ['p', 'p'], ['h', 'h'])
    singlesTensor.array = np.zeros_like(singlesTensor.array)
    doublesTensor.array = np.zeros_like(doublesTensor.array)
    Energy = contractTensorSum(CCSDEnergyEquation)
    singlesResidualTensor.array = contractTensorSum(singlesCCSDAmplitudeEquation)
    doublesResidualTensor.array = contractTensorSum(doublesCCSDAmplitudeEquation)
    doublesResidualTensor.array = (1./3.) * doublesResidualTensor.array + (1./6.) * doublesResidualTensor.array.swapaxes(0,1)
    while True:
        print(Energy)
        singlesTensor.array = iterateSinglesAmplitudes(singlesTensor, singlesResidualTensor, fockTensor.array)
        doublesTensor.array = iterateDoublesAmplitudes(doublesTensor, doublesResidualTensor, fockTensor.array)
        singlesResidualTensor.array = contractTensorSum(singlesCCSDAmplitudeEquation)
        doublesResidualTensor.array = contractTensorSum(doublesCCSDAmplitudeEquation)
        doublesResidualTensor.array = (1./3.) * doublesResidualTensor.array + (1./6.) * doublesResidualTensor.array.swapaxes(0,1)
        Energy = contractTensorSum(CCSDEnergyEquation)
        if np.all(abs(singlesResidualTensor.array) < 0.00000000000001) and np.all(abs(doublesResidualTensor.array) < 0.00000000000001):
            break
    print(Energy)
    print(singlesTensor.array)
    print(doublesTensor.array)

def convergeCCSDTAmplitudes(singlesTensor, doublesTensor, triplesTensor, CCSDTEnergyEquation, singlesCCSDTAmplitudeEquation, doublesCCSDTAmplitudeEquation, triplesCCSDTAmplitudeEquation,  fockTensor):
    singlesResidualTensor = Tensor("R", ['p'], ['h'])
    doublesResidualTensor = Tensor("R", ['p', 'p'], ['h', 'h'])
    triplesResidualTensor = Tensor("R", ['p', 'p', 'p'], ['h', 'h', 'h'])
    singlesTensor.array = np.zeros_like(singlesTensor.array)
    doublesTensor.array = np.zeros_like(doublesTensor.array)
    triplesTensor.array = np.zeros_like(triplesTensor.array)
    Energy = contractTensorSum(CCSDTEnergyEquation)
    singlesResidualTensor.array = contractTensorSum(singlesCCSDTAmplitudeEquation)
    doublesResidualTensor.array = contractTensorSum(doublesCCSDTAmplitudeEquation)
    triplesResidualTensor.array = contractTensorSum(triplesCCSDTAmplitudeEquation)
    while True:
        print(Energy)
        singlesTensor.array = iterateSinglesAmplitudes(singlesTensor, singlesResidualTensor, fockTensor.array)
        doublesTensor.array = iterateDoublesAmplitudes(doublesTensor, doublesResidualTensor, fockTensor.array)
        triplesTensor.array = iterateTriplesAmplitudes(triplesTensor, triplesResidualTensor, fockTensor.array)
        singlesResidualTensor.array = contractTensorSum(singlesCCSDTAmplitudeEquation)
        doublesResidualTensor.array = contractTensorSum(doublesCCSDTAmplitudeEquation)
        triplesResidualTensor.array = contractTensorSum(triplesCCSDTAmplitudeEquation)
        Energy = contractTensorSum(CCSDTEnergyEquation)
        if np.all(abs(singlesResidualTensor.array) < 0.00000000000001) and np.all(abs(doublesResidualTensor.array) < 0.00000000000001) and np.all(abs(triplesResidualTensor.array) < 0.00000000000001):
            break
    print(Energy)
    print(singlesTensor.array)
    print(doublesTensor.array)
    print(triplesTensor.array)