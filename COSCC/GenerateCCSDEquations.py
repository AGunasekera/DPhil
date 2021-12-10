from time import time
from GeneralisedWick import *
import CC, texify, pickle

fockTensor = Tensor("f", ['g'], ['g'])
h2Tensor = Tensor("v", ['g', 'g'], ['g', 'g'])

fockTensor.getAllDiagramsGeneral()
h2Tensor.getAllDiagramsGeneral()

t1Tensor = Tensor("{t_{1}}", ['p'], ['h'])
t2Tensor = Tensor("{t_{2}}", ['p', 'p'], ['h', 'h'])

normalOrderedHamiltonian = sum(fockTensor.diagrams) + (1. / 2.) * sum(h2Tensor.diagrams)
BCH = CC.BCHSimilarityTransform(normalOrderedHamiltonian, t1Tensor + 0.5 * t2Tensor, 4)

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
#for summand in doublesAmplitudeEquation.summandList:
#    summand.prefactor *= (1. / 2.)
t3 = time()
print("Time to find doubles amplitude equation:", t3 - t2)
print("number of terms:", len(doublesAmplitudeEquation.summandList))

d = {}
d["energyEquation"] = energyEquation
d["singlesAmplitudeEquation"] = singlesAmplitudeEquation
d["doublesAmplitudeEquation"] = doublesAmplitudeEquation
d["fockTensor"] = fockTensor
d["h2Tensor"] = h2Tensor
d["t1Tensor"] = t1Tensor
d["t2Tensor"] = t2Tensor
with open("CCSDEquations.pkl", 'wb') as f:
    p = pickle.Pickler(f)
    p.dump(d)

#def save(filename, *args):
    # Get global dictionary
#    glob = globals()
#    d = {}
#    for v in args:
        # Copy over desired values
#        d[v] = glob[v]
#    with open(filename, 'wb') as f:
        # Put them in the file 
#        pickle.dump(d, f)

#def load(filename):
    # Get global dictionary
#    glob = globals()
#    with open(filename, 'rb') as f:
#        for k, v in pickle.load(f).items():
            # Set each global variable to the value from the file
#            glob[k] = v

#save("CCSDEnergy.pkl", "energyEquation", "fockTensor", "h2Tensor", "t1Tensor", "t2Tensor")
#save("CCSDSinglesAmplitudeEquation.pkl", "singlesAmplitudeEquation", "fockTensor", "h2Tensor", "t1Tensor", "t2Tensor")
#save("CCSDDoublesAmplitudeEquation.pkl", "doublesAmplitudeEquation", "fockTensor", "h2Tensor", "t1Tensor", "t2Tensor")
texify.texify([energyEquation, singlesAmplitudeEquation, doublesAmplitudeEquation], "CCSDEquations")