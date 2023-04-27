import numpy as np
from dwave_qbsolv import QBSolv
#from dwave.system import DWaveSampler, EmbeddingComposite
import yaml
import sys

def SeparateTerms(QUBOdict):
    diag = {}
    couples = {}

    for key in QUBOdict:
        if key[0] == key[1]:
            diag[key] = QUBOdict[key]
        else:
            couples[key] = QUBOdict[key]

    return diag, couples

def CheckSolution(solutionRoutes, route_list, routesDict):
    solOccurrance = []

    for key in solutionRoutes:
        car = routesDict[key[0]%len(route_list)]

        if car in solOccurrance:
            return False

        solOccurrance.append(car)
    solOccurrance.sort()
    if solOccurrance != list(range(0, max(list(routesDict.values()))+1)):
        return False
    else:
        return True

def CheckQUBOCorrectness(QUBOdict, dim):
    #cycle under diagonal cells
    for row in range(1, dim):
        for column in range(row):
            if (row, column) in QUBOdict:
                return False
    return True

def RetrieveRoutesSolution(resultData, route_list):
    solutionRoutes = []
    chosenRoutes = []

    #cicle over result and get the correct route
    for key in resultData.keys():
        if not isinstance(key, tuple) and resultData[key] == 1:
            #Route solution found
            solutionRoutes.append((key, route_list[key % len(route_list)]))
            chosenRoutes.append(key)

    return solutionRoutes, chosenRoutes

def SolveQUBO(QPUsolve, QUBOdict, route_list, numberOfRoutes, routesDict):
    #check QUBO correctness
    IsCorrect = CheckQUBOCorrectness(QUBOdict, np.sum(numberOfRoutes) * len(route_list))

    if not IsCorrect:
        raise ValueError("Incorrect QUBO matrix")

    #instantiate variables
    iteration = 0
    isValid = False

    if QPUsolve:
        #read yaml file
        with open(r'./guild.yaml') as file:
            parameters = yaml.load(file, Loader=yaml.FullLoader)

        solver = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'chimera'}))
        diag, couples = SeparateTerms(QUBOdict)

        #range for annealing time is [0.5, 2000]
        qpu_ann_time = parameters[0]['operations']['train']['flags']['annealing_time'][2]
        reads = parameters[0]['operations']['train']['flags']['num_reads'][1]
        boltzmann = parameters[0]['operations']['train']['flags']['boltzmann']

        resultData = solver.sample_ising(diag, couples, num_reads = reads, annealing_time = qpu_ann_time, beta = boltzmann)
    else:
        resultData = QBSolv().sample_qubo(QUBOdict)

    #sample 5 result
    samples = [list(item) for item in list(resultData.record['sample'])][:5]

    Keys = list(resultData.variables)
        

    while iteration < len(samples) and not isValid:
        #return solution of QUBO problem
        solutionRoutes, chosenRoutes = RetrieveRoutesSolution(dict(zip(Keys, samples[iteration])), route_list)

        #check solution
        isValid = CheckSolution(solutionRoutes, route_list, routesDict)
        energy = list(resultData.record['energy'])[iteration]
        iteration += 1
    if not isValid:
        print("WAR: no valid solutions found!!!")
    return solutionRoutes, energy, chosenRoutes

