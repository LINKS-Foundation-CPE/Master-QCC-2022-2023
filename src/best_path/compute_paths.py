import sys
import osmnx as ox
import pickle
from datetime import datetime

def TransformPath(path):
    tuplePath = []
    for i in range(1, len(path)):
        tuplePath.append((path[i-1], path[i]))

    return tuplePath

def JaccardIndex(route1, route2):
    set1 = set(route1)
    set2 = set(route2)

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    return len(intersection)/len(union)

def ExtractMostDifferent(paths, numberOfRoutes):
    #matrix = [ [ 0 for i in range(len(paths)) ] for j in range(len(paths)) ]
    meanJaccard = [0]*len(paths)

    #calculate mean jaccard values
    for i in range(len(paths)):
        for j in range(i+1, len(paths)):
            jaccardValue = JaccardIndex(paths[i], paths[j])
            meanJaccard[i] += jaccardValue
            meanJaccard[j] += jaccardValue
    print(datetime.now(), " -> Jaccard indexes computation DONE")
    indexes = [tupla[1] for tupla in sorted(zip(meanJaccard, list(range(len(paths)))))[:numberOfRoutes]]

    resultList = []
    for index in indexes:
        resultList.append(TransformPath(paths[index]))

    return resultList

def ExtractMostDifferent_(paths, numberOfRoutes):
    #matrix = [ [ 0 for i in range(len(paths)) ] for j in range(len(paths)) ]
    jaccardValue = [0]*(len(paths)-1)

    #calculate mean jaccard values
    for i in range(1, len(paths)):
        jaccardValue[i-1] = JaccardIndex(paths[i], paths[0])
    print(datetime.now(), " -> Jaccard indexes computation DONE")
    
    indexes = [tupla[1] for tupla in sorted(zip(jaccardValue, list(range(len(paths)))))[:(numberOfRoutes-1)]]
    indexes.append(0)

    jaccardtuple = sorted(zip(jaccardValue, list(range(len(paths)))))
    
    print(sorted(jaccardValue))
    resultList = []
    for index in indexes:
        resultList.append(TransformPath(paths[index]))

    return resultList

def compute_paths(G, couples, NumberOfRoutes):
    #initialize data

    print(datetime.now(), " -> start, end nodes' IDs = " , couples)

    #generate 4 times the shortest path
    rowPaths = ox.k_shortest_paths(G, couples[0], couples[1], int(NumberOfRoutes*10))
    paths = []
    for path in rowPaths:
        paths.append(path)
    print(datetime.now(), " -> k_shortest DONE")
    #compute the jaccard index and get the most different path, evaluating the mean
    car_Routes = ExtractMostDifferent_(paths, int(NumberOfRoutes))
    print(datetime.now(), " -> extraction of most different paths DONE")
    return car_Routes
