import os
import json
import random
from datetime import datetime
from src.best_path.apiModel import apiPollution

def EvaluatePollution(G, routesPerCar):
    #retrieve content of json file
    cacheFileName = 'cacheData.json'
    cache = {}
    try:
        path = os.path.dirname(__file__)
        file = open(path + "/"+ cacheFileName)
        cache = json.load(file)
        file.close()
    except FileNotFoundError:
        print("error while loading cache")
        pass

    #flag variable to update cache file
    cacheUpdate = False

    #instantiate brezometer
    pollution = apiPollution()
    # #print the status of the Breezometer API
    pollution.apiTest()

    #instantiate data
    QUBOdict = {}
    reductionFactor = 0.

    #cycle over node
    for car, routes in enumerate(routesPerCar):
        for route in routes:
            #generate a set from the node list
            node_list = []
            node_list.append(route['route'][0][0])
            node_list.extend([nodes[1] for nodes in route['route']])

            pathTotalPollution_PM25 = 0
            pathTotalPollution_PM10 = 0

            for node in node_list:
                #check if data is in the cache
                if node not in cache:# or (datetime.strptime(cache[node]['SampleTime'], '%H:%M:%S') - datetime.now()).total_seconds() > 3600:
                    #retrieve data and add it to cache
                    nodeData = {}
                    
                    #retrieve node
                    Gnode = G.nodes[node]
                    #pollution.queryPM(Gnode['y'], Gnode['x'])

                    nodeData['PM10'] = round(random.uniform(5.0, 60.0), 1)
                    nodeData['PM25'] = round(random.uniform(5.0, 60.0), 1)
                    nodeData['SampleTime'] = datetime.now().strftime("%H:%M:%S")
                    cache[node] = nodeData
                    #cacheUpdate = True
            
            #evaluate pollution of this routes
            for segment in route['route']:
                #retrieve data
                dataNode1 = cache[segment[0]]
                dataNode2 = cache[segment[1]]

                #evaluate mean pollution and add it to counter
                meanPollution_PM25 = (dataNode1['PM25'] + dataNode2['PM25'])/2
                meanPollution_PM10 = (dataNode1['PM10'] + dataNode2['PM10'])/2

                pathTotalPollution_PM25 += G.get_edge_data(segment[0], segment[1])[0]['length'] * meanPollution_PM25
                pathTotalPollution_PM10 += G.get_edge_data(segment[0], segment[1])[0]['length'] * meanPollution_PM10

            #save values
            route['pm10'] = pathTotalPollution_PM10
            route['pm25'] = pathTotalPollution_PM25
            reductionFactor = max(pathTotalPollution_PM25, reductionFactor)

    #perturb diagonal of the matrix
    for car in routesPerCar:
        for route in car:
            QUBOdict[(route['var_num'], route['var_num'])] = route['pm25']/reductionFactor

    if cacheUpdate:
        file = open(cacheFileName, 'w')
        json_object = json.dumps(cache)
        file.write(json_object)
        file.close()

    return QUBOdict
