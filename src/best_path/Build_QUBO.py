import sys
from src.best_path.Pollution import EvaluatePollution

def routes_overlap(route_i, route_j):
    acc = 0.
    for seg in route_i:
        if seg in route_j:
            acc = acc + 1.
    return acc

def obj_QUBO(info_list):
    matrix = {}
    norm = 0.
    for i, routes in enumerate(info_list):
        for j in routes:
            for k in range(i+1, len(info_list)):
                for l in info_list[k]:
                    temp = routes_overlap(j['route'],l['route'])
                    if temp != 0.:
                        matrix[(j['var_num'],l['var_num'])] = temp
                        norm = max(norm, temp)
    matrix = {key: matrix[key] / norm for key in matrix}
    return matrix

def const1_QUBO(info_list):
    matrix = {}
    for routes in info_list:
        for i, route in enumerate(routes):
            matrix[(route['var_num'],route['var_num'])] = -1
            for route_j in routes[i+1:]:
                matrix[(route['var_num'],route_j['var_num'])] = 2
    return matrix

def GenerateQUBOmatrix_(G, info_list):
    func_matrix = obj_QUBO(info_list)
    func_matrix_poll = EvaluatePollution(G, info_list)
    Const1_matrix = const1_QUBO(info_list)
    Const1_matrix = {key: Const1_matrix[key] * 2 * len(info_list) for key in Const1_matrix}
    result = {key: func_matrix.get(key, 0) + 0*func_matrix_poll.get(key, 0) + Const1_matrix.get(key, 0) for key in set(func_matrix) | set(Const1_matrix) | set(func_matrix_poll)}
    return result