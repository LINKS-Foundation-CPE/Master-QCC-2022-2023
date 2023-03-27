import pybnb
import numpy as np
import networkx as nx
from BB_utils import *
from MIS import PulserMISSolver
from feasibility_utils import is_mIS

class BBQ_mIS(pybnb.Problem):
    
    def __init__(self, G, colors_used=0):
        self.G = G 
        self.orig_G = G.copy()     
        self.x = np.zeros((len(G.nodes),1))
        _, LB, _ = compute_LB(self.x, self.G)
        self.colors_used = colors_used
        self.edges = len(G.edges())
        self.fingerprints=set([])        
        self.lower_bound=colors_used+LB
        coloring =  dict.fromkeys(G.nodes(), -1)
        obj, coloring = compute_obj(G, colors_used, coloring)
        self.obj = obj
        self.coloring = coloring
        self.child_story={0:-1}
        self.hist_list = []
        self.solutions = None

    def sense(self):
        return pybnb.minimize

    def objective(self):             
        return self.obj

    def bound(self):
        # lower bound on the objective function
        return self.lower_bound

    def save_state(self, node, x=None, colors_used=None, lower_bound=None, H=None, obj=None, coloring=None, child_story=None, solutions = None, hist_list=None):
        if x is None:
            # root node initialization
            node.state = (self.x, self.colors_used, self.G, self.edges, self.coloring, self.child_story, self.solutions, self.hist_list)
        else:
            num_edges = len(H.edges())
            node.state = (x, colors_used, H, num_edges, coloring, child_story, solutions, hist_list)
            node.objective = obj
            node.bound = lower_bound
            UB = compute_UB(H)
            node.queue_priority = -UB*num_edges
                          
            
    def load_state(self, node):
        (self.x, self.colors_used, self.G, self.edges, self.coloring, self.child_story, self.solutions, self.hist_list) = node.state
        self.obj = node.objective
        self.lower_bound = node.bound

    def branch(self):
        pulser_MIS_solver = PulserMISSolver(self.G)
        pulser_sol = pulser_MIS_solver.solve_Pulser()
        hist_list=self.hist_list.copy()
        hist_list.append(pulser_sol)
        num_colors_child = self.colors_used+1
        solutions = []  
        for sol in pulser_sol:
            solutions.append(np.array(list(sol[0]), dtype=int))
        num_sol = len(solutions)
        child_num=0                    
        for sol in range(num_sol): 
            child_story=self.child_story.copy()
            child_story[num_colors_child]=child_num+1
            coloring_dict = self.coloring.copy() 
            x = solutions[sol]      
            if is_mIS(x, self.G):                                
                H, LB, MIS_set = compute_LB(x, self.G)
                for graph_node in MIS_set:
                    coloring_dict[graph_node]=num_colors_child                
                fp = fingerprint(H.nodes())                
                if fp not in self.fingerprints and len(H.nodes())>0:
                    child_num+=1
                    # avoid symmetries in BB
                    child = pybnb.Node()
                    obj, coloring_dict = compute_obj(H, num_colors_child, coloring_dict, device = self.device)
                    child_bound = num_colors_child+LB
                    self.save_state(child, x, num_colors_child, child_bound, H, obj, coloring_dict, child_story, pulser_sol, hist_list)
                    self.fingerprints.add(fp)
                    yield child



def BBQ_mIS_solver(G, max_nodes=20):
    problem = BBQ_mIS(G)
    solver = pybnb.Solver()
    results = solver.solve(problem, queue_strategy='custom', node_limit=max_nodes)
    best_coloring = results.best_node.state[4]
    child_story = results.best_node.state[5]
    plot_sol(G, best_coloring, results.objective)    
    print_BB_history(child_story)
    # print the MISs for the best solution
    print_mIS(best_coloring)    
    print('Coloring story {}'.format(child_story))
    print('BB found coloring with {} colors'.format(results.objective))
    return best_coloring