import pandas as pd
import time
from ud_graphs import *
import torch
from train_utils import *
from DistanceEncoderNetwork import DistanceEncoderNetwork   


class DENSolver():
    """    DEN-based solver for the constrained unit disk graph problem.
    """
    
    def __init__(self, min_dist, max_dist, max_dist_adj, set_init_pos, dim, max_epochs=3000):
        """        
        Args:
            min_dist ( numeric ): minimum allowed distance between vertexes.
            max_dist ( numeric ): maximum allowed distance between non adjacent vertexes.
            max_dist_adj ( numeric ): maximum allowed distance between adjacent vertexes.
            set_init_pos ( string ): method for computing initial positions (allowed 'scaling' or 'FR').
            dim ( int ): number of dimensions for the coordinates (allowed 2 or 3).
            max_epochs (int, optional): number of training epochs. Defaults to 3000.
        """
        self.max_epochs = max_epochs
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.max_dist_adj = max_dist_adj
        self.set_init_pos = set_init_pos
        self.dim = dim

    def solve(self, num_vertexes, sample, p_drop, lr):
        """Solve the constrained unit disk problem.

        Args:
            num_vertexes ( int ): number of vertexes of the graph (allowed 10, 20, ..., 100).
            sample ( int ): sample identifier (allowed 0,1,...,19).
            p_drop ( float ): dropout probability in [0,1].
            lr (float ): learning rate.

        Returns:
             pandas.Series: summary of the solution.
        """
        # initialize seed
        set_seed(7)
        graph_file = "graph_samples/{}G_{}.gpickle".format(num_vertexes, sample)
        # preprocessing phase
        G_init, init_pos = load_graph(graph_file, self.max_dist/2, self.set_init_pos, self.dim)  
        A, not_A= compute_adj_mtx(G_init)
        D_init = distance_matrix(init_pos, init_pos)
        d_init, D_init, D_adj_init, d_not_adj_init = compute_dists_minmax(D_init, A, not_A)
        device = torch.device('cpu')
        adj_tensor = compute_adj_tensor(A)
        # initial coordinates tensor
        init_pos_tensor = torch.Tensor(init_pos.T.flatten()).to(device)
        # initialize the DEN model
        DEN_model =  DistanceEncoderNetwork(num_vertexes, p_drop=p_drop, dim=self.dim).to(device)
        optimizer = torch.optim.AdamW(DEN_model.parameters(), lr=lr)           
        best_gap = -1
        epoch=0
        loss_vals, loss_vals_min, loss_vals_max = reset_loss_lists()
        # learning phase
        start = time.time()      
        while epoch <= self.max_epochs:            
            with torch.autograd.set_detect_anomaly(True):                              
                # train step
                DEN_model.train()
                output_dists_train = DEN_model(init_pos_tensor)      
                optimizer.zero_grad()
                loss_min, loss_max, loss_gap = embedding_loss(output_dists_train, adj_tensor, device, max_adj_feasible=self.max_dist_adj)        
                loss_val = loss_min + loss_max + loss_gap            
                loss_val.backward()
                optimizer.step()
                loss_vals, loss_vals_min, loss_vals_max = update_loss_lists(
                    loss_val, loss_min, loss_max, loss_vals, loss_vals_min, loss_vals_max)                                  
                # inference step
                DEN_model.eval()
                _ = DEN_model(init_pos_tensor) 
                den_pos = DEN_model.pos
                D = distance_matrix(den_pos, den_pos)
                found_feasible, gap = check_feasible_embedding(D, A, not_A, self.min_dist, self.max_dist, self.max_dist_adj)
                if found_feasible and gap > best_gap:
                    # optimiza the adjacency gap
                    if best_gap == -1:
                        first_feasible_epoch=epoch  
                    D_den = D    
                    best_gap = gap  
                    best_pos = den_pos     
            epoch+=1
        elapsed = time.time() - start          
        if best_gap > 0:
            # retrive the best feasible solution found so far
            d, D, D_adj, d_not_adj = compute_dists_minmax(D_den, A, not_A)
            res_series = pd.Series({'num_vertexes': num_vertexes, 'sample': sample, 'pdrop': p_drop, 'lr': lr, 'd_init': d_init,
                    'D_init': D_init, 'D_adj_init': D_adj_init, 'd_not_adj_init': d_not_adj_init, 'd': d, 'D': D, 'D_adj': D_adj,
                    'd_not_adj': d_not_adj, 'time': elapsed, 'ff_epoch':  first_feasible_epoch})
            ud_G = nx.Graph()
            
            _, feasible_pos_dict = positions_dict(G_init, init_pos, best_pos)
            max_dist_n1 = D_den.argmax()//num_vertexes
            max_dist_n2 = D_den.argmax()%num_vertexes
            center = (best_pos[max_dist_n1]+best_pos[max_dist_n2])/2

            for n, coords in feasible_pos_dict.items():
                ud_G.add_node(n,pos=tuple(coords-center))  
            ud_G.add_edges_from(G_init.edges())
            
            # save graph object
            nx.write_gpickle(ud_G, "graph_samples_ud/{}G_{}s.gpickle".format(num_vertexes, sample))


        else:
            # no feasible solution is found
            res_series = pd.Series({'num_vertexes': num_vertexes, 'sample': sample, 'pdrop': p_drop, 'lr': lr, 'd_init': d_init,
                    'D_init': D_init, 'D_adj_init': D_adj_init, 'd_not_adj_init': d_not_adj_init, 'd': None, 'D': None, 'D_adj': None,
                    'd_not_adj': None, 'time': elapsed, 'ff_epoch':  None})            
        return res_series