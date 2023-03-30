from pulser import Pulse, Sequence, Register, waveforms
from pulser.devices import Chadoq2
from pulser.simulation import Simulation
import numpy as np
import networkx as nx
import warnings
from scipy.optimize import minimize
from scipy.spatial import distance_matrix

warnings.filterwarnings('ignore')
np.random.seed(7)



def initialize_state():
    # Parameters in rad/µs and ns
    Omega_max = 2.3 * 2*np.pi
    U = Omega_max / 2.3
    delta_0 = -6 * U
    delta_f = 2 * U
    t_rise = 250
    t_sweep = (delta_f - delta_0)/(2 * np.pi * 10) * 1000
    rise = Pulse.ConstantDetuning(waveforms.RampWaveform(t_rise, 0., Omega_max), delta_0, 0.)
    sweep = Pulse.ConstantAmplitude(Omega_max, waveforms.RampWaveform(t_sweep, delta_0, delta_f), 0.)
    return rise, sweep

class PulserMISSolver:

    def __init__(self, G, penalty=10, num_layers=1):
        self.num_layers = num_layers
        self.G = G
        self.penalty=penalty
        self.max_iters=100       
        self.num_samples = 500
        self.tol =1e-5
        self.sampling_rate = 0.1
        self.time_unit = 1000
        self.max_time = 10
        self.num_params= 2
        self.omega_max = 2.3 * 2*np.pi

    def _get_cost_colouring(self, bitstring):
        z = np.array(list(bitstring), dtype=int)
        A = nx.to_numpy_matrix(self.G)
        # Add penalty and bias:
        cost = self.penalty*(z.T @ np.triu(A) @ z) - np.sum(z)
        return cost

    def _define_sequence(self, register):
        # Parametrized sequence
        seq = Sequence(register, Chadoq2)
        seq.declare_channel('ch0','rydberg_global')
        # initialize state for the register
        rise, sweep = initialize_state()
        seq.add(rise, 'ch0')
        seq.add(sweep, 'ch0')
        # add parametrize sequence
        t_list = seq.declare_variable('t_list', size=self.num_layers)
        s_list = seq.declare_variable('s_list', size=self.num_layers)
        if self.num_layers == 1:
            t_list = [t_list]
            s_list = [s_list]
        for l in range(self.num_layers):
            pulse1 = Pulse.ConstantPulse(self.time_unit*t_list[l], self.rabi_freq, 0, 0)
            pulse2 = Pulse.ConstantPulse(self.time_unit*s_list[l], self.rabi_freq, self.rabi_freq, 0)
            seq.add(pulse1, 'ch0')        
            seq.add(pulse2, 'ch0')
        seq.measure('ground-rydberg')         
        return seq
       
    def _cost_func(self, param):
        qubits={}
        for n in self.G.nodes():
            qubits[n]=self.G._node[n]['pos']
        reg = Register(qubits)
        sequence = self._define_sequence(reg)
        counter = self._experiment(sequence, param)
        cost = sum(counter[key] * self._get_cost_colouring(key) for key in counter)
        return cost / sum(counter.values())

    def _experiment(self, sequence, parameters):                
        t_params, s_params = np.reshape(np.array(parameters), (self.num_params, self.num_layers))
        assigned_sequence = sequence.build(t_list=t_params, s_list=s_params)    
        simul = Simulation(assigned_sequence, sampling_rate=self.sampling_rate)
        results = simul.run(num_cpus=4)
        return results.sample_final_state(N_samples=self.num_samples)

    def _get_param_bounds(self):
        pos_list =[]
        for n in self.G.nodes():
            pos_list.append(self.G._node[n]['pos'])
        pos=np.array(pos_list) 
        # find the rydberg blockade radius
        dist_matrix = distance_matrix(pos, pos)
        A = nx.to_numpy_matrix(self.G)
        blockade_radius = dist_matrix[A==1].max() 
        self.rabi_freq = Chadoq2.rabi_from_blockade(blockade_radius)
        # limit rabi frequency to the maxixmum value allowed for the device
        self.rabi_freq = min(self.rabi_freq, self.omega_max) 
        # Bounds for max total pulse length (machine max = 100 mus)
        dbounds = [(0.016, self.max_time), (0.016, self.max_time*0.3)]        
        return dbounds
    
    
    def _optimize_constrained(self,  bounds):
        guess = {'t': np.random.uniform(bounds[0][0], bounds[0][1], self.num_layers),
                's': np.random.uniform(bounds[1][0], bounds[1][1], self.num_layers)
                }
        res = minimize(self._cost_func,
                    method='trust-constr',
                    x0=np.r_[guess['t'], guess['s']],
                    bounds=bounds,
                    tol=self.tol,
                    options = {'maxiter': self.max_iters, 'disp': False}
                    )
        opt_params = res.x
        return opt_params

    def _get_mis_solutions(self, params):
        qubits={}
        for n in self.G.nodes():
            qubits[n]=self.G._node[n]['pos']
        reg = Register(qubits)
        sequence = self._define_sequence(reg)
        count_dict = self._experiment(sequence, params)
        sorted_dict = sorted(count_dict.items(), key=lambda item: item[1], reverse=True)
        sol_list = []  
        for sol in sorted_dict:
            sol_list.append(np.array(list(sol[0]), dtype=int)) 
        return sol_list

    def solve_Pulser(self):
        ''' Solve the MIS problem associated with the graph G'''
        bounds = self._get_param_bounds()
        opt_params = self._optimize_constrained(bounds)
        MIS_sol = self._get_mis_solutions(opt_params)
        return MIS_sol