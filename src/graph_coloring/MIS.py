from pulser import Pulse, Sequence, Register
from pulser.devices import Device
from pulser_simulation import QutipEmulator
from pulser.channels import Rydberg
import numpy as np
import networkx as nx
import warnings
from scipy.optimize import minimize, Bounds
from scipy.spatial import distance_matrix
import random

AnalogDevice = Device(
            name="AnalogDevice",
            dimensions=2,
            rydberg_level=60,
            max_atom_num=25,
            max_radial_distance=35,
            min_atom_distance=4,
            channel_objects=(
            Rydberg.Global(
                max_abs_detuning=2 * np.pi * 20,
                max_amp=2 * np.pi * 2,
                clock_period=4,
                min_duration=16,
                mod_bandwidth=8
            ),
        ),
        )


random.seed(0)
warnings.filterwarnings('ignore')
    
class PulserMISSolver:

    def __init__(self, G, penalty=10, num_layers=1, iterations_opt=100) -> None:
        self.num_layers = num_layers
        self.G = G
        self.iterations_opt=iterations_opt
        self.penalty=penalty
        self.num_samples = 100
        self.tol =1e-6
        self.time_unit = 1000
        self.max_time = 3
        self.num_params= 2
        self.omega_max = 5.4
        self.sampling_rate=0.1
        self.Delta_i_fres = 2 * np.pi * 700  # rad/us
        self.Omega_r_fres = 2 * np.pi * 30  # rad/us

    def _get_cost_colouring(self, bitstring):
        z = np.array(list(bitstring), dtype=int)
        A = nx.to_numpy_matrix(self.G)
        # Add penalty and bias:
        cost = self.penalty*(z.T @ np.triu(A) @ z) - np.sum(z)
        return cost


    def _Omega_b_from_Omega(self, Omega):
        return Omega * (2 * abs(self.Delta_i_fres) / self.Omega_r_fres)

    def _lightshift(self, Omega):
        return (self.Omega_r_fres ** 2 - self._Omega_b_from_Omega(Omega) ** 2) / (4 * self.Delta_i_fres)


    def _define_sequence(self, register):
        # Parametrized sequence
        seq = Sequence(register, AnalogDevice)
        seq.declare_channel('ch0','rydberg_global')
        # add parametrize sequence
        t_list = seq.declare_variable('t_list', size=self.num_layers)
        s_list = seq.declare_variable('s_list', size=self.num_layers)
        if self.num_layers == 1:
            t_list = [t_list]
            s_list = [s_list]
        rabi_1 = self.rabi_freq
        rabi_2 = 0
        detuning_1 = self._lightshift(rabi_1)-self._lightshift(rabi_1) 
        detuning_2 = self._lightshift(rabi_1) - self._lightshift(0)
        for l in range(self.num_layers):
            t_value = self.time_unit * t_list[l]
            s_value = self.time_unit * s_list[l]
            t_value = t_value - t_value % 4  # Round to nearest 4 ns value
            s_value = s_value - s_value % 4  # Round to nearest 4 ns value
            pulse1 = Pulse.ConstantPulse(t_value, rabi_1, detuning_1, 0)
            pulse2 = Pulse.ConstantPulse(s_value, rabi_2, detuning_2, 0)
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
        # Divide by total samples
        return cost / sum(counter.values())

    def _experiment(self, sequence, parameters):                
        t_params, s_params = np.reshape(np.array(parameters), (self.num_params, self.num_layers))
        assigned_sequence = sequence.build(t_list=t_params, s_list=s_params)    
        simul = QutipEmulator.from_sequence(assigned_sequence, sampling_rate=self.sampling_rate)
        simul.set_initial_state='all-ground'
        simul.set_evaluation_times = np.linspace(0, simul._tot_duration/1000, 10)
        results = simul.run(nsteps=10000)
        return results.sample_final_state(N_samples=self.num_samples)

    def _get_param_bounds(self):
        pos_list = []
        for n in self.G.nodes():
            pos_list.append(self.G._node[n]['pos'])
        pos=np.array(pos_list) 
        # find the rydberg blockade radius
        dist_matrix = distance_matrix(pos, pos)
        A = nx.to_numpy_matrix(self.G)
        blockade_radius = dist_matrix[A==1].max() 
        self.rabi_freq = AnalogDevice.rabi_from_blockade(blockade_radius)
        # limit rabi frequency to the maxixmum value allowed for the device
        self.rabi_freq = min(self.rabi_freq, self.omega_max) 
        # Bounds for max total pulse length (machine max = 100 mus)
        dbounds = [(0.1, self.max_time), (0.1, self.max_time)]        
        return dbounds    
    
    def _optimize_params_scipy(self,  bounds, opt_name='Nelder-Mead'):
        guess = {'t': np.random.uniform(bounds[0][0], bounds[0][1], self.num_layers),
                's': np.random.uniform(bounds[1][0], bounds[1][1], self.num_layers)
                }
        max_function_calls = self.iterations_opt
        bounds = Bounds(lb=[bounds[0][0], bounds[1][0]], ub=[bounds[0][1], bounds[1][1]], keep_feasible=True)
        res = minimize(self._cost_func,
                    method=opt_name,
                    x0=np.r_[guess['t'], guess['s']],
                    bounds=bounds,
                    tol=self.tol,
                    options = {'maxfev': max_function_calls}
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
        return sorted_dict

    def solve_Pulser(self):
        ''' Solve the MIS problem associated with the graph G'''
        bounds = self._get_param_bounds()
        opt_params = self._optimize_params_scipy(bounds)
        MIS_sol = self._get_mis_solutions(opt_params) 
        return MIS_sol