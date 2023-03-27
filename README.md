# Master-QCC-2022-2023
LINKS Course materials for Polytechnic of Turin Master in Quantum communication and computing 2022 2023


## Maximum Independent Set (MIS) tutorial 
[MIS colab](https://github.com/LINKS-Foundation-CPE/cineca_aspc/blob/main/UD-mis/UD-mis.ipynb) tutorial provides the basic tools to implement MIS problems through the [Pulser](https://pulser.readthedocs.io/) library, leveraging the Rydberg Blockade effect. Furthermore, it highlights the effect of the optimization approach to make the optimal solutions more likely to be measured.

## Graph coloring (GC) tutorial
These tutorials concern GC problems solutions through iterative MIS problem solution, exploiting [Pulser](https://pulser.readthedocs.io/) software. Two different approaches are shown:
- Greedy-it-MIS approach: solve iteratively MIS problem and assign one color at a time
- BBQ-mIS: Branch&Bound (BB) approach to explore multiple mIS solutions and find better coloring. It exploits the [PyBnB](https://pypi.org/project/pybnb/) library to model the BB exploration.

## Unit Disk (UD) graph embeddings tutorial
This tutorial implements a novel approach to retrieve UD graph embeddings starting from the adjacency matrix of a given graph. It is the official implementation of the work presented in our paper: [Neural-powered unit disk graph embedding: qubits connectivity for some QUBO problems](https://ieeexplore.ieee.org/document/9951178).