This code is about codon optimization using quantum-classical hybrid protocols, detailed in Y. K. Chung, et al. "Quantum-classical hybrid approach for codon optimization and its practical applications." [bioRxiv (2024): 2024-06.](https://www.biorxiv.org/content/10.1101/2024.06.08.598046v1.abstract)

Contribution: Dongkeun Lee, Jaehee Kim, Junho Lee

# Description

* `codon_optimization.py` : Main code involving 1. Constructing the objective function and constraints 2. Solving the constrained quadratic model 3. Conversion between DNA (or RNA) seqence and qubit vector 

* `codon_opt.ipynb` & `codon_LeaphybridCQMsolver.ipynb` : Describing functions and classes in `codon_optimization.py`

* `codon_hamiltonian_graph.ipynb` : Drawing graphes of the obejctive function for each amino acid sequence, SARS-Cov2 and insulin.

* `results_codon_optimization.ipynb` : Running `codon_optimization.py` to solve codon optimization problems using D-Wave LeapCQMHybridSolver

* `codon_table`: Codon Usage Table and Codon Pair Usage Tables obtained from COCOPUTs [[Link]](https://dnahive.fda.gov/dna.cgi?cmd=cuts_main)
