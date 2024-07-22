This code is about codon optimization using quantum-classical hybrid protocols, detailed in Ref. Y. K. Cheong, et al. bioarXiv: 

# Description

* `codon_optimization.py` : Main code involving 1. Constructing the objective function and constraints 2. Solving the constrained quadratic model 3. Conversion between Seqence and Qubit Vector 

* `codon_opt.ipynb` & `codon_LeaphybridCQMsolver.ipynb` : Describing functions and classes in `codon_optimization.py`

* `codon_hamiltonian_graph.ipynb` : Drawing graphes of the obejctive function for each amino acid sequence, SARS-Cov2 and insulin

* `results_codon_optimization.ipynb` : Executing `` to solve codon optimization problems using DWave LeapCQMHybridSolver.ipynb

* `codon_table`: Codon Usage Table and Codon Pair Usage Tables obtained from COCOPUTs [[Link]](https://dnahive.fda.gov/dna.cgi?cmd=cuts_main)
