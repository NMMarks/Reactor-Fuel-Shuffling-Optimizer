# Reactor-Fuel-Shuffling-Optimizer
Introductory Python codes utilizing modified version of Machine Learning Randomized Optimization and Search (mlrose) to redefine fuel patterns in CASMO-4E and SIMULATE-3 input files.

Codes used are broken into 3 categories:
1. The modified mlrose library package used for the main files.
2. CASMO-4E optimization file.
3. SIMULATE-3 optimization file.

## MLROSE-Package-Modifications
To overcome a limitation between the then-current version of mlrose and the project objective, an new class in "opt_probs.py" was created called *MarksOpt*.
*MarksOpt* is near identical to the class *DiscreteOpt*. New in this version, the function **random** now creates a permuted array , and **random_neighbor** instead swaps two element values in the original state.
These changes were inspired by the class *TSPOpt* that as it uses permuted arrays as states to change the order of the visited locations.

## CASMO-Optimization-File
CASMO optimization uses a provided input file with formatting akin to the examples given in this package. It reads the input file to discern its size and available assortment of fuels differentiated by ID numbers. Based on the number of fuel pins identified, the code uses 1D arrays that correspond to each pin, their IDs used to represent the fuel used for each pin. The optimization function builds a new layout of fuel pins (LFU) based on the inputted array by the optimization algorithm. The new LFU gets written into a duplicate input file that is called to be read by the local CASMO program to generated an output file. The data from the output file is placed into the empirical function to determine design quality.
The most current model is only capable of optimizing input files whose LFUs are shaped as octants of the fuel assembly.

## SIMULATE-Optimization-File
SIMULATE optimization uses a provided input file with formatting aking to the examples given in this package. 
The most current model is only capable of solving quarter-core symmetry problems and has only been tested on initial cycle files (CYC-1).
