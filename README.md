# Reactor-Fuel-Shuffling-Optimizer
Introductory Python codes utilizing modified version of Machine Learning Randomized Optimization and Search (mlrose) to redefine fuel patterns in CASMO-4E and SIMULATE-3 input files.

Codes used are broken into 3 categories:
1. The modified mlrose library package used for the main files.
2. CASMO-4E optimization file.
3. SIMULATE-3 optimization file.

## MLROSE-Package-Modifications
To overcome a limitation between the then-current version of mlrose and the project objective, a new class in "opt_probs.py" was created called *MarksOpt*.
*MarksOpt* is a combinatorial optimization problem derived from *DiscreteOpt* with the function **random** now creating a permuted array , and **random_neighbor** instead swapping two element values in the original state.
These changes were inspired by the class *TSPOpt* that as it uses permuted arrays as states to change the order of the visited locations.

## CASMO-Optimization-File
CASMO optimization uses a provided input file with formatting akin to the examples given in this package. It reads the input file to discern its size and available assortment of fuels differentiated by ID numbers. Based on the number of fuel pins identified, the code uses 1D arrays that correspond to each pin, their IDs used to represent the fuel used for each pin. The optimization function builds a new layout of fuel pins (LFU) based on the inputted array by the optimization algorithm. The new LFU gets written into a duplicate input file that is called to be read by the local CASMO program to generate an output file. The data from the output file is placed into the empirical function to determine design quality.
The most current model is only capable of optimizing input files whose LFUs are shaped as octants of the fuel assembly.

## SIMULATE-Optimization-File
SIMULATE optimization uses a provided input file with formatting aking to the examples given in this package. It reads the input file to discern the dimension of the reactor and the assembly layout in the core, memorizing which serial numbers are where in the core space. The reactor core map is then broken up into equally sized sets of assemblies to represent the four quadrant regions of the core; depending on the dimension size (the number of assemblies rows and max number of assemblies per row), division of the core space may be in either four large quadrants encompassing all assemblies except for the centermost assembly or four small quadrants and four "cross arms" encompassing all assemblies except for the the centermost assembly. For each set of spatial regions, the optimization function shuffles the layout of the assemblies and reconstructed into the original core layout shape to create new, quarter-core symmetrical core loading patterns. Each new core layout gets written into a duplicate input file that is called to be ready the local SIMULATE program to generate an output file. The data from the output file is placed into the empirical function to determine design quality.
The most current model is only capable of solving quarter-core symmetry problems and has only been tested on initial cycle files (CYC-1).
