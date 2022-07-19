# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 12:02:23 2021

@author: nmmar
"""

import numpy as np

from datetime import datetime

import sys
import six
sys.modules['sklearn.externals.six'] = six
import os
import mlrose

import matplotlib.pyplot as plt



# Inventory Counting Function
def Inventory(core): # Input - core map
    core_num=[]
    x = 0
    y = 0
    line=[]
    
    for i in range(0,len(core)):
        for j in range(0,len(core[i])):
            line.append(int(core[i][j]))
        
        core_num.append(line)
        line=[]
        
    # Count the integers
    Inv=[]
    for k in range(min(min(core_num)),max(max(core_num))+1):
        c = 0
        
        for i in range(0,len(core_num)):
            c+=core_num[i].count(k)
        
        Inv.append(c)

    return(Inv) # Output - array of inventory by core element index

# Input Matching Function
def match(x,A):
    y=0
    for i in range(0,len(x)):
        if int(x[i]) in A[i]: # if value x[i] is inside acceptable list A[i]
            continue
        else:
            y+=1 # Counter of unacceptable cases
    
    return y # value of unmatched

# Call initial input file to get parameters
file="pwrugen.inp"

lis=[]
map=[]
c=0
with open(file,'r+') as f: # file==sys.argv[1]

	for l in f:
		lis.append(l)
		if l.find('LPI')>=0 and l.find('INPUT')<0:
			c=1
			#lis.append(l)
			continue
		elif c==1:
			try:
				d=int(l.split()[0])
				map.append(l.split())                
			except ValueError:
				c=0
				continue
		else:
			continue

f.close()

core=map[:]
val=[]
idx=[]
n=0
for i in range(0,len(core)):
	for j in range(0,len(core[i])):
		val.append(int(core[i][j]))
		idx.append(n)
		n+=1

mat_max=max(val) # maximum material ID

# Acceptable list
Accept=[]
Standard=[] # no standard entry (all acceptable values based on initial core)

for i in range(1,mat_max+1):
	Standard.append(i) # all values from 1 to max in map

# No constraint; all materials accepted
for i in range(len(val)):
	Accept.append(Standard)
"""	

for i in range(len(val)):
#	if i%4==0 or i==len(val)-1:
	if i in range(0,5) or i in range(10,len(val)):
		Accept.append([mat_max]) # special condition (must be in brackets)
	else:
		Accept.append(Standard)
"""

# Empty lists to store function output values
f_1=[] # k-EOC (successful)
f_2=[] # max PPR (successful)
f_F1=[]  # weight 1
f_F2=[]  # weight 2
f_tot=[] # function
n_runs=[]# counter of runs (alternative due to Python function format)


# Optimized Function
def core_swap(Input): # Input - core-size integer array
	n_runs.append(1)
	# Step 1: Read input file and extract core map information.
	file="pwrugen.inp"

	LPI=[]
	map=[]
	c=0 # conditional variable
	ID_line=0 # line ID
	with open(file,'r+') as f: # file==sys.argv[1]

		for l in f:
			ID_line=ID_line+1
			LPI.append(l)
			if l.find('LPI')>=0 and l.find('INPUT')<0:
				c=1 # acknowledges next line is a lattice line
				Start_line=ID_line
				continue
			elif c==1:
				try:
					d=int(l.split()[0])
					map.append(l.split())                
				except ValueError:
					c=0 # stops saving lines to the lattice
					continue
			else:
				continue

	f.close()

	# Step 2: Change two elements in the core or rearrange core.
	core=map[:]
	#print(core)
	#print()
	
	# Count the inventory of the initial core map
	Inv_init=Inventory(core)
	
	val=[] # value at each position
	idx=[] # index of position
	n=0
	for i in range(0,len(core)):
		for j in range(0,len(core[i])):
			val.append(core[i][j])
			idx.append(n)
			n+=1

	# Insert new core elements			
	core_swap=[]
	for i in range(0,len(Input)):
		core_swap.append(str(Input[i]+1)) #0+1=1  2+1=3

	for i in range(0,len(core_swap)):
		val[i]=core_swap[i]
	
	
	# Reverse engineer "core"
	x = 0
	y = 0
	new_map=[]
	line=[]
	for k in range(0,len(val)):
		line.append(val[k])
		
		y+=1
		if y>x:
			y=0
			x+=1
			new_map.append(line)
			line=[]
	
	# Count inventory in the new core
	Inv_new=Inventory(new_map)

	if match(val,Accept)!=0: # if core input does not have values acceptable in all locations
		k=0; pp_avg=0			 # set "kill" values to function parameters
		sol = 0	
	else:					# if cores match inventory amounts
		# Step 3: Create a copy of the input file, inserting the new core map.
		def core_string(map):
			spacing="  "   # two spaces in between integers
			next_line="\n" # next line indicator
			
			string_core=[]
			line=str('')
			for l in map:
				
				for i in range(len(l)):
					line+=l[i]
					
					if i==len(l)-1:
						line+=next_line
					else:
						line+=spacing
						
				string_core.append(line)
				line=str('')
				
			return(string_core)
					   
		my_file = open("pwrugen.inp")
		string_list = my_file.readlines() # difference between read() and readlines()
		my_file.close()
		#print(string_list)

		# Conversion of new core map from 'list of list of str' to 'list of str'.
		string_core=core_string(new_map)

		# Changing old core map to new core map (based on original .inp format)
		for i in range(Start_line,Start_line+len(string_core)):
			string_list[i]=string_core[i-Start_line]
			

		my_file = open("pwrugen_2.inp", "w")
		new_file_contents = "".join(string_list)

		my_file.write(new_file_contents)
		my_file.close()
		
		# Step 4: Run CASMO with the new input file.
		os.system("casmo4 pwrugen_2.inp")
		#print()

		# Step 5: Delete the old input file (and unnecessary output files).
		#os.system("rm pwrugen_2.inp")
		os.system("rm pwrugen_2.cax")
		os.system("rm pwrugen_2.log")

		# Step 6: Read the output file and extract the optimized variable.
		file="pwrugen_2.out"	
		
		lis=[] #k-value data
		lip=[] #normalized power peak data
		c=0
		with open(file,'r+') as f: # file==sys.argv[1]
			
			for l in f:
				
				if l.find('BURNUP =')>=0 and l.find('K-INF =')>=0:
					c=1
					lis.append(l.split())
					continue
				elif l.find('LL =')>=0:
					c=1
					lip.append(l.split())
					
				elif c==1:
					try:
						d=int(l.split()[0])
						lis.append(l.split())
					except ValueError:
						c=0
						continue
				else:
					continue
					
					
		f.close()
					
						
		# print(lis)
		# print()

		#k=float(lis[0][6]) #BOC k
		#kdif=np.abs(k-1)
		k=float(lis[-1][6]) #EOC k
		#kdif=np.abs(k-1)
		
		pp=[]
		for i in range(0,len(lip)):
			pp.append(float(lip[i][6]))
		pp_avg=np.mean(pp) # average power peak value
		pp_max=max(pp)
		
		# Factor both flux and k into a solution
		F1=0.9*k
		F2=1/pp_max
		sol=F1+F2
		
		# Append successful output runs
		f_1.append(k)
		f_2.append(pp_max)
		
		f_F1.append(F1)
		f_F2.append(F2)
		
		f_tot.append(sol)
		
		# Step 7: Delete the output file.
		os.system("rm "+file)
	
		
	return sol # Output


# Optimizer Setup
# Initialize custom fitness function object
fitness_cust = mlrose.CustomFitness(core_swap)

problem = mlrose.DiscreteOpt(length = len(idx), fitness_fn = fitness_cust, maximize = True, max_val = mat_max) # range of 0-2 for a multi-element array

schedule = mlrose.ArithDecay() # ArithDecay()  ExpDecay() GeomDecay() init_temp, decay, min_temp, exp_const

# Define initial state
init_state = None

start=datetime.now() # start of optimization

# Solve problem using simulated annealing
best_state, best_fitness, history = mlrose.simulated_annealing(problem, schedule = schedule,
															   max_attempts = 5, max_iters = 5,
															   init_state = init_state, curve=True, random_state = 1)
# Solve problem using genetic algorithm
#best_state, best_fitness = mlrose.genetic_alg(problem, pop_size=50, mutation_prob=0.2, 
#                                               max_attempts=100, max_iters=100, curve=False, 
#                                               random_state=1)

end=datetime.now() # end of optimization

# Step 8: Save the best state and apply it to get the best solution.

def sol_swap(Input):
	# Read input file
	file="pwrugen.inp"

	LPI=[]
	map=[]
	c=0
	ID_line=0
	with open(file,'r+') as f: # file==sys.argv[1]

		for l in f:
			ID_line=ID_line+1
			LPI.append(l)
			if l.find('LPI')>=0 and l.find('INPUT')<0:
				c=1
				Start_line=ID_line
				continue
			elif c==1:
				try:
					d=int(l.split()[0])
					map.append(l.split())                
				except ValueError:
					c=0
					continue
			else:
				continue

	f.close()

	print()

	# Make the best alteration to the input deck
	core=map[:]
	#print(core)
	print()

	val=[]
	idx=[]
	n=0
	for i in range(0,len(core)):
		for j in range(0,len(core[i])):
			val.append(core[i][j])
			idx.append(n)
			n+=1

	# Insert new core elements			
	core_swap=[]
	for i in range(0,len(Input)):
		core_swap.append(str(Input[i])) #1-3

	for i in range(0,len(core_swap)):
		val[i]=core_swap[i]
	
	
	# Reverse engineer "core"
	x = 0
	y = 0
	new_map=[]
	line=[]
	for k in range(0,len(val)):
		line.append(val[k])
		
		y+=1
		if y>x:
			y=0
			x+=1
			new_map.append(line)
			line=[]
		
	# Create a copy of the input file, inserting the new core map.
	def core_string(map):
		spacing="  "   # two spaces in between integers
		next_line="\n" # next line indicator
		
		string_core=[]
		line=str('')
		for l in map:
			
			for i in range(len(l)):
				line+=l[i]
				
				if i==len(l)-1:
					line+=next_line
				else:
					line+=spacing
					
			string_core.append(line)
			line=str('')
			
		return(string_core)
				   
	my_file = open("pwrugen.inp")
	string_list = my_file.readlines() # difference between read() and readlines()
	my_file.close()
	#print(string_list)

	# Conversion of new core map from 'list of list of str' to 'list of str'.
	string_core=core_string(new_map)

	# Changing old core map to new core map
	for i in range(Start_line,Start_line+len(string_core)):
		string_list[i]=string_core[i-Start_line]
		

	my_file = open("pwrugen_opt.inp", "w")
	new_file_contents = "".join(string_list)

	my_file.write(new_file_contents)
	my_file.close()
	
	return None

print('Time Start: ',start)
print('Time End: ',end)
print('Time Elapse: ',end-start)
print()
print(best_state)
print(best_fitness)
print()
print(history)
print()

# Plug in best state (after conversion) and create optimized input file
state=[]
for i in range(0,len(best_state)):
	state.append(best_state[i]+1)

sol_swap(state)

print('Successful Runs: ',len(f_tot)) # number of successful iterations
print('# of Total Runs: ',len(n_runs)) # number of total iterations
print()

# Step 9: Display the algorithm's behavior based on output values vs. iterations

# Give the ids of the iterations with the best met conditions
print('Largest EOC k-eff at iteration #',f_1.index(max(f_1))+1)
print('Lowest maximum PP Ratio at iteration #',f_2.index(min(f_2))+1)
print('Largest weighted function at iteration #',f_tot.index(max(f_tot))+1)
print()

# Plotting successful iterations only
x_plot=range(1,len(f_tot)+1) # range from 1-total_successful_iters

plt.plot(x_plot,f_1,'r.-',lw=0.5,label='End-of-Cycle k-value')
plt.xlabel('Successful Iteration #')
plt.ylabel('EOC k-inf')
plt.show()

plt.plot(x_plot,f_2,'b.-',lw=0.5,label='Maximum Power Peak Ratio')
plt.xlabel('Successful Iteration #')
plt.ylabel('Maximum PP Ratio')
plt.show()

plt.plot(x_plot,f_F1,'r.-',lw=0.5,label='Factor 1')
plt.plot(x_plot,f_F2,'b.-',lw=0.5,label='Factor 2')
plt.xlabel('Sucessful Iteration #')
plt.ylabel('Factor Value')
plt.legend()
plt.show()

plt.plot(x_plot,f_tot,'k.-',lw=0.5,label='Weighted Function')
plt.xlabel('Successful Iteration #')
plt.ylabel('Output')
#plt.legend()
plt.show()

plt.plot(range(1,len(history)+1),history,'g.-',lw=0.5,label='Optimizer Curve')
plt.xlabel('ID')
plt.ylabel('Fitness')
plt.show()

