# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:23:10 2022

@author: nmmar
"""

"Complete SIMULATE Core Rearrangement"
"Use aspects from 'Test Quarter Symmetry Swap' and 'SIMULATE Quarter Core'."

import numpy as np

import time

import sys
import six
sys.modules['sklearn.externals.six'] = six
import os
import mlrose

import matplotlib.pyplot as plt


global core_run_list, k_run_list, PF_run_list

core_run_list=[] # empty of list of new core arrangements
k_run_list=[]
PF_run_list=[]

storage_list = []

# Start with finding the parameters needed for initializing the optimization problem.
"Step 1: Read the SIMULATE input file and extract the main information."

filename="pwr.uo2.c01-4"
#filename="optimization-3"

file = "s3."+filename+".inp"

lis=[]  # list of all lines of file
map=[]  # array of core map
c=0     # condition for fuel core (off)

ID_line=0 # starting line for identification
SYM = 0 # Default symmetry pattern if undefined.
with open(file,'r+') as f: # file==sys.argv[1]

	for l in f:
		ID_line=ID_line+1 # start with line 1
		lis.append(l)
        
        # identify fuel core size
		if l.find("'DIM.PWR'")>=0:
			DIM_line=l.split()
			
			if DIM_line[1][-1]=='/':
				L=int(DIM_line[1][:-1]) # first number (removes '/' symbol)
			else:
				L=int(DIM_line[1]) # first number after dimension
		
		# identify form of symmetry
		if l.find("'COR.SYM'")>=0:
			Sym_line=l.split()
			# Boolean system: 0 - mirror, 1 - rotation
			if Sym_line[1].find('ROT')>=0:
				SYM = 1
			else:
				SYM = 0
		
		# locate the fuel map, denoted by 'FUE.LAB' or 'FUE.SER'
		if l.find("'FUE.LAB'")>=0 or l.find("'FUE.SER'")>=0:
			LAB_line=l.split()
			
			if LAB_line[1][-1]=='/':
				s=int(LAB_line[1][:-1])
			else:
				s=int(LAB_line[1])
			
			c=1 # condition met (on)
			Start_line=ID_line 
			
			continue
		
		elif c==1:
			try:
				d=int(l.split()[0]) # if first part is an integer
				
				if d==0: # end of map denoted by ' 0 0 '
					c=0
					End_line=ID_line-1
				else:    
					map.append(l)
									
			except ValueError or l=='\n': # occurs when int(alphanumeric character) or blank line
				c=0
				End_line=ID_line-1 # confirmed end is previous line
				continue
		else:
			c=0
			continue

f.close()


# Refine map so it skips the row number & starting assembly ID
def map_refine(map):
    map_refined=[] # blank map
    order_refined=[] # blank array for labels
    
    if len(map)<10: # only 1 character for all row IDs
        c_1=1
    else:   # 2 characters for all row IDs (maximum size of PWR is 17 rows)
        c_1=2
    # Starting assembly ID begins 2 spaces after row #
    # Must always starts at assembly space #1 per row
    # Ends with one space ' ', and then the core begins
    c_2=4   # 4 characters occupy '  1 ' between assemblies and row IDs
    
    for i in range(len(map)):
        order_refined.append(map[i][0:c_1+c_2])
        
        map_refined.append(map[i][c_1+c_2:])
    
    
    return order_refined, map_refined

order,map_core=map_refine(map)
# Text in 'order' will precede PWR core layout 'map_core' in SIMULATE input file.


# Divide map into character segments based on the max characters (reverse engineer)
def map_divide(map,D,s):
    
    d=s+1       # dividing length (serial number/label + one space)
    L=D*d       # max number of characters
    map_matrix=[]
    
    for i in range(D):
        c=0 # character counter
        map_line=[]
        
        while c<L:
            e=c+d # end character index
            
            if c >= len(map[i]): # artificially add spaces for lines not at the maximum character length
                add=' '*d # add instead blank spaces equal to the dividing length
                map_line.append(add)
                
            else:
                add=map[i][c:e]
                map_line.append(add)                            
            c=e # new iteration in while loop
            
        map_matrix.append(map_line) # outside while, append list into another list
    
    return map_matrix

# Original core defined by assembly space (L-by-L)
core_original=map_divide(map_core,L,s)

"Step 2: Identify the origin of the core map."
"Step 3: Distinguish the quarter core assemblies (ignore cross assemblies)."

O=int(L/2)  # origin location (denoted as [O,O])

# Apply symmetry of core of size L via mirroring
def mirror(F,O,L):
    # Use if dimension of core is odd
    if L%2 != 0:
        # Step 1: Horizontal mirror of lower-right quadrant into lower-left quadrant
        for i in range(0,O+1):
            for j in range(0,O+1):
                F[O+i][O-j]=F[O+i][O+j]
        
        # Step 2: Vertical mirror of lower quadrants into upper quadrants
        for i in range(0,O+1):
            for j in range(0,L):
                F[O-i][j]=F[O+i][j]
    
    # Use if dimension of core is even
    if L%2 == 0:
        # Step 1: Horizontal mirror of lower-right quadrant into lower-left quadrant
        for i in range(0,O):
            for j in range(0,O):
                F[O+i][O-j-1]=F[O+i][O+j]
        
        # Step 2: Vertical mirror of lower quadrants into upper quadrants
        for i in range(0,O):
            for j in range(0,L):
                F[O-i-1][j]=F[O+i][j]
    
    return F

# Break up the core into all four quadrants
# 1 - top left     2 - top right     3 - bottom left     4 - bottom right
def Quad_Definition(core,L,O,Cross):
    Q1=[]
    Q2=[]
    Q3=[]
    Q4=[]
    
    # Method works for cores with even dimensions
    for i in range(L):
        l=[]
        n=len(core[i])
        for j in range(L):
            l.append(core[i][j])
            
            # Use if even dimension
            if n%2 == 0:
                # Line is completed when length is sufficient
                if len(l)==int(n/2):
                # Identify quadrant by relativity to origin O
                # Uses last element index as reference
                    if i<O and j<O: # Q1 - top left
                        Q1.append(l)
                        l=[]
                    
                    if i<O and j>=O: # Q2 - top right
                        Q2.append(l)
                        l=[]
                    
                    if i>=O and j<O: # Q3 - bottom left
                        Q3.append(l)
                        l=[]
                        
                    if i>=O and j>=O: # Q4 - bottom right
                        Q4.append(l)
                        l=[]
            
            # Use if odd dimension
            if n%2 != 0:
                
                if Cross != 0: # include "cross" assemblies
                    if len(l)==round(n/2)+1:
                        
                        if i<=O and j<=O: # Q1 - top left
                            Q1.append(l)
                            if i==O:
                                Q3.append(l)                            
                            l=[l[-1]]
                            
                        if i<=O and j>O: # Q2 - top right
                            Q2.append(l)
                            if i==O:
                                Q4.append(l)
                            l=[]
                            
                        if i>O and j<=O: # Q3 - bottom left
                            Q3.append(l)
                            l=[l[-1]]
                            
                        if i>O and j>O: # Q4 - bottom right
                            Q4.append(l)
                            l=[]
               
            # Use if odd dimension (exludes "cross" assemblies
                else:
                    if i==O or j==O:
                        l=[]
                        continue
                
                    else:                
                        if len(l)==int(n/2):
                            
                            if i<O and j<O: # Q1 - top left
                                Q1.append(l)
                                l=[]
                            
                            if i<O and j>O: # Q2 - top right
                                Q2.append(l)
                                l=[]
                                
                            if i>O and j<O: # Q3 - bottom left
                                Q3.append(l)
                                l=[]
                                
                            if i>O and j>O: # Q4 - bottom right
                                Q4.append(l)
                                l=[]
                        
                    
    return Q1, Q2, Q3, Q4
# If Cross=0, then the cross assemblies are ignored.  
Cross=0
Q1,Q2,Q3,Q4 = Quad_Definition(core_original,L,O,Cross)
# Blank spaces of the core are included; conversion for position ID will be needed.


"Step 4: Assign position ID order and apply to each quarter based on symmetry."
# e.g. Q4 = [[0,1],[2,3]]

# Only overwrite the serial labels, not the blank spaces.
pos_Q_ID=[] # Copy of lower-right quarter Q4

for i in range(0,len(Q4)):
    line=[]
    for j in range(0,len(Q4[i])):
        line.append(Q4[i][j])
    
    pos_Q_ID.append(line)

pos_ID=0
for i in range(0,len(Q4[i])):
    for j in range(0,len(Q4[i])):
        # Blank spaces are identified by having their first character be a space
        if pos_Q_ID[i][j][0] != ' ':
            pos_Q_ID[i][j]=pos_ID
            pos_ID+=1

n_ID = pos_ID # number of actual assemblies in the quarter (parameter)



# Main Function to Optimize
def Core_Rearrange(new_order):
    
	"Step 4: Assign position ID order and apply to each quarter based on symmetry."

	# Create an assembly position map
	# Step 1: Make a duplicate map
	core_pos_map=[]

	for i in range(0,L):
		line=[]
		for j in range(0,L):
			line.append(core_original[i][j])
		
		core_pos_map.append(line)

	# Step 2: Insert position ID of lower-right quadrant
	# Based on lower quadrant relationship to main core

	for i in range(len(Q4)):
		for j in range(len(Q4[i])):
			
			if Cross == 0 and L%2 !=0: # no cross, and odd dimensions
				core_pos_map[O+i+1][O+j+1]=pos_Q_ID[i][j]
			else:
				core_pos_map[O+i][O+j]=pos_Q_ID[i][j]

	# Step 3: Apply mirroring effect on position map 
	mirror(core_pos_map,O,L)


	# Use Quad_Definition with core_pos_map
	Qp1,Qp2,Qp3,Qp4 = Quad_Definition(core_pos_map,L,O,Cross)

	"Step 4.5: Transform mirror quarters to rotational quarters."
	def rotation(A,O,opt):
		B=[]
		for i in range(len(A)):
			l=[]
			for j in range(len(A[i])):
				l.append(A[i][j])
			B.append(l)
		
		if opt==3:
			for i in range(O):
				for j in range(0,int((O-i)/2)):
					B[j][i+j],B[-1-(i+j)][-1-j] = B[-1-(i+j)][-1-j],B[j][i+j]
			
			for i in range(1,O):
				for j in range(0,int((O-i)/2)):
					B[i+j][j],B[-1-j][-1-(i+j)] = B[-1-j][-1-(i+j)],B[i+j][j]
			
		if opt==2:
			for i in range(O):
				for j in range(0,int((O-i)/2)):
					B[i+j][j],B[-1-j][-1-(i+j)] = B[-1-j][-1-(i+j)],B[i+j][j]
			
			for i in range(1,O):
				for j in range(0,int((O-i)/2)):
					B[j][i+j],B[-1-(i+j)][-1-j] = B[-1-(i+j)][-1-j],B[j][i+j]
					
		return B

	# Q1 and Q4 are identical between rotation and mirror symmetries.
	Qp2_r=rotation(Qp2,O,2)
	Qp3_r=rotation(Qp3,O,3)


	"Step 5: Given new position ID order, apply to each quarter based on symmetry."
	# Given a new position order as an array, apply to all quadrants
	# New position order refers to Q4.

	# Step 1: Make duplicate map
	new_pos_map=[]

	for i in range(0,L):
		line=[]
		for j in range(0,L):
			line.append(core_original[i][j])
		
		new_pos_map.append(line)

	# Step 2: Insert position ID of lower-right quadrant
	# Based on lower quadrant relationship to main core
	k=0

	for i in range(len(Q4)):
		for j in range(len(Q4[i])):
			
			if Cross == 0 and L%2 !=0: # no cross, and odd dimensions
				if Q4[i][j][0] != ' ':
				   new_pos_map[O+i+1][O+j+1]=new_order[k]
				   k+=1
			else:
				if Q4[i][j][0] != ' ':
					new_pos_map[O+i][O+j]=new_order[k]
					k+=1

	# Step 3: Apply mirroring effect on position map
	mirror(new_pos_map,O,L)

	# Step 4: Break up into quadrants
	Qn1,Qn2,Qn3,Qn4 = Quad_Definition(new_pos_map,L,O,Cross)


	"Step 5.5: Transform mirror quarters to rotational quarters."
	Qn2_r=rotation(Qn2,O,2)
	Qn3_r=rotation(Qn3,O,3)


	"Step 6: Rearrange all assemblies in each quarter based on old and new positions."

	# Rearrange the assemblies of the quadrants to match the new order
	def reorder(Q,Qp,Qn):
		# Step 1: Break up matrices into 1D arrays
		Q_a=[]
		Qp_a=[]
		Qn_a=[]
		for i in range(len(Q)):
			Q_a+=Q[i]
			Qp_a+=Qp[i]
			Qn_a+=Qn[i]
			
		# Step 2: Change the order of the quadrant Q_a based on new order Qn_a
		Q_new_a=[]
		c=0
		
		while len(Q_new_a) < len(Q_a):
			if Q_a[c][0] == ' ':
				Q_new_a.append(Q_a[c])
			else:
				a=Qn_a[c]
			
				for i in range(len(Qp_a)):
					if Qp_a[i]==a:
						Q_new_a.append(Q_a[i])
						
						break # exits the for loop, but not the while loop
			c+=1
		
		# Step 3: Reconstruct in the shape of Q
		Q_new=[]
		l=[]
		j=0
		for i in range(len(Q_new_a)):
			l.append(Q_new_a[i])
			
			if len(l)==len(Q[j]):
				Q_new.append(l)
				j+=1
				l=[]
		
		
		return Q_new

	# Mirror Symmetry
	Q_new_1=reorder(Q1,Qp1,Qn1)
	Q_new_2=reorder(Q2,Qp2,Qn2)
	Q_new_3=reorder(Q3,Qp3,Qn3)
	Q_new_4=reorder(Q4,Qp4,Qn4)

	# Rotation Symmetry
	Q_new_2_r=reorder(Q2,Qp2_r,Qn2_r)
	Q_new_3_r=reorder(Q3,Qp3_r,Qn3_r)


	"Step 7: Reconstruct the core with the new quarters."
	# Using the new quadrants, create the newly arranged core.
	def core_construct(core,Q1,Q2,Q3,Q4,L,O,Cross):
		# Duplicate the original core
		core_new=[]
		for i in range(len(core)):
			l=[]
			for j in range(len(core[i])):
				l.append(core[i][j])
			
			core_new.append(l)
		
		# Apply new assembly order via quadrants to original core
		for i in range(len(core)):
			for j in range(len(core[i])):
				# Even dimensions
				if L%2 == 0:
					
					if i<O and j<O:
						core_new[i][j]=Q1[i][j]
					if i<O and j>=O:
						core_new[i][j]=Q2[i][j-O]
					if i>=O and j<O:
						core_new[i][j]=Q3[i-O][j]
					if i>=O and j>=O:
						core_new[i][j]=Q4[i-O][j-O]
				
				# Odd dimensions
				if L%2 != 0:
					
					if Cross == 0:
						if i<O and j<O:
							core_new[i][j]=Q1[i][j]
						if i<O and j>O:
							core_new[i][j]=Q2[i][j-O-1]
						if i>O and j<O:
							core_new[i][j]=Q3[i-O-1][j]
						if i>O and j>O:
							core_new[i][j]=Q4[i-O-1][j-O-1]
					
					if Cross != 0:
						if i<O and j<O:
							core_new[i][j]=Q1[i][j]
						if i<O and j>=O:
							core_new[i][j]=Q2[i][j-O]
						if i>=O and j<O:
							core_new[i][j]=Q3[i-O][j]
						if i>=O and j>=O:
							core_new[i][j]=Q4[i-O][j-O]
		
		
		return core_new
	# Mirror Symmetry
	core_new_mir=core_construct(core_original,Q_new_1,Q_new_2,Q_new_3,Q_new_4,L,O,Cross)

	# Rotation Symmetry
	core_new_rot=core_construct(core_original,Q_new_1,Q_new_2_r,Q_new_3_r,Q_new_4,L,O,Cross)


	if SYM==1:
		core_new=core_new_rot
	else:
		core_new=core_new_mir

	"Step 8: Write the new core into a duplicate input file."
	def core_string(new_map,order):
		# 12/28/21  all spaces are included with the order and new_map terms
		# Only missing the endline '\n' for all terms
			
		string_core=[]
		
			
		for i in range(len(new_map)):
			string_line=order[i] # empty array to store full lines
			
			for j in range(len(new_map[i])):
				string_line+=new_map[i][j]
			
			# Inspect the written line
			for k in range(len(string_line)):
				if k != len(string_line)-1 and string_line[k]=='\n':
					string=list(string_line) # break up string into list of characters
					string[k]=' ' # replace new line with blank space
					string_line="".join(string) # reform into single string
					
				if k == len(string_line)-1 and string_line[k]!='\n':
					string=list(string_line) # break up string into list of characters
					string[k]='\n' # insure last character starts new line
					string_line="".join(string) # reform into single string

			string_core.append(string_line) # append new string line
			
	   
		return string_core

	# Step 3: Create duplicate file to store changes to reactor core

	my_file = open(file)
	string_list = my_file.readlines() # difference between read() and readlines()
	my_file.close()
	#print(string_list)

	# Conversion of new core map from 'list of list of str' to 'list of str'.
	string_core=core_string(core_new,order)

	# Changing old core map to new core map
	for i in range(Start_line,Start_line+len(core_new)): # first index of string_list is 0
		string_list[i]=string_core[i-Start_line]
		
	copyfile="s3."+filename+"-symm"+".inp"

	my_file = open(copyfile, "w")
	new_file_contents = "".join(string_list)

	my_file.write(new_file_contents)
	my_file.close()


	"Step 9: With the new SIMULATE copyfile written, run it into SIMULATE-3."
	# Make sure the command "module load studsvik" is ran prior to calling this file.

	os.system("simulate3 -k "+copyfile)


	"Step 10: Remove the copyfile from the directory."
	# Ignore for now (3/18/2022) 

	# os.system("rm "+copyfile)

	"Step 11: Extract output data of interest for the fitness function."

	outfile="s3."+filename+"-symm.out"

	sys.argv.append(outfile)


	lis=[] # stores all output lines
	k=[]   # stores k-effective lines per depletion step
	#RPF=[] # stores max reative power fraction per depletion step
	PF=[] # stores max node-average peaking factor per depletion step


	A=1
	with open(outfile,'r+') as f: # file==sys.argv[1]

		for l in f:
			lis.append(l)
			
			if l.find('K-effective  =')>=0:
				k.append(l.split())
				continue
			elif l.find('Node-Averaged')>=0:
				PF.append(l.split())
				continue
			elif l.find('ABORT')>=0:
				A=0
			else:
				continue
				

	f.close()

	EOC_k=float(k[-1][-1]) # EOC k-effective is last object of last line read

	NPF=[]
	for i in range(len(PF)):
		NPF.append(float(PF[i][5])) # convert string numbers to float numbers
		
	max_NPF=max(NPF) # max node peaking factor using max function on array

	# Writing a weighted function to calculate for the fitness
	y=C1*EOC_k + C2/max_NPF # GOAL: Maximize

	# If the output file returns aborted or with a Nan value, ignore it by setting the fitness to zero.
	if A==0 or np.isnan(EOC_k)==True:
		y=0; A=0

	print()
	print('EOC k: ',EOC_k)
	print('max node peaking factor: ',max_NPF)
	print('Function value: ',y)
	print()


	"Step 12: Delete the output file to open space for the next."

	#os.system('rm '+outfile)

	core_run_list.append(string_core) # save the string core of the new arrangement after each run
	k_run_list.append(EOC_k)
	PF_run_list.append(max_NPF)
	
	storage_list.append([A,new_order,EOC_k,max_NPF,y,string_core])
	
	return y,EOC_k,max_NPF,A # fitness function value, boolean for successful run

# Implement a tally bin that determines whether a new arrangement was successful or not.
tally = 0 # while-loop parameter
N=1*10**2 # for-loop parameter

# Original, unchanged order.
orig_order = np.zeros(n_ID)
for i in range(n_ID):
	orig_order[i] = i
	
# Coefficient Values
C1 = 1; C2 = 1

time_start=time.perf_counter()

np.random.seed(30)

runs = []
y_run = []
"""
for i in range(N):
	new_order = np.random.permutation(n_ID)
	
	runs.append(new_order)
	
	y,k,A = Core_Rearrange(new_order)

	k_run.append(k)
	y_run.append(y)
	tally+=A

print('Runs: ')
for i in range(len(runs)):
	print(runs[i])
	print(k_run[i])
	print(y_run[i])
	print()
"""

while tally < N:
	new_order = np.random.permutation(n_ID)

	runs.append(new_order)

	y,k,PF,A = Core_Rearrange(new_order)

	y_run.append(y)
	tally+=A

print()
print('Rate of Success: ',tally,' of ',len(y_run),' runs')

time_end=time.perf_counter()
print('Time to complete: ', time_end - time_start,'s')
print()

print()
"""
for i in range(len(core_run_list[-1])):
	print(core_run_list[-1][i])
"""
#print(core_run_list[-1])
#print()
#print(storage_list[-1][5])
#print()
#print(k_run_list)
#print()
#print(PF_run_list)
#print()
#print(storage_list)
print()


# Save all the run data from storage_list into a separate text file.

# Sort the iterations in two steps.
# Step 1: Distinguish the good (A=1) runs from the bad (A=0) runs.

data_good=[]
data_bad =[]

for i in range(len(storage_list)):
    
	if storage_list[i][0] == 0:
		data_bad.append(storage_list[i])
        
	elif storage_list[i][0] == 1:
		data_good.append(storage_list[i])
        

# Step 2: For the good data only, order list from highest fitness to lowest fitness.

def List_Ordering(data,id_data):
    # data: list of numerical data
    # id_data: index of the value in each iteration that is argued for sorting.
    
	data_copy = []

	for i in range(len(data)):
		data_copy.append(data[i])
    
	list_ordered=[]

	for i in range(len(data_copy)):
		e = 0 # default index
		
		for j in range(1,len(data_copy)):
			if data_copy[j][id_data] > data_copy[e][id_data]:
				e = j
		
		list_ordered.append(data_copy[e])
		del data_copy[e]
    
    
	return list_ordered

data_good_sort = List_Ordering(data_good,4)

data_good = data_good_sort
del data_good_sort

# For the multiobjective equation in the optimizer fitness function,
# take the average of the good data runs' output variables.

# y = C1*k + C2/NPF

# Start by saving good data run variables into separate lists.
k_good = []
PF_good =[]
for i in range(len(data_good)):
	k_good.append(data_good[i][2])
	PF_good.append(data_good[i][3])

k_avg = np.average(k_good)
PF_avg = np.average(PF_good)

C1 = 1/k_avg
C2 = PF_avg

print('k-average:   ',k_avg)
print('NPF-average: ',PF_avg)
print()
print('Coefficient 1: ',C1)
print('Coefficient 2: ',C2)
print()

# Create a string object to store the iteration data, separating good from bad data.

np.set_printoptions(linewidth=np.inf) # prevents text wrapping of NumPy arrays (occurs when NumPy array characters > 75)

def Data_Writer(good_data,bad_data):
    
	text=[]

	# Initialize with title line
	text.append('Results from mlrose Optimization of SIMULATE-3 Core Arrangement \n\n')

	# Subtitle for good data
	text.append('Good Data Runs: \n')
    
    # Append each line to the text
	for i in range(len(good_data)):
		text.append('Input Arrangement: '+str(good_data[i][1])+'\n')
		text.append('EOC k-effective: '+str(round(good_data[i][2],9))+'    ')
		text.append('Max Nodal Peaking Factor: '+str(round(good_data[i][3],8))+'    ')
		text.append('Fitness Value: '+str(good_data[i][4])+'\n')
		for j in range(len(good_data[i][5])):
			text.append(good_data[i][5][j])
		text.append('\n')
        
		if i==9: # Best 10 runs saved
			text.append('\n')
        
	text.append('\n')

	# Subtitle for bad data
	text.append('Bad Data Runs: \n')

	# Append each line to the text
	for i in range(len(bad_data)):
		text.append('Input Arrangement: '+str(bad_data[i][1])+'\n')
		text.append('EOC k-effective: '+str(round(bad_data[i][2],9))+'    ')
		text.append('Max Nodal Peaking Factor: '+str(round(bad_data[i][3],8))+'    ')
		text.append('Fitness Value: '+str(bad_data[i][4])+'\n')
		text.append('\n')

	text.append('\n')
    
	return text

true_text = Data_Writer(data_good,data_bad)

file = open("mlrose_results.txt", "w+")

new_file_contents = "".join(true_text)

file.write(new_file_contents)
file.close()


# Plot the results.
f1 = plt.figure(1)
plt.plot(range(1,len(y_run)+1),y_run,'g.-',lw=0.5,label='Optimizer Curve')
plt.xlabel('ID - Iteration #')
plt.ylabel('Fitness')
#plt.ylim([0.9,1.5])
f1.show()

f2 = plt.figure(2)
plt.plot(range(1,len(k_run_list)+1),k_run_list,'r.-',lw=0.5,label='k-effective (Manual Tracking)')
plt.plot([1,len(k_run_list)],[k_avg,k_avg],'k--',lw=0.3,label='Average')
plt.xlabel('ID - Iteration #')
plt.ylabel('End-of-Cycle k-effective')
#plt.ylim([0.9,1.5])
f2.show()

f3 = plt.figure(3)
plt.plot(range(1,len(PF_run_list)+1),PF_run_list,'b.-',lw=0.5,label='Power Peaking Factor (Manual Tracking)')
plt.plot([1,len(PF_run_list)],[PF_avg,PF_avg],'k--',lw=0.3,label='Average')
plt.xlabel('ID - Iteration #')
plt.ylabel('Maximum Nodal Peaking Factor')
#plt.ylim([0.9,1.5])
f3.show()

plt.show()
