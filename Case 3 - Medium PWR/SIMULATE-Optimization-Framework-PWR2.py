# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 21:16:34 2022

@author: nmmar
"""

"""
Complete SIMULATE Core Rearrangement

Initialization - 
	Get information of the initial reactor core model based on selected file name;
	Set limits to enrichment and burnable absorber ranges relative to the initial;
	Automatically run a series of random rearrangements to define fitness equation coefficients.

Optimization - Optimize the core model in sequential order:
	Quadrant Assembly Arrangement
	Cross Assembly Arrangement (if applicable)
	Fuel Assembly Enrichment
	Fuel Assembly Burnable Absorber Concentration (where applicable)

Post-Optimization - 
	Generate text files that arrange the results in order of "optimized";
	Print out all possible plots relative to optimization processes
"""
# Libraries
import numpy as np # mathematical formulas and calculations

import time # tracking time

# Required for mlrose optimization
import sys
import six
sys.modules['sklearn.externals.six'] = six
import os
import mlrose

import matplotlib.pyplot as plt # post-optimization plotting


# (6/19/23) Hard limits to enr and psn
min_e = 1; max_e = 5; r_e = max_e-min_e # enrichment weight perecent minimum and maximum limits
min_p = 0.1; max_p = 5; r_p = max_p-min_p # burnable absorber weight percent minimum and maximum limits
NPF_limit = 5    # maximum desired limit

# Quadrant lists of output results (all reactor models)
core_run_q_list=[] # empty list of new core arrangements
k_run_q_list=[] # empty list of EOC k-eff
PP_run_q_list=[]# empty list of max NPF
F1_run_q_list=[]# empty list of factor 1
F2_run_q_list=[]# empty list of factor 2
y_run_q_list=[] # empty list of quality factor

storage_q_list = [] # empty list of all important data

# Cross lists of output results (only odd-dimension models)
core_run_c_list=[] # empty list of new core arrangements
k_run_c_list=[]    # empty list of EOC k-eff
PP_run_c_list=[]   # empty list of max NPF
F1_run_c_list=[]   # empty list of factor 1
F2_run_c_list=[]   # empty list of factor 2
y_run_c_list=[]    # empty list of quality factor

storage_c_list = [] # empty list of all important data

# 9/11/2022 - Enrichment storage lists (all assemblies unless changed)
seg_run_e_list=[] # empty list of new enrichments
k_run_e_list=[] # empty list of EOC k-eff
PP_run_e_list=[]# empty list of max NPF
F1_run_e_list=[]# empty list of factor 1
F2_run_e_list=[]# empty list of factor 2
y_run_e_list=[] # empty list of quality factor

storage_e_list=[] # empty list of all important data

# 10/16/2022 - Burnable poison concentration storage lists (only certain assemblies unless changed)
seg_run_p_list=[] # empty list of new enrichments
k_run_p_list=[] # empty list of EOC k-eff
PP_run_p_list=[]# empty list of max NPF
F1_run_p_list=[]# empty list of factor 1
F2_run_p_list=[]# empty list of factor 2
y_run_p_list=[] # empty list of quality factor

storage_p_list=[] # empty list of all important data

# Set time intervals to zero
time_opt = 0; time_pre = 0; time_q = 0; time_c = 0; time_e = 0; time_p = 0


# Start with finding the parameters needed for initializing the optimization problem.
"Step 1: Read the SIMULATE input file and extract the main information."

filename="Gen_PWR_Model_2" # naming convention for located original design and creating new designs
file = "s3."+filename+".inp"

lis=[]  # list of all lines of file
map=[]  # array of core map
c=0     # condition for fuel core (off)

"Update 7/26/2022: Optimize for fuel assembly enrichment by extracting data from lines with 'SEG.DAT'."
seg_line_original=[] # list of lines of the original assemblies
seg_original=[]  # list of lines starting with 'SEG.DAT' representing fuel assembly types.
seg_base=[]      # list of lines starting with 'SEG.DAT' (changeable)
e=[]    # list of enrichments
l_seg=[]# list of line IDs correspondant to 'SEG.DAT' lines.

"Update 10/12/2022: Optimize for fuel assembly burnable poison weight percentage by '''."
p=[]    # list of burnable poison weight percentages


# Retrieve all original reactor design characteristics for initialization
ID_line=0 # starting line for identification
SYM = 0 # Default symmetry pattern if undefined.

with open(file,'r+') as f: # file==sys.argv[1]

	for l in f: # read line-by-line
		ID_line=ID_line+1 # starting ID as line 1
		lis.append(l)
		
		# identify fuel core size
		if l.find("'DIM.PWR'")>=0:
			DIM_line=l.split()
			
			if DIM_line[1][-1]=='/' or DIM_line[1][-1]==',': # if non-numerical symbol is present at the end due to formatting
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
		
		# 7/26/22 - read and save fuel assembly type data
		# 10/12/22 - read wt perc. data
		if l.find("'SEG.DAT'")>=0:
			seg_line=l.split()
			
			seg_line_original.append(l)
			seg_original.append(seg_line)
			seg_base.append(seg_line)
			l_seg.append(ID_line)
			
			if seg_line[2][-1]==',': # if comma is attached to number
				e.append(float(seg_line[2][:-1]))
			else:
				e.append(float(seg_line[2]))
			
			if seg_line[3][-1]==',': # if comma is attached to number
				p.append(float(seg_line[3][:-1]))
			else:
				p.append(float(seg_line[3]))
			
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
									
			except ValueError or l=='\n': # occurs when int(alphanumeric variable) or blank line
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
        c_1=2
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
core_original=map_divide(map_core,L,s) # object to change/alter within functions
core_original_copy = map_divide(map_core,L,s) # copy for direct comparison

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
                            l=[] # restart for next line
                            
                        if i>O and j<=O: # Q3 - bottom left
                            Q3.append(l)
                            l=[l[-1]]
                            
                        if i>O and j>O: # Q4 - bottom right
                            Q4.append(l)
                            l=[] # restart for next line
               
            # Use if odd dimension (exludes "cross" assemblies)
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

"All cross assemblies are located such that one index is O."
# Extract cross assemblies from core, maintaining shape of segments.
# 1 - top     2 - left      3 - right     4 - bottom
def Cross_Definition(core,L,O):
    # Use ONLY if L is odd.
    cross_1=[]
    cross_2=[]
    cross_3=[]
    cross_4=[]
    
    # Append 
    for i in range(O): # number of assemblies along each cross arm (excludes center)
        cross_2.append([core[O][i]])
        cross_3.append([core[O][L-O+i]])
        
        cross_1.append([core[i][O]])
        cross_4.append([core[L-O+i][O]])
    
    return cross_1, cross_2, cross_3, cross_4

# Cross assemblies
if L%2 == 1:
    A1,A2,A3,A4 = Cross_Definition(core_original,L,O)


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

n_Q_ID = pos_ID # number of actual assemblies in the quarter (parameter)

if L%2 == 1:
	# Count the number of assemblies per arm of the cross (priority: A3)
	if SYM == 0: # mirror symmetry
		# Set the size to be the length of 2 arms
		n_C_ID = 2*len(A3)
	elif SYM == 1: # rotational symmetry
		# Set the size to be the length of 1 arm
		n_C_ID = len(A3)


# Segments from Optimization Functions (6/13/2022)

# Construct new core after shuffling quadrant assemblies.
def new_core_quadrant(new_order):
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

	return core_new

# Construct new core after shuffling cross assemblies.
def new_core_cross(new_order):
	"Step 4: Assign position ID order and apply to each cross arm based on symmetry."

	# Step 1: Make a duplicate map
	core_pos_map=[]

	for i in range(0,L):
		line=[]
		for j in range(0,L):
			line.append(core_original[i][j])
		
		core_pos_map.append(line)

	# Step 2: Insert position ID to arm(s)

	pos_C_ID=[]

	for i in range(n_C_ID):
		pos_C_ID.append(i)

	if SYM==0: # mirror
		# Position ID to right arm
		for i in range(0,len(A3)):
			core_pos_map[O][O+1+i]=pos_C_ID[i]
		# Position ID to bottom arm
		for i in range(len(A3),2*len(A3)):
			core_pos_map[1+i][O]=pos_C_ID[i]

	elif SYM==1: # rotational
		for i in range(0,len(A3)):
			core_pos_map[O][O+1+i]=pos_C_ID[i] # Position ID to right arm
			core_pos_map[O+1+i][O]=pos_C_ID[i] # Position ID to bottom arm


	# Step 3: Apply mirroring effect on position map
	mirror(core_pos_map,O,L)

	# Use Cross_Definition with core_pos_map
	Ap1,Ap2,Ap3,Ap4 = Cross_Definition(core_pos_map,L,O) # works for both symmetry forms
		

	"Step 5: Given new position ID order, apply to each cross arm."
	#new_order = np.random.permutation(n_C_ID)

	# Step 1: Make duplicate map
	new_pos_map=[]

	for i in range(0,L):
		line=[]
		for j in range(0,L):
			line.append(core_original[i][j])
		
		new_pos_map.append(line)

	# Step 2: Insert position ID of cross arms

	if SYM==0: # mirror
		# Position ID to right arm
		for i in range(0,len(A3)):
			new_pos_map[O][O+1+i]=new_order[i]
		# Position ID to bottom arm
		for i in range(len(A3),2*len(A3)):
			new_pos_map[1+i][O]=new_order[i]

	elif SYM==1: # rotational
		for i in range(0,len(A3)):
			new_pos_map[O][O+1+i]=new_order[i] # Position ID to right arm
			new_pos_map[O+1+i][O]=new_order[i] # Position ID to bottom arm

	# Step 3: Apply mirroring effect on position map
	mirror(new_pos_map,O,L)

	# Step 4: Break up into quadrants
	An1,An2,An3,An4 = Cross_Definition(new_pos_map,L,O)


	"Step 6: Rearrange all assemblies in each quarter based on old and new positions."
	# Rearrange the assemblies of the cross arms to match the new order
	# Rotational symmetry only
	def reorder_cross_rot(Q,Qp,Qn):
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
			if Q_a[c] == ' ':
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
		
		for i in range(len(Q_new_a)):
			Q_new.append(Q_new_a[i])

		return Q_new

	# Mirror Symmetry Only
	def reorder_cross_mir(Q1,Q2,Qp1,Qp2,Qn1,Qn2):
		# Step 1: Break up into 1D arrays
		Q_a=[]
		Qp_a=[]
		Qn_a=[]
		
		for i in range(len(Q1)):
			Q_a+=Q1[i]
			Qp_a+=Qp1[i]
			Qn_a+=Qn1[i]
			
		for i in range(len(Q2)):
			Q_a+=Q2[i]
			Qp_a+=Qp2[i]
			Qn_a+=Qn2[i]
		
		# Step 2: Change order of the cross assemblies Q_a based on new order Qn_a
		# In mirror symmetry, right-bottom and left-top are connections.
		Q_new_a=[]
		c=0    
		
		while len(Q_new_a) < len(Q_a):
			if Q_a[c] == ' ':
				Q_new_a.append(Q_a[c])
			else:
				a=Qn_a[c]
			
				for i in range(len(Qp_a)):
					if Qp_a[i]==a:
						Q_new_a.append(Q_a[i])
						
						break # exits the for loop, but not the while loop
			c+=1    
		
		# Step 3: Separate arms at the midpoint
		Q_new_1=[]
		Q_new_2=[]
		
		for i in range(len(Q_new_a)):
			if i < len(Q1):
				Q_new_1.append(Q_new_a[i])
			
			else:
				Q_new_2.append(Q_new_a[i])
		
		return Q_new_1, Q_new_2

	if SYM==0:
		# Mirror Symmetry
		A_new_1,A_new_2=reorder_cross_mir(A1,A2,Ap1,Ap2,An1,An2)
		A_new_3,A_new_4=reorder_cross_mir(A3,A4,Ap3,Ap4,An3,An4)


	elif SYM==1:
		# Rotational Symmetry
		A_new_1=reorder_cross_rot(A1,Ap1,An1)
		A_new_2=reorder_cross_rot(A2,Ap2,An2)
		A_new_3=reorder_cross_rot(A3,Ap3,An3)
		A_new_4=reorder_cross_rot(A4,Ap4,An4)


	"Step 7: Reconstruct the core with the new cross region."
	def core_construct_cross(core,A1,A2,A3,A4,L,O):
		# Duplicate the original core
		core_new=[]
		for i in range(len(core)):
			l=[]
			for j in range(len(core[i])):
				l.append(core[i][j])
			
			core_new.append(l)
		
		# Apply new assembly order to via cross arms
		
		# A1
		for i in range(len(A1)):
			core_new[i][O] = A1[i]
			
		# A2
		for i in range(len(A2)):
			core_new[O][i] = A2[i]
		
		# A3
		for i in range(len(A3)):
			core_new[O][O+1+i] = A3[i]
		
		# A4
		for i in range(len(A4)):
			core_new[O+1+i][O] = A4[i]
		
		return core_new

	core_new = core_construct_cross(core_original,A_new_1,A_new_2,A_new_3,A_new_4,L,O)	
	
	return core_new

# Main Function to Optimize (all reactor models)
def Quadrant_Rearrange(new_order):
	"Steps 4-7 are managed through the outside function new_core_quadrant(new_order)."
    
	core_new = new_core_quadrant(new_order)
	
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

	"Step 10: Remove the copyfile from the directory." # Unnecessary, will be written over
	# Ignore for now (3/18/2022) 
	# os.system("rm "+copyfile)

	"Step 11: Extract output data of interest for the fitness function."
	outfile="s3."+filename+"-symm.out"
	sys.argv.append(outfile)

	lis=[] # stores all output lines
	k=[]   # stores k-effective lines per depletion step
	PF=[]  # stores max node-average peaking factor per depletion step

	# Extract output data and examine whether data is good or not.
	A=1 # boolean value (good = 1, bad = 0)
	with open(outfile,'r+') as f: # file==sys.argv[1]

		for l in f:
			lis.append(l)
			
			if l.find('K-effective  =')>=0:
				k.append(l.split())
				continue
			elif l.find('Node-Averaged')>=0: #l.find('Node-Averaged Peaking Factor =')>=0:
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
	y=C1*EOC_k + C2/max_NPF
	F1 = C1*EOC_k
	F2 = C2/max_NPF
	
	# If the output file returns aborted or with a Nan value, ignore it by setting the fitness to zero.
	if A==0 or np.isnan(EOC_k)==True or max_NPF > NPF_limit:
		y=0; A=0; F1 = 0; F2 = 0

	print()
	print('EOC k: ',EOC_k)
	print('max node peaking factor: ',max_NPF)
	print('Function value: ',y)
	print()

	"Step 12: Delete the output file to open space for the next." # Unnecessary, will be written over
	#os.system('rm '+outfile)

	core_run_q_list.append(new_order)
	k_run_q_list.append(EOC_k)
	PP_run_q_list.append(max_NPF)
	F1_run_q_list.append(F1)
	F2_run_q_list.append(F2)
	y_run_q_list.append(y)
	
	storage_q_list.append([A,new_order,EOC_k,max_NPF,y,string_core+seg_line_original])
	
	return y #,EOC_k,A # fitness function value, boolean for successful run

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
		
# New Function to Optimize Cross Assemblies (only such that L is odd).
def Cross_Rearrange(new_order):
	# To use for both mirror & rotational symmetry.
	# Determination is based on length of new_order.
	
	"Steps 4-7 are performed in the external function new_core_cross."
	
	core_new = new_core_cross(new_order)
	
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
	PF=[] # stores max node-average peaking factor per depletion step

	A=1
	with open(outfile,'r+') as f: # file==sys.argv[1]

		for l in f:
			lis.append(l)
			
			if l.find('K-effective  =')>=0:
				k.append(l.split())
				continue
			elif l.find('Node-Averaged')>=0: #l.find('Node-Averaged Peaking Factor =')>=0:
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
	y=C1*EOC_k + C2/max_NPF
	F1 = C1*EOC_k
	F2 = C2/max_NPF
	
	# If the output file returns aborted or with a Nan value, ignore it by setting the fitness to zero.
	if A==0 or np.isnan(EOC_k)==True or max_NPF > NPF_limit:
		y=0; A=0; F1=0; F2=0

	print()
	print('EOC k: ',EOC_k)
	print('max node peaking factor: ',max_NPF)
	print('Function value: ',y)
	print()
    
	"Step 12: Delete the output file to open space for the next."
	#os.system('rm '+outfile)
    
	core_run_c_list.append(string_core) # save the string core of the new arrangement after each run
	k_run_c_list.append(EOC_k)
	PP_run_c_list.append(max_NPF)
	F1_run_c_list.append(F1)
	F2_run_c_list.append(F2)
	y_run_c_list.append(y)
	
	storage_c_list.append([A,new_order,EOC_k,max_NPF,y,string_core+seg_line_original])
	
	return y #,EOC_k,max_NPF,A


# 7/26/22 - Change the desired 'SEG.DAT' enrichment values.
def seg_string(e_new,e_ID,size):
	"""
	Parameters
	----------
	e_new : list of float numbers
		List of new enrichments for SEG.DAT lines in copy file.
	e_ID : list of int
		List of integers representing specific IDs of SEG.DAT lines.
	size : int
		Number of characters that string of e_new can have for formatting.

	Returns
	-------
	seg_lines : list of strings
		List of new SEG.DAT lines to be written into the copy file.

	"""
    # Copy of segment data lines
	seg=[]
	line=[]
	for i in range(len(seg_base)):
		for j in range(len(seg_base[i])):
			line.append(seg_base[i][j])
		seg.append(line)
		line=[]
    
	seg_lines=[]
    
	e_copy=[]
	for i in range(len(e_new)):
		e_copy.append(e_new[i])
		
	# Convert all objects in enrichment list into string characters
	for a in range(len(e_new)):
		e_str = str(e_new[a])
		
		# Add trailing zeros to the strings
		if len(e_str) < size:
			e_str += '0'*(size-len(e_str))
		
		elif len(e_str) > size:
			e_str = e_str[0:size]
		
		e_str+=','
		
		e_copy[a] = e_str
		
	n = 0 # e_new index
	#for i in range(len(e_new)):
	for i in e_ID:
		# Replace old enrichment with new enrichment
		seg[i][2] = e_copy[n]
		
		for j in range(len(seg[i])):
			seg[i][j] += ' ' # add space after each string of characters
			
			if j == 0:
				seg[i][j] += ' ' # add another space after header
			
			if j == len(seg[i])-1:
				seg[i][j]+='\n' # newline to end of line
			
		string_seg="".join(seg[i])
		seg_lines.append(string_seg)
		
		n += 1
		
	return seg_lines

# 10/11/2022 - Change the desired "SEG.DAT" burnable poison weight percentages
def seg_string_2(p_new,p_ID,size):
	"""
	Parameters
	----------
	p_new : list of float numbers
		List of new burnable poison weight percentages for SEG.DAT lines in copy file.
	p_ID : list of int
		List of integers representing specific IDs of SEG.DAT lines.
	size : int
		Number of characters that string of e_new can have for formatting.

	Returns
	-------
	seg_lines : list of strings
		List of new SEG.DAT lines to be written into the copy file.
	"""
	# Copy of segment data lines
	seg=[]
	line=[]
	for i in range(len(seg_base)):
		for j in range(len(seg_base[i])):
			line.append(seg_base[i][j])
		seg.append(line)
		line=[]
    
	seg_lines=[]

	p_copy=[]
	for i in range(len(p_new)):
		p_copy.append(p_new[i])
		
	# Convert all objects in percent list into string characters
	for a in range(len(p_new)):
		p_str = str(p_new[a])
		
		# Add trailing zeros to the strings
		if len(p_str) < size:
			p_str += '0'*(size-len(p_str))
		
		elif len(p_str) > size:
			p_str = p_str[0:size]
		
		p_str+=','
		p_copy[a] = p_str
		
		
	n = 0 # p_new index
	for i in p_ID:
		# Replace old weight percentage with new weight percentage
		seg[i][3] = p_copy[n]
		
		for j in range(len(seg[i])):
			seg[i][j] += ' ' # add space after each string of characters
			
			if j == 0:
				seg[i][j] += ' ' # add another space after header
			
			if j == len(seg[i])-1:
				seg[i][j]+='\n' # newline to end of line
			
		string_seg="".join(seg[i])
		seg_lines.append(string_seg)
		
		n += 1
		
	return seg_lines


def Conversion_1(x,min_y):
	"""
	x(numpy array of float) - list of values
	min_y(float) - minimum value (Note: max_y is handled in optimization problem)
	"""
	y = min_y + x/100
	
	for i in range(len(y)):
		y[i] = format(y[i],'.'+str(2)+'f')
	return y

def Conversion_2(y,min_y):
	"""
	y(numpy array of float) - list of values
	min_y(float) - minimum value (Note: max_y is handled in optimization problem)
	"""
	x = 100*(y - min_y)
	
	x = x.astype(int) # convert to array of int32
	return x


# Main Function to Optimize
def Enrichment_Change(p_r):
	e_n = Conversion_1(p_r,min_e) # Convert to enrichment

	# Step 3: Create duplicate file to store changes to reactor core
	seg_run_e_list.append(e_n*100)
	
	# 9/11/2022 - Internal conversion: Divide e_new by 100
	e_new=[]
	for i in range(len(e_n)):
		e_new.append(float(e_n[i]))#/100)	
	
	# Take the original input file.
	my_file = open(file)
	string_list = my_file.readlines() # difference between read() and readlines()
	my_file.close()
	
	# Plug in the optimal core arrangement to the original text.
	string_core=core_string(core_optimal,order)

	for i in range(Start_line,Start_line+len(core_optimal)): # first index of string_list is 0
		string_list[i]=string_core[i-Start_line]
	
	# Plug in the new 'SEG.DAT' lines to the original text.
	seg_lines = seg_string(e_new,e_ID,4)

	for i in range(len(e_new)):
		string_list[l_seg[e_ID[i]]-1] = seg_lines[i]

	copyfile="s3."+filename+"-enr"+".inp"

	my_file = open(copyfile, "w")
	new_file_contents = "".join(string_list)

	my_file.write(new_file_contents)
	my_file.close()

	"Step 9: With the new SIMULATE copyfile written, run it into SIMULATE-3."
	# Make sure the command "module load studsvik" is ran prior to calling this file.
	os.system("simulate3 -k "+copyfile)

	"Step 11: Extract output data of interest for the fitness function."
	outfile="s3."+filename+"-enr.out"

	sys.argv.append(outfile)

	lis=[] # stores all output lines
	k=[]   # stores k-effective lines per depletion step
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
	F1 = C1*EOC_k
	F2 = C2/max_NPF
	
	# If the output file returns aborted or with a Nan value, ignore it by setting the fitness to zero.
	if A==0 or np.isnan(EOC_k)==True or max_NPF > NPF_limit:
		y=0; A=0; F1=0; F2=0

	print()
	print('EOC k: ',EOC_k)
	print('max node peaking factor: ',max_NPF)
	print('Function value: ',y)
	print()

	"Step 12: Delete the output file to open space for the next."
	k_run_e_list.append(EOC_k)
	PP_run_e_list.append(max_NPF)
	F1_run_e_list.append(F1)
	F2_run_e_list.append(F2)
	y_run_e_list.append(y)
	
	storage_e_list.append([A,p_r,EOC_k,max_NPF,y,string_core_optimal+seg_lines])
	
	return y # fitness function value, boolean for successful run

# Main Function to Optimize
def Poison_Change(p_r):
	p_n = Conversion_1(p_r,min_p) # Convert to BPN
	
	# Step 3: Create duplicate file to store changes to reactor core	
	# 9/11/2022 - Internal conversion: Divide p_new by 100
	p_new=[]
	for i in range(len(p_n)):
		p_new.append(float(p_n[i]))#/100)	
	
	# Step 3: Create duplicate file to store changes to reactor core
	seg_run_p_list.append(p_new)
	
	my_file = open(file)
	string_list = my_file.readlines() # difference between read() and readlines()
	my_file.close()

	# Plug in the optimal core arrangement to the original text.
	string_core=core_string(core_optimal,order)
	
	for i in range(Start_line,Start_line+len(core_optimal)): # first index of string_list is 0
		string_list[i]=string_core[i-Start_line]
	
	# Plug in the optimal enrichment 'SEG.DAT' lines
	seg_e_lines = seg_string(e_final,e_ID,4) # enrichment line
	for i in range(len(e_final)):
		string_list[l_seg[e_ID[i]]-1] = seg_e_lines[i]
	
	# Plug in the new burnable poison 'SEG.DAT' lines.
	seg_p_lines = seg_string_2(p_new,p_ID,2+2) # burnable poison line
	for i in range(len(p_new)):
		string_list[l_seg[p_ID[i]]-1] = seg_p_lines[i]

	# Save all lines, unchanged and changed
	seg_lines=[]
	
	# Save all lines from original
	for i in range(len(seg_original)):
		seg_lines.append(seg_original[i])
		
	for i in range(len(e_ID)):
		seg_lines[e_ID[i]]=seg_e_lines[i]
		
	for j in range(len(p_ID)):
		seg_lines[p_ID[j]]=seg_p_lines[j]


	copyfile="s3."+filename+"-psn"+".inp"

	my_file = open(copyfile, "w")
	new_file_contents = "".join(string_list)

	my_file.write(new_file_contents)
	my_file.close()


	"Step 9: With the new SIMULATE copyfile written, run it into SIMULATE-3."
	# Make sure the command "module load studsvik" is ran prior to calling this file.
	os.system("simulate3 -k "+copyfile)

	"Step 11: Extract output data of interest for the fitness function."
	outfile="s3."+filename+"-psn.out"

	sys.argv.append(outfile)

	lis=[] # stores all output lines
	k=[]   # stores k-effective lines per depletion step
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
	F1=C1*EOC_k
	F2=C2/max_NPF
	
	# If the output file returns aborted or with a Nan value, ignore it by setting the fitness to zero.
	if A==0 or np.isnan(EOC_k)==True or max_NPF > NPF_limit:
		y=0; A=0; F1=0; F2=0

	print()
	print('EOC k: ',EOC_k)
	print('max node peaking factor: ',max_NPF)
	print('Function value: ',y)
	print()

	"Step 12: Delete the output file to open space for the next."
	k_run_p_list.append(EOC_k)
	PP_run_p_list.append(max_NPF)
	F1_run_p_list.append(F1)
	F2_run_p_list.append(F2)
	y_run_p_list.append(y)
	
	storage_p_list.append([A,p_r,EOC_k,max_NPF,y,string_core_optimal+seg_lines])
	
	return y #,EOC_k,max_NPF,A # fitness function value, boolean for successful run

"Pre-Evaluation of Core Design"
np.random.seed(100) # RNG seed if desired; comment out if undesired.
# Note: The chosen RNG seed will affect the randomness of the optimization algorithms,
#	    regardless if the "set_state" values are defined or not. But, by defining the 
#	    "set_state" values, this changes/restarts the seeded RNG sequence.

"Pre-optimization for full-core assembly shuffling."
C1 = 1; C2 = 1 # Preset default values - Automatically change after pre-optimization

# Step 1: Run a series of good runs determined randomly.
# If L is odd, shuffle both quadrants and cross before testing.
# If L is even, shuffle only quadrants (no cross present).

N = 1000 # sample limit of good runs
tally = 0 # number of good runs
sample_count = 0 # number of total runs, good and bad

time_pre_start = time.perf_counter() # timer start

# while-loop until number of good runs is achieved.
while tally < N:
	sample_input_q = np.random.permutation(n_Q_ID) # randomly generated quadrant arrangement
	
	if L%2 == 0: # even dimension - just quadrant shuffle
		y = Quadrant_Rearrange(sample_input_q)
		A = storage_q_list[-1][0] # 0 if bad run; 1 if good run
		tally+=A
		sample_count+=1
		
	if L%2 == 1: # odd dimension - quadrant shuffle, then cross shuffle
		# Temporary change to core_original to cross rearrange
		core_original = new_core_quadrant(sample_input_q) # set new beginning core arrangement to have quadrant shuffle
		
		sample_input_c = np.random.permutation(n_C_ID) # randomly generated cross arrangement
		y = Cross_Rearrange(sample_input_c)
		A = storage_c_list[-1][0] # 0 if bad run; 1 if good run
		tally+=A
		sample_count+=1
		
		# Reset core_original
		core_original = map_divide(map_core,L,s) # reset core arrangement for next iteration (automatic for L%2==0)
		
time_pre_end = time.perf_counter() # timer stop

# Step 2: Isolate good runs from the bad runs based on variable A		
sample_good=[]
for i in range(len(storage_q_list)):
        
	if storage_q_list[i][0] == 1:
		sample_good.append(storage_q_list[i])
		
if L%2 == 1:
	for i in range(len(storage_c_list)):
		
		if storage_c_list[i][0] == 1:
			sample_good.append(storage_c_list[i])

# Step 3: Extract EOC k-eff and max NPF from good sample list.
sample_k = [] # EOC k-eff
sample_PF = []# max NPF
for i in range(len(sample_good)):
	sample_k.append(sample_good[i][2])
	sample_PF.append(sample_good[i][3])

# Step 4: Average output variable lists and define the coefficients.
k_avg = np.average(sample_k)
PF_avg = np.average(sample_PF)

# Step 5: Reset the in-function lists for the optimizer runs.
core_run_q_list=[]
k_run_q_list=[]
PP_run_q_list=[]
F1_run_q_list=[]
F2_run_q_list=[]
y_run_q_list=[]
storage_q_list=[]

core_run_c_list=[] 
k_run_c_list=[] 
PP_run_c_list=[]
F1_run_c_list=[]
F2_run_c_list=[]
y_run_c_list=[] 
storage_c_list = []

time_pre = time_pre_end-time_pre_start # pre-optimization time interval

"Optimization for Reactor Core Shuffling"
time_start = time.perf_counter() # timer start
# Set the driving coefficients for the quadrant optimization function
S1 = 100; S2 = 50 # driving coefficients to affect the weights of k-EOC and max NPF (user defined)

C1 = S1/k_avg  # coefficient for k-EOC
C2 = S2*PF_avg # coefficient for max NPF
# Note: Coefficients C1 & C2 can be changed to user specificity. If user wants to skip
#		the pre-optimization portion, comment out pre-optimization Steps 1-5 using """ lines.

"Optimizer Setup for Quadrant Shuffling"
# Initialize custom fitness function object
fitness_cust_q = mlrose.CustomFitness(Quadrant_Rearrange)

# Core_Rearrange
problem_q = mlrose.MarksOpt(length = n_Q_ID, fitness_fn = fitness_cust_q, maximize = True) # range of 0-(length-1) as integers once in each state
schedule_q = mlrose.ExpDecay() # ArithDecay()  ExpDecay() GeomDecay() init_temp, decay, min_temp, exp_const

# Define initial state
init_state = []
for i in range(n_Q_ID):
	init_state.append(i)

init_state = np.array(init_state) # must convert to a NumPy array

time_q_start=time.perf_counter() # timer start

# Solve problem using simulated annealing
best_state_q, best_fitness_q, history_q = mlrose.simulated_annealing(problem_q, schedule = schedule_q,
															   max_attempts = 10, max_iters = 1000,
															   init_state = init_state, curve=True, random_state = 46)

time_q_end=time.perf_counter() # timer stop

time_q = time_q_end-time_q_start # quadrant shuffle time interval

# Construct the optimal quadrant arrangement as the new core pattern basis.
core_optimal = new_core_quadrant(best_state_q)
string_core_optimal=core_string(core_optimal,order)

"Pre-optimization for Cross Shuffling 'Cross_Rearrange'."
"Only if core dimension is odd."
if L%2 == 1: # odd
	"With quadrant shuffling done first, make core_original = optimal_core."
	core_original = new_core_quadrant(best_state_q)
	core_quad_shuffle = new_core_quadrant(best_state_q)

	"Optimizer Setup for Quadrant Shuffling"
	# Initialize custom fitness function object
	fitness_cust_c = mlrose.CustomFitness(Cross_Rearrange)

	# Core_Rearrange
	problem_c = mlrose.MarksOpt(length = n_C_ID, fitness_fn = fitness_cust_c, maximize = True) # range of 0-(length-1) as integers once in each state

	schedule_c = mlrose.ExpDecay() # ArithDecay()  ExpDecay() GeomDecay() init_temp, decay, min_temp, exp_const

	# Define initial state
	init_state = []
	for i in range(n_C_ID):
		init_state.append(i)

	init_state = np.array(init_state) # must convert to a NumPy array

	time_c_start=time.perf_counter() # timer start

	# Solve problem using simulated annealing
	best_state_c, best_fitness_c, history_c = mlrose.simulated_annealing(problem_c, schedule = schedule_c,
																   max_attempts = 5, max_iters = 500,
																   init_state = init_state, curve=True, random_state = 84)
	
	time_c_end=time.perf_counter() # timer stop
	
	time_c = time_c_end-time_c_start # cross shuffle time interval
	
	# Construct the optimal cross arrangement as the new core pattern basis.
	core_optimal = new_core_cross(best_state_c)
	string_core_optimal=core_string(core_optimal,order)


# 9/11/2022 - Create an optimization algorithm for changing enrichment of a fuel type.
# Initialize custom fitness function object
fitness_cust_e = mlrose.CustomFitness(Enrichment_Change)

# Enrichment Change
e_ID = [0,1,2,3,4] # element indexes to change in problem
# Range of values (Update 6/19/2023)
problem_e = mlrose.DiscreteOpt(length = len(e_ID), fitness_fn = fitness_cust_e, maximize = True, max_val = round(r_e*100)+1)
schedule_e = mlrose.ExpDecay() # ArithDecay()  ExpDecay() GeomDecay() init_temp, decay, min_temp, exp_const

# Define initial state
e_init = []
for i in range(len(e_ID)):
	e_init.append(e[e_ID[i]])
e_init = np.array(e_init)

# (6/19/2023) - New initial state to reflect range limit
r_init = Conversion_2(e_init,min_e)

time_e_start=time.perf_counter() # timer start

# Solve problem using simulated annealing
best_state_e, best_fitness_e, history_e = mlrose.simulated_annealing(problem_e, schedule = schedule_e,
															   max_attempts = 20, max_iters = 1000,
															   init_state = r_init, curve=True, random_state = 56)

time_e_end=time.perf_counter() # timer stop

time_e = time_e_end-time_e_start

# Final enrichment values based on optimizer results.
e_final = Conversion_1(best_state_e,min_e)

# 10/18/2022 - Construct the optimal enrichment values. (Updated 12/8/2022)
seg_e_container = seg_string(e_final,e_ID,4)

# Redefine the enrichment values written
for i in range(len(e_ID)):
	seg_base[i] = seg_e_container[e_ID[i]].split()


# 10/17/2022 - Create an optimization algorithm for changing burnable poison concentration of a fuel type.
# Initialize custom fitness function object
fitness_cust_p = mlrose.CustomFitness(Poison_Change)

# BPN Change (Update 6/19/2023)
p_ID = [1,2,4] # element indexes to change in problem
problem_p = mlrose.DiscreteOpt(length = len(p_ID), fitness_fn = fitness_cust_p, maximize = True, max_val = round(r_p*100)+1) # range of 0-(length-1) as integers once in each state
schedule_p = mlrose.ExpDecay() # ArithDecay()  ExpDecay() GeomDecay() init_temp, decay, min_temp, exp_const

# Define initial state
p_init = []
for i in range(len(p_ID)):
	p_init.append(float(p[p_ID[i]]))
p_init = np.array(p_init)

# (6/19/2023) - New initial state to reflect range limit
s_init = Conversion_2(p_init,min_p)

time_p_start=time.perf_counter() # timer start

# Solve problem using simulated annealing
best_state_p, best_fitness_p, history_p = mlrose.simulated_annealing(problem_p, schedule = schedule_p,
															   max_attempts = 20, max_iters = 1000,
															   init_state = s_init, curve=True, random_state = 96)

time_p_end=time.perf_counter() # timer stop

time_p=time_p_end-time_p_start

# Final enrichment values based on optimizer results.
p_final = Conversion_1(best_state_p,min_p)

# 10/18/2022 - Construct the optimal enrichment values. (Updated 12/8/2022)
seg_p_container = seg_string_2(p_final,p_ID,4)

time_end=time.perf_counter() # timer stop
# Post-Optimization Data Logging

# Time to complete optimization phase
time_opt = time_end - time_start


# Main Function to Write & Run Optimal Reactor Design
storage_post_list=[]  # storage of initial and final data

def Core_Write(core_new,e_new,p_new):
	#core_new - fuel assembly map
	#e_new - SEG.DAT line data following FE optimization
	#p_new - SEG.DAT line data following BP optimization
	

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

	# Conversion of new core map from 'list of list of str' to 'list of str'.
	string_core=core_string(core_new,order)

	# Changing old core map to new core map
	for i in range(Start_line,Start_line+len(core_new)): # first index of string_list is 0
		string_list[i]=string_core[i-Start_line]
	
	
	# Changing old SEG.DAT enrichments to new concentrations
	# Reset seg_base to original using seg_original
	seg_base=[]
	for i in range(len(seg_original)):
		line=[]
		for j in range(len(seg_original[i])):
			line.append(seg_original[i][j])
		seg_base.append(line)
	
	seg_lines = seg_string(e_new,e_ID,4) # enrichment lines
	
	# Write the new lines into the script.
	for i in range(len(e_new)):
		string_list[l_seg[e_ID[i]]-1] = seg_lines[i]
	
	seg_lines = seg_string_2(p_new,p_ID,4) # burnable absorber lines
	
	# Write the new lines into the script.
	for i in range(len(p_new)):
		string_list[l_seg[p_ID[i]]-1] = seg_lines[i]
	
	# Creation of Optimal File
	copyfile="s3."+filename+"-symm-opt"+".inp"

	my_file = open(copyfile, "w")
	new_file_contents = "".join(string_list)
    
	my_file.write(new_file_contents)
	my_file.close()


	os.system("simulate3 -k "+copyfile)

	outfile="s3."+filename+"-symm-opt.out"

	sys.argv.append(outfile)

	lis=[] # stores all output lines
	k=[]   # stores k-effective lines per depletion step
	PF=[]  # stores max node-average peaking factor per depletion step

	# Extract output data and examine whether data is good or not.
	A=1 # boolean value (good = 1, bad = 0)
	with open(outfile,'r+') as f: # file==sys.argv[1]

		for l in f:
			lis.append(l)
			
			if l.find('K-effective  =')>=0:
				k.append(l.split())
				continue
			elif l.find('Node-Averaged')>=0:# Peaking Factor =')>=0:
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
		
	max_NPF=max(NPF) # max node peaking factor usin

	y = C1*EOC_k + C2/max_NPF

	storage_post_list.append([A,core_new,EOC_k,max_NPF,y,string_core+seg_lines])

	# Writing a weighted function to calculate for the fitness
	if L%2==1: # odd-dimension design - evaluate both quad and cross functions		
		y1 = C1*EOC_k + C2/max_NPF
		return EOC_k, max_NPF, y1
	
	else:
		y1 = C1*EOC_k + C2/max_NPF
		return EOC_k, max_NPF, y1

# Write the optimal arrangement in an input file.
initial_results = Core_Write(core_original_copy,e_init,p_init) # returns a tuple with data based on core design
final_results = Core_Write(core_optimal,e_final,p_final) # assures optimal run file contains final design


"Reorganize all iteration lists to remove duplicates."
# Isolate the results of the initial run
data_init_q=storage_q_list[0] 
if L%2==1:
	data_init_c=storage_c_list[0]
data_init_e=storage_e_list[0]
data_init_p=storage_p_list[0]

data_init=storage_q_list[0] # initial run of all algorithms (prior to reorganizing)

# Remove all duplicate runs in case the optimizer ran the function twice per iteration.
# Main List - input arrangement
def Clear_Dupes(main_list,dupes_list):
	clean_list=[]
	
	for i in range(1,len(main_list)): # ignore first/initial run data
		if i==1: # first iteration after initial
			clean_list.append(dupes_list[i])
		
		elif np.array_equal(main_list[i],main_list[i-1])==False:
			clean_list.append(dupes_list[i])

	return clean_list

# Remove the duplicate runs from the lists.
y_new_q_list=Clear_Dupes(core_run_q_list, y_run_q_list) # storage of fitnesses
a_new_q_list=Clear_Dupes(core_run_q_list, core_run_q_list) # storage of states
k_new_q_list=Clear_Dupes(core_run_q_list, k_run_q_list) # storage of EOC k-effective
PP_new_q_list=Clear_Dupes(core_run_q_list, PP_run_q_list)# storage of max node-average peaking factor
F1_new_q_list=Clear_Dupes(core_run_q_list, F1_run_q_list)
F2_new_q_list=Clear_Dupes(core_run_q_list, F2_run_q_list)

storage_new_q_list=Clear_Dupes(core_run_q_list, storage_q_list)

storage_q_list = storage_new_q_list
del storage_new_q_list

if L%2==1:
	y_new_c_list=Clear_Dupes(core_run_c_list, y_run_c_list) # storage of fitnesses
	a_new_c_list=Clear_Dupes(core_run_c_list, core_run_c_list) # storage of states
	k_new_c_list=Clear_Dupes(core_run_c_list, k_run_c_list) # storage of EOC k-effective
	PP_new_c_list=Clear_Dupes(core_run_c_list, PP_run_c_list)# storage of max node-aveage peaking factor
	F1_new_c_list=Clear_Dupes(core_run_c_list, F1_run_c_list)
	F2_new_c_list=Clear_Dupes(core_run_c_list, F2_run_c_list)

	storage_new_c_list=Clear_Dupes(core_run_c_list, storage_c_list)
	
	storage_c_list = storage_new_c_list
	del storage_new_c_list

y_new_e_list=Clear_Dupes(seg_run_e_list, y_run_e_list) # storage of fitnesses
a_new_e_list=Clear_Dupes(seg_run_e_list, seg_run_e_list) # storage of states
k_new_e_list=Clear_Dupes(seg_run_e_list, k_run_e_list) # storage of EOC k-effective
PP_new_e_list=Clear_Dupes(seg_run_e_list, PP_run_e_list)# storage of max node-average peaking factor
F1_new_e_list=Clear_Dupes(seg_run_e_list, F1_run_e_list)
F2_new_e_list=Clear_Dupes(seg_run_e_list, F2_run_e_list)

storage_new_e_list=Clear_Dupes(seg_run_e_list, storage_e_list)

storage_e_list = storage_new_e_list
del storage_new_e_list

y_new_p_list=Clear_Dupes(seg_run_p_list, y_run_p_list) # storage of fitnesses
a_new_p_list=Clear_Dupes(seg_run_p_list, seg_run_p_list) # storage of states
k_new_p_list=Clear_Dupes(seg_run_p_list, k_run_p_list) # storage of EOC k-effective
PP_new_p_list=Clear_Dupes(seg_run_p_list, PP_run_p_list)# storage of max node-average peaking factor
F1_new_p_list=Clear_Dupes(seg_run_p_list, F1_run_p_list)
F2_new_p_list=Clear_Dupes(seg_run_p_list, F2_run_p_list)

storage_new_p_list=Clear_Dupes(seg_run_p_list, storage_p_list)

storage_p_list = storage_new_p_list
del storage_new_p_list


# (6/18/2022)
# For plotting purposes, combine iteration results and specify where one algorithm run ends & the other begins.
if L%2 == 1:
	y_full_list = y_new_q_list + y_new_c_list
	k_full_list = k_new_q_list + k_new_c_list
	PP_full_list = PP_new_q_list + PP_new_c_list
	
	y_asby_list = y_new_e_list + y_new_p_list
	k_asby_list = k_new_e_list + k_new_p_list
	PP_asby_list = PP_new_e_list + PP_new_p_list
		
	y_new_list = y_new_q_list + y_new_c_list + y_new_e_list + y_new_p_list
	k_new_list = k_new_q_list + k_new_c_list + k_new_e_list + k_new_p_list
	PP_new_list = PP_new_q_list + PP_new_c_list + PP_new_e_list + PP_new_p_list
else:
	y_asby_list = y_new_e_list + y_new_p_list
	k_asby_list = k_new_e_list + k_new_p_list
	PP_asby_list = PP_new_e_list + PP_new_p_list
	
	y_new_list = y_new_q_list + y_new_e_list + y_new_p_list
	k_new_list = k_new_q_list + k_new_e_list + k_new_p_list
	PP_new_list = PP_new_q_list + PP_new_e_list + PP_new_p_list

n_inputs = len(y_new_list)

# Save all the run data from storage_list into a separate text file.
# Sort the iterations in two steps.
# Step 1: Distinguish the good (A=1) runs from the bad (A=0) runs.

data_q_good=[]
data_q_bad =[]
data_c_good=[]
data_c_bad =[]
data_e_good=[]
data_e_bad =[]
data_p_good=[]
data_p_bad =[]

F1_q_good=[]
F2_q_good=[]
F1_c_good=[]
F2_c_good=[]
F1_e_good=[]
F2_e_good=[]
F1_p_good=[]
F2_p_good=[]

for i in range(len(storage_q_list)):
	if storage_q_list[i][0] == 0:
		data_q_bad.append(storage_q_list[i])
		
	elif storage_q_list[i][0] == 1:
		data_q_good.append(storage_q_list[i])
		F1_q_good.append(F1_new_q_list[i])
		F2_q_good.append(F2_new_q_list[i])

for i in range(len(storage_c_list)):
	if storage_c_list[i][0] == 0:
		data_c_bad.append(storage_c_list[i])
		
	elif storage_c_list[i][0] == 1:
		data_c_good.append(storage_c_list[i])
		F1_c_good.append(F1_new_c_list[i])
		F2_c_good.append(F2_new_c_list[i])	

for i in range(len(storage_e_list)):
	if storage_e_list[i][0] == 0:
		data_e_bad.append(storage_e_list[i])
		
	elif storage_e_list[i][0] == 1:
		data_e_good.append(storage_e_list[i])
		F1_e_good.append(F1_new_e_list[i])
		F2_e_good.append(F2_new_e_list[i])

for i in range(len(storage_p_list)):
	if storage_p_list[i][0] == 0:
		data_p_bad.append(storage_p_list[i])
		
	elif storage_p_list[i][0] == 1:
		data_p_good.append(storage_p_list[i])
		F1_p_good.append(F1_new_p_list[i])
		F2_p_good.append(F2_new_p_list[i])
		  
# Step 2: For the good data only, order list from highest fitness to lowest fitness.

def List_Ordering(data,id_data): # function to order from highest to lowest.
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

data_q_good_sort = List_Ordering(data_q_good,4)
data_q_good = data_q_good_sort
del data_q_good_sort

if L%2==1:
	data_c_good_sort = List_Ordering(data_c_good,4)
	data_c_good = data_c_good_sort
	del data_c_good_sort

data_e_good_sort = List_Ordering(data_e_good,4)
data_e_good = data_e_good_sort
del data_e_good_sort

data_p_good_sort = List_Ordering(data_p_good,4)
data_p_good = data_p_good_sort
del data_p_good_sort

# 6/24/2022
# Find the data runs with best values of fitness, k-eff, and NPF.

# Concatenate all iterations of good results.
if L%2==1:
	data_core = data_q_good + data_c_good
	data_complete = data_q_good + data_c_good + data_e_good + data_p_good
else:
	data_complete = data_q_good + data_e_good + data_p_good


# Sort them based on the three primary outputs: fitness, EOC k, and max NPF.
# Indexes ares 4, 2, and 3, respectively.
data_by_f = List_Ordering(data_complete, 4)
data_by_k = List_Ordering(data_complete, 2)
data_by_P = List_Ordering(data_complete, 3)

# Extract the best runs and place into a new file.
data_best_runs=[]

data_best_runs.append(data_by_f[0]) # 0 = first = highest
data_best_runs.append(data_by_k[0])
data_best_runs.append(data_by_P[-1]) # -1 = last = lowest

# Save min and max output values from good data runs
k_good_q_data=[]
PP_good_q_data=[]
y_good_q_data=[]

k_good_c_data=[]
PP_good_c_data=[]
y_good_c_data=[]

k_good_e_data=[]
PP_good_e_data=[]
y_good_e_data=[]

k_good_p_data=[]
PP_good_p_data=[]
y_good_p_data=[]

for i in range(len(data_q_good)):
	k_good_q_data.append(round(data_q_good[i][2],6))
	PP_good_q_data.append(round(data_q_good[i][3],5))
	y_good_q_data.append(round(data_q_good[i][4],3))


for i in range(len(data_c_good)):
	k_good_c_data.append(round(data_c_good[i][2],6))
	PP_good_c_data.append(round(data_c_good[i][3],5))
	y_good_c_data.append(round(data_c_good[i][4],3))

for i in range(len(data_e_good)):
	k_good_e_data.append(round(data_e_good[i][2],6))
	PP_good_e_data.append(round(data_e_good[i][3],5))
	y_good_e_data.append(round(data_e_good[i][4],3))

for i in range(len(data_p_good)):
	k_good_p_data.append(round(data_p_good[i][2],6))
	PP_good_p_data.append(round(data_p_good[i][3],5))
	y_good_p_data.append(round(data_p_good[i][4],3))


if L%2 == 1:
	y_good_core_data = y_good_q_data + y_good_c_data
	k_good_core_data = k_good_q_data + k_good_c_data
	PP_good_core_data = PP_good_q_data + PP_good_c_data
	
	y_good_asby_data = y_good_e_data + y_good_p_data
	k_good_asby_data = k_good_e_data + k_good_p_data
	PP_good_asby_data = PP_good_e_data + PP_good_p_data
	
	y_good_data = y_good_q_data + y_good_c_data + y_good_e_data + y_good_p_data
	k_good_data = k_good_q_data + k_good_c_data + k_good_e_data + k_good_p_data
	PP_good_data = PP_good_q_data + PP_good_c_data + PP_good_e_data + PP_good_p_data
	
else:
	y_good_asby_data = y_good_e_data + y_good_p_data
	k_good_asby_data = k_good_e_data + k_good_p_data
	PP_good_asby_data = PP_good_e_data + PP_good_p_data

	y_good_data = y_good_q_data + y_good_e_data + y_good_p_data
	k_good_data = k_good_q_data + k_good_e_data + k_good_p_data
	PP_good_data = PP_good_q_data + PP_good_e_data + PP_good_p_data

# (6/20/2023) Good data organized. Retrieve initial and final results.

final_model = Poison_Change(best_state_p) # call last opt function with best state from algorithm
data_final = storage_p_list[-1] # final data

del storage_p_list[-1] # remove the duplicate best state run after saving result


# Create string objects to store the iteration data, separating good from bad data.

np.set_printoptions(linewidth=np.inf) # prevents text wrapping of NumPy arrays (occurs when NumPy array characters > 75)

# Used for core arrangement results.
def Data_Writer(initial_data, good_data,bad_data,time):
    
	text=[]

	# Initialize with title line
	text.append('Results from mlrose Optimization of SIMULATE-3 Core Arrangement \n\n')
	
	# Display the period to complete optimization
	text.append('Number of Inputs: '+str(len(good_data)+len(bad_data))+'\n')
	text.append('Algorithm Period: '+str(time)+'\n\n')
	
	# Declare the initial design
	text.append('Starting Design: \n')
	text.append('Input Arrangement: '+str(initial_data[1])+'\n')
	text.append('EOC k-effective: '+str(round(initial_data[2],9))+'    ')
	text.append('Max Nodal Peaking Factor: '+str(round(initial_data[3],8))+'    ')
	text.append('Fitness Value: '+str(initial_data[4])+'\n')
	for j in range(len(initial_data[5])):
		text.append(initial_data[5][j])
	text.append('\n\n')
	
	# Header for good data
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

	# Header for bad data
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

def Data_Writer_2(initial_data, final_data, good_data):
	
	text=[]

	# Initialize with title line
	text.append('Results from mlrose Optimization of SIMULATE-3 Core Arrangement \n\n')
	
	# Display the period to complete optimization
	text.append('Number of Inputs: '+str(n_inputs)+'\n')
	
	text.append('Time to Complete: \n')
	text.append('Pre-optimization: '+str(time_pre)+'\n')
	text.append('Optimization: '+str(time_opt)+'\n')
	text.append('Total Period: '+str(time_pre+time_opt)+'\n\n')
	
	# Declare the initial design
	text.append('Starting Design: \n')
	text.append('EOC k-effective: '+str(round(initial_data[2],9))+'    ')
	text.append('Max Nodal Peaking Factor: '+str(round(initial_data[3],8))+'    ')
	text.append('Fitness Value: '+str(initial_data[4])+'\n')
	for j in range(len(initial_data[5])):
		text.append(initial_data[5][j])
	text.append('\n\n')

	# Declare the final design
	text.append('Final Design: \n')
	text.append('EOC k-effective: '+str(round(final_data[2],9))+'    ')
	text.append('Max Nodal Peaking Factor: '+str(round(final_data[3],8))+'    ')
	text.append('Fitness Value: '+str(final_data[4])+'\n')
	for j in range(len(final_data[5])):
		text.append(final_data[5][j])
	text.append('\n\n')

	# Header for best data
	text.append('Best Data Runs: \n')

	# Append each line to the text
	for i in range(len(good_data)):
		if i==0:
			text.append('Best Fitness: \n')
		if i==1:
			text.append('Best EOC k: \n')
		if i==2:
			text.append('Best max NPF: \n')
		
		#text.append('Input Arrangement: '+str(good_data[i][1])+'\n')
		text.append('EOC k-effective: '+str(round(good_data[i][2],9))+'    ')
		text.append('Max Nodal Peaking Factor: '+str(round(good_data[i][3],8))+'    ')
		text.append('Fitness Value: '+str(good_data[i][4])+'\n')
		for j in range(len(good_data[i][5])):
			text.append(good_data[i][5][j])
		text.append('\n')
	
	return text


# Write the text into a file to store all quadrant changes
true_text_q = Data_Writer(data_init_q,data_q_good,data_q_bad,time_q)

file_1 = open("mlrose_results_quadrants.txt", "w+")

new_file_contents = "".join(true_text_q)

file_1.write(new_file_contents)
file_1.close()

# Write the text into a file to store all cross changes
if L%2 == 1:
	true_text_c = Data_Writer(data_init_c,data_c_good,data_c_bad,time_c)
	
	file_2 = open("mlrose_results_cross.txt", "w+")

	new_file_contents = "".join(true_text_c)

	file_2.write(new_file_contents)
	file_2.close()

# Write the text into a file to store all enrichment changes
true_text_e = Data_Writer(data_init_e,data_e_good,data_e_bad,time_e)

file_3 = open("mlrose_results_enrichment.txt", "w+")

new_file_contents = "".join(true_text_e)

file_3.write(new_file_contents)
file_3.close()

# Write the text into a file to store all burnable absorber changes
true_text_p = Data_Writer(data_init_p,data_p_good,data_p_bad,time_p)

file_4 = open("mlrose_results_poison.txt", "w+")

new_file_contents = "".join(true_text_p)

file_4.write(new_file_contents)
file_4.close()


# Write the text into a file to store the best patterns.
true_text_complete = Data_Writer_2(data_init,data_final,data_best_runs)

file_5 = open("mlrose_results_best.txt", "w+")

new_file_contents = "".join(true_text_complete)

file_5.write(new_file_contents)
file_5.close()

# (10/28/2023) Write the output values (fitness, k_EOC, NPF_max) into a text file.

file_6 = open("mlrose_output.txt", "w+")

file_6.write("Output Values per Iteration \n") # Title
file_6.write("Iter.#\tFitness\t\tk_EOC\t\tNPF_max"+"\n") # Headers

for i in range(n_inputs):
	file_6.write(str(i+1)+"\t")
	file_6.write("{0:.6f}".format(y_new_list[i])+"\t")
	file_6.write("{0:.6f}".format(k_new_list[i])+"\t")
	file_6.write("{0:.3f}".format(PP_new_list[i])+"\n")
file_6.close()


# Check Results

print()
print('Pre-Optimization Data:')
print()
print('Quadrant Shuffle Sampling Results:')
print('Number of Good Samples: ',tally,' of ',sample_count)
#print('Pre-Optimization Time: ',time_pre_end-time_pre_start)
print()
print('Driving Coefficients: ',S1,' & ',S2)
print('k-average: ',k_avg,'  Empirical coefficient 1: ',C1)
print('PF-average:',PF_avg,'  Empirical coefficient 2: ',C2)
print()
if L%2 == 1:
	print()
	print('Cross Shuffle Sampling Results:')
	print('Number of Good Samples: ',tally,' of ',sample_count)
	print()
	print('Driving Coefficients: ',S1,' & ',S2)
	print('k-average: ',k_avg,'  Empirical coefficient 1: ',C1)
	print('PF-average:',PF_avg,'  Empirical coefficient 2: ',C2)
	print()
print()

print('Initial Core Design:')
print()
print('Initial EOC k-effective: ',data_init_q[2])
print('Initial max node-average peaking factor: ',data_init_q[3])
print('Initial Fitness: ',data_init_q[4])
print()
print('Initial Enrichments: ',e_init)
print('Initial Burnable Poison Concentrations: ',p_init)
print()
print('Initial Segment lines: ',seg_original)
print()
print()
print('Optimization Results:')
print()
print('Quadrant Optimizer Results:')
print('Number of iterations: ',len(storage_q_list))
print('Number from Curve:    ',len(history_q))
print('Time to complete: ', time_q,'s')
print()
print('Best Pattern: ',best_state_q)
print('Best Fitness: ',best_fitness_q)
print()
if L%2 == 1:
	print()
	print('Cross Optimizer Results:')
	print('Number of iterations: ',len(storage_c_list))
	print('Number from Curve:    ',len(history_c))
	print('Time to complete: ', time_c,'s')
	print()
	print('Best Pattern: ',best_state_c)
	print('Best Fitness: ',best_fitness_c)
	print()
print()
print('Enrichment Optimizer Results:')
print('Number of iterations: ',len(storage_e_list))
print('Number from Curve:    ',len(history_e))
print('Time to complete: ', time_e,'s')
print()
print('Best Combination: ',best_state_e)
print('Best Fitness:     ',best_fitness_e)
print('Final Enrichments: ', e_final)
print('Optimal "SEG.DAT": ', seg_e_container)
print()
print()
print('Burnable Poison Optimizer Results:')
print('Number of iterations: ',len(storage_p_list))
print('Number from Curve:    ',len(history_p))
print('Time to complete: ', time_p,'s')
print()
print('Best Combination: ',best_state_p)
print('Best Fitness:     ',best_fitness_p)
print('Final BPN wt%:     ', p_final)
print('Optimal "SEG.DAT": ', seg_p_container)
print()
print()
print('Final Core Design:')
print()
print('Final EOC k-effective: ',final_results[0])
print('Final max node-average peaking factor: ',final_results[1])
print('Final Fitness: ',final_results[2])
print()
print()
print('Design Steps:')
print()
for i in range(len(core_original_copy)):
	print(core_original_copy[i])
print()
for i in range(len(core_optimal)):
	print(core_optimal[i])
print()

# Plot the results.

# Quadrant Optimization
f1 = plt.figure(1)
plt.plot(range(1,len(history_q)+1),history_q,'r.-',lw=0.5,label='Optimizer Save')
plt.plot(range(1,len(y_new_q_list)+1),y_new_q_list,'b.',lw=0.5,label='Manual Save')
plt.xlabel('ID - Iteration #')
plt.ylabel('Fitness')
plt.ylim(bottom = min(y_good_q_data)-1, top = max(np.concatenate((y_good_q_data,history_q)))+1)
plt.legend()
plt.title('Quadrant Shuffle: Fitness')
f1.show()
f1.savefig('Quadrant Shuffle Fitness.png')

f2 = plt.figure(2)
plt.plot(range(1,len(k_new_q_list)+1),k_new_q_list,'r.',lw=0.5,label='k-effective (Manual Tracking)')
plt.xlabel('ID - Iteration #')
plt.ylabel('End-of-Cycle k-effective')
plt.ylim(bottom = min(k_good_q_data)-0.002, top = max(k_good_q_data)+0.002)
plt.title('Quadrant Shuffle: End-of-Cycle k-effective')
f2.show()
f2.savefig('Quadrant Shuffle k-EOC.png')

f3 = plt.figure(3)
plt.plot(range(1,len(PP_new_q_list)+1),PP_new_q_list,'b.',lw=0.5,label='Power Peaking Factor (Manual Tracking)')
plt.xlabel('ID - Iteration #')
plt.ylabel('Maximum Nodal Peaking Factor')
plt.ylim(bottom = min(PP_good_q_data)-0.2, top = min([NPF_limit,max(PP_good_q_data)+0.2]))
plt.title('Quadrant Shuffle: Max Node-Average Peaking Factor')
f3.show()
f3.savefig('Quadrant Shuffle Max NPF.png')

# Enrichment Optimization
f4 = plt.figure(4)
plt.plot(range(1,len(history_e)+1),history_e,'r.-',lw=0.5,label='Optimizer Save')
plt.plot(range(1,len(y_new_e_list)+1),y_new_e_list,'b.',lw=0.5,label='Manual Save')
plt.xlabel('ID - Iteration #')
plt.ylabel('Fitness')
plt.ylim(bottom = min(y_good_e_data)-1, top = max(np.concatenate((y_good_e_data,history_e)))+1)
plt.legend()
plt.title('Enrichment Optimization: Fitness')
f4.show()
f4.savefig('Enrichment Fitness.png')

f5 = plt.figure(5)
plt.plot(range(1,len(k_new_e_list)+1),k_new_e_list,'r.',lw=0.5,label='k-effective (Manual Tracking)')
plt.xlabel('ID - Iteration #')
plt.ylabel('End-of-Cycle k-effective')
plt.ylim(bottom = min(k_good_e_data)-0.002, top = max(k_good_e_data)+0.002)
plt.title('Enrichment Optimization: End-of-Cycle k-effective')
f5.show()
f5.savefig('Enrichment k-EOC.png')

f6 = plt.figure(6)
plt.plot(range(1,len(PP_new_e_list)+1),PP_new_e_list,'b.',lw=0.5,label='Power Peaking Factor (Manual Tracking)')
plt.xlabel('ID - Iteration #')
plt.ylabel('Maximum Nodal Peaking Factor')
plt.ylim(bottom = min(PP_good_e_data)-0.2, top = min([NPF_limit,max(PP_good_e_data)+0.2]))
plt.title('Enrichment Optimization: Max Node-Average Peaking Factor')
f6.show()
f6.savefig('Enrichment Max NPF.png')

# Burnable Poison Optimization
f7 = plt.figure(7)
plt.plot(range(1,len(history_p)+1),history_p,'r.-',lw=0.5,label='Optimizer Save')
plt.plot(range(1,len(y_new_p_list)+1),y_new_p_list,'b.',lw=0.5,label='Manual Save')
plt.xlabel('ID - Iteration #')
plt.ylabel('Fitness')
plt.ylim(bottom = min(y_good_p_data)-1, top = max(np.concatenate((y_good_p_data,history_p)))+1)
plt.legend()
plt.title('Burnable Poison Optimization: Fitness')
f7.show()
f7.savefig('Burnable Poison Fitness.png')

f8 = plt.figure(8)
plt.plot(range(1,len(k_new_p_list)+1),k_new_p_list,'r.',lw=0.5,label='k-effective (Manual Tracking)')
plt.xlabel('ID - Iteration #')
plt.ylabel('End-of-Cycle k-effective')
plt.ylim(bottom = min(k_good_p_data)-0.0002, top = max(k_good_p_data)+0.0002)
plt.title('Burnable Poison Optimization: End-of-Cycle k-effective')
f8.show()
f8.savefig('Burnable Poison k-EOC.png')

f9 = plt.figure(9)
plt.plot(range(1,len(PP_new_p_list)+1),PP_new_p_list,'b.',lw=0.5,label='Power Peaking Factor (Manual Tracking)')
plt.xlabel('ID - Iteration #')
plt.ylabel('Maximum Nodal Peaking Factor')
plt.ylim(bottom = min(PP_good_p_data)-0.2, top = min([NPF_limit,max(PP_good_p_data)+0.2]))
plt.title('Burnable Poison Optimization: Max Node-Average Peaking Factor')
f9.show()
f9.savefig('Burnable Poison Max NPF.png')

# Assembly Optimization
f22 = plt.figure(22)
plt.axvline(x = 0,color='b',alpha=0.5)
plt.axvline(x = len(k_new_e_list),color='g',alpha=0.5)
plt.plot(range(1,len(k_asby_list)+1),k_asby_list,'r.',lw=0.5,label='k-effective (Manual Tracking)')
plt.xlabel('ID - Iteration #')
plt.ylabel('End-of-Cycle k-effective')
plt.ylim(bottom = min(k_good_asby_data)-0.002, top = max(k_good_asby_data)+0.002)
plt.title('Assembly Optimization Profile: End-of-Cycle k-effective')
f22.show()
f22.savefig('Assembly Opt k-EOC.png')

f23 = plt.figure(23)
plt.axvline(x = 0,color='b',alpha=0.5)
plt.axvline(x = len(PP_new_e_list),color='g',alpha=0.5)
plt.plot(range(1,len(PP_asby_list)+1),PP_asby_list,'b.',lw=0.5,label='Power Peaking Factor (Manual Tracking)')
plt.xlabel('ID - Iteration #')
plt.ylabel('Maximum Nodal Peaking Factor')
plt.ylim(bottom = min(PP_good_asby_data)-0.2, top = min([NPF_limit,max(PP_good_asby_data)+0.2]))
plt.title('Assembly Optimization Profile: Max Node-Average Peaking Factor')
f23.show()
f23.savefig('Assembly Opt Max NPF.png')

f24 = plt.figure(24)
plt.axvline(x = 0,color='b',alpha=0.5, label='Start of Enrichment Optimizer')
plt.axvline(x = len(y_new_e_list),color='g',alpha=0.5, label='Start of BPN Optimizer')
plt.plot(range(1,len(history_e)+len(history_p)+1),np.concatenate((history_e,history_p)),'r.-',lw=0.5,label='Optimizer Save')
plt.plot(range(1,len(y_asby_list)+1),y_asby_list,'b.',lw=0.5,label='Manual Save')
plt.xlabel('ID - Iteration #')
plt.ylabel('Fitness')
plt.ylim(bottom = min(y_good_asby_data)-1, top = max(np.concatenate((y_good_asby_data,history_e,history_p)))+1)
plt.legend()
plt.title('Assembly Optimization Profile: Fitness')
f24.show()
f24.savefig('Assembly Opt Fitness.png')

# If no cross-assembly region, concatenate quadrant, enrichment, and burnable poison optimization results.
if L%2 == 0:
	f13 = plt.figure(13)
	plt.axvline(x = 0,color='k',alpha=0.5)
	plt.axvline(x = len(k_new_q_list),color='b',alpha=0.5)
	plt.axvline(x = len(k_new_q_list)+len(k_new_e_list),color='g',alpha=0.5)
	plt.plot(range(1,len(k_new_list)+1),k_new_list,'r.',lw=0.5,label='k-effective (Manual Tracking)')
	plt.xlabel('ID - Iteration #')
	plt.ylabel('End-of-Cycle k-effective')
	plt.ylim(bottom = min(k_good_data)-0.002, top = max(k_good_data)+0.002)
	plt.title('Complete Optimization Profile: End-of-Cycle k-effective')
	f13.show()
	f13.savefig('Complete Opt k-EOC.png')
	
	f14 = plt.figure(14)
	plt.axvline(x = 0,color='k',alpha=0.5)
	plt.axvline(x = len(PP_new_q_list),color='b',alpha=0.5)
	plt.axvline(x = len(PP_new_q_list)+len(PP_new_e_list),color='g',alpha=0.5)
	plt.plot(range(1,len(PP_new_list)+1),PP_new_list,'b.',lw=0.5,label='Power Peaking Factor (Manual Tracking)')
	plt.xlabel('ID - Iteration #')
	plt.ylabel('Maximum Nodal Peaking Factor')
	plt.ylim(bottom = min(PP_good_data)-0.2, top = min([NPF_limit,max(PP_good_data)+0.2]))
	plt.title('Complete Optimization Profile: Max Node-Average Peaking Factor')
	f14.show()	
	f14.savefig('Complete Opt Max NPF.png')
	
	f15 = plt.figure(15)
	plt.axvline(x = 0,color='k',alpha=0.5, label='Start of Quadrant Optimizer')
	plt.axvline(x = len(y_new_q_list),color='b',alpha=0.5, label='Start of Enrichment Optimizer')
	plt.axvline(x = len(y_new_q_list)+len(y_new_e_list),color='g',alpha=0.5, label='Start of BPN Optimizer')
	plt.plot(range(1,len(history_q)+len(history_e)+len(history_p)+1),np.concatenate((history_q,history_e,history_p)),'r.-',lw=0.5,label='Optimizer Save')
	plt.plot(range(1,len(y_new_list)+1),y_new_list,'b.',lw=0.5,label='Manual Save')
	plt.legend()
	plt.xlabel('ID - Iteration #')
	plt.ylabel('Fitness')
	plt.ylim(bottom = min(y_good_data)-1, top = max(np.concatenate((y_good_data,history_q,history_e,history_p)))+1)
	plt.title('Complete Optimization Profile: Fitness')
	f15.show()
	f15.savefig('Complete Opt Fitness.png')

	f28, ax = plt.subplots()
	NPF = ax.twinx()
	plt.axvline(x = 0,color='k',alpha=0.5)
	plt.axvline(x = len(PP_new_q_list),color='b',alpha=0.5)
	plt.axvline(x = len(PP_new_q_list)+len(PP_new_e_list),color='g',alpha=0.5)
	ax.plot(range(1,len(k_new_list)+1),k_new_list,'r.',markersize=5)
	NPF.plot(range(1,len(PP_new_list)+1),PP_new_list,'b.',markersize=5)
	ax.set_xlabel('ID - Iteration #')
	ax.set_ylabel('End-of-cycle k-effective',color='r')
	ax.tick_params(axis='y', labelcolor='r')
	ax.set_ylim(bottom = min(k_good_data)-0.002, top = max(k_good_data)+0.002)
	NPF.set_ylabel('Maximum Nodal Peaking Factor',color='b')
	NPF.tick_params(axis='y',labelcolor='b')
	NPF.set_ylim(bottom = min(PP_good_data)-0.2, top = min([NPF_limit,max(PP_good_data)+0.2]))
	plt.title('Complete Opt Output Variables')
	f28.show()
	f28.savefig('Complete Opt Output.png')


	
# If cross-assembly region is present, concatenate quadrant, cross, enrichment, & burnable poison optimization results.
if L%2 == 1:
	# Cross Optimization
	f16 = plt.figure(16)
	plt.plot(range(1,len(k_new_c_list)+1),k_new_c_list,'r.',lw=0.5,label='Manual Save')
	plt.xlabel('ID - Iteration #')
	plt.ylabel('End-of-Cycle k-effective')
	plt.ylim(bottom = min(k_good_c_data)-0.002, top = max(k_good_c_data)+0.002)
	plt.title('Cross Shuffle: End-of-Cycle k-effective')
	f16.show()
	f16.savefig('Cross Shuffle k-EOC.png')
	
	f17 = plt.figure(17)
	plt.plot(range(1,len(PP_new_c_list)+1),PP_new_c_list,'b.',lw=0.5,label='Manual Save')
	plt.xlabel('ID - Iteration #')
	plt.ylabel('Maximum Nodal Peaking Factor')
	plt.ylim(bottom = min(PP_good_c_data)-0.2, top = min([NPF_limit,max(PP_good_c_data)+0.2]))
	plt.title('Cross Shuffle: Max Node-Average Peaking Factor')
	f17.show()
	f17.savefig('Cross Shuffle Max NPF.png')

	f18 = plt.figure(18)
	plt.plot(range(1,len(history_c)+1),history_c,'r.-',lw=0.5,label='Optimizer Save')
	plt.plot(range(1,len(y_new_c_list)+1),y_new_c_list,'b.',lw=0.5,label='Manual Save')
	plt.xlabel('ID - Iteration #')
	plt.ylabel('Fitness')
	plt.ylim(bottom = min(y_good_c_data)-1, top = max(np.concatenate((y_good_c_data,history_c)))+1)
	plt.legend()
	plt.title('Cross Shuffle: Fitness')
	f18.show()
	f18.savefig('Cross Shuffle Fitness.png')
	"""
	"""
	# Full-Core Optimization
	f19 = plt.figure(19)
	plt.axvline(x = 0,color='k',alpha=0.5)
	plt.axvline(x = len(k_new_q_list),color='r',alpha=0.5)
	plt.plot(range(1,len(k_full_list)+1),k_full_list,'r.',lw=0.5,label='k-effective (Manual Tracking)')
	plt.xlabel('ID - Iteration #')
	plt.ylabel('End-of-Cycle k-effective')
	plt.ylim(bottom = min(k_good_core_data)-0.002, top = max(k_good_core_data)+0.002)
	plt.title('Full-Core Optimization Profile: End-of-Cycle k-effective')
	f19.show()
	f19.savefig('Full-Core Opt k-EOC.png')
	
	f20 = plt.figure(20)
	plt.axvline(x = 0,color='k',alpha=0.5)
	plt.axvline(x = len(PP_new_q_list),color='r',alpha=0.5)
	plt.plot(range(1,len(PP_full_list)+1),PP_full_list,'b.',lw=0.5,label='Power Peaking Factor (Manual Tracking)')
	plt.xlabel('ID - Iteration #')
	plt.ylabel('Maximum Nodal Peaking Factor')
	plt.ylim(bottom = min(PP_good_core_data)-0.2, top = min([NPF_limit,max(PP_good_core_data)+0.2]))
	plt.title('Full-Core Optimization Profile: Max Node-Average Peaking Factor')
	f20.show()
	f20.savefig('Full-Core Opt Max NPF.png')
	
	f21 = plt.figure(21)
	plt.axvline(x = 0,color='k',alpha=0.5,label='Start of Quadrant Optimizer')
	plt.axvline(x = len(y_new_q_list),color='r',alpha=0.5, label='Start of Cross Optimizer')
	plt.plot(range(1,len(history_c)+len(history_q)+1),np.concatenate((history_q,history_c)),'r.-',lw=0.5,label='Optimizer Save')
	plt.plot(range(1,len(y_full_list)+1),y_full_list,'b.',lw=0.5,label='Manual Save')
	plt.xlabel('ID - Iteration #')
	plt.ylabel('Fitness')
	plt.ylim(bottom = min(y_good_core_data)-1, top = max(np.concatenate((y_good_core_data,history_q,history_c)))+1)
	plt.legend()
	plt.title('Full-Core Optimization Profile: Fitness')
	f21.show()	
	f21.savefig('Full-Core Opt Fitness.png')
	
	# Complete Optimization
	f25 = plt.figure(25)
	plt.axvline(x = 0,color='k',alpha=0.5)
	plt.axvline(x = len(k_new_q_list),color='r',alpha=0.5)
	plt.axvline(x = len(k_new_q_list)+len(k_new_c_list),color='b',alpha=0.5)
	plt.axvline(x = len(k_new_q_list)+len(k_new_c_list)+len(k_new_e_list),color='g',alpha=0.5)
	plt.plot(range(1,len(k_new_list)+1),k_new_list,'r.',lw=0.5,label='k-effective (Manual Tracking)')
	plt.xlabel('ID - Iteration #')
	plt.ylabel('End-of-Cycle k-effective')
	plt.ylim(bottom = min(k_good_data)-0.002, top = max(k_good_data)+0.002)
	plt.title('Complete Optimization Profile: End-of-Cycle k-effective')
	f25.show()
	f25.savefig('Complete Opt k-EOC.png')
	
	f26 = plt.figure(26)
	plt.axvline(x = 0,color='k',alpha=0.5)
	plt.axvline(x = len(PP_new_q_list),color='r',alpha=0.5)
	plt.axvline(x = len(PP_new_q_list)+len(PP_new_c_list),color='b',alpha=0.5)
	plt.axvline(x = len(PP_new_q_list)+len(PP_new_c_list)+len(PP_new_e_list),color='g',alpha=0.5)
	plt.plot(range(1,len(PP_new_list)+1),PP_new_list,'b.',lw=0.5)#,label='Power Peaking Factor (Manual Tracking)')
	plt.xlabel('ID - Iteration #')
	plt.ylabel('Maximum Nodal Peaking Factor')
	plt.ylim(bottom = min(PP_good_data)-0.2, top = min([NPF_limit,max(PP_good_data)+0.2]))
	plt.title('Complete Optimization Profile: Max Node-Average Peaking Factor')
	f26.show()
	f26.savefig('Complete Opt Max NPF.png')

	f27 = plt.figure(27)
	plt.axvline(x = 0,color='k',alpha=0.5, label='Start of Quadrant Optimizer')
	plt.axvline(x = len(y_new_q_list), color='r', alpha=0.5, label='Start of Cross Optimizer')
	plt.axvline(x = len(y_new_c_list)+len(y_new_q_list), color='b', alpha=0.5, label='Start of Enrichment Optimizer')
	plt.axvline(x = len(y_new_e_list)+len(y_new_c_list)+len(y_new_q_list), color='g', alpha=0.5, label='Start of BPN Optimizer')
	plt.plot(range(1,len(history_q)+len(history_c)+len(history_e)+len(history_p)+1),np.concatenate((history_q,history_c,history_e,history_p)),'r.-',lw=0.5,label='Optimizer Save')
	plt.plot(range(1,len(y_new_list)+1),y_new_list,'b.',lw=0.5,label='Manual Save')
	plt.xlabel('ID - Iteration #')
	plt.ylabel('Fitness')
	plt.ylim(bottom = min(y_good_data)-1, top = max(np.concatenate((y_good_data,history_q,history_c,history_e,history_p)))+1)
	plt.legend()
	plt.title('Complete Optimization Profile: Fitness')
	f27.show()
	f27.savefig('Complete Opt Fitness.png')
	
	f29, ax = plt.subplots()
	NPF = ax.twinx()
	plt.axvline(x = 0,color='k',alpha=0.5)
	plt.axvline(x = len(PP_new_q_list),color='r',alpha=0.5)
	plt.axvline(x = len(PP_new_q_list)+len(PP_new_c_list),color='b',alpha=0.5)
	plt.axvline(x = len(PP_new_q_list)+len(PP_new_c_list)+len(k_new_e_list),color='g',alpha=0.5)
	ax.plot(range(1,len(k_new_list)+1),k_new_list,'r.',markersize=5)
	NPF.plot(range(1,len(PP_new_list)+1),PP_new_list,'b.',markersize=5)
	ax.set_xlabel('ID - Iteration #')
	ax.set_ylabel('End-of-cycle k-effective',color='r')
	ax.tick_params(axis='y', labelcolor='r')
	ax.set_ylim(bottom = min(k_good_data)-0.002, top = max(k_good_data)+0.002)
	NPF.set_ylabel('Maximum Nodal Peaking Factor',color='b')
	NPF.tick_params(axis='y',labelcolor='b')
	NPF.set_ylim(bottom = min(PP_good_data)-0.2, top = min([NPF_limit,max(PP_good_data)+0.2]))
	plt.title('Complete Opt Output Variables')
	f29.show()
	f29.savefig('Complete Opt Output.png')

plt.show()


