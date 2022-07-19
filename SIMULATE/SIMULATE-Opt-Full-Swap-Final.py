# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 08:08:43 2022

@author: nmmar
"""

"Complete SIMULATE Core Rearrangement"
"Use aspects from 'SIMULATE-Cross-Swap.py' and 'SIMULATE-Opt-Swap-Eval.py'."

import numpy as np

import time

import sys
import six
sys.modules['sklearn.externals.six'] = six
import os
import mlrose

import matplotlib.pyplot as plt


# Quadrant lists of output results (all reactor models)
core_run_q_list=[] # empty list of new core arrangements
k_run_q_list=[] # empty list of EOC k-eff
PF_run_q_list=[]# empty list of max NPF
F1_run_q_list=[]# empty list of factor 1
F2_run_q_list=[]# empty list of factor 2
y_run_q_list=[] # empty list of quality factor

storage_q_list = [] # empty list of all important data

# Cross lists of output results (only odd-dimension models)
core_run_c_list=[] # empty list of new core arrangements
k_run_c_list=[] # empty list of EOC k-eff
PF_run_c_list=[]# empty list of max NPF
F1_run_c_list=[]# empty list of factor 1
F2_run_c_list=[]# empty list of factor 2
y_run_c_list=[] # empty list of quality factor

storage_c_list = [] # empty list of all important data


# Start with finding the parameters needed for initializing the optimization problem.
"Step 1: Read the SIMULATE input file and extract the main information."

filename="pwr.uo2.c01-1"

file = "s3."+filename+".inp"

lis=[]  # list of all lines of file
map=[]  # array of core map
c=0     # condition for fuel core (off)

ID_line=0 # starting line for identification

SYM = 1 # Default symmetry pattern if undefined.

with open(file,'r+') as f: # file==sys.argv[1]

	for l in f:
		ID_line=ID_line+1 # start with line 1
		lis.append(l)
		
		# identify fuel core size
		if l.find("'DIM.PWR'")>=0:
			DIM_line=l.split()
			
			if DIM_line[1][-1]=='/' or DIM_line[1][-1]==',':
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
                            l=[]
                            
                        if i>O and j<=O: # Q3 - bottom left
                            Q3.append(l)
                            l=[l[-1]]
                            
                        if i>O and j>O: # Q4 - bottom right
                            Q4.append(l)
                            l=[]
               
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
		
		# A1 (top)
		for i in range(len(A1)):
			core_new[i][O] = A1[i]
			
		# A2 (left)
		for i in range(len(A2)):
			core_new[O][i] = A2[i]
		
		# A3 (right)
		for i in range(len(A3)):
			core_new[O][O+1+i] = A3[i]
		
		# A4 (bottom)
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


	"Step 10: Remove the copyfile from the directory."
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

	# Writing a weighted function to calculate for the fitness (6/12/2022 - changed coefficients from C1,C2 to CQ1,CQ2)
	#y=CQ1*EOC_k + CQ2/max_NPF # GOAL: Maximize
	#F1 = CQ1*EOC_k
	#F2 = CQ2/max_NPF
	
	y=C1*EOC_k + C2/max_NPF
	F1 = C1*EOC_k
	F2 = C2/max_NPF
	
	# If the output file returns aborted or with a Nan value, ignore it by setting the fitness to zero.
	if A==0 or np.isnan(EOC_k)==True:
		y=0; A=0; F1 = 0; F2 = 0

	print()
	print('EOC k: ',EOC_k)
	print('max node peaking factor: ',max_NPF)
	print('Function value: ',y)
	print()


	"Step 12: Delete the output file to open space for the next."

	#os.system('rm '+outfile)
	
	"Save all input and output data into lists."
	core_run_q_list.append(new_order)
	k_run_q_list.append(EOC_k)
	PF_run_q_list.append(max_NPF)
	F1_run_q_list.append(F1)
	F2_run_q_list.append(F2)
	y_run_q_list.append(y)
	
	storage_q_list.append([A,new_order,EOC_k,max_NPF,y,string_core])
	
	return y #,EOC_k,A # fitness function value, boolean for successful run


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
	#RPF=[] # stores max reative power fraction per depletion step
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
	#y=CC1*EOC_k + CC2/max_NPF # GOAL: Maximize
	#F1 = CC1*EOC_k
	#F2 = CC2/max_NPF

	y=C1*EOC_k + C2/max_NPF
	F1 = C1*EOC_k
	F2 = C2/max_NPF
	
	# If the output file returns aborted or with a Nan value, ignore it by setting the fitness to zero.
	if A==0 or np.isnan(EOC_k)==True:
		y=0; A=0; F1=0; F2=0

	print()
	print('EOC k: ',EOC_k)
	print('max node peaking factor: ',max_NPF)
	print('Function value: ',y)
	print()
    
	"Step 12: Delete the output file to open space for the next."
    
	#os.system('rm '+outfile)
	
	"Save all input and output data into lists."
	core_run_c_list.append(string_core) # save the string core of the new arrangement after each run
	k_run_c_list.append(EOC_k)
	PF_run_c_list.append(max_NPF)
	F1_run_c_list.append(F1)
	F2_run_c_list.append(F2)
	y_run_c_list.append(y)

	storage_c_list.append([A,new_order,EOC_k,max_NPF,y,string_core])

	return y #,EOC_k,max_NPF,A


"Pre-Evaluation of Core Design"
np.random.seed(1000)

"Pre-optimization for full-core assembly shuffling."
C1 = 1; C2 = 1

# Step 1: Run a series of good runs determined randomly.
# If L is odd, shuffle both quadrants and cross before testing.
# If L is even, shuffle only quadrants (no cross present).

N = 1000 # sample limit of good runs
tally = 0
sample_count = 0

time_pre_start = time.perf_counter()

while tally < N: # until enough good samples has been reached
	sample_input_q = np.random.permutation(n_Q_ID) # random quadrant pattern
	
	if L%2 == 0: # even-dimension - just quadrant shuffle
		y = Quadrant_Rearrange(sample_input_q)
		A = storage_q_list[-1][0] # determiner of good or bad pattern
		tally+=A
		sample_count+=1
		
	if L%2 == 1: # odd-dimension - quadrant shuffle and cross shuffle
		# Temporary change to core_original to cross rearrange
		core_original = new_core_quadrant(sample_input_q)
		
		sample_input_c = np.random.permutation(n_C_ID) # random cross pattern
		y = Cross_Rearrange(sample_input_c)
		A = storage_c_list[-1][0] # determiner of good or bad pattern
		tally+=A
		sample_count+=1
		
		# Reset core_original
		core_original = map_divide(map_core,L,s)
		
time_pre_end = time.perf_counter()

# Step 2: Isolate good runs from the bad runs based on variable A		
sample_good=[]
for i in range(len(storage_q_list)):
        
	if storage_q_list[i][0] == 1: # ensure good runs only
		sample_good.append(storage_q_list[i]) # good runs from even-dimension samples
		
if L%2 == 1:
	for i in range(len(storage_c_list)):
		
		if storage_c_list[i][0] == 1: # ensure good runs only
			sample_good.append(storage_c_list[i]) # good runs from odd-dimension samples

# Step 3: Extract EOC k-eff and max NPF from good sample list.
sample_k = []
sample_PF = []
for i in range(len(sample_good)):
	sample_k.append(sample_good[i][2])
	sample_PF.append(sample_good[i][3])

# Step 4: Average output variable lists and define the coefficients.
k_avg = np.average(sample_k)
PF_avg = np.average(sample_PF)

# Step 5: Reset the in-function lists for the optimizer runs.
core_run_q_list=[]
k_run_q_list=[]
PF_run_q_list=[]
F1_run_q_list=[]
F2_run_q_list=[]
y_run_q_list=[]
storage_q_list=[]

core_run_c_list=[] 
k_run_c_list=[] 
PF_run_c_list=[]
F1_run_c_list=[]
F2_run_c_list=[]
y_run_c_list=[] 
storage_c_list = []

print()
print('Pre-Optimization Time: ',time_pre_end-time_pre_start)
print()


"Optimization for Reactor Core Shuffling"
# Slaving/Driving Coefficients
S1 = 100; S2 = 1

C1 = S1/k_avg  # Modified Coefficient 1
C2 = S2*PF_avg # Modified Coefficient 2


"Optimizer Setup for Quadrant Shuffling"
# Initialize custom fitness function object
fitness_cust_q = mlrose.CustomFitness(Quadrant_Rearrange)

# Core_Rearrange
problem_q = mlrose.MarksOpt(length = n_Q_ID, fitness_fn = fitness_cust_q, maximize = True) # range of 0-(length-1) as integers once in each state

# Decay schedule
schedule_q = mlrose.ExpDecay() # ArithDecay()  ExpDecay() GeomDecay() init_temp, decay, min_temp, exp_const

# Define initial state
init_state = []
for i in range(n_Q_ID):
	init_state.append(i)

init_state = np.array(init_state)

time_q_start=time.perf_counter()

# Solve problem using simulated annealing
best_state_q, best_fitness_q, history_q = mlrose.simulated_annealing(problem_q, schedule = schedule_q,
															   max_attempts = 10, max_iters = 1000,
															   init_state = init_state, curve=True, random_state = 1000)

time_q_end=time.perf_counter()

# Construct the optimal core.
core_optimal = new_core_quadrant(best_state_q)

"Pre-optimization for Cross Shuffling 'Cross_Rearrange'."
"Only if core dimension is odd."
if L%2 == 1: # odd
	"With quadrant shuffling done first, make core_original = optimal_core."
	core_original = new_core_quadrant(best_state_q)
	core_quad_shuffle = new_core_quadrant(best_state_q) # copy to save for post-opt analysis

	"Optimizer Setup for Cross Shuffling"
	# Initialize custom fitness function object
	fitness_cust_c = mlrose.CustomFitness(Cross_Rearrange)

	# Core_Rearrange
	problem_c = mlrose.MarksOpt(length = n_C_ID, fitness_fn = fitness_cust_c, maximize = True) # range of 0-(length-1) as integers once in each state
	
	# Decay Schedule
	schedule_c = mlrose.ExpDecay() # ArithDecay()  ExpDecay() GeomDecay() init_temp, decay, min_temp, exp_const

	# Define initial state
	#init_state = None
	init_state = []
	for i in range(n_C_ID):
		init_state.append(i)

	init_state = np.array(init_state)

	time_c_start=time.perf_counter()

	# Solve problem using simulated annealing
	best_state_c, best_fitness_c, history_c = mlrose.simulated_annealing(problem_c, schedule = schedule_c,
																   max_attempts = 10, max_iters = 1000,
																   init_state = init_state, curve=True, random_state = 1000)
	
	time_c_end=time.perf_counter()
	
	# Construct the optimal core.
	core_optimal = new_core_cross(best_state_c)


# Post-Optimization Data Logging

# Main Function to Write & Run Optimal Reactor Design
def Core_Write(core_new):
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
		
	max_NPF=max(NPF) # max node peaking factor usin

	# Writing a weighted function to calculate for the fitness
	if L%2==1: # odd-dimension design - evaluate both quad and cross functions
		y1 = C1*EOC_k + C2/max_NPF
		y2 = C1*EOC_k + C2/max_NPF
		return EOC_k, max_NPF, y1, y2
	
	else:
		y1 = C1*EOC_k + C2/max_NPF
		return EOC_k, max_NPF, y1

# Write the optimal arrangement in an input file.
initial_results = Core_Write(core_original_copy) # returns a tuple with data based on core design
final_results = Core_Write(core_optimal) # assures optimal run file contains final design

"Reorganize all iteration lists to remove duplicates."

data_init_q=storage_q_list[0] # Isolate the results of the initial run

if L%2==1:
	data_init_c=storage_c_list[0]

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
PF_new_q_list=Clear_Dupes(core_run_q_list, PF_run_q_list)# storage of max node-average peaking factor
F1_new_q_list=Clear_Dupes(core_run_q_list, F1_run_q_list)# storage of fitness factor 1
F2_new_q_list=Clear_Dupes(core_run_q_list, F2_run_q_list)# storage of fitness factor 2

storage_new_q_list=Clear_Dupes(core_run_q_list, storage_q_list)

storage_q_list = storage_new_q_list
del storage_new_q_list

if L%2==1:
	y_new_c_list=Clear_Dupes(core_run_c_list, y_run_c_list) # storage of fitnesses
	a_new_c_list=Clear_Dupes(core_run_c_list, core_run_c_list) # storage of states
	k_new_c_list=Clear_Dupes(core_run_c_list, k_run_c_list) # storage of EOC k-effective
	PF_new_c_list=Clear_Dupes(core_run_c_list, PF_run_c_list)# storage of max node-aveage peaking factor
	F1_new_c_list=Clear_Dupes(core_run_c_list, F1_run_c_list)# storage of fitness factor 1
	F2_new_c_list=Clear_Dupes(core_run_c_list, F2_run_c_list)# storage of fitness factor 2

	storage_new_c_list=Clear_Dupes(core_run_c_list, storage_c_list)
	
	storage_c_list = storage_new_c_list
	del storage_new_c_list

# (6/18/2022)
# For plotting purposes, combine iteration results and specify where one algorithm run ends & the other begins.

if L%2 == 1:
	y_new_list = y_new_q_list + y_new_c_list
	k_new_list = k_new_q_list + k_new_c_list
	PF_new_list = PF_new_q_list + PF_new_c_list

# Save all the run data from storage_list into a separate text file.
# Sort the iterations in two steps.
# Step 1: Distinguish the good (A=1) runs from the bad (A=0) runs.
data_q_good=[]
data_q_bad =[]
data_c_good=[]
data_c_bad =[]

F1_q_good=[]
F2_q_good=[]
F1_c_good=[]
F2_c_good=[]

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

# 6/24/2022
# Find the data runs with best values of fitness, k-eff, and NPF.

# Concatenate all iterations of good results.
if L%2==1:
	data_complete = data_q_good + data_c_good
else:
	data_complete = data_q_good

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
PF_good_q_data=[]
y_good_q_data=[]

k_good_c_data=[]
PF_good_c_data=[]
y_good_c_data=[]

for i in range(len(data_q_good)):
	k_good_q_data.append(round(data_q_good[i][2],6))
	PF_good_q_data.append(round(data_q_good[i][3],5))
	y_good_q_data.append(round(data_q_good[i][4],3))


for i in range(len(data_c_good)):
	k_good_c_data.append(round(data_c_good[i][2],6))
	PF_good_c_data.append(round(data_c_good[i][3],5))
	y_good_c_data.append(round(data_c_good[i][4],3))


if L%2 == 1:
	y_good_data = y_good_q_data + y_good_c_data
	k_good_data = k_good_q_data + k_good_c_data
	PF_good_data = PF_good_q_data + PF_good_c_data
	

# Create string objects to store the iteration data, separating good from bad data.

np.set_printoptions(linewidth=np.inf) # prevents text wrapping of NumPy arrays (occurs when NumPy array characters > 75)

def Data_Writer(initial_data, good_data,bad_data):
    
	text=[]

	# Initialize with title line
	text.append('Results from mlrose Optimization of SIMULATE-3 Core Arrangement \n\n')
	
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

def Data_Writer_2(initial_data, good_data):
	
	text=[]

	# Initialize with title line
	text.append('Results from mlrose Optimization of SIMULATE-3 Core Arrangement \n\n')

	# Declare the initial design
	text.append('Starting Design: \n')
	text.append('Input Arrangement: '+str(initial_data[1])+'\n')
	text.append('EOC k-effective: '+str(round(initial_data[2],9))+'    ')
	text.append('Max Nodal Peaking Factor: '+str(round(initial_data[3],8))+'    ')
	text.append('Fitness Value: '+str(initial_data[4])+'\n')
	for j in range(len(initial_data[5])):
		text.append(initial_data[5][j])
	text.append('\n\n')

	# Header for best data
	text.append('Good Data Runs: \n')

	# Append each line to the text
	for i in range(len(good_data)):
		if i==0:
			text.append('Best Fitness: \n')
		if i==1:
			text.append('Best EOC k: \n')
		if i==2:
			text.append('Best max NPF: \n')
		
		text.append('Input Arrangement: '+str(good_data[i][1])+'\n')
		text.append('EOC k-effective: '+str(round(good_data[i][2],9))+'    ')
		text.append('Max Nodal Peaking Factor: '+str(round(good_data[i][3],8))+'    ')
		text.append('Fitness Value: '+str(good_data[i][4])+'\n')
		for j in range(len(good_data[i][5])):
			text.append(good_data[i][5][j])
		text.append('\n')
	
	return text

# Write the text into a file to store all quadrant changes
true_text_q = Data_Writer(data_init_q,data_q_good,data_q_bad)

file_1 = open("mlrose_results_quadrants.txt", "w+")

new_file_contents = "".join(true_text_q)

file_1.write(new_file_contents)
file_1.close()

# Write the text into a file to store all cross changes
if L%2 == 1:
	true_text_c = Data_Writer(data_init_c,data_c_good,data_c_bad)
	
	file_2 = open("mlrose_results_cross.txt", "w+")

	new_file_contents = "".join(true_text_c)

	file_2.write(new_file_contents)
	file_2.close()

# Write the text into a file to store the best designs by one output variable.
true_text_best = Data_Writer_2(data_init_q,data_best_runs)

file_3 = open("mlrose_results_best.txt","w+")

new_file_contents = "".join(true_text_best)

file_3.write(new_file_contents)
file_3.close()

# Check Results
SQ1 = S1; SQ2 = S2; SC1 = S1; SC2 = S2
CQ1 = C1; CQ2 = C2; CC1 = C1; CC2 = C2
k_q_avg = k_avg; PF_q_avg = PF_avg
k_c_avg = k_avg; PF_c_avg = PF_avg

tally_q = tally; tally_c = tally
sample_count_q = sample_count; sample_count_c = sample_count

print()
print('Pre-Optimization Data:')
print()
print('Quadrant Shuffle Sampling Results:')
print('Number of Good Samples: ',tally_q,' of ',sample_count_q)
#print('Pre-Optimization Time: ',time_pre_end-time_pre_start)
print()
print('Slaving Coefficients: ',SQ1,' & ',SQ2)
print('k-average: ',k_q_avg,'  Empirical coefficient 1: ',CQ1)
print('PF-average:',PF_q_avg,'  Empirical coefficient 2: ',CQ2)
print()
if L%2 == 1:
	print()
	print('Cross Shuffle Sampling Results:')
	print('Number of Good Samples: ',tally_c,' of ',sample_count_c)
	print()
	print('Slaving Coefficients: ',SC1,' & ',SC2)
	print('k-average: ',k_c_avg,'  Empirical coefficient 1: ',CC1)
	print('PF-average:',PF_c_avg,'  Empirical coefficient 2: ',CC2)
	print()
print()
print('Initial Core Design:')
print()
print('Initial EOC k-effective: ',initial_results[0])
print('Initial max node-average peaking factor: ',initial_results[1])
print('Initial Fitness: ',initial_results[2])
print()
print()
print('Optimization Results:')
print()
print('Quadrant Optimizer Results:')
print('Number of iterations: ',len(storage_q_list))
print('Number from Curve:    ',len(history_q))
print('Time to complete: ', time_q_end - time_q_start,'s')
print()
print('Best Pattern: ',best_state_q)
print('Best Fitness: ',best_fitness_q)
print()
if L%2 == 1:
	print()
	print('Cross Optimizer Results:')
	print('Number of iterations: ',len(storage_c_list))
	print('Number from Curve:    ',len(history_c))
	print('Time to complete: ', time_c_end - time_c_start,'s')
	print()
	print('Best Pattern: ',best_state_c)
	print('Best Fitness: ',best_fitness_c)
	print()

print()
print('Final Core Design:')
print()
print('Final EOC k-effective: ',final_results[0])
print('Final max node-average peaking factor: ',final_results[1])
print('Final Fitness: ',final_results[2])
#if L%2 == 1:
#	print('Final Cross Fitness: ',final_results[3])
print()

print()
print('Design Steps:')
print()
print('Original: ')
for i in range(len(core_original_copy)):
	print(core_original_copy[i])
print()
if L%2 == 1:
	print('After Quadrant Shuffle: ')
	print()
	for i in range(len(core_quad_shuffle)):
		print(core_quad_shuffle[i])
	print()
print('Final: ')
print()
for i in range(len(core_optimal)):
	print(core_optimal[i])
print()


# Plot the results.
f1 = plt.figure(1)
plt.plot(range(1,len(history_q)+1),history_q,'r.-',lw=0.5,label='Optimizer Save')
plt.plot(range(1,len(y_new_q_list)+1),y_new_q_list,'b.',lw=0.5,label='Manual Save')
plt.xlabel('ID - Iteration #')
plt.ylabel('Fitness')
#plt.ylim([S1+S2-(S1+S2)/8,S1+S2+(S1+S2)/8])
plt.ylim(bottom = min(y_good_q_data)-0.2, top = max(y_good_q_data)+0.2)
plt.legend()
plt.title('Quadrant Shuffle: Fitness')
f1.show()

if L%2 == 0:
	f2 = plt.figure(2)
	plt.plot(range(1,len(k_new_q_list)+1),k_new_q_list,'r.',lw=0.5,label='k-effective (Manual Tracking)')
	plt.xlabel('ID - Iteration #')
	plt.ylabel('End-of-Cycle k-effective')
	plt.ylim(bottom = min(k_good_q_data)-0.002, top = max(k_good_q_data)+0.002)
	plt.title('Quadrant Shuffle: End-of-Cycle k-effective')
	f2.show()

	f3 = plt.figure(3)
	plt.plot(range(1,len(PF_new_q_list)+1),PF_new_q_list,'b.',lw=0.5,label='Power Peaking Factor (Manual Tracking)')
	plt.xlabel('ID - Iteration #')
	plt.ylabel('Maximum Nodal Peaking Factor')
	plt.ylim(bottom = min(PF_good_q_data)-0.2)
	plt.title('Quadrant Shuffle: Max Node-Average Peaking Factor')
	f3.show()

if L%2 == 1:	
	f4 = plt.figure(4)
	plt.plot(range(1,len(history_c)+1),history_c,'r.-',lw=0.5,label='Optimizer Save')
	plt.plot(range(1,len(y_new_c_list)+1),y_new_c_list,'b.',lw=0.5,label='Manual Save')
	plt.xlabel('ID - Iteration #')
	plt.ylabel('Fitness')
	plt.ylim(bottom = min(y_good_c_data)-0.2, top = max(y_good_c_data)+0.2)
	plt.legend()
	plt.title('Cross Shuffle: Fitness')
	f4.show()
	
	f5 = plt.figure(5)
	plt.axvline(x = len(k_new_q_list))
	plt.plot(range(1,len(k_new_list)+1),k_new_list,'r.',lw=0.5,label='k-effective (Manual Tracking)')
	plt.xlabel('ID - Iteration #')
	plt.ylabel('End-of-Cycle k-effective')
	plt.ylim(bottom = min(k_good_data)-0.002, top = max(k_good_data)+0.002)
	plt.title('Quadrant & Cross Shuffle: End-of-Cycle k-effective')
	f5.show()
	
	f6 = plt.figure(6)
	plt.axvline(x = len(PF_new_q_list))
	plt.plot(range(1,len(PF_new_list)+1),PF_new_list,'b.',lw=0.5,label='Power Peaking Factor (Manual Tracking)')
	plt.xlabel('ID - Iteration #')
	plt.ylabel('Maximum Nodal Peaking Factor')
	plt.ylim(bottom = min(PF_good_data)-0.2)
	plt.title('Quadrant & Cross Shuffle: Max Node-Average Peaking Factor')
	f6.show()
	
	# Concatenated plots
	f7 = plt.figure(7)
	plt.axvline(x = len(history_q), label='Start of Cross Optimizer')
	plt.plot(range(1,len(history_c)+len(history_q)+1),np.concatenate((history_q,history_c)),'r.-',lw=0.5,label='Optimizer Save')
	plt.plot(range(1,len(y_new_list)+1),y_new_list,'b.',lw=0.5,label='Manual Save')
	plt.xlabel('ID - Iteration #')
	plt.ylabel('Fitness')
	plt.ylim(bottom = min(y_good_data)-0.2, top = max(y_good_data)+0.2)
	plt.legend()
	plt.title('Quadrant & Cross Shuffle: Fitness')
	f7.show()


plt.show()
