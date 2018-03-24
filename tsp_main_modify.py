import numpy
from numpy import ndarray
import itertools
import matplotlib.pyplot as plt
import pickle
import simplejson
import time
import json

def tsp_greedy():
	global city_dists
	global city_size
	total_greedy = 0.0
	greedy_dist = city_dists
	for ii in range(0, city_size):
		for jj in range(0, city_size):
			if ii == jj:
				greedy_dist[ii][jj] = float('inf')

	#print greedy_dist			
	CitiesList = set(range(city_size))
	#print CitiesList
	last_city = 0 # initialize the City
	greedy_route = []
	greedy_route.append(0)
	CitiesList = CitiesList.difference(set([last_city]))
	while CitiesList != set():
		tmp_dist = float('inf')
		for x in CitiesList:
			if greedy_dist[last_city][x] < tmp_dist:
				tmp_dist = greedy_dist[last_city][x]
				tmp_city = x
		CitiesList = CitiesList.difference([tmp_city])	
		total_greedy = total_greedy + tmp_dist
		greedy_route.append(tmp_city)
	total_greedy = total_greedy + city_dists[0][tmp_city]	
	greedy_route.append(0)	
	#print 'Greedy Distance is: ', total_greedy
	return greedy_route, total_greedy		





def subsetToInt(tmp_subset):
	# This function transforms the subset into an index, used in B_matrix, L_matrix construction
	global city_size
	int_string = '0'*(city_size-1)
	int_list = list(int_string)
	for ele in tmp_subset:
		int_list[ele-1] = '1'
	int_string = ''.join(int_list)
	# print type(int_string), int_string	
	return int(int_string, 2)

def BestDP(s, X, t):
	# This is a recursive function that calcualte: B(s, X, t)
	global city_dists
	global B_matrix
	if X == set():
		return city_dists[s][t] # If X is an empty set, return Best(s,X,t) = distance between s <---> t.
	else:
		return_val = float('inf')
		for x in X:
			#print x, type(x)
			#############################################################################
			# This implementation is actually very inefficient, however to enable trace back extra effort is required
			X_tmp = X
			new_X = X_tmp.difference(set([x]))
			tmp_val = BestDP(s, new_X, x) + city_dists[x][t]
			if tmp_val < return_val:
				return_val = tmp_val
				city_before = x
		idxBsub = subsetToInt(X)
		if B_matrix[idxBsub][t-1] > return_val:
			#print 'Executed?'
			B_matrix[idxBsub][t-1] = return_val
			L_matrix[idxBsub][t-1] = city_before
		# if B_matrix[idxBsub][idxBx-1] > return_val:
		# 	print 'ever executed?' 	
		# 	B_matrix[idxBsub][idxBx-1] = return_val	
		#################################################################################	
		return return_val # Return the minimum values found

def tsp_dp(city_size): # This function acts as the main() of the Dynamic Programming Version of TSP
	global city_dists
	Cities = set(range(city_size))
	#print 'Cities:', Cities
	remaining_City = Cities.difference(set([0]))
	global_min = float('inf')
	for t in remaining_City:
		# print t, type(t)
		global_tmp = BestDP(0, remaining_City, t) + city_dists[0][t]
		if global_tmp < global_min:
			global_min = global_tmp
	return global_min		
	
def Best_efficient(s, city_remain, t, city_dists): # This is a very efficient implemenation proposed by MIT
	#global city_dists
	if city_remain == set():
		return city_dists[s][t]
	else:
		return min([Best_efficient(s, city_remain.difference(set([x])), x, city_dists) + city_dists[x][t] for x in city_remain])	


def tsp_dp_efficient(): # This is a very efficient implemenation proposed by MIT
	global city_size
	global city_dists
	Cities = set(range(city_size))
	return min([Best_efficient(0, Cities.difference(set([0,t])), t, city_dists) + city_dists[0][t] for t in Cities.difference(set([0]))])	

def tsp_optimal(city_size, city_dists): # This function acts as a main for the Brute Force Version of TSP
	all_routes = list(itertools.permutations(range(0,city_size))) 
	
	vec_dist = numpy.matrix(city_dists.reshape((city_size,city_size)))
	max_dis = vec_dist.max()

	optimal_len = max_dis * city_size

	#print 'starting max length is', optimal_len
	optimal_route = []
	#print type(all_routes)
	# f = open("route.txt", "w")
	# simplejson.dump(all_routes, f)
	# f.close
	# # fp = open("test.txt", "w")
	# # for a_route in all_routes:
	# # 	sr = str(a_route)
	# # 	print type(sr), sr
	# # 	fp.write("%s\n", sr)
	# # fp.close()	
	# with open("route.txt") as f:
	# 	for line in f:
	# 		print line, type(line)

	for single_route in all_routes:
		single_len = float(0)
		current_path = []
		for ii in range(1, city_size):
			origin = single_route[ii-1]
			destination = single_route[ii]
			current_path.append(origin)
			#print 'Origin:', origin, 'destination:', destination , 'distances:', city_dists[origin][destination]
			single_len = single_len + city_dists[origin][destination]
			if (ii == city_size -1):
				origin = single_route[ii]				
				destination = single_route[0]
				current_path.append(origin)
				current_path.append(destination)
				#print 'Origin:', origin, 'destination:', destination , 'distances:', city_dists[origin][destination]
				single_len = single_len + city_dists[origin][destination]

		#print 'Single length for route', current_path,single_route, 'is', single_len	
		if (single_len < optimal_len):
			optimal_len = single_len
			optimal_route = current_path

		# del origin
		# del destination
		# del single_route
		# del current_path
		# del single_len
		# gc.collect()
	
	#print 'The Optimal Length is:', optimal_len	
	#print 'The Optimal Route is:', optimal_route
	return [optimal_route, optimal_len]

def plot_greedy(greedy_route): # This function illustrate the optimal path
	#print type(optimal_route)
	#print 'The Optimal Length is:', optimal_len	
	#print 'The Optimal Route is:', optimal_route
	global city_coordinates
	plt.plot(city_coordinates[0], city_coordinates[1], 'ro')	
	plt.axis([0, 100, 0, 100])
	for ii in range(0, len(greedy_route)-1):
		id1 = greedy_route[ii]
		id2 = greedy_route[ii + 1]
		x1 = city_coordinates[0][id1]
		x2 = city_coordinates[0][id2]
		y1 = city_coordinates[1][id1]
		y2 = city_coordinates[1][id2]
		plt.plot([x1, x2], [y1, y2], 'r--')
		plt.xlabel('Geological Coordinates x of Cities')
		plt.ylabel('Geological Coordinates y of Cities')
		plt.title('Optimal Solution for TSP, \n Assuming Straight Line Distances Between Cities')
		plt.grid(True)

# use keyword args


def plot_optimal(optimal_route): # This function illustrate the optimal path
	#print type(optimal_route)
	#print 'The Optimal Length is:', optimal_len	
	#print 'The Optimal Route is:', optimal_route
	global city_coordinates
	plt.plot(city_coordinates[0], city_coordinates[1], 'ro')	
	plt.axis([0, 100, 0, 100])
	for ii in range(0, len(optimal_route)-1):
		id1 = optimal_route[ii]
		id2 = optimal_route[ii + 1]
		x1 = city_coordinates[0][id1]
		x2 = city_coordinates[0][id2]
		y1 = city_coordinates[1][id1]
		y2 = city_coordinates[1][id2]
		plt.plot([x1, x2], [y1, y2], 'b--')
		plt.xlabel('Geological Coordinates x of Cities')
		plt.ylabel('Geological Coordinates y of Cities')
		plt.title('Optimal Solution for TSP, \n Assuming Straight Line Distances Between Cities')
		plt.grid(True)

# use keyword args
	plt.show() # Finally, show the image


# Random Initialize the coordinates of cities
#################################################################################################
city_input = input('Please type the cities that salesman want to travel...:  ')
city_size = int(city_input)
city_coordinates = numpy.zeros((2,city_size))
for idxj in range(0, city_size):
	for idxi in range(0,2):
		city_coordinates[idxi,idxj] = 100*numpy.random.random_sample()
	x_coor = city_coordinates[0,idxj]
	y_coor = city_coordinates[1,idxj]
	
#print city_coordinates.transpose()
	
city_dists = numpy.zeros((city_size,city_size))

# Generating the distances between random two cities
for ii in range(0, city_size):
	for jj in range(0, city_size):
		city_dists[ii][jj] = ((city_coordinates[0][ii] - city_coordinates[0][jj]) ** (2) + (city_coordinates[1][ii] - city_coordinates[1][jj]) ** (2))**(0.5)

total_sub = subsetToInt(set(range(city_size)))
#print total_sub
B_matrix = numpy.zeros((total_sub+1,city_size-1))
L_matrix = numpy.zeros((total_sub+1,city_size-1))

for ii in range(0, total_sub+1):
	for jj in range(0, city_size-1):
		B_matrix[ii][jj] = float('inf')
		L_matrix[ii][jj] = float('inf')

#print B_matrix.shape
# B_matrix[total_sub][1] = 1
# print B_matrix
#################################################################################
start_time = time.time()
optimal_dp = tsp_dp(city_size)
# print '-----------------------------------------------------------------------------'
# print 'The optimal distances obatined using dp is:', optimal_dp 
# print 'The running time using DP program is: ', time.time() - start_time, 'seconds.'
# print '-----------------------------------------------------------------------------'
#print B_matrix, L_matrix

final = float('inf')
for ii in range(0, total_sub+1):
	for jj in range(0, city_size-1):
		Bt = B_matrix[ii][jj]
		idxt = L_matrix[ii][jj]
		try:
			final = Bt + city_dists[0][int(idxt)]
			if final == optimal_dp:
				#print 'candaidate final distance: ', final
				#print 'We found it!', final, optimal_dp
				final_end = int(idxt)
				break
		except:
			mood = 'happy' 

#print 'final end point found: ', final_end		

# Finally constrct the route obtained using dynamic programing
dp_route = [0]
dp_route.append(int(final_end))
#print dp_route
C_back = set(range(city_size))
C_Back_update = C_back.difference(set([0,final_end]))
#print C_Back_update
last_end = final_end
while C_Back_update != set():
	ele_t = last_end-1
	subsetIdx = subsetToInt(C_Back_update)
	pop_out = L_matrix[subsetIdx][ele_t]
	last_end =  int(pop_out)
	dp_route.append(int(last_end))
	C_Back_tmp = C_Back_update.difference(set([last_end]))
	C_Back_update = C_Back_tmp
	#print dp_route, C_Back_update
dp_route.append(int(0))
#print dp_route, type(dp_route)
#plot_optimal(city_coordinates, dp_route, optimal_dp)

print '-----------------------------------------------------------------------------'
print 'The optimal distances obatined using dp is:', optimal_dp , 'with route: ', dp_route
print 'The running time using DP program is: ', time.time() - start_time, 'seconds.'
print '-----------------------------------------------------------------------------'

dp_efficient_results = tsp_dp_efficient()
print dp_efficient_results

[greedy_route, total_greedy] = tsp_greedy()
plot_greedy(greedy_route)

plot_optimal(dp_route)

def city_map(city_size):
	city_coordinates = numpy.zeros((2,city_size))
	for idxj in range(0, city_size):
		for idxi in range(0,2):
			city_coordinates[idxi,idxj] = 100*numpy.random.random_sample()
	x_coor = city_coordinates[0,idxj]
	y_coor = city_coordinates[1,idxj]
#print city_coordinates.transpose()
	city_dists = numpy.zeros((city_size,city_size))
	# Generating the distances between random two cities
	for ii in range(0, city_size):
		for jj in range(0, city_size):
			city_dists[ii][jj] = ((city_coordinates[0][ii] - city_coordinates[0][jj]) ** (2) + (city_coordinates[1][ii] - city_coordinates[1][jj]) ** (2))**(0.5)
	return city_dists

for ii in range(5, 10000, 100):
	total = 0
	city_dists = city_map(ii)
	city_size = ii
	start_time = time.time()
	for jj in range(0,5):	
		[greedy_route , greedy_results] = tsp_greedy()
	print ii, (time.time() - start_time)/5


for ii in range(5, 12):
	for jj in range(0,1):
		city_size = ii
		city_dists = city_map(city_size)
		start_time = time.time()
		dp_efficient_results = tsp_dp_efficient()
		dp_time = time.time() - start_time
		start_time = time.time()
		#print 'The computing time of ', city_size, 'cities is: ', time.time() - start_time
		brute_reulst = tsp_optimal(city_size, city_dists)
		brute_time = time.time() - start_time
		start_time = time.time()
		[greedy_route , greedy_results] = tsp_greedy()
		greedy_time = time.time() - start_time
		print city_size, jj, dp_time, dp_efficient_results, brute_time, brute_reulst[1], greedy_time, greedy_results

# start_time = time.time()
# [optimal_route, optimal_len] =  tsp_optimal(city_size, city_dists)
# print 'The running time using Brute Force program is: ', time.time() - start_time, 'seconds.'
# print '-----------------------------------------------------------------------------'
# plot_optimal(city_coordinates, optimal_route, optimal_len)

# print dp_route, type(dp_route)
# print optimal_route, type(optimal_route)
