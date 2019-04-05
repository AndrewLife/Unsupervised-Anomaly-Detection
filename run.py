
import argparse
import sys
import matplotlib
import numpy as np
import lib.helper as hp
import lib.scaler as scaler
import lib.dir_manager as dm
import lib.morisita_index as mi
import lib.grapher
import lib.training
import os
import collections
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt

if __name__ == '__main__':
	# Parse CLI arguments
	parser = argparse.ArgumentParser(description='Perform data anomaly detection with LCE')
	parser.add_argument(dest='dataset', type=str, help='Choose: AAPL, GOOG, FB, IBM')
	parser.add_argument(dest='sample_size', type=int, help='the size of a training data')
	parser.add_argument(dest='end_sub_sample', type=int, help='maximum data to observe')
	args = parser.parse_args()

	# Index of first sample to check
	sample_start = 0
	
	# Number of samples per bucket
	sample_size = args.sample_size
	
	# Number samples to parse from the dataset
	cap_data = args.end_sub_sample

	dims = []

	m = 2
	i = 1
	benchmarks = 15
	result_list = []

	# Read comma-separated twitter dataset
	dataset = hp.csv_extraction('./nab/realTweets/realTweets/Twitter_volume_'+str(args.dataset)+'.csv',1)
	# Get the list of all timestamps
	timestamp = hp.get_col(dataset, 0)
	# Get the list of all tweetcounts in a given 5-second interval
	tweetCount = hp.get_col(dataset, 1)

	# arr_month -> month of each tweet
	# arr_date  -> day of each tweet
	# arr_time  -> start-time for 5-second interval
	# arr_tweet -> Number of tweets in given 5-second interval
	arr_month, arr_date, arr_time, arr_tweet = hp.disect_dimensions(timestamp, tweetCount, 0, cap_data+2)

	arr_tweet = list(map(int, arr_tweet))
	dims.append(arr_tweet)


	# Initialize the range of interval to observe data.
	# For example, if sample_size = 100, we start observe sample_size+1 because
	# sample_size has a stable data we trust for cluster. cap_data
	# represents the cap of data we want to examine.
	total_timestamp = list(range(sample_size+1, cap_data+1))
	mid_graph = max(dims[0][sample_size+1:cap_data+2]) / 2

	main_bucket_size = [0, sample_size]
	observe_1_bucket_size = [0, sample_size + 1]
	observe_2_bucket_size = [0, sample_size + 2]

	while(i < cap_data - sample_size + 1):
		print('\n\n\n============= Timestamp Number: ', str(i+sample_size), ' =============')
		# Definted window's sizes
		main_bucket = [sample_start+i, sample_size+i]
		observe_1_bucket = [sample_start+i, sample_size+i +1]
		observe_2_bucket = [sample_start+i, sample_size+i +2]

		# Get each window into 3 instances
		main_dims = mi.get_sub_dims(dims, main_bucket)
		observe_1_dims = mi.get_sub_dims(dims, observe_1_bucket)
		observe_2_dims = mi.get_sub_dims(dims, observe_2_bucket)

		# Pre-define the scale for this current iteration 
		# We take max and min from all 3 windows at this iteration because
		# 	we do not want to shift around the data within the interval.
		#	Remember, LCE is purely based on data stabality on the dimension,
		#	thus, NO data should be rescaled or move when comparison between
		#	windows happen.
		all_min = min(min(main_dims[0]), min(observe_1_dims[0]), min(observe_2_dims[0]))
		all_max = max(max(main_dims[0]), max(observe_1_dims[0]), max(observe_2_dims[0]))


		# Scale 3 windows based on min and max of their 3 windows.
		scaled_main = scaler.scale_data(main_dims[0], all_min, all_max)
		scaled_obs_1 = scaler.scale_data(observe_1_dims[0], all_min, all_max)
		scaled_obs_2 = scaler.scale_data(observe_2_dims[0], all_min, all_max)

		print('Max observe data : ', observe_2_dims)
		# print('Scaled Main      : ', scaled_main)
		# print('Scaled Obs 1     : ', scaled_obs_1)
		# print('Scaled Obs 2     : ', scaled_obs_2)

		print('All min: ', all_min)
		print('All max: ', all_max)

		print()

		# Extra, no need to worry about this for now
		MI_main = mi.MINDID_Multi_Dims([scaled_main], m, 15, main_bucket_size)
		MI_obs_1 = mi.MINDID_Multi_Dims([scaled_obs_1], m, 15, observe_1_bucket_size)
		MI_obs_2 = mi.MINDID_Multi_Dims([scaled_obs_2], m, 15, observe_2_bucket_size)

		# Removing duplicates so that each precise number is unique and does not
		#	cause infinite LCE value
		main_dims = [list(dict.fromkeys(scaled_main))]
		observe_1_dims = [list(dict.fromkeys(scaled_obs_1))]
		observe_2_dims = [list(dict.fromkeys(scaled_obs_2))]

		# print('Non dupe Main      : ', main_dims)
		# print('Non dupe Observe 1 : ', observe_1_dims)
		# print('Non dupe Observe 2 : ', observe_2_dims)

		# Define bucket sizes after duplicates have been removed
		main_bucket_size = [0, len(main_dims[0])]
		observe_1_bucket_size = [0, len(observe_1_dims[0])]
		observe_2_bucket_size = [0, len(observe_2_dims[0])]

		
		# Compute the score of LCE, this will result in list of cluster sum
		CS_main_LCE = mi.MINDID_Multi_Dims_Norminator(main_dims, m, 10, main_bucket_size)
		CS_obs_1_LCE = mi.MINDID_Multi_Dims_Norminator(observe_1_dims, m, 10, observe_1_bucket_size)
		CS_obs_2_LCE = mi.MINDID_Multi_Dims_Norminator(observe_2_dims, m, 10, observe_2_bucket_size)

		# Get the LCE index
		t0_LCE = hp.compute_LCE(CS_main_LCE)
		t1_LCE = hp.compute_LCE(CS_obs_1_LCE)
		t2_LCE = hp.compute_LCE(CS_obs_2_LCE)


		# print('MI_t0: ', CS_main_LCE[0])
		# print('MI_t1: ', CS_obs_1_LCE[0])
		# print('MI_t2: ', CS_obs_2_LCE[0])


		print('#######################\n')


		print('Length t0: ', len(main_dims[0]))
		print('Length t1: ', len(observe_1_dims[0]))
		print('Length t2: ', len(observe_2_dims[0]))

		print('t0 LCE: ',CS_main_LCE[0])
		print('t1 LCE: ',CS_obs_1_LCE[0])
		print('t2 LCE: ',CS_obs_2_LCE[0])

		print()
		print('#######################\n')


		# Start codition
		if( (len(main_dims[0]) == len(observe_1_dims[0]) or len(observe_1_dims[0]) == len(observe_2_dims[0])) or
			(CS_main_LCE[0][t0_LCE] < CS_obs_1_LCE[0][t1_LCE] or CS_obs_1_LCE[0][t1_LCE] < CS_obs_2_LCE[0][t2_LCE])):
			result_list.append(0)
		else:
			result_list.append(mid_graph)

		print('Result', result_list)

		i += 1

	N = 0
	A = 0

	for d in result_list:
		if(d == 0):
			N += 1
		else:
			A += 1

	print("=== Total classification ===")
	print('Normal: ', N)
	print('Anomaly: ', A)

	fig = plt.figure()

	# Plot anomalies
	result, = plt.plot(total_timestamp, result_list, '^', markersize=np.sqrt(10.), c='r')
	
	# Plot 'normal' data
	origin, = plt.plot(total_timestamp, dims[0][sample_size:cap_data], 'o', markersize=np.sqrt(10.), c='b')
	
	plt.title('Anomaly Detection on NAB on '+ str(args.dataset))
	plt.ylabel('Tweet Numbers')
	plt.xlabel('Timestamp')
	plt.legend([result, origin], ['Anomaly', 'Data'])
	
	dm.mkdir('results')
	fig.savefig('./results/'+str(args.dataset)+"_"+str(sample_size)+"_stables", dpi=500)
	plt.clf()
	
