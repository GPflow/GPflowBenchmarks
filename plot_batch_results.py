from matplotlib import pylab as plt
import numpy as np
import os
import consts
import csv

def parseResultsFile(fileName):
	file_reader = csv.reader(open(fileName,'r'))
	software_dict = {}
	for row in file_reader:
		software,str_num_threads,str_seconds = row
		num_threads = int(str_num_threads)
		seconds = float(str_seconds)
		if software not in software_dict:
			software_dict[software] = {}
		if num_threads not in software_dict[software]:
			software_dict[software][num_threads] = []
		software_dict[software][num_threads].append( seconds )
	return software_dict

def plotResults(results,plotFileName,device_type):
	for software in results:
		thread_list = []
		mean_list = []
		std_list = []
		for num_threads in results[software]:
			thread_list.append(num_threads)
			iterationsPerSecond = consts.num_timed_iterations*np.array(results[software][num_threads])**(-1)
			mean_list.append( iterationsPerSecond.mean() )
			std_list.append( iterationsPerSecond.std() )
			label = software+' '+device_type
		plt.errorbar(np.array(thread_list), np.array(mean_list), yerr=np.array(std_list), fmt='o', label=label)
			
def processResults():
	plt.figure()
	cpuFileName = consts.batch_results_file_name
	cpuResults = parseResultsFile(cpuFileName)
	plotResults(cpuResults,'batch_results_plot','CPU')

	gpuFileName = consts.GPU_results_file_name
	if os.path.exists(gpuFileName):
		gpuResults = parseResultsFile(gpuFileName)
		plotResults(gpuResults,'batch_results_plot','GPU')
	plt.xlabel('Number of devices')
	plt.ylabel('Iterations per second')
	plt.xlim([0.5,6.5])
	plt.legend()
	plt.show()

if __name__ == '__main__':
	processResults()
