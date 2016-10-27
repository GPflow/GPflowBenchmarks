from matplotlib import pylab as plt
import numpy as np
import os
import consts
import csv
from matplotlib2tikz import save as save_tikz

x_limits = [0.5,6.5]

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

def plotResults(ax,results,plotFileName,device_type,horizontal_line=False):
	for software in results:
		thread_list = []
		mean_list = []
		std_list = []
		for num_threads in results[software]:
			thread_list.append(num_threads)
			iterationsPerSecond = consts.num_timed_iterations*np.array(results[software][num_threads])**(-1)
			mean_list.append( iterationsPerSecond.mean() )
			std_list.append( iterationsPerSecond.std() )
			label = software+': '+device_type
		if horizontal_line==False:
			ax.errorbar(np.array(thread_list), np.array(mean_list), yerr=np.array(std_list), fmt='o', label=label)
		else:
			assert( len(mean_list)== 1 )
			n_lattice = 100
			x_points = np.linspace( x_limits[0], x_limits[1], n_lattice )
			upper_bound = mean_list[0] + std_list[0] 
			lower_bound = mean_list[0] - std_list[0] 
			cycle = ax._get_lines.color_cycle
			color = next(cycle)
			ax.plot( x_points, np.ones( x_points.shape )*upper_bound, color=color, linestyle='dashed' )
			ax.plot( x_points, np.ones( x_points.shape )*lower_bound, label=label, color=color, linestyle='dashed' )
			
def processResults():
	fig, ax = plt.subplots()
	cpuFileName = os.path.join('timing_results_no_avx', consts.batch_results_file_name)
	cpuResults = parseResultsFile(cpuFileName)
	plotResults(ax,cpuResults,'batch_results_plot','N CPU threads.')

	gpuStandardFileName = os.path.join('timing_results',consts.GPU_results_file_name)
	if os.path.exists(gpuStandardFileName):
		gpuResults = parseResultsFile(gpuStandardFileName)
		plotResults(ax,gpuResults,'batch_results_plot','GPU and all CPU threads.', horizontal_line=True)
		
	#gpuProjectedFileName = os.path.join('TimingData','grothendieck_gpu_hack',consts.GPU_results_file_name)
	#if os.path.exists(gpuProjectedFileName):
	#	gpuResults = parseResultsFile(gpuProjectedFileName)
	#	plotResults(ax,gpuResults,'batch_results_plot','GPU and all CPU threads. Projected', horizontal_line=True)
	plt.xlabel('N CPU threads')
	plt.ylabel('Iterations per second')
	plt.xlim(x_limits)
	plt.legend(loc=5)
	save_tikz('GPflow_timing.tikz',figurewidth='\\figurewidth', figureheight = '\\figureheight') 
	plt.show()

if __name__ == '__main__':
	processResults()
