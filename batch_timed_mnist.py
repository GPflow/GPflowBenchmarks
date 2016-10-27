import sys
import subprocess
import os
import consts

def runBatchExperiments(descriptor):
	if descriptor=="short":
		lower_num_threads = 5
		max_num_threads = 6
		num_repeats = 2
	elif descriptor=="full":
		lower_num_threads = 1
		max_num_threads = 6
		num_repeats = 5
	else:
		raise NotImplementedError

	f= open(consts.batch_results_file_name,'w')
	for repeat_index in range(num_repeats):
		for num_threads in range(lower_num_threads,max_num_threads+1):
			for software in ["GPy","GPflow"]:
				os.environ['OMP_NUM_THREADS'] = str(num_threads)
				text_out = subprocess.check_output(["python","timed_mnist.py",software,str(num_threads)])
				comma = ","
				line = software+comma+str(num_threads)+comma+text_out.decode()
				f.write(line)
	f.close()

if __name__ == '__main__':
	runBatchExperiments(sys.argv[1])
