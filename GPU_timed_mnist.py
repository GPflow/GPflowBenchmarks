import sys
import subprocess
import os
import consts

def runGPUExperiments(descriptor):
	if descriptor=="short":
		num_repeats = 2
	elif descriptor=="full":
		num_repeats = 5
	else:
		raise NotImplementedError

	software = "GPflow"
	no_limit = -1
	f= open(consts.GPU_results_file_name,'w')
	for repeat_index in range(num_repeats):
		text_out = subprocess.check_output(["python","timed_mnist.py",software,str(no_limit)])
		comma = ","
		line = software+comma+str(1)+comma+text_out.decode()
		f.write(line)
	f.close()

if __name__ == '__main__':
	runGPUExperiments(sys.argv[1])
