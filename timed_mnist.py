import numpy as np
import GPflow
#import climin
import sys
#import cPickle as pickle
import time
import os.path
import tensorflow as tf
import time
import consts

try:
	import multiclassLikelihood_GPy
	import GPy
except:
	pass

def getGPflowModel( X_train, Y_train, initZ, batch_size, nClasses, num_threads ):
	def getKernel():
		k = GPflow.kernels.RBF(X_train.shape[1], ARD=False) + GPflow.kernels.White(1, 1e-3)
		return k
	if num_threads==-1:
		session_config = None
	else:
		session_config = tf.ConfigProto(intra_op_parallelism_threads=num_threads,inter_op_parallelism_threads=num_threads)
	m_vb = GPflow.svgp.SVGP(X=X_train, Y=Y_train.astype(np.int64), kern=getKernel(), likelihood=GPflow.likelihoods.MultiClass(nClasses), num_latent=nClasses, Z=initZ.copy(), minibatch_size=batch_size, whiten=False, session_config=session_config)
	m_vb.likelihood.invlink.epsilon = 1e-3
	m_vb.likelihood.fixed=True
	return m_vb

def getGPflowTimedFunction( model, optimizer, n_iterations):
	opt_step = model._compile(optimizer)
	def timed_function():
		for iteration_index in range(n_iterations):
			model._session.run(opt_step, feed_dict=model.get_feed_dict())
	return timed_function
	
def getGPyModel( X_train, Y_train, initZ, batch_size ):
	def getKernel():
		k = GPy.kern.RBF(X_train.shape[1], ARD=False) + GPy.kern.White(1, 1e-3)
		return k
	lik = multiclassLikelihood_GPy.Multiclass()
	m_vb = GPy.core.SVGP(X=X_train, Y=Y_train, kernel=getKernel(), likelihood=lik.copy(), num_latent_functions=10, Z=initZ.copy(), batchsize=batch_size)
	m_vb.likelihood.delta.fix(1e-3)
	return m_vb
	
def getGPyTimedFunction( model, n_iterations ):
	def timed_function():
		for iteration_index in range(n_iterations):
			model.stochastic_grad( model.optimizer_array) 
	return timed_function

def timeFunction(function):
	t0 = time.time()
	function()
	t1 = time.time()
	return t1-t0

def sharedCode():
	mnist_file_name = consts.mnist_file_name

	np.random.seed(0)
	num_inducing = 500
	nClasses = 10

	vb_batchsize = 1000

	npy = '.npy'
	X_train, Y_train, X_test, Y_test = np.load('Xtrain'+npy), np.load('Ytrain'+npy), np.load('Xtest'+npy), np.load('Ytest'+npy)
	X_train = X_train.reshape(X_train.shape[0], -1)
	X_test = X_test.reshape(X_test.shape[0], -1)

	#scale data
	X_train = X_train/255.0
	X_train = X_train*2. - 1.
	X_test = X_test/255.0
	X_test = X_test*2. - 1.
	
	initZ = np.load('initZ.npy')
	return X_train, Y_train, initZ, vb_batchsize, nClasses

def runExperiments(setting,flow_num_threads):
	#GPy num threads needs to be set as the environment variable. OMP_NUM_THREADS
	if setting=='GPflow':
		use_GPflow=True
		use_GPy=False
	elif setting=='GPy':
		use_GPflow=False
		use_GPy=True
	else:
		raise IOError
	
	X_train, Y_train, initZ, vb_batchsize, nClasses = sharedCode()
				
	n_timed_iterations = consts.num_timed_iterations
	
	if use_GPflow:
		GPflow_model = getGPflowModel( X_train, Y_train, initZ, batch_size=vb_batchsize, nClasses=nClasses, num_threads = flow_num_threads)
		tf_optimizer = tf.train.AdadeltaOptimizer(learning_rate=1e-3, rho=0.9, epsilon=1e-4, use_locking=True)
		GPflow_timed_function = getGPflowTimedFunction( GPflow_model, tf_optimizer, n_timed_iterations)
		GPflow_time = timeFunction( GPflow_timed_function )
		print(GPflow_time)
	
	if use_GPy:
		GPy_model = getGPyModel( X_train, Y_train, initZ, batch_size=vb_batchsize )
		GPy_timed_function = getGPyTimedFunction( GPy_model, n_timed_iterations )
		timed_process = lambda x : timeFunction( GPy_timed_function )
		GPy_time = timeFunction( GPy_timed_function )
		print(GPy_time)

if __name__ == '__main__':
	runExperiments(sys.argv[1],int(sys.argv[2]))
