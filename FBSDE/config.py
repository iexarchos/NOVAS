"""
Parameters for running each experiment.

"""
import numpy as np
import math
import torch

class Config(object):
	batch_size = 128
	test_size = 128
	valid_size = 128
	num_iterations = 3000
	logging_frequency_valid = 100
	logging_frequency_train = 20
	V_init_range = [0, 1]
	Vx_init_range = [-1, 1]
	max_gradient_norm = 5.0
	optimizer_name="Adam" #"Adagrad", "SGD", "RMSprop"
	learning_rate_RAdam=5e-3
	random_seed = 0
	initialization = {'LSTM': 'xavier_normal', 'FC': 'xavier_normal', \
					  'h_init': 'xavier_normal', 'qp_FC': 'xavier_normal'} #xavier_normal, xavier_uniform, zeros 
	float_type = None

	# Non-convex layer optimization parameters:
	n_sample_train = 50
	n_sample_infer = 200
	n_iter_train = 5
	n_iter_infer = 50
	init_sigma = 10.0
	init_mu = 0.0
	opt = 'min'
	use_gpu = True
	if use_gpu:
		device = 'cuda:0'
	else:
		device = 'cpu'
	kappa = torch.Tensor([35.0]).to(device)
	alpha = torch.Tensor([1.0]).to(device)
	shape_func = 'standard_exponential' # soft_ce or standard_exponential
	use_Hessian = False
	train_kappa = False
	train_alpha = False
	use_linesearch = True

class InvertedPendulumConfig(Config):
	logging_frequency_train = 1
	logging_frequency_valid = 10
	
	# For swing-up:
	total_time = 1.5  #for random seed test experiment
	num_time_interval = int(75) #for random seed test experiment
	
	# For balancing:
	# total_time = 1.0  #for random seed test experiment
	# num_time_interval = int(50) #for random seed test experiment	

	delta_t = total_time / float(num_time_interval)
	batch_size = 256
	num_iterations = 20 # for true experiment
	# num_iterations = 500 #for test random seed experiment
	lr_values = 1e-3
	# lr_values = list(np.array([1e-3, 5e-4, 1e-4, 1e-5]))
	hidden_dims = [16, 16]
	V_init_range = [0, 1]

	# Set initial condition:
	x0 = [0.0, 0.0] # for swing up task
	# x0 = [math.pi, 0] # intial condition for balancing task

	sigma = 1.0
	# sigma = 0.1
	weight_V = 1.0
	weight_Vx = 1.0
	weight_V_true = 1.0
	weight_Vx_true = 1.0
	exploration = 1.0
	running_cost = True
	target_states = np.array([[np.pi, 0.0]])
	target_states = np.repeat(target_states,num_time_interval+1,axis=0)
	use_TD_error=False
	input_targets=False
	use_lstm_batch_norm=False
	fixed_lr = True
	max_gradient_norm = 5.0
	random_seed = 49

	message = "using,\n" + \
	str(hidden_dims) + " neurons for hidden layers" + "\n" +\
	"sigma = " + str(sigma) + "\n"+\
	"weight on V = " + str(weight_V) + "\n" + \
	"weight on Vx = " + str(weight_Vx) + "\n" + \
	"weight on true V = " + str(weight_V_true) + "\n" + \
	"weight on true Vx = " + str(weight_Vx_true) + "\n" + \
	"total_time = " + str(total_time) + "\n" + \
	"delta_t = " + str(delta_t) + "\n" + \
	"num_time_interval = " + str(num_time_interval) + "\n" + \
	"exploration = " + str(exploration) + "\n" +\
	"fixed_lr = " + str(fixed_lr) + "\n" + \
	"max_gradient_norm = " + str(max_gradient_norm) + "\n" + \
	"random_seed: " +str(random_seed)+"\n" 

class CartPoleConfig(Config):

	logging_frequency_train = 20
	logging_frequency_valid = 100
	total_time = 1.5  #for random seed test experiment
	num_time_interval = int(75) #for random seed test experiment
	delta_t = total_time / float(num_time_interval)
	batch_size = 128
	test_size = 256
	valid_size = 256
	num_iterations = 6000 #for test random seed experiment
	milestones = [55, 58]
	lr_values = 5e-3
	hidden_dims = [16, 16]
	V_init_range = [0, 1]
	x0 = [0.0, 0.0, 0.0, 0.0] #for swing up task
	sigma = 1.0
	
	# Loss weights:
	weight_V = 1.0
	weight_Vx = 1.0
	weight_Vxx_col = 1.0
	weight_V_true = 1.0 #1.0 must be 0 for Merton Problem!
	weight_Vx_true = 0.0
	weight_Vxx_col_true = 0.0
	use_abs_loss = False
	pre_compute_init = True
	training_with_linesearch = True
	
	exploration = 1.0
	running_cost = True
	target_states = np.array([[0.0, np.pi, 0.0, 0.0]])
	target_states = np.repeat(target_states,num_time_interval+1, axis=0)
	use_TD_error=False
	input_targets=False
	use_lstm_batch_norm=False
	fixed_lr = False
	max_gradient_norm = 5.0
	random_seed = 49

	message = "using,\n" + \
	str(hidden_dims) + " neurons for hidden layers" + "\n" +\
	"sigma = " + str(sigma) + "\n"+\
	"Training batch_size = " + str(batch_size) + "\n"+\
	"weight on V = " + str(weight_V) + "\n" + \
	"weight on Vx = " + str(weight_Vx) + "\n" + \
	"weight on true V = " + str(weight_V_true) + "\n" + \
	"weight on true Vx = " + str(weight_Vx_true) + "\n" + \
	"weight on Vxx_col = " + str(weight_Vxx_col) + "\n" + \
	"weight on true Vxx = " + str(weight_Vxx_col_true) + "\n" + \
	"pre_compute_init? " + str(pre_compute_init) + "\n" + \
	"training_with_linesearch? " + str(training_with_linesearch) + "\n" + \
	"total_time = " + str(total_time) + "\n" + \
	"delta_t = " + str(delta_t) + "\n" + \
	"num_time_interval = " + str(num_time_interval) + "\n" + \
	"exploration = " + str(exploration) + "\n" +\
	"fixed_lr = " + str(fixed_lr) + "\n" + \
	"max_gradient_norm = " + str(max_gradient_norm) + "\n" + \
	"random_seed: " +str(random_seed)+"\n" 


class FinanceConfig(Config):
	logging_frequency_train = 20
	logging_frequency_valid = 100
	total_time = 1 # in years 
	# The unit of time is years because the rate of returns in Primbs' paper is yearly
	# so dt must also have a unit in years. 
	# If you want to step montly, use 1 month = 1/12 years, so dt = 1/12 years
	# If you want to step weekly, use 1 week = 1/52 years, so dt = 1/52 years
	# If you want to step daily, use 1 day = 1/365 years, so dt = 1/365 years
	num_time_interval = total_time * 52 # stepping weekly 
	delta_t = total_time / float(num_time_interval) 
	batch_size = 32
	test_size = 512
	valid_size = 256	
	lr_values = 1e-2
	num_iterations = 5000
	milestones = [40, 45]
	hidden_dims = [16, 16]
	# hidden_dims = [32, 32]
	# hidden_dims = [64, 64]
	y_init_range = [0.0, 1.0]
	x0 = [1.0] # dummy variable to work with rest of code 
	sigma = 1.0 # dummy variable to work with rest of code
	
	# Loss weights:
	weight_V = 1.0
	weight_Vx = 1.0
	weight_Vxx_col = 1.0
	weight_V_true = 1.0 #1.0 must be 0 for Merton Problem!
	weight_Vx_true = 0.0
	weight_Vxx_col_true = 0.0
	use_abs_loss = False
	pre_compute_init = True
	training_with_linesearch = True

	running_cost = True
	exploration = 1.0
	use_TD_error = False
	input_targets = False
	use_lstm_batch_norm = False
	use_Xavier_initializer = True
	target_states = None # For index tracking
	fixed_lr = False
	max_gradient_norm = 5.0
	random_seed = 0

	message = "**** NOT DIFFERENTIATING THROUGH LINESEARCH ********\n" + \
	"using,\n" + \
	str(hidden_dims) + " neurons for hidden layers" + "\n" +\
	"Training batch_size = " + str(batch_size) + "\n"+\
	"Validation batch_size = " + str(valid_size) + "\n"+\
	"Testing batch_size = " + str(test_size) + "\n"+\
	"weight on V = " + str(weight_V) + "\n" + \
	"weight on Vx = " + str(weight_Vx) + "\n" + \
	"weight on true V = " + str(weight_V_true) + "\n" + \
	"weight on true Vx = " + str(weight_Vx_true) + "\n" + \
	"weight on Vxx_col = " + str(weight_Vxx_col) + "\n" + \
	"weight on true Vxx = " + str(weight_Vxx_col_true) + "\n" + \
	"total_time = " + str(total_time) + "\n" + \
	"delta_t = " + str(delta_t) + "\n" + \
	"num_time_interval = " + str(num_time_interval) + "\n" + \
	"exploration = " + str(exploration) + "\n" +\
	"fixed_lr = " + str(fixed_lr) + "\n" + \
	"max_gradient_norm = " + str(max_gradient_norm) + "\n" + \
	"use_abs_loss = " + str(use_abs_loss) + "\n" + \
	"pre_compute_init? " + str(pre_compute_init) + "\n" + \
	"training_with_linesearch? " + str(training_with_linesearch) + "\n" + \
	"random_seed: " +str(random_seed)+"\n" 

def get_config(name):
	try:
		return globals()[name+'Config']
	except KeyError:
		raise KeyError("Config for the required problem not found.")
