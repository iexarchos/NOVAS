from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import os,sys,time
import options,util
import json
from config import get_config
from graphs.get_graph import get_graph
from systems.get_system import get_system
from ipdb import set_trace as debug
import torch
import os
from time import time
from datetime import datetime
from utils.plotting import *
import pdb
from adaptive_stochastic_search import ad_stoch_search

opt = options.set()

def main():
	problem_name = opt.problem_name
	config = get_config(problem_name)
	if config.use_gpu:
		print("Setting default tensor type to CUDA tensor with device:", config.device)
		torch.set_default_tensor_type('torch.cuda.DoubleTensor')
		torch.cuda.set_device(config.device)
	else:
		torch.set_default_tensor_type('torch.DoubleTensor')

	#Create save path
	stamp = datetime.fromtimestamp(time()).strftime('%Y%m%d-%H%M%S')  # Experiment time stamp
	if problem_name == "InvertedPendulum":
		plot_path = "plots/InvertedPendulum/" + opt.graph + '/' + stamp + "/"
		if not os.path.exists(plot_path):
			os.makedirs(plot_path)
	elif problem_name == "CartPole":
		plot_path = "plots/CartPole/" + opt.graph + '/' + stamp + "/"
		if not os.path.exists(plot_path):
			os.makedirs(plot_path)
	elif problem_name == "Finance":
		plot_path = "plots/Finance/" + opt.graph + '/' + stamp + "/"
		if not os.path.exists(plot_path):
			os.makedirs(plot_path)
	else:
		raise KeyError("Required system not found")
	root_path = os.getcwd() + '/'

	if opt.load_timestamp is not None:
		loadpath = root_path + "plots/" + problem_name + "/" + opt.graph + "/" + opt.load_timestamp + "/" 
	else:
		loadpath = None

	torch.manual_seed(config.random_seed)
	np.random.seed(config.random_seed) #numpy random seed
	np.set_printoptions(precision=10, linewidth=1000)

	dynamics = get_system(problem_name, config.x0, config.delta_t, \
		config.num_time_interval, config.sigma, config.target_states,\
		config.running_cost, config.float_type, savepath=root_path+plot_path, loadpath=loadpath)

	graph = get_graph(opt.graph, config, dynamics, True, ad_stoch_search)

	if opt.test_only=="True":
		if opt.load_timestamp is not None:
			print("Generating test noise ...")
			dw_test = torch.randn((config.test_size, dynamics._state_dim,\
								  config.num_time_interval))*np.sqrt(config.delta_t)			
			full_load_path = root_path + "plots/" + problem_name + "/" + opt.graph + "/" + opt.load_timestamp + "/" 
			print("Building model and testing for saved model parameters in ", full_load_path)
			graph.build(savepath=root_path+plot_path)
			print("Testing plots will be stored in ", root_path + plot_path)	
			# Building the model will initialize the weights. Call load model will overwrite those weights:
			graph.load_model(load_path=full_load_path)	
						
			# test the model: 
			traj, controls, traj_costs = graph.test(load_path=full_load_path, dw_test=dw_test)
			if dynamics._sys_name is not "Finance":
				traj = np.array(traj)
			controls = np.array(controls)
			np.savez(plot_path + 'traj_costs', traj_costs=traj_costs)			
		else:
			raise KeyError("In test only mode, you must pass the load_timestamp command line argument.")
	elif opt.test_only=="False" and opt.load_timestamp is not None:
		dw_test = torch.randn((config.test_size, dynamics._state_dim,\
							  config.num_time_interval))*np.sqrt(config.delta_t)		
		full_load_path = root_path + "plots/" + problem_name + "/" + opt.graph + "/" + opt.load_timestamp + "/" 
		print("Building model and resuming training for saved model parameters in ", full_load_path)
		graph.build(savepath=root_path+plot_path)
		print("New model and plots after testing will be stored in ", root_path + plot_path)
		graph.load_model(load_path=full_load_path)
		valid_total_losses, valid_terminal_losses = graph.train(root_path+plot_path)

		#Test the model 
		traj, controls, traj_costs = graph.test(root_path+plot_path, dw_test=dw_test)
		if dynamics._sys_name is not "Finance":
			traj = np.array(traj)
		controls = np.array(controls)
		plot_loss(valid_total_losses, valid_terminal_losses, plot_path)
		np.savez(plot_path + 'loss_data', valid_loss=valid_total_losses, term_loss=valid_terminal_losses)
		np.savez(plot_path + 'traj_costs', traj_costs=traj_costs)			
	else:
		dw_test = torch.randn((config.test_size, dynamics._state_dim,\
							  config.num_time_interval))*np.sqrt(config.delta_t) 		
		print("Training model from scratch. Model and plots will be saved in ", root_path + plot_path)
		graph.build(savepath=root_path+plot_path)
		valid_total_losses, valid_terminal_losses = graph.train(root_path+plot_path)

		#Test the model		
		traj, controls, traj_costs = graph.test(root_path+plot_path, dw_test=dw_test)
		if dynamics._sys_name is not "Finance":
			traj = np.array(traj)
		controls = np.array(controls)
		plot_loss(valid_total_losses, valid_terminal_losses, plot_path)		
		np.savez(plot_path + 'loss_data', valid_loss=valid_total_losses, term_loss=valid_terminal_losses)
		np.savez(plot_path + 'traj_costs', traj_costs=traj_costs)			

	#Plot system specific state trajectories:
	if opt.problem_name == "InvertedPendulum":
		plot_InvertedPendulum(traj, controls, config.target_states, config.num_time_interval, \
			graph._delta_t, root_path+plot_path, dynamics)
	elif opt.problem_name == "CartPole":
		plot_CartPole(traj, controls, config.target_states, config.num_time_interval, \
			graph._delta_t, root_path+plot_path, dynamics)
	elif opt.problem_name == "Finance":
		plot_Finance(traj, controls, config.target_states, config.num_time_interval, \
			graph._delta_t, root_path+plot_path, dynamics)
	else:
		raise KeyError("Required system not found")

	# Saving the configuration
	parameters = dict()
	for name in dir(config):
		if not (name.startswith('__') or name == 'target_states' or name == 'float_type' \
				or name == 'kappa' or name == 'alpha'):
			parameters[name] = getattr(config, name)
	if opt.problem_name == "Finance":
		parameters['State_Cost'] = None
		parameters['Control_Cost'] = None
	else:
		parameters['State_Cost'] = dynamics.Q
		parameters['Control_Cost'] = dynamics.R
	parameters['kappa'] = config.kappa.item()
	parameters['alpha'] = config.alpha.item()
	with open('{}_config.json'.format(plot_path + problem_name), 'w') as outfile:
		json.dump(parameters, outfile, indent=2)

	if dynamics._sys_name is not "Finance":
		np.savez(plot_path + 'data', state=traj, control=controls)
	print("Plots saved at " + plot_path)


main()
