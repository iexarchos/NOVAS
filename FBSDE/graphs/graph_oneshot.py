from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from utils.losses import *
import options,util
from collections import OrderedDict
import os,sys,time
from ipdb import set_trace as debug
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
from pdb import set_trace as bp
from adaptive_stochastic_search import *
 
# from barebones_adaptive_stochastic_search import bb_ad_stoch_search
# print("++++++++ Using barebones version of adaptive_stochastic_search.py ++++++++++")

class FBSDE_Model(torch.nn.Module):
	def __init__(self, dynamics, config, savepath=None):
		super(FBSDE_Model, self).__init__()
		self._dynamics = dynamics
		self._config = config
		self._dtype = self._config.float_type
		self.lstm_cells = []
		self.hidden_dims = config.hidden_dims
		input_size = dynamics._state_dim
		for i in range(len(self.hidden_dims)):
			output_size = self.hidden_dims[i]
			self.lstm_cells.append(nn.LSTMCell(input_size, output_size))
			input_size = output_size
		self.lstm_layers = nn.ModuleList(self.lstm_cells)
		self.FC_Vx = nn.Linear(input_size, dynamics.state_dim)
		self.FC_Vxx_col = nn.Linear(input_size, dynamics.state_dim)
		self._V_init = Parameter(torch.rand(1))
		self.h_and_c_list = []
		if self._config.train_kappa:
			self.kappa = Parameter(self._config.kappa)
		else:
			self.kappa = self._config.kappa

		if self._config.train_alpha:
			self.alpha = Parameter(self._config.alpha)
		else:
			self.alpha = self._config.alpha

		for i in range(len(self._config.hidden_dims)):
			h = Parameter(torch.zeros((1,self._config.hidden_dims[i])))
			c = Parameter(torch.zeros((1,self._config.hidden_dims[i])))
			self.h_and_c_list.append((h,c))
			self.register_parameter(name="h_init"+str(i),param=h)
			self.register_parameter(name="c_init"+str(i),param=c)

	def LSTM_layer(self, inputs, qp_control=None):
		x, lstm_states = inputs
		hidden_list = []
		for i in range(len(self.hidden_dims)):
			h_and_c = lstm_states[i]
			h, c = self.lstm_cells[i](x, h_and_c)
			x = h
			hidden_list.append((h,c))
		Vx = self.FC_Vx(x)
		Vxx_col = self.FC_Vxx_col(x)
		return (Vx, Vxx_col, hidden_list)

	def forward(self, dw, save_path, n_sample, n_iter, device, training):
		#Broadcast all initializations to batch size
		bs = dw.shape[0] # batch size
		all_one_vec = torch.ones((bs, 1))
		x = all_one_vec * self._dynamics.x0.unsqueeze(0)
		V = all_one_vec * self._V_init
		traj, controls = [], []		
		h_and_c_list = []
		for i in range(len(self._config.hidden_dims)):
			h_temp, c_temp = self.h_and_c_list[i]
			h_temp = all_one_vec * h_temp
			c_temp = all_one_vec * c_temp
			h_and_c_list.append((h_temp, c_temp))

		traj.append(x)

		if not training and self._config.use_linesearch:
			# print("Using linesearch")
			linesearch_bool=True
		else:
			linesearch_bool=False
			
		#Propagate the network forward in time 
		for t in range(self._config.num_time_interval):
			layer_inputs = (x, h_and_c_list)
			Vx, Vxx_col, h_and_c_list = self.LSTM_layer(layer_inputs)
			f = self._dynamics.hamiltonian
			
			if self._config.pre_compute_init and n_iter>1:
				with torch.no_grad():
					mu_init, _, sigma_init, _ = ad_stoch_search(f, x, Vx, Vxx_col, n_batch=bs, \
								nu=self._dynamics._control_dim, n_sample=n_sample, n_iter=n_iter-1, \
								alpha=self.alpha, kappa=self.kappa, init_sigma=self._config.init_sigma, \
								init_mu=self._config.init_mu, opt=self._config.opt, device=device, \
								shape_func=self._config.shape_func, use_Hessian=self._config.use_Hessian, 
								linesearch=linesearch_bool)
					
				# Perform 1 inner-loop iteration with mu_init and sigma_init that records gradients 
				u, _, _, _ = ad_stoch_search(f, x, Vx, Vxx_col, n_batch=bs, \
								nu=self._dynamics._control_dim, n_sample=n_sample, n_iter=1, \
								alpha=self.alpha, kappa=self.kappa, init_sigma=sigma_init, \
								init_mu=mu_init, opt=self._config.opt, device=device, \
								shape_func=self._config.shape_func, use_Hessian=self._config.use_Hessian, 
								linesearch=linesearch_bool)									
			else:
				u, delta_fmu, avg_sigma, last_iter = ad_stoch_search(f, x, Vx, Vxx_col, n_batch=bs, \
							nu=self._dynamics._control_dim, n_sample=n_sample, n_iter=n_iter, \
							alpha=self.alpha, kappa=self.kappa, init_sigma=self._config.init_sigma, \
							init_mu=self._config.init_mu, opt=self._config.opt, device=device, \
							shape_func=self._config.shape_func, use_Hessian=self._config.use_Hessian, 
							linesearch=linesearch_bool)
			
			V, x = self._dynamics.value_and_dynamics_prop(x, u, V, Vx, dw[:,:,t], t)
			traj.append(x)
			controls.append(u)
		layer_inputs = (x, h_and_c_list)
		Vx, Vxx_col, _ = self.LSTM_layer(layer_inputs)

		return x, V, Vx, Vxx_col, traj, controls
#-------------------------------------------------------------------------------------------------------------------------

		
class FBSDESolverGraphLSTM(object):
	def __init__(self, config, dynamics_model, solver):
		print(util.yellow("Using oneshot graph with lstm"))

		self._config 			= config
		self._dynamics 			= dynamics_model
		self._state_dim 		= dynamics_model.state_dim
		self._control_dim 		= dynamics_model.control_dim
		self._num_time_interval = config.num_time_interval
		self._total_time 		= config.total_time
		self._dtype				= config.float_type
		self._initialization 	= config.initialization
		self._gradient_clip		= config.max_gradient_norm
		self._solver 			= solver

		self._delta_t 			= config.delta_t
		self._sqrt_delta_t		= np.sqrt(self._delta_t)
		self._saver 			= None
		self._random_seed       = config.random_seed

		self._weight_V			= config.weight_V  # weight for value function difference
		self._weight_Vx			= config.weight_Vx  # weight for value function gradient difference
		self._weight_V_true		= config.weight_V_true  # weight for true value function
		self._weight_Vx_true	= config.weight_Vx_true  # weight for true value function gradient
		self._weight_Vxx_col	= config.weight_Vxx_col
		self._weight_Vxx_col_true = config.weight_Vxx_col_true
		self._exploration 		= config.exploration # exploration factor during training
		self._use_TD_error 		= self._config.use_TD_error
		self._input_targets 	= self._config.input_targets
		self._use_lstm_batch_norm = self._config.use_lstm_batch_norm
		self._use_abs_loss = self._config.use_abs_loss
		print(util.yellow("%s" %(self._config.message)))
		ad_stoch_search_params_print ="For ad_stoch_search using,\n" + \
		"n_sample_train = " + str(config.n_sample_train) + "\n"+\
		"n_sample_infer = " + str(config.n_sample_infer) + "\n"+\
		"n_iter_train = " + str(config.n_iter_train) + "\n" + \
		"n_iter_infer = " + str(config.n_iter_infer) + "\n" + \
		"shape_func = " + str(config.shape_func) + "\n" + \
		"use_Hessian = " + str(config.use_Hessian) + "\n" + \
		"kappa = " + str(config.kappa) + "\n" + \
		"alpha = " + str(config.alpha) + "\n" + \
		"use_gpu = " + str(config.use_gpu) + "\n" + \
		"device = " + str(config.device) + "\n" + \
		"train_kappa = " + str(config.train_kappa) + "\n" + \
		"train_alpha = " + str(config.train_alpha) + "\n" + \
		"use_linesearch = " + str(config.use_linesearch) + "\n" + \
		"init_sigma = " + str(config.init_sigma) + "\n" 
		print(util.yellow("%s"%(ad_stoch_search_params_print)))

	def optimizer_init(self, name, params, lr=1e-3, weight_decay=1e-5):
		if name == "Adam":
			opt = optim.Adam(params=params, lr=lr, weight_decay=weight_decay)
		elif name == "Adagrad":
			opt = optim.Adagrad(params=params, lr=lr, weight_decay=weight_decay)
		elif name == "RMSprop":
			opt = optim.RMSprop(params=params, lr=lr, weight_decay=weight_decay)
		elif name == "SGD":
			opt = optim.SGD(params=params, lr=lr, weight_decay=weight_decay)
		else:
			raise keyError("Optimizer not available!")
		if self._config.fixed_lr:
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.999999, patience=100, 
																threshold=1e-2, threshold_mode='rel', min_lr=1e-3)
		else:
			# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, 
			# 													threshold_mode='rel', min_lr=1e-6)			
			print("Using MultiStepLR learning rate decay with milestones = ", self._config.milestones)
			scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=self._config.milestones)
		return opt, scheduler

	def train(self, save_path):
		#Initialize checkpoint saver
		best_loss = None
		
		training_loss = []
		true_loss = []
		start_time	= time.time()
		min_loss=1e10
		lr_used = self._config.lr_values

		Validation_total_losses, Validation_terminal_losses = [], []
		#Generate validation noise:
		dw_valid = torch.randn((self._config.valid_size, self._state_dim, self._num_time_interval)) * self._sqrt_delta_t 

		if self._config.train_kappa:
			print("Kappa = ", self.model.kappa.item())
		if self._config.train_alpha:
			print("Alpha = ", self.model.alpha.item())
		for step in range(self._config.num_iterations+1):
			if step == 0:
				f = open(save_path+"log.txt","a")
				print("%s" %(self._config.message), file=f)
				f.close()
			dw_train = torch.randn((self._config.batch_size, self._state_dim, self._num_time_interval)) * self._sqrt_delta_t 
			train_iter_start_time = time.time()

			if self._config.training_with_linesearch:
				x, V, Vx, Vxx_col, _, _ = self.model(dw_train, save_path,\
													n_sample=self._config.n_sample_train,\
													n_iter=self._config.n_iter_train,\
													device=self._config.device, training=False)
			else:
				x, V, Vx, Vxx_col, _, _ = self.model(dw_train, save_path,\
													n_sample=self._config.n_sample_train,\
													n_iter=self._config.n_iter_train,\
													device=self._config.device, training=True)

			V_true = self._dynamics.terminal_cost(x)
			Vx_true = self._dynamics.terminal_cost_gradient(x)
			Vxx_col_true = self._dynamics.terminal_Vxx_col(x)

			loss, V_dif_loss, V_abs_loss, Vx_dif_loss, Vx_abs_loss, Vxx_dif_loss, Vxx_abs_loss = V_Vx_custom_loss(V_true, V, \
											Vx_true, Vx, \
											Vxx_col_true, Vxx_col,\
											weight_V=self._weight_V, weight_Vx=self._weight_Vx, \
											weight_V_true=self._weight_V_true, weight_Vx_true=self._weight_Vx_true,\
											weight_Vxx_col=self._weight_Vxx_col, \
											weight_Vxx_col_true=self._weight_Vxx_col_true,\
											use_abs_loss=self._use_abs_loss)
			avg_term_cost = torch.mean(V_true).item()
			self.optimizer.zero_grad()
			loss.backward()
			nn.utils.clip_grad_norm_(self.model.parameters(), self._gradient_clip)
			self.optimizer.step()
	
			if (step+1)%self._config.logging_frequency_train==0:
				f = open(save_path+"log.txt","a")
				print("Train Step:%d/%d" % (step+1, self._config.num_iterations) ,", loss=%10.2e," % loss.item(),\
				 "Vdiff=%10.2e," % V_dif_loss.item(), "V=%10.2e," % V_abs_loss.item(), \
				 "Vxdiff=%10.2e," % Vx_dif_loss.item(), "Vx = %10.2e," % Vx_abs_loss.item(), \
				 "Term. Cost=%10.2e" % avg_term_cost, \
				 ",lr:", lr_used, ",time:%2.3fs"%(time.time()-train_iter_start_time), file=f)
				
				print("Step:%d/%d," % (step+1, self._config.num_iterations) ,"loss=%8.2e," % loss.item(),\
				 "dV=%8.2e," % V_dif_loss.item(), "V=%8.2e," % V_abs_loss.item(), "dVx=%8.2e," % Vx_dif_loss.item(), \
				 "Vx=%8.2e," % Vx_abs_loss.item(), "dVxx=%8.2e,"% Vxx_dif_loss.item(), \
				 "Vxx=%8.2e,"% Vxx_abs_loss.item(),\
				 "lr:%.2e"% lr_used, ",time:%2.3fs"%(time.time()-train_iter_start_time))
				f.close()
			#Validation step
			if (step+1)%self._config.logging_frequency_valid==0:
				valid_start_time = time.time()

				with torch.no_grad():
					x_valid, V_valid, Vx_valid, Vxx_col_valid, _, _ = self.model(dw_valid, save_path, \
																		n_sample=self._config.n_sample_infer,\
																		n_iter=self._config.n_iter_infer,\
																		device=self._config.device,\
																		training=False)
				V_true_valid = self._dynamics.terminal_cost(x_valid)
				Vx_true_valid = self._dynamics.terminal_cost_gradient(x_valid)
				Vxx_col_true_valid = self._dynamics.terminal_Vxx_col(x_valid)

				avg_term_cost_valid = V_true_valid.mean().item()

				valid_loss, _, _, _, _, _, _ = V_Vx_custom_loss(V_true_valid, V_valid, \
											Vx_true_valid, Vx_valid, \
											Vxx_col_true_valid, Vxx_col_valid,\
											weight_V=self._weight_V, weight_Vx=self._weight_Vx, \
											weight_V_true=self._weight_V_true, weight_Vx_true=self._weight_Vx_true,\
											weight_Vxx_col=self._weight_Vxx_col, \
											weight_Vxx_col_true=self._weight_Vxx_col_true,\
											use_abs_loss=self._use_abs_loss)
				self.scheduler.step()
				lr_used = self.scheduler.state_dict()['_last_lr'][0]				

				f = open(save_path+"log.txt","a")
				print("Validation Step:%d/%d" % (step+1, self._config.num_iterations),\
				", Validation total loss=%10.2e," % valid_loss.item(), \
				"avg. term. Cost=%10.2e" % avg_term_cost_valid, \
				",time:%2.3fs"%(time.time()-valid_start_time),file=f)
				print("Validation Step:%d/%d" % (step+1, self._config.num_iterations),\
				", Validation total loss=%10.2e," % valid_loss.item(), \
				"avg. term. Cost=%10.2e" % avg_term_cost_valid, ",time:%2.3fs"%(time.time()-valid_start_time),
				"bs:", self._config.valid_size)
				if self._config.train_kappa:
					print("Kappa = ", self.model.kappa.item())
				if self._config.train_alpha:
					print("Alpha = ", self.model.alpha.item())					
				f.close()				

				Validation_total_losses.append(valid_loss.item())
				Validation_terminal_losses.append(avg_term_cost_valid)

				if valid_loss < min_loss:
					min_loss = valid_loss
					f = open(save_path+"log.txt","a")
					print("Lowest Validation loss so far. Saving weights", file=f)
					print("Lowest Validation loss so far. Saving weights")
					f.close()
					torch.save(self.model.state_dict(), save_path+'best_model')

		f = open(save_path+"log.txt","a")
		print("Total time taken for training = ", time.time()-start_time, "s", file=f)
		print("Total time taken for training = ", time.time()-start_time, "s")
		f.close()
		return np.array(Validation_total_losses), np.array(Validation_terminal_losses)

	def init_weights(self, model):
		for name, param in model.named_parameters():
			if param.requires_grad:
				print(name, param.shape,param.device)
				if 'lstm_layers' in name and 'weight' in name:
					if self._initialization['LSTM'] == 'xavier_normal':
						nn.init.xavier_normal_(param)
					elif self._initialization['LSTM'] == 'xavier_uniform':
						nn.init.xavier_uniform_(param)
					elif self._initialization['LSTM'] == 'zeros':
						nn.init.zeros_(param)
					else:
						raise keyError('LSTM weight initialization not available')
				if 'FC' in name and 'weight' in name:
					if self._initialization['FC'] == 'xavier_normal':
						nn.init.xavier_normal_(param)
					elif self._initialization['FC'] == 'xavier_uniform':
						nn.init.xavier_uniform_(param)
					elif self._initialization['FC'] == 'zeros':
						nn.init.zeros_(param)
					else:
						raise keyError('FC weight initialization not available')
				if 'h_init' in name or 'c_init' in name:
					if self._initialization['h_init'] == 'xavier_normal':
						nn.init.xavier_normal_(param)
					elif self._initialization['h_init'] == 'xavier_uniform':
						nn.init.xavier_uniform_(param)
					elif self._initialization['h_init'] == 'zeros':
						nn.init.zeros_(param)
					else:
						raise keyError('h_init weight initialization not available')
				if 'qp_FC' in name and 'weight' in name:
					if self._initialization['qp_FC'] == 'xavier_normal':
						nn.init.xavier_normal_(param)
					elif self._initialization['qp_FC'] == 'xavier_uniform':
						nn.init.xavier_uniform_(param)
					elif self._initialization['qp_FC'] == 'zeros':
						nn.init.zeros_(param)
					else:
						raise keyError('qp_FC weight initialization not available')
				# print(param.data)

	def build(self, savepath=None):
		start_time = time.time()
		self.model = FBSDE_Model(self._dynamics, self._config, savepath=savepath)
		self.init_weights(self.model)
		self.optimizer, self.scheduler = self.optimizer_init(self._config.optimizer_name, self.model.parameters(),\
		 self._config.lr_values)
		print("Total time taken for building the network = ", time.time()-start_time, "s")

	def test(self, load_path, dw_test):
		print("Testing with batch size of", self._config.test_size)
		self.model.load_state_dict(torch.load(load_path+'best_model', map_location=self._config.device))
		if self._config.train_kappa:
			print("Best Kappa = ", self.model.kappa.item())
		if self._config.train_alpha:
			print("Best Alpha = ", self.model.alpha.item())			

		with torch.no_grad():
			_, _, _, _, traj_out, control_out = self.model(dw_test, load_path,\
															n_sample=self._config.n_sample_infer,\
															n_iter=self._config.n_iter_infer,\
															device=self._config.device,\
															training=False)

		if self._dynamics._sys_name == "Finance":
			if self._dynamics.Merton_problem:
				print('Warning: Testing Merton Problem')
				u_const_eq = torch.Tensor([1.0-0.8888888888888888,0.8888888888888888]).unsqueeze(0).repeat(self._config.test_size,1)
			else:
				print("Testing with constant and equal controls ...") 
				u_const_eq = torch.Tensor(np.ones((self._config.test_size, self._dynamics._control_dim)))
			all_one_vec = torch.ones((self._config.test_size, 1))
			traj_const_eq = [all_one_vec * self._dynamics.x0.unsqueeze(0)] # initial state
			for t_ in range(self._config.num_time_interval): # propagate trajectories with constant control
				x_test_eq, _ = self._dynamics.dynamics_prop(traj_const_eq[-1], u_const_eq, dw_test[:,:,t_])
				traj_const_eq.append(x_test_eq)
			traj_const_eq = torch.stack(traj_const_eq).detach().cpu().numpy()

			print("******* Testing with ALL random controls *********")
			u_const_rand = torch.Tensor(np.random.rand(self._config.test_size, self._dynamics._control_dim, \
														self._config.num_time_interval))
			traj_const_rand = [all_one_vec * self._dynamics.x0.unsqueeze(0)] # initial state
			for t_ in range(self._config.num_time_interval): # propagate trajectories with constant control
				x_test_rand, _ = self._dynamics.dynamics_prop(traj_const_rand[-1], \
															  u_const_rand[:,:,t_], dw_test[:,:,t_])
				traj_const_rand.append(x_test_rand) 
			traj_const_rand = torch.stack(traj_const_rand).detach().cpu().numpy()

		# Evaluate trajectory cost:
		costs = []
		for i in range(len(traj_out)-1): # iterate over time dimension: (T+1) steps
			costs.append(self._dynamics.q_t(traj_out[i], i).squeeze(dim=-1)) 
		costs.append(self._dynamics.terminal_cost(traj_out[-1]).squeeze(dim=-1))

		if self._dynamics._sys_name == "Finance":
			traj_output = {"fbsde":torch.stack(traj_out).detach().cpu().numpy(), \
						"const_eq": traj_const_eq,\
						"const_rand": traj_const_rand}
		else:
			traj_output = torch.stack(traj_out).detach().cpu().numpy()	

		control_out = torch.stack(control_out).detach().cpu().numpy()
		costs = torch.stack(costs).detach().cpu().numpy() # shape : (T+1, valid_size)

		return traj_output, control_out, costs

	def load_model(self, load_path):
		print("\nLoading model from ", load_path)
		self.model.load_state_dict(torch.load(load_path+'best_model', map_location=self._config.device))
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def oneshot(config, dynamics_model, lstm, solver):
	if lstm:
		return FBSDESolverGraphLSTM(config, dynamics_model, solver), 'LSTM/'
	else:
		return FBSDESolverGraph(config, dynamics_model), 'FC/'
