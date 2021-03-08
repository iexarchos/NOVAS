import matplotlib.pyplot as plt
import numpy as np
from pdb import set_trace as bp
import torch 
from matplotlib import ticker as mticker
import seaborn as sns
import pandas as pd

def plot_InvertedPendulum(x, u, targets, num_time_interval, dt, path, dynamics):
	# shape of x: (T+1, batch_size, state_dim)
	# shape of u: (T, batch_size, control_dim) 

	T = np.array(range(num_time_interval+1))*dt # time horizon
	bs = x.shape[1] # batch size

	plt.figure(figsize=(15, 10))
	if x.ndim > 2:
		for i in range(bs):
			traj, = plt.plot(T, x[:, i, 0], 'b', label='output')
	else:
		traj, = plt.plot(T, x[:, 0], 'b', label='output')

	target, = plt.plot(T, np.zeros(len(T)) + targets[:, 0], 'r', label='target')
	plt.legend([traj, target], ['output', 'target'])
	plt.xlabel('Time (s)')
	plt.ylabel('Angle (rad)')
	plt.savefig(path + 'Inverted Pendulum Angle.png')
	plt.clf()

	plt.figure(figsize=(15, 10))

	if x.ndim > 2:
		for i in range(bs):
			traj, = plt.plot(T, x[:, i, 1], 'b', label='output')
	else:
		traj, = plt.plot(T, x[:, 1], 'b', label='output')

	target, = plt.plot(T, np.zeros(len(T)) + targets[:, 1], 'r', label='target')
	plt.xlabel('Time (s)')
	plt.ylabel('Anglular Velocity (rad/s)')
	plt.savefig(path + 'Inverted Pendulum Angular Velocity.png')
	plt.clf()

	T = np.array(range(num_time_interval))*dt
	plt.figure(figsize=(15, 10))
	if u.ndim > 2:
		for i in range(bs):
			traj, = plt.plot(T, u[:,i,0], 'b', label='output')
	else:
		traj, = plt.plot(T, u[:, 0], 'b', label='output') 
	plt.xlabel('Time (s)')
	plt.ylabel('Control')
	plt.savefig(path + 'Control.png')


def plot_CartPole(x, u, targets, num_time_interval, dt, path, dynamics):
	# shape of x: (T+1, batch_size, state_dim)
	# shape of u: (T, batch_size, control_dim) 

	T = np.array(range(num_time_interval+1)) * dt
	bs = x.shape[1] # batch size

	plt.figure(figsize=(15, 10))
	if x.ndim > 2:
		for i in range(bs):
			plt.plot(T, x[:,i,0], 'b', label='output')
	else:
		plt.plot(T, x[:, 0], 'b', label='output')

	plt.xlabel('Time (s)')
	plt.ylabel('Position (m)')
	plt.savefig(path + 'Cart Pole Position.png')
	plt.clf()

	plt.figure(figsize=(15, 10))
	if x.ndim > 2:
		for i in range(bs):
			plt.plot(T, x[:,i,1], 'b', label='output')
	else:
		plt.plot(T, x[:, 1], 'b', label='output')

	target, = plt.plot(T, np.zeros(len(T)) + targets[:, 1], 'r', label='target')
	# plt.legend([traj, target], ['output', 'target'])
	plt.xlabel('Time (s)')
	plt.ylabel('Angle (rad)')
	plt.savefig(path + 'Cart Pole Angle.png')
	plt.clf()

	plt.figure(figsize=(15, 10))
	if x.ndim > 2:
		for i in range(bs):
			plt.plot(T, x[:,i,2], 'b', label='output')
	else:
		plt.plot(T, x[:, 2], 'b', label='output')

	target, = plt.plot(T, np.zeros(len(T)) + targets[:, 2], 'r', label='target')
	# plt.legend([traj, target], ['output', 'target'])
	plt.xlabel('Time (s)')
	plt.ylabel('Velocity (m/s)')
	plt.savefig(path + 'Cart Pole Velocity.png')
	plt.clf()

	plt.figure(figsize=(15, 10))
	if x.ndim > 2:
		for i in range(bs):
			plt.plot(T, x[:,i,3], 'b', label='output')
	else:
		plt.plot(T, x[:, 3], 'b', label='output')

	target, = plt.plot(T, np.zeros(len(T)) + targets[:, 3], 'r', label='target')
	# plt.legend([traj, target], ['output', 'target'])
	plt.xlabel('Time (s)')
	plt.ylabel('Angular Velocity (rad/s)')
	plt.savefig(path + 'Cart Pole Angular Velocity.png')
	plt.clf()

	T = np.array(range(num_time_interval))*dt
	plt.figure(figsize=(15, 10))
	if u.ndim > 2:
		for i in range(bs):
			traj, = plt.plot(T, u[:,i,0], 'b', label='output')
	else:
		traj, = plt.plot(T, u[:, 0], 'b', label='output') 
	plt.xlabel('Time (s)')
	plt.ylabel('Control')
	plt.savefig(path + 'Control.png')


def plot_Finance(traj, u, targets, num_time_interval, dt, path, dynamics):
	
	plt.rcParams.update({'font.size': 20})
	plt.rcParams["axes.labelsize"] = 20

	if type(traj) is dict:
		print("Extracting test trajectories from dictionary ... ")
		x_const_eq = traj['const_eq']
		x_const_rand = traj['const_rand']
		x = traj['fbsde']
	else:
		raise KeyError("Test data must be a dictionary")

	# shape of x: (T+1, batch_size, state_dim)
	# shape of u: (T, batch_size, control_dim) 

	T = np.array(range(num_time_interval+1)) * dt
	bs = x.shape[1] # batch size


	if not dynamics.Merton_problem:
		fbsde_costs = np.zeros((bs,num_time_interval+1))
		const_eq_costs = np.zeros((bs,num_time_interval+1))
		const_rand_costs = np.zeros((bs,num_time_interval+1))
	
	computed_index = np.zeros((bs,num_time_interval+1))
	computed_index_check = np.zeros((bs,num_time_interval+1))

	for i in range(bs):
		S = x[:,i,:-1]
		computed_index[i,:] = np.mean(S, axis=1)

		S_check = x_const_eq[:,i,:-1]
		computed_index_check[i,:] = np.mean(S, axis=1)

		if not dynamics.Merton_problem:
			if dynamics.alternate_cost_func:
				if i==0:
					print("Using Alternate Cost Function")
				fbsde_costs[i,:-1] = dynamics._Q_t * (torch.nn.functional.relu(torch.Tensor(computed_index[i,:-1] \
													- x[:-1,i,-1]))**2).detach().cpu().numpy()
				fbsde_costs[i,-1] = dynamics._Q_T * (torch.nn.functional.relu(torch.Tensor(computed_index[i,-1] \
													- x[-1,i,-1]))**2).detach().cpu().numpy()

				const_eq_costs[i,:-1] = dynamics._Q_t * (torch.nn.functional.relu(torch.Tensor(computed_index[i,:-1] \
													- x_const_eq[:-1,i,-1]))**2).detach().cpu().numpy()
				const_eq_costs[i,-1] = dynamics._Q_T * (torch.nn.functional.relu(torch.Tensor(computed_index[i,-1] \
													- x_const_eq[-1,i,-1]))**2).detach().cpu().numpy()

				const_rand_costs[i,:-1] = dynamics._Q_t * (torch.nn.functional.relu(torch.Tensor(computed_index[i,:-1] \
													- x_const_rand[:-1,i,-1]))**2).detach().cpu().numpy()
				const_rand_costs[i,-1] = dynamics._Q_T * (torch.nn.functional.relu(torch.Tensor(computed_index[i,-1] \
													- x_const_rand[-1,i,-1]))**2).detach().cpu().numpy()

			elif dynamics.softplus_cost_func:
				raise KeyError("Cost computation using different running and terminal costs not implemented for softplus_cost_func")
				if i==0:
					print("Using the soft plus function")
				fbsde_costs[i,:] = dynamics._Q * torch.nn.functional.softplus(torch.Tensor(computed_index[i,:] \
													- x[:,i,-1]), beta=dynamics._beta, \
													threshold=dynamics._threshold).detach().cpu().numpy()
				const_eq_costs[i,:] = dynamics._Q * torch.nn.functional.softplus(torch.Tensor(computed_index[i,:] \
													- x_const_eq[:,i,-1]), beta=dynamics._beta, \
													threshold=dynamics._threshold).detach().cpu().numpy()
				const_rand_costs[i,:] = dynamics._Q * torch.nn.functional.softplus(torch.Tensor(computed_index[i,:] \
													- x_const_rand[:,i,-1]), beta=dynamics._beta, \
													threshold=dynamics._threshold).detach().cpu().numpy()									
			elif dynamics.softplus_squared:
				if i==0:
					print("Using the sqaured softplus cost function")
				diff = torch.Tensor(computed_index[i,:] - x[:,i,-1])
				SP = torch.nn.functional.softplus(diff, beta=dynamics._beta, \
					threshold=dynamics._threshold).detach().cpu().numpy()			
				fbsde_costs[i,:-1] = dynamics._Q_t * dynamics._beta * (SP[0:-1]**2)
				fbsde_costs[i,-1] = dynamics._Q_T * dynamics._beta * (SP[-1]**2)

				diff = torch.Tensor(computed_index[i,:] - x_const_eq[:,i,-1])
				SP = torch.nn.functional.softplus(diff, beta=dynamics._beta, \
					threshold=dynamics._threshold).detach().cpu().numpy()			
				const_eq_costs[i,:-1] = dynamics._Q_t * dynamics._beta * (SP[0:-1]**2)
				const_eq_costs[i,-1] = dynamics._Q_T * dynamics._beta * (SP[-1]**2)				

				diff = torch.Tensor(computed_index[i,:] - x_const_rand[:,i,-1])
				SP = torch.nn.functional.softplus(diff, beta=dynamics._beta, \
					threshold=dynamics._threshold).detach().cpu().numpy()			
				const_rand_costs[i,:-1] = dynamics._Q_t * dynamics._beta * (SP[0:-1]**2)
				const_rand_costs[i,-1] = dynamics._Q_T * dynamics._beta * (SP[-1]**2)

			elif dynamics.exponential_cost_func:
				if i==0:
					print("Using the exponential cost function")				
				fbsde_costs_all_time = (torch.exp(torch.Tensor(computed_index[i,:] \
															- x[:,i,-1]))).detach().cpu().numpy()
				fbsde_costs[i,:-1] = dynamics._Q_t * fbsde_costs_all_time[0:-1]
				fbsde_costs[i,-1] = dynamics._Q_T * fbsde_costs_all_time[-1]

				const_eq_costs_all_time = (torch.exp(torch.Tensor(computed_index[i,:] \
															- x_const_eq[:,i,-1]))).detach().cpu().numpy()
				const_eq_costs[i,:-1] = dynamics._Q_t * const_eq_costs_all_time[0:-1]
				const_eq_costs[i,-1] = dynamics._Q_T * const_eq_costs_all_time[-1]

				const_rand_costs_all_time = (torch.exp(torch.Tensor(computed_index[i,:] \
															- x_const_rand[:,i,-1]))).detach().cpu().numpy()
				const_rand_costs[i,:-1] = dynamics._Q_t * const_rand_costs_all_time[0:-1]
				const_rand_costs[i,-1] = dynamics._Q_T * const_rand_costs_all_time[-1]

			else:
				raise KeyError("Cost computation using different running and terminal costs not implemented for original quadratic cost")
				fbsde_costs[i,:] = dynamics._Q * (x[:,i,-1] - dynamics.index_mult * computed_index[i,:])**2 
				const_eq_costs[i,:] = dynamics._Q * (x_const_eq[:,i,-1] - dynamics.index_mult * computed_index[i,:])**2 
				const_rand_costs[i,:] = dynamics._Q * (x_const_rand[:,i,-1] - dynamics.index_mult * computed_index[i,:])**2

	if dynamics.Merton_problem:
		print('Creating plots for Merton Problem...')
		fbsde_wealth = x[-1,:,-1]
		const_eq_wealth = x_const_eq[-1,:,-1]
		const_rand_wealth = x_const_rand[-1,:,-1]
		total_fbsde_costs = -dynamics._Q_T*2.0*np.sqrt(x[-1,:,-1])
		total_const_eq_costs = -dynamics._Q_T*2.0*np.sqrt(x_const_eq[-1,:,-1])
		total_const_rand_costs=-dynamics._Q_T*2.0*np.sqrt(x_const_rand[-1,:,-1])		 
		sort_indices = np.argsort(total_fbsde_costs) # sorts in ascending order
		print('Mean fbsde wealth: ', np.mean(fbsde_wealth))
		print('Mean optimal control wealth: ', np.mean(const_eq_wealth))
		print('Mean random control wealth: ', np.mean(const_rand_wealth))

	if not dynamics.Merton_problem:
		# shape of x: (T+1, batch_size, state_dim)
		terminal_index = computed_index[:,-1]
		fbsde_terminal_wealth = x[-1,:,-1]
		const_eq_terminal_wealth = x_const_eq[-1,:,-1]
		const_rand_terminal_wealth = x_const_rand[-1,:,-1]		

		fbsde_terminal_diff = fbsde_terminal_wealth - terminal_index
		const_eq_terminal_diff = const_eq_terminal_wealth - terminal_index
		const_rand_terminal_diff = const_rand_terminal_wealth - terminal_index

		fbsde_total_diff = np.sum(x[:,:,-1] - computed_index.T, axis=0)
		const_eq_total_diff = np.sum(x_const_eq[:,:,-1] - computed_index.T, axis=0)
		const_rand_total_diff = np.sum(x_const_rand[:,:,-1] - computed_index.T, axis=0)

		# Sum the costs along the time dimension:
		total_fbsde_costs = np.sum(fbsde_costs, axis=1)
		total_const_eq_costs = np.sum(const_eq_costs, axis=1)
		total_const_rand_costs = np.sum(const_rand_costs, axis=1)
		sort_indices = np.argsort(total_fbsde_costs) # sorts in ascending order

	np.savez(path+'losses_for_violins.npz',\
		fbsde=total_fbsde_costs, \
		const_eq=total_const_eq_costs, 
		const_rand=total_const_rand_costs)

	legend_properties = {'weight':'bold'}
	plt.figure(figsize=(15, 10))
	plt.plot(T, x[:,sort_indices[0],-1],'b')
	plt.plot(T, x_const_eq[:,sort_indices[0],-1],'g')
	plt.plot(T, x_const_rand[:,sort_indices[0],-1],'k')
	plt.plot(T, computed_index[sort_indices[0],:], 'r')
	plt.legend(['FBSDE', 'CONSTANT', 'RANDOM', 'Index'], prop=legend_properties)

	plt.title('Best Case Trajectories', fontweight='bold')
	plt.xlabel('Time (years)', fontweight='bold')
	plt.ylabel('Amount in dollars', fontweight='bold')
	plt.savefig(path + 'best_case.png')

	plt.figure(figsize=(15, 10))
	plt.plot(T, x[:,sort_indices[int(bs/2)],-1],'b')
	plt.plot(T, x_const_eq[:,sort_indices[int(bs/2)],-1],'g')
	plt.plot(T, x_const_rand[:,sort_indices[int(bs/2)],-1],'k')
	plt.plot(T, computed_index[sort_indices[int(bs/2)],:], 'r')
	plt.legend(['FBSDE', 'CONSTANT', 'RANDOM', 'Index'], prop=legend_properties)
	plt.xlabel('Time (years)', fontweight='bold')
	plt.ylabel('Amount in dollars', fontweight='bold')	
	plt.title('Median Case Trajectories', fontweight='bold')
	plt.savefig(path + 'median_case.png')

	plt.figure(figsize=(15, 10))
	plt.plot(T, x[:,sort_indices[-1],-1],'b')
	plt.plot(T, x_const_eq[:,sort_indices[-1],-1],'g')
	plt.plot(T, x_const_rand[:,sort_indices[-1],-1],'k')
	plt.plot(T, computed_index[sort_indices[-1],:], 'r')
	plt.legend(['FBSDE', 'CONSTANT', 'RANDOM', 'Index'], prop=legend_properties)
	plt.title('Worst Case Trajectories', fontweight='bold')
	plt.xlabel('Time (years)', fontweight='bold')
	plt.ylabel('Amount in dollars', fontweight='bold')	
	plt.savefig(path + 'worst_case.png')

	ctrl_legend = ['u_'+str(i) for i in range(u.shape[-1])]
	u = (torch.nn.functional.softmax(torch.Tensor(u), dim=2)).detach().cpu().numpy()
	plt.figure(figsize=(15, 10))
	T = np.array(range(num_time_interval))*dt
	plt.plot(T, u[:,sort_indices[0],:])
	plt.legend(ctrl_legend)
	plt.title('best case controls')
	plt.savefig(path + 'best_controls.png')

	plt.figure(figsize=(15, 10))
	plt.plot(T, u[:,sort_indices[-1],:])
	plt.legend(ctrl_legend)
	plt.title('worst case controls')
	plt.savefig(path + 'worst_controls.png')

	# Plotting violin plots:
	if not dynamics.Merton_problem:
		plt.figure(figsize=(15,10))
		sns.set(style="whitegrid")
		sns.set(rc={"xtick.bottom" : True, "ytick.left" : True})

		# log transformed data:
		try:
			f, ce, cr = total_fbsde_costs, total_const_eq_costs, total_const_rand_costs
			pd_data = pd.DataFrame({"FBSDE":np.log10(f), \
								"CONSTANT":np.log10(ce),\
								"RANDOM":np.log10(cr)})
			ax = sns.violinplot(data=pd_data)
			all_min = np.min([np.min(f), np.min(ce), np.min(cr)])
			all_max = np.max([np.max(f), np.max(ce), np.max(cr)])

			lb = all_min * 0.1
			ub = all_max * 10.0
			print("lb =", lb, "ub = ", ub)			
		except:
			print("Exact 0s detected !")
			f, ce, cr = total_fbsde_costs, total_const_eq_costs, total_const_rand_costs
			f_temp = np.where(f==0.0, np.ones_like(f)*1e10, f)
			ce_temp = np.where(ce==0.0, np.ones_like(ce)*1e10, ce)
			cr_temp = np.where(cr==0.0, np.ones_like(cr)*1e10, cr)

			all_min = np.min([np.min(f_temp), np.min(ce_temp), np.min(cr_temp)])
			all_max = np.max([np.max(f), np.max(ce), np.max(cr)])

			lb = all_min * 0.1
			ub = all_max * 10.0

			f_new = np.where(f_temp==1e10, lb*np.ones_like(f), f)
			ce_new = np.where(ce_temp==1e10, lb*np.ones_like(ce), ce)
			cr_new = np.where(cr_temp==1e10, lb*np.ones_like(cr), cr)

			f_log = np.log10(f_new)
			ce_log = np.log10(ce_new)
			cr_log = np.log10(cr_new)	
			pd_data_log = pd.DataFrame({"FBSDE":f_log, \
					"CONSTANT":ce_log,\
					"RANDOM":cr_log})
			ax = sns.violinplot(data=pd_data_log)
			print("Costs have exact zero values, replacing with a lower bound of", lb)

		ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
		ax.yaxis.set_ticks([np.log10(x) for p in range(int(np.floor(np.log10(lb))), int(np.ceil(np.log10(ub))) ) \
					for x in np.linspace(10**p, 10**(p+1), 10)], minor=True)


		ax.set(ylabel='costs')
		if dynamics.alternate_cost_func:
			ax.set(title="alternate cost function")
		elif dynamics.softplus_cost_func:
			ax.set(title="softplus cost function")
		elif dynamics.exponential_cost_func:
			ax.set(title="exponential cost function")
		elif dynamics.softplus_squared:
			ax.set(title="squared softplus cost function")
		else:
			ax.set(title="original cost function")
		plt.savefig(path + 'violin_plot_log_comp.png')


		# This plot is for us to check if we use the correct cost function
		plt.figure(figsize=(15,10))
		# sns.set(style="whitegrid")
		pd_data = pd.DataFrame({"FBSDE":total_fbsde_costs, \
							"CONSTANT":total_const_eq_costs,\
							"RANDOM":total_const_rand_costs})
		ax = sns.violinplot(data=pd_data)
		ax.set(ylabel='costs')
		if dynamics.alternate_cost_func:
			ax.set(title="alternate cost function")
		elif dynamics.softplus_cost_func:
			ax.set(title="softplus cost function")			
		elif dynamics.softplus_squared:
			ax.set(title="squared softplus cost function")			
		elif dynamics.exponential_cost_func:
			ax.set(title="exponential cost function")			
		else:
			ax.set(title="original cost function")
		plt.savefig(path + 'violin_plot_comp.png')		

		sns.set(font_scale=2)
		# This plot is for the paper 
		plt.figure(figsize=(15,10))
		# sns.set(style="whitegrid")
		pd_data = pd.DataFrame({"FBSDE":total_fbsde_costs, \
							"CONSTANT":total_const_eq_costs,\
							"RANDOM":total_const_rand_costs})
		ax = sns.violinplot(data=pd_data)
		ax.axes.set_title("Cost function performance", fontweight='bold')
		ax.set_xlabel("Strategies", fontweight='bold')
		ax.set_ylabel("Total Cost", fontweight='bold')
		plt.savefig(path + 'cost_performance.png')


		# # Terminal wealth comparison plot for paper:
		# plt.figure(figsize=(15,10))
		# pd_data = pd.DataFrame({"Index":terminal_index,\
		# 					"FBSDE":fbsde_terminal_wealth, \
		# 					"CONSTANT":const_eq_terminal_wealth,\
		# 					"RANDOM":const_rand_terminal_wealth})
		# # sns.set(style="whitegrid")
		# ax = sns.violinplot(data=pd_data)
		# ax.axes.set_title("Investment performance", fontweight='bold')
		# ax.set_xlabel("Strategies", fontweight='bold')
		# ax.set_ylabel("Terminal Wealth", fontweight='bold')
		# plt.savefig(path + 'terminal_wealth_comp.png')	

		# Terminal difference comparison plot for paper:
		plt.figure(figsize=(15,10))
		pd_data = pd.DataFrame({"FBSDE":fbsde_terminal_diff, \
							"CONSTANT":const_eq_terminal_diff,\
							"RANDOM":const_rand_terminal_diff})
		# sns.set(style="whitegrid")
		ax = sns.violinplot(data=pd_data)
		ax.axes.set_title("Investment performance - Terminal difference from index", fontweight='bold')
		ax.set_xlabel("Strategies", fontweight='bold')
		ax.set_ylabel("Terminal Wealth", fontweight='bold')
		plt.savefig(path + 'terminal_wealth_diff.png')

		# Total difference comparison plot for paper:
		plt.figure(figsize=(15,10))
		pd_data = pd.DataFrame({"FBSDE":fbsde_total_diff, \
							"CONSTANT":const_eq_total_diff,\
							"RANDOM":const_rand_total_diff})
		# sns.set(style="whitegrid")
		ax = sns.violinplot(data=pd_data)
		ax.axes.set_title("Investment performance - Total difference from index", fontweight='bold')
		ax.set_xlabel("Strategies", fontweight='bold')
		ax.set_ylabel("Terminal Wealth", fontweight='bold')
		plt.savefig(path + 'total_wealth_diff.png')



	elif dynamics.Merton_problem:
		plt.figure(figsize=(15,10))
		pd_data = pd.DataFrame({"fbsde":total_fbsde_costs, \
							"const_equal":total_const_eq_costs,\
							"const_random":total_const_rand_costs})
		sns.set(style="whitegrid")
		ax = sns.violinplot(data=pd_data)
		ax.set(ylabel='costs')
		ax.set(title="Merton Problem Costs (switched sign for min")
		plt.savefig(path + 'violin_plot_comp.png')

		plt.figure(figsize=(15,10))
		pd_data = pd.DataFrame({"fbsde":fbsde_wealth, \
							"const_equal":const_eq_wealth,\
							"const_random":const_rand_wealth})
		sns.set(style="whitegrid")
		ax = sns.violinplot(data=pd_data)
		ax.set(ylabel='Wealth')
		ax.set(title="Merton Problem Wealth")
		plt.savefig(path + 'violin_plot_comp2.png')	


	Nbins = 100
	bins1 = np.linspace(np.floor(np.min(total_const_rand_costs)),np.ceil(np.max(total_const_rand_costs)),Nbins)
	bins2 = np.linspace(np.floor(np.min(total_const_eq_costs)),np.ceil(np.max(total_const_eq_costs)),Nbins)
	bins3 = np.linspace(np.floor(np.min(total_fbsde_costs)),np.ceil(np.max(total_fbsde_costs)),Nbins)
	#bins = np.linspace(300,1200,100)
	plt.figure()
	plt.hist(total_const_rand_costs,bins1, label='Constant random', facecolor='g', alpha=0.2)
	plt.hist(total_const_eq_costs, bins2, label='Constant equal', facecolor='r', alpha=0.2)
	plt.hist(total_fbsde_costs, bins3, label='FBSDE', facecolor='b', alpha=0.2)
	plt.legend(loc='upper right')
	plt.title("Exp cost function")
	plt.xlabel("Cost Value")
	plt.ylabel("Frequency")	
	plt.savefig(path + 'histogram')	


def plot_loss(valid_total_losses, valid_terminal_losses, path):
	plt.figure(figsize=(15, 10))
	xaxis = np.arange(1,len(valid_total_losses)+1)
	plt.semilogy(xaxis, valid_total_losses, '-r')
	plt.semilogy(xaxis, valid_terminal_losses, '-b')
	plt.legend(['total loss', 'mean terminal cost'])
	plt.xlabel('Iteration')
	plt.ylabel('loss')
	plt.savefig(path + 'loss_plot.png')
	plt.clf()
