import numpy as np
import numpy.random as npr

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import autograd


import csv
import os

import pickle as pkl


from adaptive_stochastic_search import ad_stoch_search


from time import time


import matplotlib.pyplot as plt
from pdb import set_trace as bp


def main(cfg):
	sys.excepthook = ultratb.FormattedTB(mode='Verbose',
        color_scheme='Linux', call_pdb=1)
	print('Current dir: ', os.getcwd())
	# bp()
	print("\nTraining from scratch")
	exp = RegressionExp(cfg)
	exp.run()
	exp.PlotEnergyFunction()



class RegressionExp():
	def __init__(self, cfg):
		self.cfg = cfg
		self.n_batch = cfg.n_batch

		
		shape_func = self.cfg.shape_func
		use_Hessian = self.cfg.use_Hessian
		num_samples = self.cfg.n_sample
		self.exp_dir = os.path.join(os.getcwd(), shape_func+"_Hessian_"+str(use_Hessian)+"_"\
				+str(num_samples)+"_samples_"+str(self.cfg.n_batch)+"_bs")
		

		torch.manual_seed(cfg.seed)
		npr.seed(cfg.seed)

		if self.cfg.use_gpu:
			print(f"Using GPU: {self.cfg.gpu_no}")
			self.device = torch.device(f"cuda:{self.cfg.gpu_no}")
		else:
			print("Using CPU")
			self.device = torch.device("cpu")

		print(f"Using a batch size of {self.n_batch}")
		self.Enet = EnergyNet(n_in=1, n_out=1, n_hidden=cfg.n_hidden).to(self.device)
		self.model = UnrollEnergyAdaStochSearch(Enet = self.Enet, n_sample = num_samples, \
			n_iter=self.cfg.n_iter, alpha = self.cfg.alpha, kappa=self.cfg.kappa, sigma=self.cfg.sigma, \
			opt=self.cfg.opt, shape_func=self.cfg.shape_func, use_Hessian=self.cfg.use_Hessian,\
			pre_compute_init=self.cfg.pre_compute_init)
		
		self.load_data()
		print("\nModel trainable parameters are:")
		for name, param in self.model.named_parameters():
			if param.requires_grad:
				print (name, ", shape:", list(param.shape))
		print("\n")

	def __getstate__(self):
		d = copy.copy(self.__dict__)
		del d['x_train']
		del d['y_train']
		return d

	def __setstate__(self, d):
		self.__dict__ = d
		self.load_data()

	def load_data(self):
		# Generate all the data:
		data = torch.linspace(0., 2.*np.pi, steps=self.cfg.n_samples).to(self.device)
		# Generate random indices:
		rand_idx = np.random.choice(self.cfg.n_samples, size=self.cfg.n_samples, replace=False)
		valid_idx = np.ceil(self.cfg.valid_frac*self.cfg.n_samples).astype(np.int32)
		self.x_valid = data[rand_idx[:valid_idx]]
		self.y_valid = self.x_valid * torch.sin(self.x_valid)

		self.x_train = data[rand_idx[valid_idx:]]
		self.y_train = self.x_train*torch.sin(self.x_train)
		self.num_train_data, self.num_valid_data  = self.x_train.shape[0], self.x_valid.shape[0]
		print("Number of data points for, training:", self.num_train_data, \
			", and validation:", self.num_valid_data)

#	def test(self):
#		print("Testing for ", self.cfg.model.tag)
#		print("Parameters:")
#		for key, value in self.cfg.model.params.items():
#			print(key, ":", value)
#
#		x_test = torch.linspace(0.0, 2.0*np.pi, steps=self.cfg.n_test_samples).to(self.device)
#		y_test = x_test*torch.sin(x_test)
#
#		loss_data = np.zeros((20,1))
#		loss_data2 = np.zeros((20,1))
#		for niters in range(1,21):
#			self.model.n_iter = niters
#			if self.cfg.model.tag == 'gd':
#				y_preds = self.model(x_test.view(-1, 1))
#			else:
#				y_preds, _, _, _ = self.model(x_test.view(-1, 1))
#
#			loss = F.mse_loss(input=y_preds.squeeze(), target=y_test)
#			loss.backward()
#			loss_data[niters-1] = loss.item()
#			print(f'Number of inner iterations: {self.model.n_iter}, Average Test Loss: {loss:.5f}')
#
#			if self.gd_at_test == True:
#				self.model_test.n_iter = niters
#				y_preds2 = self.model_test(x_test.view(-1, 1))
#				loss2 = F.mse_loss(input=y_preds2.squeeze(),target = y_test)
#				loss2.backward()
#				loss_data2[niters-1]=loss2.item()
#				print(f'Number of GD inner iterations: {self.model_test.n_iter}, Average Test Loss: {loss2:.5f}')



#		savepath = os.path.join(self.exp_dir, 'test_data')
#		np.save(savepath, loss_data)
#		print("Test data saved to", savepath+".npy")

	def PlotEnergyFunction(self):
		#bp()
		print("Plotting Energy Function ...")
		Nx = 100
		Ny = 200
		x,y = torch.meshgrid([torch.linspace(0., 2.*np.pi, steps=Nx),torch.linspace(-2.*np.pi,2*np.pi,steps=Ny)])
		x = x.reshape(Nx*Ny,1).to(self.device)
		y = y.reshape(Nx*Ny,1).to(self.device)
		energy = self.Enet.forward(x,y)
		x = x.reshape(Nx,Ny).cpu().data.numpy()
		y = y.reshape(Nx,Ny).cpu().data.numpy()
		energy = energy.reshape(Nx,Ny).cpu().data.numpy()
		plt.figure()
		# plt.contourf(x,y, np.log(np.clip(energy - energy.min() + 1e-4, 1.5, 10.00)), levels=200)
		#plt.contourf(x,y, np.clip(np.log(energy - energy.min() + 1e-4), 0.5, 3.0), levels=200)
		#bp()
		en_min = np.min(energy,axis = 1)
		en_min = np.repeat(en_min.reshape(Nx,1),Ny, axis =1)
		energy = energy - en_min
		en_max = np.max(energy,axis =1)
		en_max = np.repeat(en_max.reshape(Nx,1),Ny,axis=1)
		energy = energy/en_max + 1e-4
		#plt.contourf(x,y,energy, levels = 200)
		plt.contourf(x,y, np.log(energy), levels=200)
		xt = np.linspace(0.,2*np.pi,1000)
		plt.plot(xt,xt*np.sin(xt),'k')
		plt.colorbar()
		#plt.savefig(self.exp_dir+'/energy.png')
		plt.savefig('energy.png')
		#print("Plot saved to:", self.exp_dir+'/energy.png')
		

	def run(self):
		opt = optim.Adam(self.model.parameters(), lr=self.cfg.lr_start_value)
		if self.cfg.fixed_lr:
			print("Using a fixed learning rate of:", self.cfg.lr_start_value)
			lr_sched = ReduceLROnPlateau(opt, 'min', patience=20, factor=0.999999, verbose=True)
		else:
			lr_sched = ReduceLROnPlateau(opt, 'min', patience=20, factor=0.5, verbose=True)

		fieldnames = ['iter', 'loss']

		step = 0
		while step < self.cfg.n_update:
			iter_start_time = time()
			assert self.num_train_data > self.n_batch
			j = npr.choice(self.num_train_data, size=self.n_batch, replace=False)
			for i in range(self.cfg.n_inner_update):

				if self.cfg.verbose:
					y_preds, Delta_fmu, sigma_av, term_iter = self.model(self.x_train[j].view(-1,1))
					print(f'train step: {step}, fdiff = {Delta_fmu:.5f}, average sigma: {sigma_av:.2f}, last iter: {term_iter}')
				else:
					y_preds, _, _, _ = self.model(self.x_train[j].view(-1,1))
						
				y_preds.squeeze_(1)					
				loss = F.mse_loss(input=y_preds, target=self.y_train[j])
				opt.zero_grad()
				loss.backward(retain_graph=True)

				if self.cfg.clip_norm:
					nn.utils.clip_grad_norm_(self.Enet.parameters(), 1.0)
				opt.step()
				step += 1

			if step % 100 == 0: # VALIDATION STEP
				if self.num_valid_data == 0: # incase valid_frac = 0.0
					if step==100:
						print("No validation data. Using training data to validate")
					self.x_valid = self.x_train
					self.y_valid = self.y_train 

				
				y_preds, _, _, _ = self.model(self.x_valid.view(-1, 1))


				loss = F.mse_loss(input=y_preds.squeeze(), target=self.y_valid)
				lr_sched.step(loss)
				iter_time = time() - iter_start_time


				if self.cfg.verbose:
					print("\n")
					print(f'validation step: {step}, loss: {loss:.5f}, iter time: {iter_time:.5f} s')
					print("\n")
				else:
					print(f'validation step: {step}, loss: {loss:.5f}, iter time: {iter_time:.5f} s')

class EnergyNet(nn.Module):
	def __init__(self, n_in: int, n_out: int, n_hidden: int = 256):
		super().__init__()
		self.n_in = n_in
		self.n_out = n_out
		self.E_net = nn.Sequential(
			nn.Linear(n_in+n_out, n_hidden),
			nn.Softplus(),
			nn.Linear(n_hidden, n_hidden),
			nn.Softplus(),
			nn.Linear(n_hidden, n_hidden),
			nn.Softplus(),
			nn.Linear(n_hidden, 1),
		)

	def forward(self, x, y):
		z = torch.cat((x, y), dim=-1)
		E = self.E_net(z)
		return E

class UnrollEnergyAdaStochSearch(nn.Module):
	def __init__(self, Enet: EnergyNet, n_sample, n_iter,alpha, kappa, sigma, opt,  \
		shape_func, use_Hessian, pre_compute_init):
		super().__init__()
		self.Enet = Enet
		self.n_sample = n_sample
		self.kappa = kappa
		self.alpha = alpha
		self.n_iter = n_iter
		self.sigma = sigma
		self.opt = opt # will be either 'min' or 'max' depending on whether it's minimization or maximization
		self.shape_func = shape_func
		self.use_Hessian = use_Hessian
		self.pre_compute_init = pre_compute_init

		print("Param values:")
		print("alpha: ", self.alpha)
		print("kappa: ", self.kappa)
		print("n_sample: ", self.n_sample)
		print("n_iter: ", self.n_iter)
		print("sigma: ", self.sigma)
		print("shape function: ", self.shape_func)
		print("Use Hessian: ", self.use_Hessian )
		print("pre_compute_init: ", self.pre_compute_init)


	def forward(self, x):
		b = x.ndimension() > 1
		if not b:
			x = x.unsqueeze(0)
		assert x.ndimension() == 2
		nbatch = x.size(0)

		def f(y):
			_x = x.unsqueeze(1).repeat(1, y.size(1), 1) # repeat the same x for every sample 
			Es = self.Enet(_x.view(-1, 1), y.view(-1, 1)).view(y.size(0), y.size(1))
			return Es

		if self.pre_compute_init and self.n_iter>1:
			with torch.no_grad():
				mu_init, _, sigma_init, _ = ad_stoch_search(
					f, n_batch=nbatch, nx = 1, n_sample = self.n_sample,
					n_iter = self.n_iter-1, alpha = self.alpha, kappa=self.kappa, init_sigma=self.sigma, opt = self.opt,
					device=x.device, shape_func=self.shape_func, use_Hessian=self.use_Hessian)
			# Perform 1 inner-loop iteration that records gradients 
			#with torch.autograd.enable_grad():
			yhat = ad_stoch_search(
				f, n_batch=nbatch, nx=1, n_sample=self.n_sample, n_iter=1, alpha=self.alpha, \
				kappa=self.kappa, init_mu=mu_init, init_sigma=sigma_init, opt=self.opt,\
				device=x.device, shape_func=self.shape_func, use_Hessian=self.use_Hessian)			
		else:			
			yhat = ad_stoch_search(
				f, n_batch=nbatch, nx = 1, n_sample = self.n_sample,
				n_iter = self.n_iter,alpha = self.alpha, kappa=self.kappa, init_sigma=self.sigma, opt = self.opt,
				device=x.device, shape_func=self.shape_func, use_Hessian=self.use_Hessian)

		return yhat
	

if __name__ == '__main__':
	import sys
	from IPython.core import ultratb
	sys.excepthook = ultratb.FormattedTB(mode='Verbose',
		color_scheme='Linux', call_pdb=1)
	from regression_conf import config
	cfg = config()
	main(cfg)
