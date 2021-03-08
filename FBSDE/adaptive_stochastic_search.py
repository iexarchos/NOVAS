import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from pdb import set_trace as bp

def ad_stoch_search(
	f,
	x, 
	Vx,
	Vxx_col,
	nu,
	opt,
	n_batch = 1,
	n_sample = 20,
	n_iter = 10,
	kappa = 1., #transformation "temp"
	alpha = 1., # adaptive stochastic search learning rate
	lb = None,
	ub = None,
	init_mu = None,
	init_sigma = None,
	device = None,
	iter_eps = 1e-4,
	nesterov = False,
	momentum = 0.0,
	shape_func = 'standard_exponential',
	use_Hessian = True,
	linesearch = False
):
	if init_mu is None:
		init_mu = 0.

	if isinstance(init_mu, torch.Tensor):
		init_mu.requires_grad = True
		mu = init_mu.clone()
	elif isinstance(init_mu, float):
		mu = init_mu*torch.ones((n_batch, nu,), requires_grad=True, device=device)
	else:
		assert False

	if use_Hessian:
		if init_sigma < 10.:
		  raise KeyError('For use of Hessian, init_sigma should be at least 10.')   
		if init_sigma is None:
			init_sigma = 10.
	else:
		if init_sigma is None:
			init_sigma = 10.

	if isinstance(init_sigma, torch.Tensor):
		init_sigma.requires_grad = True
		sigma = init_sigma.clone()
	elif isinstance(init_sigma, float):
		sigma = init_sigma*torch.ones((n_batch, nu),requires_grad=True, device=device) 
	else:
		assert False

	assert mu.size() == (n_batch, nu)
	assert sigma.size() == (n_batch,nu)

	# if lb is not None:
	# 	assert isinstance(lb, float)

	# if ub is not None:
	# 	assert isinstance(ub, float)
	# 	assert ub > lb

	# fmu_before = f(x=x, u=mu, Vx=Vx, Vxx_col=Vxx_col)
	mu_opt = mu.clone()
	sigma_opt = sigma.clone()
	#fmu_opt = fmu_before.mean()
	# fmu_opt = fmu_before.clone()
	# var = sigma**2
	# theta1_k = mu/var #n_batch, nu
	# theta2_k = -0.5*torch.reciprocal(var)
	# theta_k = torch.cat((theta1_k,theta2_k),1) # n_batch, 2nx
	
	# repeat the same x and Vx for every sample 
	x_copies = x.unsqueeze(1).repeat(1, n_sample, 1) # (n_batch, n_sample, state_dim)
	Vx_copies = Vx.unsqueeze(1).repeat(1, n_sample, 1) # (n_batch, n_sample, state_dim)
	Vxx_col_copies = Vxx_col.unsqueeze(1).repeat(1, n_sample, 1)
	state_dim = x.shape[-1]

	# for linesearch:
	if linesearch:
		num_lrs, lrs0, factor = 20, 1, 0.5
		lrs = np.empty((num_lrs,))
		lrs[0], lrs[1:] = lrs0, factor
		lrs = torch.tensor(np.cumprod(lrs))
		lrs = torch.reshape(lrs, (-1,1,1)).to(device) # (num_lrs, 1, 1)	        

	for i in range(n_iter):
		#Delta_U is dimension (n_sample, n_batch, nu)
		Delta_U = Normal(torch.zeros((n_batch, nu), device=device),sigma).rsample((n_sample,)).to(device) 

		U = mu + Delta_U
		U = U.transpose(0,1).to(device) # (n_batch, n_sample, nu)
		U = U.contiguous()

		if opt == 'min':
			fU = -f(x=x_copies.view(-1,state_dim), u=U.view(-1,nu), \
					Vx=Vx_copies.view(-1,state_dim), Vxx_col=Vxx_col_copies.view(-1,state_dim))
		elif opt == 'max':
			fU = f(x=x_copies.view(-1,state_dim), u=U.view(-1,nu), \
				   Vx=Vx_copies.view(-1,state_dim), Vxx_col=Vxx_col_copies.view(-1,state_dim))
		else:
			raise KeyError('No optimization type provided.')
		
		U, fU = U.view(n_batch, n_sample, nu), fU.view(n_batch, n_sample)
	
		minfU = fU.min(dim=1, keepdim=True).values #get the minimum value for normalization
		maxfU = fU.max(dim=1, keepdim=True).values #get max
		_fU = (fU - minfU)/(maxfU-minfU) # normalize. Shapes: (n_batch, n_sample) - (n_batch, 1)
	
		if shape_func == 'standard_exponential':
			S = torch.nn.functional.softmax(torch.nn.functional.relu(kappa).to(device)*_fU, dim=1) 
			# S is dimension (n_batch x n_sample)
			
			# the 3 following lines perform the exact same operation as above, but have different numerical behavior
			#S = torch.exp(-kappa*_fU)
			#norm = S.sum(dim=1)
			#S = S/norm #TODO: careful with n_batch > 1! Should be element-wise

		elif shape_func ==  'soft_ce':
			# print("using soft_ce")
			S0 = 1000
			elite = int(0.1*n_sample)
			fU_sorted, _ = fU.sort(dim=1, descending=True) #Sort in a descending order so it goes from best to worse trajectory
			gamma = fU_sorted[:, elite-1].unsqueeze(-1) #extract the cutoff threshold
			sig = torch.nn.Sigmoid()
			S = _fU * sig(S0 * (fU - gamma))#_fU / (1 + torch.exp(-S0 * (_fU - gamma))) #unnormalized
			S /= S.sum(dim=1, keepdim=True)
		
		S = S.transpose(0,1).unsqueeze(2) # final shape: (n_sample, n_batch, 1)
			  
		old_mu = mu.clone()
 
		#---------------- Adapt Stoch Search update rule using Hessian ---------------------
		if use_Hessian:
			#bp()
			CU_k = torch.cat((U,U*U),2) # n_batch, n_sample, 2*nu
			#                                                    S'* CU_k
			#               S             copy 2nx times                  elementwise mult                  .sum(dim=1) 
			#        (n_batch, n_sample, 1) ------->     (n_batch,n_sample,2*nu) * (n_batch, n_sampe, 2*nu) ------->   (n_batch, 1, 2*nu)             
			grad_k = (torch.cat(2*nu*[S.transpose(0,1)],2)*CU_k).sum(dim=1,keepdim=True)
			grad_k = grad_k - torch.cat((mu.unsqueeze(1),sigma.unsqueeze(1)**2+mu.unsqueeze(1)**2),2)
			
				# Hes_k = -covariance(CU_k)
			m = CU_k - torch.mean(CU_k,dim=1,keepdim=True) #n_batch, n_sample, 2nx
			mmt = torch.einsum('bij,bjk->bik',m.transpose(1,2),m)  #n_batch,2nx, 2nx
			Hes_k = - (1.0/(n_sample-1.0))*mmt #+ 1e-4*torch.eye(nu).reshape((1,nu,nu)).repeat(n_batch,1,1) # scale by 1/(n_sample-1) and add small number in diagonal
			
			# with Hessian inversion use this:
			Hes_inv_k = torch.inverse(Hes_k)                       #n_batch, 2nx, 2nx, n_batch,2*nu, 1 --> n_batch, 2*nu, 1  --> n_batch,2*nu (squeeze)
			theta_k = theta_k - alpha*(torch.einsum('bij,bjk->bik',Hes_inv_k, grad_k.transpose(1,2))).squeeze(2) #n_batch, 2nx
			
			# if you want to avoid Hessian inversion, use this:
			#scaling = torch.diagonal(Hes_k, dim1=1,dim2=2)
			#scaling = torch.reciprocal(scaling)
			#theta_k = theta_k - alpha*scaling*grad_k.squeeze(1)
			
			theta1_k = theta_k[:,0:nu].clone()
			theta2_k = theta_k[:,nu:].clone()
			var = torch.clamp(-0.5*torch.reciprocal(theta2_k),min=0.001,max=1000) # clamp between 0.001 and 1000, just in case
			sigma = var.sqrt()
			mu = theta1_k*var
		

		#---------------- Simplest update rule (no Hessian) ----------------------
		else:
			# shape of S is (n_sample, n_batch, 1)
			# shape of Delta_U is (n_sample, n_batch, nu)
			
			#elementwise_product = S*Delta_U #TODO: careful with nu > 1 !!!  
			if linesearch:
				gradient = (S*Delta_U).sum(0)
				with torch.no_grad():
					fmu_before_step = f(x=x, u=mu, Vx=Vx, Vxx_col=Vxx_col)# (n_batch,)
					# Generate candidate mus:
					mus = mu.unsqueeze(0) + lrs * gradient.unsqueeze(0) # (num_lrs,1,1)*(1,n_batch,nu)=(num_lrs,n_batch,nu)
					mus = mus.transpose(0,1) # (n_batch,num_lrs,nu)

					# Evaluate the candidate mus:
					fmus = f(x=x_copies[:,:num_lrs].reshape(-1,state_dim), u=mus.reshape(-1,nu), \
							Vx=Vx_copies[:,:num_lrs].reshape(-1,state_dim), \
							Vxx_col=Vxx_col_copies[:,:num_lrs].reshape(-1,state_dim))
					fmus_diff = fmu_before_step.unsqueeze(1) - fmus.reshape(n_batch, num_lrs) # (n_batch, num_lrs)
					fmus_max_diff = (fmus_diff).max(dim=1, keepdim=True).values # (n_batch, 1)
					# We are interested in the maximum improvement. Even if there is no improvement we take the
					# step which has the least worsening of the objective function value
					bool_tensor = (fmus_max_diff==fmus_diff).float() # (n_batch, num_lrs)
					
					# In some cases we have multiple max values and so multiple mus            
					bool_tensor = bool_tensor/bool_tensor.sum(dim=1, keepdim=True) 
					
					########### UNCOMMENT TO DIFFERENTIATE THROUGH LINESEARCH ##############
					# bool_tensor.unsqueeze_(-1) # (n_batch,num_lrs,1) 
					# mu = (mus * bool_tensor).sum(dim=1) # (n_batch,nu) 
					########################################################################

					# If we do not want gradients to flow through the linesearch procedure, 
					# we need to compute a batch on alphas as a result of the linesearch procedure:
					squeezed_lrs = lrs.squeeze() # (num_lrs,)
					alphas = (bool_tensor * squeezed_lrs.unsqueeze(dim=0)).sum(dim=1) # (n_batch, num_lrs)

				# Record gradients for update step only:
				mu += alphas.unsqueeze(dim=1) * gradient 
			else:
				mu += torch.abs(alpha)*(S*Delta_U).sum(0)
			sigma = ((S.transpose(0,1)*(U - mu.unsqueeze(1))**2).sum(dim=1) + 0.001).sqrt() # final shape: (n_batch, nu)
		
		
		# KEEPING TRACK OF OPTIMAL MU's. This does not influence the optimization process, only the final reported values.
		# if n_iter > 1:
		# 	fmu = f(x=x, u=mu, Vx=Vx, Vxx_col=Vxx_col) # (batch_size,)
		# 	repeated_fmu = fmu.unsqueeze(1).repeat(1,nu)
		# 	repeated_fmu_opt = fmu_opt.unsqueeze(1).repeat(1,nu)

		# 	mu_opt = torch.where(repeated_fmu<repeated_fmu_opt, mu, mu_opt) 
		# 	sigma_opt = torch.where(repeated_fmu<repeated_fmu_opt, sigma, sigma_opt)
		# 	fmu_opt = torch.where(fmu<fmu_opt, fmu, fmu_opt)
		# else:
		mu_opt = mu
		sigma_opt = sigma		
		
		if (mu-old_mu).norm() < iter_eps:
			break

	# fmu_after = f(x=x, u=mu_opt, Vx=Vx, Vxx_col=Vxx_col)
	# delta_fmu = (fmu_before-fmu_after).mean()
	delta_fmu = torch.tensor([0.])
		 

	# if lb is not None or ub is not None:
	# 	mu = torch.clamp(mu, lb, ub)
	
	return mu_opt, delta_fmu, sigma_opt, i
