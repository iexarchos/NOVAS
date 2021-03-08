import numpy as np

import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from pdb import set_trace as bp
#from optimizers import SGD, Adam


def ad_stoch_search(
    f,
    nx,
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
    shape_func = 'standard_exponential',
    use_Hessian = True,
):
    if init_mu is None:
        init_mu = 0.

    if isinstance(init_mu, torch.Tensor):
        init_mu.requires_grad = True
        mu = init_mu.clone()
    elif isinstance(init_mu, float):
        mu = init_mu*torch.ones((n_batch, nx,), requires_grad=True, device=device)
    else:
        assert False

    if use_Hessian:
        #if init_sigma < 10.:
        #  raise KeyError('For use of Hessian, init_sigma should be at least 10.')   
        if init_sigma is None:
            init_sigma = 10.
    else:
        if init_sigma is None:
            init_sigma = 10.

    if isinstance(init_sigma, torch.Tensor):
        init_sigma.requires_grad = True
        sigma = init_sigma.clone()
    elif isinstance(init_sigma, float):
        sigma = init_sigma*torch.ones((n_batch, nx),requires_grad=True, device=device) # for soft_ce
    else:
        assert False

    assert mu.size() == (n_batch, nx)
    assert sigma.size() == (n_batch,nx)

    if lb is not None:
        assert isinstance(lb, float)

    if ub is not None:
        assert isinstance(ub, float)
        assert ub > lb

    fmu_before = f(mu)
    mu_opt = mu.clone()
    sigma_opt = sigma.clone()
    fmu_opt = fmu_before.clone()

    if use_Hessian: # initialize variables for Hessian case
        var = sigma**2
        theta1_k = mu/var #n_batch, nx
        theta2_k = -0.5*torch.reciprocal(var)
        theta_k = torch.cat((theta1_k,theta2_k),1) # n_batch, 2nx
        

    for i in range(n_iter):
        Delta_X = Normal(torch.zeros((n_batch, nx), device=device),sigma).rsample((n_sample,)).to(device) # shape: (n_sample, n_batch, nx)

        X = mu + Delta_X
        X = X.transpose(0,1).to(device) # (n_batch, n_sample, nx)
        X = X.contiguous()

        if opt == 'min':
            fX = -f(X)
        elif opt == 'max':
            fX = f(X)
        else:
            raise KeyError('No optimization type provided.')
        
        X, fX = X.view(n_batch, n_sample, -1), fX.view(n_batch, n_sample)
    
    
        minfX = fX.min(dim=1, keepdim=True).values #get the minimum value for normalization
        maxfX = fX.max(dim=1, keepdim=True).values
        _fX = (fX - minfX)/(maxfX-minfX) # subtract baseline. Shapes: (n_batch, n_sample) - (n_batch, 1)

        if shape_func == 'standard_exponential':
            S = torch.nn.functional.softmax(kappa*_fX, dim=1) # S is dimension (n_batch x n_sample)
            # the 3 following lines perform the exact same operation as above, but have different numerical behavior
            #S = torch.exp(-kappa*_fX)
            #norm = S.sum(dim=1)
            #S = S/norm #TODO: careful with n_batch > 1! Should be element-wise

        elif shape_func ==  'soft_ce':
            S0=1000
            elite=int(0.1*n_sample)
            fX_sorted, _ = fX.sort(dim=1, descending=True) #Sort in a descending order so it goes from best to worse trajectory
            gamma = fX_sorted[:, elite-1].unsqueeze(-1) #extract the cutoff threshold
            sig = torch.nn.Sigmoid()
            S = _fX * sig(S0 * (fX - gamma))#_fX / (1 + torch.exp(-S0 * (_fX - gamma))) #unnormalized
            S /= S.sum(dim=1, keepdim=True)
        
        S = S.transpose(0,1).unsqueeze(2) # final shape: (n_sample, n_batch, 1)
              
        old_mu = mu.clone()
            
        
        
        #---------------- Adapt Stoch Search update rule using Hessian ---------------------
        if use_Hessian:

            CX_k = torch.cat((X,X*X),2) # n_batch, n_sample, 2*nx
            #                                                    S'* CX_k
            #               S             copy 2nx times                  elementwise mult                  .sum(dim=1) 
            #        (n_batch, n_sample, 1) ------->     (n_batch,n_sample,2*nx) * (n_batch, n_sampe, 2*nx) ------->   (n_batch, 1, 2*nx)             
            grad_k = (torch.cat(2*nx*[S.transpose(0,1)],2)*CX_k).sum(dim=1,keepdim=True)
            grad_k = grad_k - torch.cat((mu.unsqueeze(1),sigma.unsqueeze(1)**2+mu.unsqueeze(1)**2),2)
            
                # Hes_k = -covariance(CX_k)
            m = CX_k - torch.mean(CX_k,dim=1,keepdim=True) #n_batch, n_sample, 2nx
            mmt = torch.einsum('bij,bjk->bik',m.transpose(1,2),m)  #n_batch,2nx, 2nx
            Hes_k = - (1.0/(n_sample-1.0))*mmt #+ 1e-4*torch.eye(nx).reshape((1,nx,nx)).repeat(n_batch,1,1) # scale by 1/(n_sample-1) and add small number in diagonal
            
            # with Hessian inversion use this:
            Hes_inv_k = torch.inverse(Hes_k)                       #n_batch, 2nx, 2nx, n_batch,2*nx, 1 --> n_batch, 2*nx, 1  --> n_batch,2*nx (squeeze)
            theta_k = theta_k - alpha*(torch.einsum('bij,bjk->bik',Hes_inv_k, grad_k.transpose(1,2))).squeeze(2) #n_batch, 2nx
            
            # if you want to avoid Hessian inversion, use this:
            #scaling = torch.diagonal(Hes_k, dim1=1,dim2=2)
            #scaling = torch.reciprocal(scaling)
            #theta_k = theta_k - alpha*scaling*grad_k.squeeze(1)
            
            theta1_k = theta_k[:,0:nx].clone()
            theta2_k = theta_k[:,nx:].clone()
            var = torch.clamp(-0.5*torch.reciprocal(theta2_k),min=0.001,max=1000) # clamp between 0.001 and 1000, just in case
            sigma = var.sqrt()
            mu = theta1_k*var
        

        #---------------- Simplest update rule (no Hessian) ----------------------
        else:
            #elementwise_product = S*Delta_X #TODO: careful with nx > 1 !!!  
            mu += alpha*(S*Delta_X).sum(0)
            sigma = ((S.transpose(0,1)*(X - mu.unsqueeze(1))**2).sum(dim=1) + 0.001).sqrt() # final shape: (n_batch, nx)
        
        # KEEPING TRACK OF OPTIMAL MU's. This does not influence the optimization process, \
        # only the final reported values.
        #if n_iter > 1:                
        #    fmu = f(mu)
        #    mu_opt = torch.where(fmu<fmu_opt,mu,mu_opt) #replace only those mu's in the batch that improved
        #    fmu_opt = torch.where(fmu<fmu_opt,fmu,fmu_opt)
        #    sigma_opt = torch.where(fmu<fmu_opt,sigma,sigma_opt)
        #else:
        mu_opt = mu
        sigma_opt = sigma
        
        if (mu-old_mu).norm() < iter_eps:
            break

    fmu_after = f(mu_opt)
    delta_fmu = (fmu_before-fmu_after).mean()
         

    if lb is not None or ub is not None:
        mu = torch.clamp(mu, lb, ub)
    return mu_opt, delta_fmu, sigma_opt, i
