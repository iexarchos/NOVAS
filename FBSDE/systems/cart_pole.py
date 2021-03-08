from systems.base_dynamics import Dynamics
import numpy as np
import torch
GRAVITY = 9.81
from pdb import set_trace as bp

class CartPole(Dynamics):
	"""Cart pole system"""
	def __init__(self, x0, delta_t, num_time_interval, sigma, target, run_cost, float_type,\
				 savepath=None, loadpath=None, mp=0.01, mc=1.0, l=0.5):
		super(CartPole, self).__init__(x0=x0, state_dim=4, control_dim=1,
											   delta_t=delta_t, num_time_interval=num_time_interval, float_type=float_type)
		self._sys_name = "CartPole"
		self._Sigma = torch.Tensor(np.diag([0.0, 0.0, sigma, sigma]))
		self._sigma = sigma
		self._mp = mp  # Pole mass
		self._mc = mc  # Cart mass
		self._l = l  # Pole length
		self._I = mp*(l**2)  # Inertia
		self._R_out = 0.1
		self._R = torch.Tensor([self._R_out]).unsqueeze(dim=-1)  # Control cost
		self._target_out = target
		self._target = torch.Tensor(self._target_out)  # Target state
		self._Q_out = [0.0, 9.0, 1.0, 0.5]
		self._Q_out_T = self._Q_out
		self._Q_t = torch.Tensor(np.diag(self._Q_out))  # State cost
		self._Q_T = torch.Tensor(np.diag(self._Q_out))  # State cost
		self._run_cost = run_cost
		print("control cost = ", self._R_out)
		print("state cost diag:", self._Q_out)
		print("Using Vxx_col as diag(Vxx)")

	@property
	def target(self):
		return self._target_out

	@property
	def R(self):
		return self._R_out

	@property
	def R_tf(self):
		return self._R

	@property
	def Q(self):
		return self._Q_out

	@property
	def u_max(self):
		return self._u_constraint

	def q_t(self, x, t):
		q = 0.5 * torch.matmul((x - self._target[t, :])**2, self._Q_t).sum(dim=1, keepdim=True)
		if self._run_cost:
			return q
		else:
			return torch.zeros_like(q)
	
	def terminal_cost(self, x):
		q_T = 0.5 * torch.matmul((x - self._target[-1, :])**2, self._Q_T).sum(dim=1, keepdim=True)
		return q_T

	def terminal_cost_gradient(self, x):
		qx_T = torch.matmul(x - self._target[-1, :], self._Q_T)
		return qx_T
	
	# states:   [x1, x2, x3, x4] = [cart position, angular position, cart velocity, angular velocity]
	# indicies: [0 , 1 , 2 , 3]
	
	def G(self, x):
		theta = x[:, 1]
		sint = torch.sin(theta)
		cost = torch.cos(theta)
		denominator = self._mc + self._mp*(sint**2)
		zeros_tensor = torch.zeros_like(denominator)
		ones_tensor = torch.ones_like(denominator)
		G = torch.stack([zeros_tensor, zeros_tensor, ones_tensor/denominator, \
					-cost*ones_tensor/(denominator*self._l)], dim=-1)
		return G.unsqueeze(dim=-1)

	def terminal_Vxx_col(self, x): # for cartpole Vxx_col is actually diag(Vxx)
		temp = torch.zeros_like(x)
		temp[:,0], temp[:,1], temp[:,2], temp[:,3] = self._Q_T[0,0], self._Q_T[1,1],\
													 self._Q_T[2,2], self._Q_T[3,3]  
		return temp


	def hamiltonian(self, x, u, Vx, Vxx_col): # for cartpole Vxx_col is actually diag(Vxx)
		# VxTf = 
		Vx_T_G = torch.matmul(Vx.unsqueeze(1), self.G(x)).squeeze(dim=1) # (batch_size,control_dim)
		Vx_T_G_u = (Vx_T_G * u).sum(dim=-1, keepdim=False)
		H = 0.5 * torch.matmul(u**2, self._R).sum(dim=1, keepdim=False) + Vx_T_G_u 
		
		trace_term = 0.5 * (self._sigma**2) * (Vxx_col[:,2] + Vxx_col[:,3])
		H = H + trace_term
		return H

	def h_t(self, x, t, u):  
		h = self.q_t(x, t) + 0.5 * torch.matmul(u**2, self._R).sum(dim=1, keepdim=True)
		# (batch_size,1)
		return h

	def dynamics_prop(self, x, u, dw):
		theta = x[:, 1]
		theta_dot = x[:, 3]
		sint = torch.sin(theta)
		cost = torch.cos(theta)
		denominator = self._mc + self._mp*(sint**2)

		x1 = x[:, 0] + x[:, 2]*self._delta_t
		x2 = x[:, 1] + x[:, 3]*self._delta_t
		x3 = x[:, 2] + self._mp*sint*(self._l*(theta_dot**2) + GRAVITY*cost)*self._delta_t/denominator \
					 + u.squeeze(dim=-1)*self._delta_t/denominator  + self._sigma*dw[:,2]
		
		x4 = x[:, 3] + (-self._mp*self._l*(theta_dot**2)*cost*sint \
					 - (self._mc+self._mp)*GRAVITY*sint)*self._delta_t/(denominator*self._l)\
					 + (-(u.squeeze(dim=-1))*cost)*self._delta_t/(denominator*self._l) + self._sigma*dw[:,3]
						 
		x_next = torch.stack([x1, x2, x3, x4], dim=1)
		dummy = torch.zeros_like(x_next)
		return x_next, dummy

	def value_prop(self, x, V, Vx, dw, u, t): 
		Vx_T_Sigma = torch.matmul(Vx, self._Sigma.transpose(0, 1))
		V_next = V - self.h_t(x, t, u)*self._delta_t + Vx_T_Sigma*dw.sum(dim=1, keepdim=True)
		return V_next

	def value_and_dynamics_prop(self, x, u, V, Vx, dw, t):
		theta = x[:, 1]
		theta_dot = x[:, 3]
		sint = torch.sin(theta)
		cost = torch.cos(theta)
		denominator = self._mc + self._mp*(sint**2)

		x1 = x[:, 0] + x[:, 2]*self._delta_t
		x2 = x[:, 1] + x[:, 3]*self._delta_t
		x3 = x[:, 2] + self._mp*sint*(self._l*(theta_dot**2) + GRAVITY*cost)*self._delta_t/denominator \
					 + u.squeeze(dim=-1)*self._delta_t/denominator  + self._sigma*dw[:,2]
		
		x4 = x[:, 3] + (-self._mp*self._l*(theta_dot**2)*cost*sint \
					 - (self._mc+self._mp)*GRAVITY*sint)*self._delta_t/(denominator*self._l)\
					 + (-(u.squeeze(dim=-1))*cost)*self._delta_t/(denominator*self._l) + self._sigma*dw[:,3]
						 
		x_next = torch.stack([x1, x2, x3, x4], dim=1)

		Vx_T_Sigma = torch.matmul(Vx, self._Sigma.transpose(0, 1))
		V_next = V - self.h_t(x, t, u)*self._delta_t + Vx_T_Sigma*dw.sum(dim=1, keepdim=True)
		
		return V_next, x_next