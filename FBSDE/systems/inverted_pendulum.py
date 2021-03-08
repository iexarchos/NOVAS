from systems.base_dynamics import Dynamics
import numpy as np
import torch
import utils
from pdb import set_trace as bp
import math
import pdb
GRAVITY = 9.81

class InvertedPendulum(Dynamics):
	"""Inverted pendulum system"""

	def __init__(self, x0, delta_t, num_time_interval, sigma, target, run_cost, float_type, m=2.0, l=0.5, b=0.1):
		super(InvertedPendulum, self).__init__(x0=x0, state_dim=2, control_dim=1,
											   delta_t=delta_t, num_time_interval=num_time_interval, float_type=float_type)
		self._Sigma = torch.Tensor(np.diag([0.0, sigma]))
		self._sigma = sigma
		self._m = m  # Pendulum mass
		self._l = l  # Pendulum length
		self._b = b  # Damping coefficient
		self._I = m*(l**2)  # Inertia
		self._R_out = 0.1
		self._R = torch.Tensor([self._R_out]).unsqueeze(dim=-1)  # Control cost
		self._target_out = target
		self._target = torch.Tensor(self._target_out)  # Target state
		self._Q_out = [3.5, 0.05]
		self._Q_t = torch.Tensor(np.diag(self._Q_out)) # State cost
		self._Q_T = torch.Tensor(np.diag(self._Q_out))  # State cost
		self._G = torch.Tensor([0.0, 1.0 / (m * (l ** 2))]).unsqueeze(dim=-1)
		self._run_cost = run_cost

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

	# --------------------------------- Alternate Formulation -------------------------------------------------------

	def hamiltonian(self, x, u, Vx):
		# VxTf = 
		Vx_T_G = torch.matmul(Vx.unsqueeze(1), self.G(x)).squeeze(dim=1) # (batch_size,control_dim)
		Vx_T_G_u = (Vx_T_G * u).sum(dim=-1, keepdim=False)
		H = 0.5 * torch.matmul(u**2, self._R).sum(dim=1, keepdim=False) + Vx_T_G_u 
		return H

	def G(self, x):
		return torch.ones_like(x).unsqueeze(dim=-1) * self._G.unsqueeze(dim=0)

	def h_t(self, x, t, u):  # Added contribution of Gamma term
		h = self.q_t(x, t) + (0.5 * torch.matmul(u**2, self._R).sum(dim=1, keepdim=False)).unsqueeze(dim=-1)  
		# (batch_size,1)
		return h
	
	def dynamics_prop(self, x, u, dw): # Corrected for gravity term in F_x, coefficient of noise (should not have 1/I multiplied)
		
		sinx    = torch.sin(x[:, 0])
		F_x     = (-self._b/self._I)*x[:, 1] - (GRAVITY/self._l)*sinx
			
		x1      = x[:, 0] + x[:, 1]*self._delta_t
		x2      = x[:, 1] + F_x*self._delta_t + (u.squeeze(dim=-1)/self._I)*self._delta_t + self._sigma*dw[:,1]
	
		x_next  = torch.stack([x1, x2], dim=1)
		return x_next
	
	def value_prop(self, x, V, Vx, dw, u, t): 
		Vx_T_Sigma = torch.matmul(Vx, self._Sigma.transpose(0, 1))
		V_next = V - self.h_t(x, t, u)*self._delta_t + Vx_T_Sigma*dw.sum(dim=1, keepdim=True)
		return V_next


