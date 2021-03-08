import torch
import numpy as np
GRAVITY = 9.81


class Dynamics(object):
	"""Base class for system dynamics"""

	def __init__(self, x0, state_dim, control_dim, delta_t, num_time_interval, float_type):
		self._x0 = torch.Tensor(x0)
		self._dtype = float_type
		self._state_dim = state_dim
		self._control_dim = control_dim
		self._num_time_interval = num_time_interval
		self._delta_t = delta_t
		self._sqrt_delta_t = np.sqrt(self._delta_t)

	def q_t(self, x, t):
		"""Running state cost"""
		raise NotImplementedError

	def dynamics_prop(self, x, u, dw):
		"""Forward propagate dynamics SDE"""
		raise NotImplementedError

	def value_prop(self, x, V, Vx, dw, u, t):
		"""Forward propagate value function SDE"""
		raise NotImplementedError

	def terminal_cost(self, x, t):
		"""Terminal state cost"""
		raise NotImplementedError

	# All these below are getter methods (to access the private values)
	# which allow the function names to be used as attributes.
	@property
	def x0(self):
		return self._x0

	@property
	def state_dim(self):
		return self._state_dim

	@property
	def control_dim(self):
		return self._control_dim

	@property
	def total_time(self):
		return self._total_time

	@property
	def num_time_interval(self):
		return self._num_time_interval

	@property
	def delta_t(self):
		return self._delta_t
