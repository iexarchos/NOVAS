from systems.inverted_pendulum import InvertedPendulum
from systems.cart_pole import CartPole
from systems.finance import Finance

def get_system(name, x0, delta_t, num_time_interval, sigma, target, run_cost, dtype, savepath, loadpath):
	if name == 'InvertedPendulum':
		return InvertedPendulum(x0, delta_t, num_time_interval, sigma, target, run_cost, dtype, savepath, loadpath)
	elif name == 'CartPole':
		return CartPole(x0, delta_t, num_time_interval, sigma, target, run_cost, dtype, savepath, loadpath)
	elif name == 'Finance':
		return Finance(x0, delta_t, num_time_interval, sigma, target, run_cost, dtype, savepath, loadpath)
	else:
		raise KeyError("Required system not found.")
