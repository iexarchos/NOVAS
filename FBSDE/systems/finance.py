from systems.base_dynamics import Dynamics
import numpy as np
import torch 
import itertools 
from pdb import set_trace as bp
from time import time

class Finance(Dynamics):
	
	def __init__(self, x0, delta_t, num_time_interval, sigma, \
				target, run_cost, float_type, savepath, loadpath, N_stocks=50, N_traded=10, \
				selection=None, correlated_noise=True):
		super(Finance, self).__init__(x0=x0, state_dim=N_stocks+1, control_dim=N_traded+1,
						   delta_t=delta_t, num_time_interval=num_time_interval, float_type=float_type)
		# state dimension is the number of stocks in the index + the wealth state
		# control dimension is the number of stocks we are trading + risk-free asset
		# selection: indeces of stocks we are trading. If none, a random choice will be made	
		# 
		# S0:N stocks comprising index
		# 
		# (S1,...S_N_stocks): risky assets (stocks), 
		# state is x = [S1, S2, .... S_{N_stocks}, Wealth] : N_stocks + 1
		# control is u = [p_risk-free, p_1, ...,p_N_traded] unnormalized percentages of 
		# risk-free and stock selection: N_traded + 1

		self._sys_name = "Finance"
		self._savepath = savepath
		self._loadpath = loadpath
		self.Merton_problem = False
		self.abs_loss = False
		self.alternate_cost_func = False
		self.softplus_cost_func = False
		self.softplus_squared = True
		self.exponential_cost_func = False
		self._beta = 10.0
		self._threshold = 100.0
		self.use_Primbs = False
		self.N_stocks = N_stocks
		self.N_traded = N_traded
		self.selection = selection
		self.correlated_noise = correlated_noise

		self.r = torch.Tensor([0.05]) #risk-free asset rate
		
		if not self.correlated_noise:

			# Merton's portfolio selection Problem values
			if self.Merton_problem:
				assert self.N_stocks == 1
				assert self.N_traded == 1
				mu = 0.09 #careful! If changing these values, need to update the test section of graph_oneshot with the new u_opt! 
				sigma = 0.30
				self.selection = 0
				self.u_opt = (mu -0.05)/(0.5*sigma**2)
				print('Optimal control: ', self.u_opt)
				mu = [mu]
				sigma = [sigma]
			else:
	        	# Primbs' values:
				assert self.N_stocks == 5
				assert self.N_traded == 3
				mu = [0.0916, 0.1182, 0.1462, 0.0924, 0.1486]
				# sigma = np.sqrt(np.array([0.09401, 0.05062, 0.09017, 0.09922, 0.07318]))
				sigma = np.array([0.30659419, 0.22498889, 0.3002832, 0.31499206, 0.27051802])
				self.selection = np.array([0,1,2])

				#or use random values:
				mu = np.random.uniform(low=0.05,high = 0.15,size =N_stocks) # rate for each stock (should be at least as high as risk-free rate)
				sigma = np.sqrt(0.5*mu)				

		elif self.correlated_noise:
			def MatrixSquareRoot(M):
			# Enter symmetric pos-def matrix M and get matrix M_sqrt s.t. M_sqrt = M_sqrt.T and M = M_sqrt*M_sqrt
				U,S,VT = np.linalg.svd(M)
				M_sqrt = np.matmul(np.matmul(U,np.diag(np.sqrt(S))),VT)
				assert (np.max(np.abs(M_sqrt-M_sqrt.T))<1e-10) # make sure M_sqrt = M_sqrt.T 
				assert (np.max(np.abs(M-np.matmul(M_sqrt,M_sqrt)))<1e-10) #make sure M = M_sqrt*M_sqrt
				return M_sqrt

			Primbs_Sigma2 = np.array([[0.09401, 0.01374, 0.01452, 0.01237, 0.01838],
                   [0.01374, 0.05062, 0.01475, 0.02734, 0.02200],
                   [0.01452, 0.01475, 0.09017, 0.01029, 0.01286],
                   [0.01237, 0.02734, 0.01029, 0.09922, 0.02674],
                   [0.01838, 0.02200, 0.01286, 0.02674, 0.07318]])

			if self.use_Primbs:
				# Primbs' values:
				assert self.N_stocks == 5
				assert self.N_traded == 3
				mu = [0.0916,0.1182,0.1462,0.0924,0.1486]
				Sigma = MatrixSquareRoot(Primbs_Sigma2)
				self.selection = np.array([0,1,2])
				print("\nPrimbs Sigma = \n", Sigma)
			else:
				if self._loadpath is not None:
					print("************* Loading saved rates and volatility data ********************")
					mus_and_sigmas = np.load(self._loadpath+'mus_and_sigmas.npz')
					mu = mus_and_sigmas['mu']
					Sigma = mus_and_sigmas['Sigma']
					print("Sigma = \n", Sigma)
					self.selection = np.load(self._loadpath+'selection.npz')['selection']
				else:
					print("Generating random return rates and voltility covariance matrix")
					# or use random values:
					multiplier = 10.0
					diag_multiplier = 0.4
					all_divider = 6.0
					mu = np.random.uniform(low=0.05,high = 0.15,size =N_stocks)
					Sigma = np.random.uniform(size=(N_stocks,N_stocks)) # generate random matrix
					Sigma2 = np.matmul(Sigma,Sigma.T) # use random matrix to generate random symmetric pos def matrix
					Sigma = np.abs(multiplier*(1.0/N_stocks)*MatrixSquareRoot(Sigma2)) # get matrix square root
					temp = np.ones((N_stocks, N_stocks))
					np.fill_diagonal(temp, diag_multiplier)
					Sigma = Sigma * temp/all_divider					
					
					np.set_printoptions(precision=3, linewidth=100000)
					print("Sigma (with doubled variances) = \n", Sigma)
					print("Varinaces are: ", np.diag(Sigma))
					print('Trace of Sigma = ', np.trace(Sigma))
					print('Column-sums of Sigma = ', np.sum(Sigma,axis=0))
					print('Average column-sum of Sigma = ', np.mean(np.sum(Sigma,axis=0)))
					print('Trace over ACS ratio= ', np.trace(Sigma)/np.mean(np.sum(Sigma,axis=0)))
					print("\nPrimbs Sigma = \n", MatrixSquareRoot(Primbs_Sigma2))
					print("\nmultiplier used = ", multiplier)
					print("diag_multiplier used = ", diag_multiplier)
					print("all_divider used = ", all_divider)

					# for logging to file 
					f = open(self._savepath+"log.txt","a")
					print("Sigma (with doubled variances) = \n", Sigma, file=f)
					print("Varinaces are: ", np.diag(Sigma), file=f)
					print('Trace of Sigma = ', np.trace(Sigma), file=f)
					print('Column-sums of Sigma = ', np.sum(Sigma,axis=0), file=f)
					print('Average column-sum of Sigma = ', np.mean(np.sum(Sigma,axis=0)), file = f)
					print('Trace over ACS ratio= ', np.trace(Sigma)/np.mean(np.sum(Sigma,axis=0)),file = f)
					print("\nPrimbs Sigma = \n", MatrixSquareRoot(Primbs_Sigma2), file=f)
					print("\nmultiplier used = ", multiplier, file=f)					
					print("diag_multiplier used = ", diag_multiplier, file=f)
					print("all_divider used = ", all_divider, file=f)
					f.close()
					np.savez(self._savepath+'mus_and_sigmas', mu=mu, Sigma=Sigma)
					bp()

		if self.selection is None:
			self.selection  = np.random.choice(range(N_stocks), size=N_traded, replace=False)
			np.savez(self._savepath+'selection', selection=self.selection)

			
		self.rates = torch.Tensor(mu)
		if self.correlated_noise:
			self.Sigma = torch.Tensor(Sigma)
		else:
			self.sigma = torch.Tensor(sigma)
		# self._run_cost = run_cost 
		
		self._x0 = torch.Tensor([1.0]*(self.N_stocks+1)) # Normalize everything by its initial value
		
		self._Q_t = 0.0
		self._Q_T = 50.0
		self.index_mult	= 1.0
		print("Use correlated noise? ", self.correlated_noise)
		print("State cost on wealth: Running cost Q_t = ", self._Q_t, "and terminal cost Q_T = ", self._Q_T)		
		print("Return rates:", mu)
		print("Volatility:", sigma)
		print("Use alternate cost function? ", self.alternate_cost_func)
		print("Use softplus cost function? ", self.softplus_cost_func, "with beta = ", self._beta,\
		 " and threshold = ", self._threshold)
		print("Use softplus squared? ", self.softplus_squared)
		print("Solve Merton's Portfolio Selection problem? ", self.Merton_problem)
		print("use absolute loss? ", self.abs_loss)
		print("Use exponential cost function? ", self.exponential_cost_func)
		print("use Primbs? ", self.use_Primbs)		
		print("Stocks selected for trading:", self.selection)
		print("dynamics inside hamiltonian, inline run cost for value prop, faster computation of trace term")
		print("")

	def compute_index(self, x):
		# extract stock prices:
		S = x[:,:self.N_stocks] # (batch_size, total_stock)
		
		# Compute the index to track = average of all stock prices
		return S.mean(dim=1, keepdim=True) # (batch_size, 1)

	def q_t(self, x, t): # Running cost computation
		if self.alternate_cost_func:
			q = self._Q_t * torch.nn.functional.relu(self.compute_index(x) - x[:,-1].unsqueeze(dim=-1))**2
		elif self.exponential_cost_func:
			q = self._Q_t * torch.exp(self.compute_index(x) - x[:,-1].unsqueeze(dim=-1))
		elif self.softplus_cost_func:
			diff = self.compute_index(x) - x[:,-1].unsqueeze(dim=-1)
			q = self._Q_t * torch.nn.functional.softplus(diff, beta=self._beta, threshold=self._threshold)
		elif self.softplus_squared:
			diff = self.compute_index(x) - x[:,-1].unsqueeze(dim=-1)
			SP = torch.nn.functional.softplus(diff, beta=self._beta, threshold=self._threshold)			
			q = self._Q_t * self._beta * (SP**2)
		elif self.Merton_problem:
			q = torch.zeros(x[:,-1].unsqueeze(dim=-1).shape) #zeros			
		elif self.abs_loss:
			q = self._Q_t * torch.abs(x[:,-1].unsqueeze(dim=-1) - self.compute_index(x))
		else:
			q =  self._Q_t * (x[:,-1].unsqueeze(dim=-1) - self.index_mult * self.compute_index(x))**2
		return q
		
	def terminal_cost(self, x):
		if self.alternate_cost_func:
			q = self._Q_T * torch.nn.functional.relu(self.compute_index(x) - x[:,-1].unsqueeze(dim=-1))**2
		elif self.exponential_cost_func:
			q = self._Q_T * torch.exp(self.compute_index(x) - x[:,-1].unsqueeze(dim=-1))			
		elif self.softplus_cost_func:
			diff = self.compute_index(x) - x[:,-1].unsqueeze(dim=-1)
			q = self._Q_T * torch.nn.functional.softplus(diff, beta=self._beta, threshold=self._threshold)			
		elif self.softplus_squared:
			diff = self.compute_index(x) - x[:,-1].unsqueeze(dim=-1)
			SP = torch.nn.functional.softplus(diff, beta=self._beta, threshold=self._threshold)			
			q = self._Q_T * self._beta * (SP**2)			
		elif self.Merton_problem:
			q = -self._Q_T*2.0*torch.sqrt(x[:,-1].unsqueeze(dim=-1))
		elif self.abs_loss:
			q = self._Q_T * torch.abs(x[:,-1].unsqueeze(dim=-1) - self.compute_index(x))			
		else:
			q =  self._Q_T * (x[:,-1].unsqueeze(dim=-1) - self.index_mult * self.compute_index(x))**2 
		return q

	def terminal_cost_gradient(self, x): 
		bs = x.shape[0]
		if self.alternate_cost_func:
			W = x[:,-1]
			I = self.compute_index(x).squeeze(dim=-1)
			q = torch.ones(size=[bs])*self._Q_T
			stock_terms = torch.ones(size=[bs]) * (1.0/self.N_stocks)	
			VxT_Q = torch.stack([stock_terms*q]*(self.N_stocks)+[-q], dim=1)
			all_zeros = torch.stack([torch.zeros(size=[bs])]*(self._state_dim), dim=1)
			
			repeated_I = I.unsqueeze(1).repeat(1,self._state_dim)
			repeated_W = W.unsqueeze(1).repeat(1,self._state_dim)
			# out = torch.where(repeated_I>repeated_W, VxT_Q, all_zeros)
			out = torch.where(repeated_I>repeated_W, VxT_Q, all_zeros)*\
				2.0* torch.nn.functional.relu(self.compute_index(x) - x[:,-1].unsqueeze(dim=-1))
			return out
		elif self.exponential_cost_func:
			common_term = self._Q_T * torch.exp(self.compute_index(x).squeeze(dim=-1) - x[:,-1])			
			stock_terms = torch.ones(size=[bs]) * (1.0/self.N_stocks)	
			out = torch.stack([stock_terms*common_term]*(self.N_stocks)+[-common_term], dim=1)
			return out			
		elif self.softplus_cost_func:
			diff = self.compute_index(x) - x[:,-1].unsqueeze(dim=-1)
			common_term = (self._Q_T * torch.sigmoid(self._beta * diff)).squeeze(1)
			stock_terms = torch.ones(size=[bs]) * (1.0/self.N_stocks)	
			out = torch.stack([stock_terms*common_term]*(self.N_stocks)+[-common_term], dim=1)
			return out
		elif self.softplus_squared:
			diff = self.compute_index(x) - x[:,-1].unsqueeze(dim=-1)
			SP = torch.nn.functional.softplus(diff, beta=self._beta, threshold=self._threshold)			
			g = torch.sigmoid(self._beta * diff)
			common_term = (2.0 * self._Q_T * self._beta * SP *	g).squeeze(1)
			stock_terms = torch.ones(size=[bs]) * (1.0/self.N_stocks)	
			out = torch.stack([stock_terms*common_term]*(self.N_stocks)+[-common_term], dim=1)
			return out			
		elif self.Merton_problem:
			W = x[:,-1]
			# q = torch.reciprocal(torch.sqrt(W)) # 1/sqrt(w)
			q = -self._Q_T*1.0/torch.sqrt(W)
			stock_terms = torch.zeros(size=[bs])
			out = torch.stack([stock_terms]*(self.N_stocks)+[q], dim=1)	
			return out 
		elif self.abs_loss:
			W = x[:,-1]
			I = self.compute_index(x)
			q = torch.ones(size=[bs])*self._Q_T
			stock_terms = torch.ones(size=[bs]) * (-1.0/self.N_stocks)
			grad = torch.stack([stock_terms*q]*(self.N_stocks)+[q], dim=1)
			repeated_I = I.repeat(1,self._state_dim)
			repeated_W = W.unsqueeze(1).repeat(1,self._state_dim)
			out = torch.where(repeated_W>=repeated_I, grad, -grad)
			return out			
		else:
			q = 2.0*self._Q_T*(x[:,-1].unsqueeze(dim=-1) - self.index_mult*self.compute_index(x)).squeeze(dim=1) 
			stock_terms = torch.ones(size=[bs]) * (-self.index_mult/self.N_stocks)
			VxT_Q = torch.stack([stock_terms*q]*(self.N_stocks)+[q], dim=1)
			return VxT_Q

	def terminal_Vxx_col(self, x):
		bs = x.shape[0]
		if self.alternate_cost_func:
			W = x[:,-1]
			I = self.compute_index(x).squeeze(dim=-1)
			all_zeros = torch.stack([torch.zeros(size=[bs])]*(self._state_dim), dim=1)
			repeated_I = I.unsqueeze(1).repeat(1,self._state_dim)
			repeated_W = W.unsqueeze(1).repeat(1,self._state_dim)			
			ones = 2.0*torch.ones(size=[bs]) * self._Q_T 
			stock_terms = torch.ones(size=[bs]) * (-1/self.N_stocks)
			Vxx_col = torch.stack([stock_terms*ones]*(self.N_stocks)+[ones], dim=1)
			out = torch.where(repeated_I>repeated_W, Vxx_col, all_zeros)
			return out
		elif self.exponential_cost_func:
			common_term = self._Q_T * torch.exp(self.compute_index(x).squeeze(dim=-1) - x[:,-1])
			stock_terms = torch.ones(size=[bs]) * (-1.0/self.N_stocks)	
			out = torch.stack([stock_terms*common_term]*(self.N_stocks)+[common_term], dim=1)
			return out									
		elif self.softplus_cost_func:
			diff = self.compute_index(x) - x[:,-1].unsqueeze(dim=-1)
			g = torch.sigmoid(self._beta * diff)
			common_term = (self._Q_T * (-self._beta) * g * (1 - g)).squeeze(1)
			stock_terms = torch.ones(size=[bs]) * (1.0/self.N_stocks)	
			out = torch.stack([stock_terms*common_term]*(self.N_stocks)+[-common_term], dim=1)
			return out
		elif self.softplus_squared:
			diff = self.compute_index(x) - x[:,-1].unsqueeze(dim=-1)
			SP = torch.nn.functional.softplus(diff, beta=self._beta, threshold=self._threshold)			
			g = torch.sigmoid(self._beta * diff)
			common_term = 2.0*self._Q_T*self._beta*(self._beta*g*(1-g)*SP + g**2)
			common_term = common_term.squeeze(1)
			stock_terms = torch.ones(size=[bs]) * (-1.0/self.N_stocks)	
			out = torch.stack([stock_terms*common_term]*(self.N_stocks)+[common_term], dim=1)
			return out												
		elif self.Merton_problem:
			W = x[:,-1]
			stock_terms = torch.zeros(size=[bs])
			# w_terms = -0.5*torch.reciprocal(torch.sqrt(torch.pow(W,3.0))) # -0.5*x^{-3/2}
			w_terms = -self._Q_T*(-0.5)*(1/torch.sqrt(torch.pow(W,3.0))) # -0.5*x^{-3/2}
			Vxx_col = torch.stack([stock_terms]+[w_terms], dim=1)
			return Vxx_col	
		elif self.abs_loss:
			return torch.stack([torch.zeros(size=[bs])]*(self._state_dim), dim=1)
		else:
			ones = 2.0*torch.ones(size=[bs]) * self._Q_T 
			stock_terms = torch.ones(size=[bs]) * (-self.index_mult/self.N_stocks)
			Vxx_col = torch.stack([stock_terms*ones]*(self.N_stocks)+[ones], dim=1)
			return Vxx_col

	def h_t(self, x, t, u): 
		# h = self.q_t(x, t) + 0.5 * torch.matmul(u**2, self._R).sum(dim=1, keepdim=True)
		h = self.q_t(x, t) # (batch_size,1)
		return h

	def hamiltonian(self, x, u, Vx, Vxx_col):

		S = x[:,:self.N_stocks] # stock price processes
		W = x[:,-1] # wealth process
		perc = torch.nn.functional.softmax(u, dim=1) # shape: (batch_size, control_dim=N_traded+1)
		Vxx_col_N_entries = Vxx_col[:,:self.N_stocks]
		Vxx_col_last_entry = Vxx_col[:,-1]

		bs = x.shape[0]

		########################### UNCOMMENT FOR ORIGINAL CODE ######################
		# _, F = self.dynamics_prop(x, u, torch.zeros_like(x))
		##############################################################################
		
		# Including code to compute F to avoid function calls:
		F_S = S * self.rates.unsqueeze(dim=0)
		F_W = W * (self.r * perc[:,0] \
			  + (self.rates[self.selection].unsqueeze(dim=0)*perc[:,1:]).sum(dim=1,keepdim=False))
		F = torch.cat([F_S,F_W.unsqueeze(dim=1)],dim=1) 

		VxT_F = (Vx*F).sum(dim=1, keepdim=False)
		control_cost = torch.zeros(size=[bs])

		hW = perc[:,1:] * W.unsqueeze(dim=1) # convert risky fractions into amounts	

		# Alternate calculation:
		if not self.correlated_noise:
			if self.Merton_problem:
				temp_1 = S[:,self.selection] * hW.squeeze(dim=1) * (self.sigma[self.selection].unsqueeze(dim=0))**2
				temp_2 = temp_1 * Vxx_col_N_entries[:, self.selection]
			else:
				temp_1 = S[:,self.selection] * hW * (self.sigma[self.selection].unsqueeze(dim=0))**2
				temp_2 = (temp_1 * Vxx_col_N_entries[:, self.selection]).sum(dim=1, keepdim=False)
			temp_3 = Vxx_col_last_entry * ((hW * self.sigma[self.selection].unsqueeze(dim=0))**2).sum(dim=1, keepdim=False)
			trace_term = 0.5 * (2.0*temp_2 + temp_3)

		elif self.correlated_noise:	

			# ############################ UNCOMMENT FOR ORIGINAL CODE ##########################################
			# def get_sigmas(s_id): # s_id is the number of the particular stock
			# 	sigmas = self.Sigma[s_id,:].unsqueeze(dim=0) * S[:,s_id].unsqueeze(dim=1)
			# 	return sigmas # shape: (batch_size, N_stocks)

			# sigma_tildes = (hW.unsqueeze(dim=-1) * self.Sigma[self.selection,:].unsqueeze(dim=0)).sum(dim=1)
			# # (bs,N_traded,1)*(1,N_traded,N_stocks) = (bs,N_traded,N_stocks) --sum--> (bs, N_stocks)
			
			# sigma_sums = [(get_sigmas(n_)*sigma_tildes).sum(dim=1, keepdim=True) for n_ in range(self.N_stocks)] 
			# Stocks_trace_terms = (Vxx_col_N_entries * torch.cat(sigma_sums, dim=1)).sum(dim=1, keepdim=False)
			# ##################################################################################################

			############################ ALTERNATE VECTORIZED FORMULATION ####################################
			sigma_tildes = (hW.unsqueeze(dim=-1) * self.Sigma[self.selection,:].unsqueeze(dim=0)).sum(dim=1)

			Sigma_S = S.unsqueeze(dim=2) * self.Sigma.unsqueeze(dim=0)
			# (bs,N_stocks,1)*(1,N_stocks,N_stocks)=(bs,N_stocks,N_stocks)
			Sigma_S_Sigma_tilde = Sigma_S * sigma_tildes.unsqueeze(dim=1)
			# (bs,N_stocks,N_stocks) * (bs,1,N_stocks) = (bs,N_stocks,N_stocks)
			sigma_sums = Sigma_S_Sigma_tilde.sum(dim=2, keepdim=False)
			Stocks_trace_terms = (Vxx_col_N_entries * sigma_sums).sum(dim=1, keepdim=False)
			##################################################################################################

			Wealth_trace_term = Stocks_trace_terms + Vxx_col_last_entry * (sigma_tildes**2).sum(dim=1)
			trace_term = 0.5*(Stocks_trace_terms + Wealth_trace_term)

		H = trace_term + VxT_F + control_cost 
		return H

	def value_and_dynamics_prop(self, x, u, V, Vx, dw, t):

		# Perform common operations:
		S 		= x[:,:self.N_stocks] # stock price processes
		W 		= x[:,-1] # wealth process
		perc	= torch.nn.functional.softmax(u, dim=1) # shape: (batch_size, control_dim=N_traded+1)
		dw_Ns 	= dw[:,:-1] # (batch_size, N_stocks) 
		bs 		= x.shape[0]
		
		if not self.correlated_noise: # DIFFUSION FOR UNCORRELATED NOISE
			# Compute Sigma matrix terms for stocks:
			volatility = S * self.sigma.unsqueeze(dim=0)*dw_Ns # (batch_size, N_stocks)		
			if self.Merton_problem:
				wealth_noise = W.unsqueeze(dim=1)*perc[:,1:]*self.sigma[self.selection].unsqueeze(dim=0) \
									* dw_Ns[:,self.selection].unsqueeze(dim=1) # (batch_size, control_dim-1 = N_traded)
			else:
				wealth_noise = W.unsqueeze(dim=1)*perc[:,1:]*self.sigma[self.selection].unsqueeze(dim=0) \
									* dw_Ns[:,self.selection] # (batch_size, control_dim-1 = N_traded)
			wealth_noise = wealth_noise.sum(dim=1, keepdim=True)

		elif self.correlated_noise:
			diffusion = (self.Sigma.unsqueeze(dim=0) * dw_Ns.unsqueeze(dim=1)).sum(dim=-1)
			volatility = S * diffusion
			wealth_noise = W.unsqueeze(dim=1)*perc[:,1:]* diffusion[:,self.selection]		
			wealth_noise = wealth_noise.sum(dim=1,keepdim=True)
			
		# First we propagate the value function:
		# Construct the vector (Sigma * dw):
		Sigma_dw = torch.cat([volatility, wealth_noise], dim=1) # (bs, state_dim)

		Vx_T_Sigma_dw = (Vx*Sigma_dw).sum(dim=1, keepdim=True)
		
		#################### UNCOMMENT FOR ORIGINAL CODE ######################################
		# V_next = V - self.q_t(x, t)*self._delta_t + Vx_T_Sigma_dw
		# WARNING THE ABOVE CANNOT WORK WHEN YOU HAVE RUNNING CONTROL COST
		#######################################################################################

		################# ALTERNATE INLINE COMPUTATION ########################################
		INDEX = S.mean(dim=1, keepdim=True) # (batch_size, 1)
		if self.alternate_cost_func:
			q = self._Q_t * torch.nn.functional.relu(INDEX - x[:,-1].unsqueeze(dim=-1))**2
		elif self.exponential_cost_func:
			q = self._Q_t * torch.exp(INDEX - x[:,-1].unsqueeze(dim=-1))			
		elif self.softplus_cost_func:
			diff = INDEX - x[:,-1].unsqueeze(dim=-1)
			q = self._Q_t * torch.nn.functional.softplus(diff, beta=self._beta, threshold=self._threshold)
		elif self.softplus_squared:
			# print("using softplus squared")
			diff = INDEX - x[:,-1].unsqueeze(dim=-1)
			SP = torch.nn.functional.softplus(diff, beta=self._beta, threshold=self._threshold)			
			q = self._Q_t * self._beta * (SP**2)			
		elif self.Merton_problem:
			q = torch.zeros(x[:,-1].unsqueeze(dim=-1).shape) #zeros			
		elif self.abs_loss:
			q = self._Q_t * torch.abs(x[:,-1].unsqueeze(dim=-1) - INDEX)
		else:
			q =  self._Q_t * (x[:,-1].unsqueeze(dim=-1) - self.index_mult * INDEX)**2

		V_next = V - q*self._delta_t + Vx_T_Sigma_dw
		########################################################################################

		# Next we propagate the dynamics:
		wealth_noise = wealth_noise.squeeze(dim=1)
		F_S = S * self.rates.unsqueeze(dim=0)
		F_W = W * (self.r * perc[:,0] \
			  + (self.rates[self.selection].unsqueeze(dim=0)*perc[:,1:]).sum(dim=1,keepdim=False))
		S 	= S + F_S * self._delta_t + volatility #check dimensions
		W 	= W + F_W * self._delta_t + wealth_noise 

		x_next  = torch.cat([S,W.unsqueeze(dim=1)], dim=1)

		return V_next, x_next	

	def dynamics_prop(self, x, u, dw): 

		S 		= x[:,:self.N_stocks] # stock price processes
		W 		= x[:,-1] # wealth process
		perc	= torch.nn.functional.softmax(u, dim=1) # shape: (batch_size, control_dim=N_traded+1)
		dw_Ns 	= dw[:,:-1] # (batch_size, N_stocks)  -1 because  we dont want the noise sampled for wealth as wealth noise is a weighted combination  of noise values in stocks
		bs = x.shape[0]

		if not self.correlated_noise: # DIFFUSION FOR UNCORRELATED NOISE
			# Compute Sigma matrix terms for stocks:
			volatility = S * self.sigma.unsqueeze(dim=0)*dw_Ns # (batch_size, N_stocks)		
			if self.Merton_problem:
				wealth_noise = W.unsqueeze(dim=1)*perc[:,1:]*self.sigma[self.selection].unsqueeze(dim=0) \
									* dw_Ns[:,self.selection].unsqueeze(dim=1) # (batch_size, control_dim-1 = N_traded)
			else:
				wealth_noise = W.unsqueeze(dim=1)*perc[:,1:]*self.sigma[self.selection].unsqueeze(dim=0) \
									* dw_Ns[:,self.selection] # (batch_size, control_dim-1 = N_traded)
			wealth_noise = wealth_noise.sum(dim=1, keepdim=False)

		elif self.correlated_noise: # DIFFUSION FOR CORRELATED NOISE:
			# volatility = S*torch.einsum('bij,bjk->bik',self.Sigma.unsqueeze(dim=0).repeat(bs,1,1),dw_Ns.unsqueeze(2)).squeeze(2)
			# wealth_noise = W.unsqueeze(dim=1)*perc[:,1:]*torch.einsum('bij,bjk->bik',self.Sigma_wealth.unsqueeze(dim=0).repeat(bs,1,1,),dw_Ns[:,self.selection].unsqueeze(2)).squeeze(2)
			diffusion = (self.Sigma.unsqueeze(dim=0) * dw_Ns.unsqueeze(dim=1)).sum(dim=-1)
			volatility = S * diffusion
			wealth_noise = W.unsqueeze(dim=1)*perc[:,1:]* diffusion[:,self.selection]
			wealth_noise = wealth_noise.sum(dim=1,keepdim=False)


		F_S = S * self.rates.unsqueeze(dim=0)
		F_W = W * (self.r * perc[:,0] \
			  + (self.rates[self.selection].unsqueeze(dim=0)*perc[:,1:]).sum(dim=1,keepdim=False))
		S 	= S + F_S * self._delta_t + volatility #check dimensions
		W 	= W + F_W * self._delta_t + wealth_noise 

		x_next  = torch.cat([S,W.unsqueeze(dim=1)], dim=1)
		F 		= torch.cat([F_S,F_W.unsqueeze(dim=1)],dim=1) 
		return x_next, F

'''
# FOR REFERENCE


	def value_prop(self, x, V, Vx, dw, u, t):

		S 		= x[:,:self.N_stocks] # stock price processes
		W 		= x[:,-1] # wealth process
		perc	= torch.nn.functional.softmax(u, dim=1) # shape: (batch_size, control_dim=N_traded+1)
		dw_Ns 	= dw[:,:-1] # (batch_size, N_stocks) 
		bs = x.shape[0]
		
		if not self.correlated_noise: # DIFFUSION FOR UNCORRELATED NOISE
			# Compute Sigma matrix terms for stocks:
			volatility = S * self.sigma.unsqueeze(dim=0)*dw_Ns # (batch_size, N_stocks)		
			if self.Merton_problem:
				wealth_noise = W.unsqueeze(dim=1)*perc[:,1:]*self.sigma[self.selection].unsqueeze(dim=0) \
									* dw_Ns[:,self.selection].unsqueeze(dim=1) # (batch_size, control_dim-1 = N_traded)
			else:
				wealth_noise = W.unsqueeze(dim=1)*perc[:,1:]*self.sigma[self.selection].unsqueeze(dim=0) \
									* dw_Ns[:,self.selection] # (batch_size, control_dim-1 = N_traded)
			wealth_noise = wealth_noise.sum(dim=1, keepdim=True)

		elif self.correlated_noise:
			# volatility = S*torch.einsum('bij,bjk->bik',self.Sigma.unsqueeze(dim=0).repeat(bs,1,1),dw_Ns.unsqueeze(2)).squeeze(2)
			# wealth_noise = W.unsqueeze(dim=1)*perc[:,1:]*torch.einsum('bij,bjk->bik',self.Sigma_wealth.unsqueeze(dim=0).repeat(bs,1,1,),dw_Ns[:,self.selection].unsqueeze(2)).squeeze(2)
			
			diffusion = (self.Sigma.unsqueeze(dim=0) * dw_Ns.unsqueeze(dim=1)).sum(dim=-1)
			volatility = S * diffusion
			wealth_noise = W.unsqueeze(dim=1)*perc[:,1:]* diffusion[:,self.selection]		
			wealth_noise = wealth_noise.sum(dim=1,keepdim=True)
			
		# Construct the vector (Sigma * dw):
		Sigma_dw = torch.cat([volatility, wealth_noise], dim=1) # (bs, state_dim)

		Vx_T_Sigma_dw = (Vx*Sigma_dw).sum(dim=1, keepdim=True)
		V_next = V - self.h_t(x, t, u)*self._delta_t + Vx_T_Sigma_dw
		return V_next

'''		
