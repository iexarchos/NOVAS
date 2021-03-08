To run experiments navigate to the root directory and in command line type:

python main.py --problem_name=<enter problem name here>

	** options for problem names include:
	1. CartPole
	2. Finance

* NOVAS related parameters are set in config.py in the Config base class
	
	n_sample_train : number of samples for NOVAS during training
	n_sample_infer : number of samples for NOVAS during inference
	n_iter_train : number of inner-loop iterations for NOVAS during training
	n_iter_infer : number of inner-loop iterations for NOVAS during inference
	init_sigma : initial values of sigma for NOVAS
	init_mu : initial values of mu for NOVAS
	opt : 'min' for minimization and 'max' for maximization
	use_gpu : boolean to use gpu or not 
	if use_gpu:
		device = 'cuda:0'
	else:
		device = 'cpu'
	kappa = torch.Tensor([35.0]).to(device) : temperature parameter when using exp(x) as shape function
	alpha = torch.Tensor([1.0]).to(device) : learning rate for NOVAS
	shape_func = 'standard_exponential' # soft_ce or standard_exponential
	use_Hessian : boolean to compute and use Hessian for NOVAS (default: False)  
	train_kappa : boolean to make the temperature parameter trainable (default: False) 
	train_alpha : boolean to make the learning rate trainable (default: False)
	use_linesearch : boolean to use linesearch to determine alpha instead of fixed value (default: True)

* Problem specific parameters are set in config.py in problem specific classes that inherit from Config base class
	
	** options for problem specific configuration classes include:
	1. CartPoleConfig
	2. FinanceConfig

	Description of some important hyperparameters:

		total_time : Time horizon
		num_time_interval : Number of time intervals 
		delta_t : time discretization
		batch_size : training batch size
		test_size : testing batch size
		valid_size : validation batch size
		num_iterations : number of training iterations of Adam
		milestones : (iteration_number)/(logging_frequency_valid) at which learning rate is reduced by factor of 0.1  
		lr_values : starting value of learning rate
		hidden_dims : list of number of neurons in each layer of LSTM
	
		# Loss weights (refer to supplementary material B.4):
		weight_V : l_1
		weight_Vx : l_2
		weight_Vxx_col : l_3 
		weight_V_true : l_4
		weight_Vx_true : l_5
		weight_Vxx_col_true l_6
		pre_compute_init : boolean to perform (n-1) iterations of NOVAS off-graph
		training_with_linesearch : boolean to perform linesearch to determine alpha during training 

(current values are those used in our experiments)

** list of packages in conda environment **

Package          Version            
---------------- -------------------
backcall         0.1.0              
certifi          2020.4.5.1         
cffi             1.14.0             
cycler           0.10.0             
decorator        4.4.2              
future           0.18.2             
higher           0.2                
hydra-core       0.11.3             
ipdb             0.12.3             
ipython          7.13.0             
ipython-genutils 0.2.0              
jedi             0.17.0             
kiwisolver       1.2.0              
lml              0.0.1              
matplotlib       3.1.3              
mkl-fft          1.0.15             
mkl-random       1.1.1              
mkl-service      2.3.0              
numpy            1.18.1             
omegaconf        1.4.1              
pandas           1.0.3              
parso            0.7.0              
pexpect          4.8.0              
pickleshare      0.7.5              
Pillow           7.1.2              
pip              20.0.2             
prompt-toolkit   3.0.4              
ptyprocess       0.6.0              
pycparser        2.20               
Pygments         2.6.1              
pyparsing        2.4.7              
python-dateutil  2.8.1              
pytz             2020.1             
PyYAML           5.3.1              
scipy            1.4.1              
seaborn          0.10.1             
semantic-version 2.8.5              
setproctitle     1.1.10             
setuptools       46.2.0.post20200511
six              1.14.0             
termcolor        1.1.0              
torch            1.6.0a0+d1eeb3b    
tornado          6.0.4              
traitlets        4.3.3              
wcwidth          0.1.9              
wheel            0.34.2 
