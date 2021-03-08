import numpy as np
import os
import argparse
import util
from ipdb import set_trace as debug
import random

def set():

	# parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_name", default="Finance", help="name for system to control")
    parser.add_argument("--graph", default="oneshot",help="which graph to use")
    parser.add_argument("--debug", action="store_true", help="")
    parser.add_argument("--gpu", default="False", help="use cuda tensors")
    parser.add_argument("--recurrent", action="store_true", help="recurrent")
    parser.add_argument("--load_timestamp", default=None, help="timestamp of previous experiment you want to load")
    parser.add_argument("--test_only", default="False", help="Run test only. For this load_timestamp is mandatory")

    args = parser.parse_args()

	# --- below are automatically set ---
    # if args.seed is not None:
    #     seed = args.seed
    #     random.seed(seed)
    #     os.environ['PYTHONHASHSEED'] = str(seed)
    #     np.random.seed(seed)
        # args.problem_name += "_seed{}".format(args.seed)

    opt_dict    ={name: value for (name, value) in args._get_kwargs()}
    opt = argparse.Namespace(**opt_dict)

    # print configurations
    for o in vars(opt):
        print(util.green(o),":",util.yellow(getattr(opt,o)))
    print()

    return opt
