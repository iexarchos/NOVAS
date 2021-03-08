import numpy as np

import os,sys,time
import termcolor

# convert to colored strings
def red(content): return termcolor.colored(str(content),"red",attrs=["bold"])
def green(content): return termcolor.colored(str(content),"green",attrs=["bold"])
def blue(content): return termcolor.colored(str(content),"blue",attrs=["bold"])
def cyan(content): return termcolor.colored(str(content),"cyan",attrs=["bold"])
def yellow(content): return termcolor.colored(str(content),"yellow",attrs=["bold"])
def magenta(content): return termcolor.colored(str(content),"magenta",attrs=["bold"])

def get_time(sec):
	h = int(sec//3600)
	m = int((sec//60)%60)
	s = sec%60
	return h,m,s

def restore_checkpoint(opt,model,load_name,keys):
	print(magenta("loading checkpoint {}...".format(load_name)))
	with torch.cuda.device(opt.gpu):
		checkpoint = torch.load(load_name,map_location=opt.device)
		for k in keys:
			getattr(model,k).load_state_dict(checkpoint[k])

def save_checkpoint(opt,model,keys,ep):
	os.makedirs("checkpoint/{0}/{1}".format(opt.group,opt.name),exist_ok=True)
	checkpoint = {}
	with torch.cuda.device(opt.gpu):
		for k in keys:
			checkpoint[k] = getattr(model,k).state_dict()
		torch.save(checkpoint,"checkpoint/{0}/{1}/ep{2}.npz".format(opt.group,opt.name,ep))
	print(green("checkpoint saved: ({0}) {1}, epoch {2}".format(opt.group,opt.name,ep)))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def to_numpy(var):
#     return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

# def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
#     return Variable(
#         torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
#     ).type(dtype)