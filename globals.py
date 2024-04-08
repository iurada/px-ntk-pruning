import torch
import torch.backends.mps

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

CONFIG = dotdict({})

if torch.cuda.is_available():
    CONFIG.device = 'cuda'
elif torch.backends.mps.is_available() and \
    torch.backends.mps.is_built():
    CONFIG.device = 'mps'
else:
    CONFIG.device = 'cpu'

CONFIG.dtype = torch.float32