import torch
import numpy as np

def log_var(text, variable = 'None', depth = 0, default = torch.Tensor, additional = None):
    if additional is not None:
        text = text + ' ' + str(additional)
    if variable == 'None':
        print('\t' * depth, text)
    elif variable is None:
        print('\t' * depth + '  ', text, 'None')
    elif hasattr(variable, 'shape'):
        print('\t' * depth + '  ', text, variable.shape, variable.dtype, type(variable) if type(variable) != default else '')
    elif isinstance(variable, dict):
        print('\t' * depth + '  ', text, type(variable))
        for k, v in variable.items():
            log_var(k, v, depth = depth + 1)
    elif isinstance(variable, list) or isinstance(variable, tuple):
        print('\t' * depth + '  ', text, type(variable))
        for i, v in enumerate(variable):
            log_var('%d' % i , v, depth = depth + 1)
    elif np.isscalar(variable):
        print('\t' * depth + '  ', text, variable, type(variable))
    else:
        print('\t' * depth + '  ', text, type(variable))

def print_shape(obj, name = None, layer = 0):
    if isinstance(obj, dict):
        print('\t'*layer + '%s(dict):' % name if name else '')
        for k, v in obj.items():
            print_shape(v, k, layer = layer+1)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        print('\t'*layer + '%s(iter):' % name if name else '')
        for v in obj:
            print_shape(v, layer = layer+1)
    else:
        if hasattr(obj, 'shape'):
            print('\t'*layer + "%s: %s %s" % (name if name else '', type(obj), str(obj.shape)))
        else:
            print('\t'*layer + "%s: %s %s" % (name if name else '', type(obj), str(obj)))