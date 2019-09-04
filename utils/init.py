import torch

import os
import sys

def initialize_weights(model, init = "xavier"):    
    init_func = None
    if init == "xavier":
        init_func = torch.nn.init.xavier_normal_
    elif init == "kaiming":
        init_func = torch.nn.init.kaiming_normal_
    elif init == "gaussian" or init == "normal":
        init_func = torch.nn.init.normal_
      
    if init_func is not None:
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) \
                or isinstance(module, torch.nn.ConvTranspose2d):
                    init_func(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
    elif os.path.exists(init):
        weights = torch.load(init, map_location={'cuda:1':'cuda:0'})
        model.load_state_dict(weights["model_state_dict"])        
    else:
        print("Error when initializing model's weights, {} either doesn't exist or is not a valid initialization function.".format(init), \
            file=sys.stderr)

