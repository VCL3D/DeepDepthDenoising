import torch.optim as optim

import sys

class OptimizerParameters(object):
    def __init__(self, learning_rate=0.001, momentum=0.9, momentum2=0.999,\
        epsilon=1e-8, weight_decay=0.0005, damp=0):
        super(OptimizerParameters, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.momentum2 = momentum2
        self.epsilon = epsilon
        self.damp = damp
        self.weight_decay = weight_decay

    def get_learning_rate(self):
        return self.learning_rate

    def get_momentum(self):
        return self.momentum

    def get_momentum2(self):
        return self.momentum2

    def get_epsilon(self):
        return self.epsilon

    def get_weight_decay(self):
        return self.weight_decay

    def get_damp(self):
        return self.damp

def get_optimizer(opt_type, model_params, opt_params):
    if opt_type == "adam":
        return optim.Adam(model_params, \
            lr=opt_params.get_learning_rate(), \
            betas=(opt_params.get_momentum(), opt_params.get_momentum2()), \
            eps=opt_params.get_epsilon(),
            weight_decay = opt_params.get_weight_decay() \
        )
    else:
        print("Error when initializing optimizer, {} is not a valid optimizer type.".format(opt_type), \
            file=sys.stderr)
        return None

def adjust_learning_rate(optimizer, epoch, scale=2):
    # Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    for param_group in optimizer.param_groups:
        lr =  param_group['lr']
        lr = lr * (0.1 ** (epoch // scale))
        param_group['lr'] = lr