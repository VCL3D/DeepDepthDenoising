import torch
from os import path


def save_network_state(model, optimizer, epoch , name , save_path):
    if not path.exists(save_path):
        raise ValueError("{} not a valid path to save model state".format(save_path))
    torch.save(
        {
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict()
        }, path.join(save_path, name))

