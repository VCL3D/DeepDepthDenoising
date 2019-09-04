import torch
import torch.nn.functional as F

def gradient_x(tensor):
        # pad input to keep output size consistent
        tensor = F.pad(tensor, (0, 1, 0, 0), mode="replicate")
        gx = tensor[:, :, :, :-1] - tensor[:, :, :, 1:]  # NCHW
        return gx

def gradient_y(tensor):
        # pad input to keep output size consistent
        tensor = F.pad(tensor, (0, 0, 0, 1), mode="replicate")
        gy = tensor[:, :, :-1, :] - tensor[:, :, 1:, :]  # NCHW
        return gy


def masked_gradient_x(tensor):
        # pad input to keep output size consistent
        tensor = F.pad(tensor, (0, 1, 0, 0), mode="replicate")
        gx = tensor[:, :, :, :-1] - tensor[:, :, :, 1:]  # NCHW
        mask = (tensor[:, :, :, :-1] != 0) * (tensor[:, :, :, 1:] != 0)
        return gx, mask

def masked_gradient_y(tensor):
        # pad input to keep output size consistent
        tensor = F.pad(tensor, (0, 0, 0, 1), mode="replicate")
        gy = tensor[:, :, :-1, :] - tensor[:, :, 1:, :]  # NCHW
        mask = (tensor[:, :, :-1, :] != 0) * (tensor[:, :, 1:, :] != 0)
        return gy, mask
