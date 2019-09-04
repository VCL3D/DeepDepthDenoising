import torch

from .losses import *
from .splatting import *

# https://discuss.pytorch.org/t/how-to-implement-accumulated-gradient/3822/8 ?

def depth_regularisation(inout):
    i = iter(inout.keys())
    key = next(i)
    loss, loss_map = inverse_huber_loss_masked(inout[key]["depth"]["original"],\
        inout[key]["depth"]["prediction"], inout[key]["depth"]["mask"],\
        inout[key]["depth"]["count"])
    for k in i:
        loss_tuple = inverse_huber_loss_masked(inout[k]["depth"]["original"],\
            inout[k]["depth"]["prediction"], inout[k]["depth"]["mask"],\
            inout[k]["depth"]["count"])
        loss += loss_tuple[0]
        loss_map += loss_tuple[1]
    return loss / len(inout.keys()), loss_map / len(inout.keys())

def robust_photometric_supervision_splat(inout, kernel_size=5, std=1.5, alpha=0.85, mode='gaussian'):
    i = iter(inout.keys())
    key = next(i)            
    mask = (inout[key]["color"]["splatted"] != 0).detach()\
        .type(inout[key]["color"]["splatted"].dtype)
    count = torch.sum(mask)
    masked_color = inout[key]["color"]["original"] * mask
    l1_l_map = mask * torch.abs(masked_color\
        - inout[key]["color"]["splatted"])
    l1_l_map = L1_Charbonnier_premasked(l1_l_map, epsilon_squared=0.2)
    l1_l = torch.sum(l1_l_map) / count
    ssim_l_map = mask * torch.clamp(\
        1 - ssim_loss(inout[key]["color"]["splatted"],\
        masked_color, kernel_size=kernel_size,\
        std=std, mode=mode) / 2, 0, 1)
    ssim_l_map = tukey_loss_map(ssim_l_map)
    ssim_l = torch.sum(ssim_l_map) / count
    for k in i:
        mask = (inout[k]["color"]["splatted"] != 0).detach()\
        .type(inout[k]["color"]["splatted"].dtype)
        count = torch.sum(mask)
        masked_color = inout[k]["color"]["original"] * mask
        l1_l_map_k = mask * torch.abs(\
            masked_color - inout[k]["color"]["splatted"])
        l1_l_map_k = L1_Charbonnier_premasked(l1_l_map_k, epsilon_squared=0.2)
        l1_l_map += l1_l_map_k
        l1_l += torch.sum(l1_l_map_k) / count
        ssim_l_map_k = mask * torch.clamp(\
            1 - ssim_loss(inout[k]["color"]["splatted"],\
            masked_color, kernel_size=kernel_size,\
            std=std, mode=mode) / 2, 0, 1)
        ssim_l_map_k = tukey_loss_map(ssim_l_map_k)
        ssim_l_map += ssim_l_map_k
        ssim_l += torch.sum(ssim_l_map_k) / count
    photo_l = l1_l / len(inout.keys()) * (1 - alpha)\
        + ssim_l / len(inout.keys()) * alpha
    photo_loss_map = (l1_l_map / len(inout.keys()) * (1 - alpha)\
        + ssim_l_map / len(inout.keys()) * alpha)
    return photo_l, photo_loss_map

def surface_smoothness_prior(inout):
    i = iter(inout.keys())
    key = next(i)
    loss, loss_map = normal_smoothness_loss(inout[key]["n3d"]["prediction"])
    for k in i:
        loss_tuple = normal_smoothness_loss(inout[k]["n3d"]["prediction"])
        loss += loss_tuple[0]
        loss_map += loss_tuple[1]
    return loss / len(inout.keys()), loss_map / len(inout.keys())