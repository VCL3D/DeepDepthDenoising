from .projections import *
from .transformations import *
from .ssim import *
from .smoothness import *

import torch

def get_mask(depth, min_threshold=0.1, max_threshold=6.5):    
    mask = ((depth < max_threshold) & (depth > min_threshold)).detach().type(depth.dtype)
    count = torch.sum(mask).item()
    return mask, count

def inverse_huber_loss_masked(gt, pred, mask, count):
    diff = gt - pred
    abs_diff = torch.abs(diff)
    c = torch.max(abs_diff).item() / 5
    leq = (abs_diff <= c).float()
    l2_losses = (diff**2 + c**2) / (2 * c)
    loss = leq * abs_diff + (1 - leq) * l2_losses
    _, c, __, ___ = loss.size()
    masked_loss = loss * mask
    return torch.sum(masked_loss) / count, masked_loss  

def L1_Charbonnier_premasked(map_loss, epsilon_squared=0.1):
    error = torch.sqrt(map_loss * map_loss + epsilon_squared)
    error = error - numpy.sqrt(epsilon_squared)
    return error

def negate_byte_mask(mask):
        return mask.int() ^ 1

def normal_smoothness_loss(pred_normals):
    # ATTENTION : assuming that pred_normals is normized
    #             computing the sum of cos(theta) in a neighbourhood between normal
    #             vectors around a 3D point
    extended_normals = torch.nn.functional.pad(pred_normals,(1,1,1,1)) #mode = "constant" , value = 0

    b,c,h,w = pred_normals.shape
    
    # non_zero_neighbours = torch.zeros((b,1,h,w)).to(torch.device("cuda:0"))
    non_zero_neighbours = torch.zeros((b,1,h,w)).to(pred_normals.device)
    # up-left
    loss = torch.abs(torch.sum(extended_normals[:,:,:-2,:-2] * pred_normals , 1)).unsqueeze(1) ##inner product
    non_zero_neighbours += (torch.sum(extended_normals[:,:,:-2,:-2],1) != 0).float().unsqueeze(1)
    # up
    loss += torch.abs(torch.sum(extended_normals[:,:,:-2,1:-1] * pred_normals , 1)).unsqueeze(1) ##inner product
    non_zero_neighbours += (torch.sum(extended_normals[:,:,:-2,1:-1],1) != 0).float().unsqueeze(1)
    # up-right
    loss += torch.abs(torch.sum(extended_normals[:,:,:-2,2:] * pred_normals , 1)).unsqueeze(1) ##inner product
    non_zero_neighbours += (torch.sum(extended_normals[:,:,:-2,2:],1) != 0).float().unsqueeze(1)
    # right
    loss += torch.abs(torch.sum(extended_normals[:,:,1:-1,:-2] * pred_normals , 1)).unsqueeze(1) ##inner product
    non_zero_neighbours += (torch.sum(extended_normals[:,:,1:-1,:-2],1) != 0).float().unsqueeze(1)
    # left
    loss += torch.abs(torch.sum(extended_normals[:,:,1:-1,2:] * pred_normals , 1)).unsqueeze(1) ##inner product
    non_zero_neighbours += (torch.sum(extended_normals[:,:,1:-1,2:],1) != 0).float().unsqueeze(1)
    # down-left
    loss += torch.abs(torch.sum(extended_normals[:,:,2:,:-2] * pred_normals , 1)).unsqueeze(1) ##inner product
    non_zero_neighbours += (torch.sum(extended_normals[:,:,2:,:-2],1) != 0).float().unsqueeze(1)
    # down
    loss += torch.abs(torch.sum(extended_normals[:,:,2:,1:-1] * pred_normals , 1)).unsqueeze(1) ##inner product
    non_zero_neighbours += (torch.sum(extended_normals[:,:,2:,1:-1],1) != 0).float().unsqueeze(1)
    # down-right
    loss += torch.abs(torch.sum(extended_normals[:,:,2:,2:] * pred_normals , 1)).unsqueeze(1) ##inner product
    non_zero_neighbours += (torch.sum(extended_normals[:,:,2:,2:],1) != 0).float().unsqueeze(1)

    # now loss contains the cos(theta) losses of the normals with every neighbour
    # now it has to be normilized, diveded by zero_neighbours,but it may contain zeros

    # where zero neighbours is zero, replace it with 1
    zero_mask = (non_zero_neighbours == 0).float()
    non_zero_mask = negate_byte_mask(zero_mask).float()
    loss = loss / (non_zero_neighbours + zero_mask)
   
    loss_map = (1 - loss) * non_zero_mask
   
    return torch.sum(loss_map) / torch.sum(non_zero_mask), loss_map

def tukey_loss_map(x, c=2.2):
    y = (c**2 / 6) * (1. - (1. - (x / c)**2)**3)
    y = torch.clamp(y, 0, c**2 / 6)
    return y

class AverageMeter(object):
    # Computes and stores the average and current value
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = torch.tensor(0.0)
        self.avg = torch.tensor(0.0)
        self.sum = torch.tensor(0.0)
        self.count = torch.tensor(0.0)

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count