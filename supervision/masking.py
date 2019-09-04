import torch

def get_mask(depth, min_threshold=0.1, max_threshold=6.5):    
    mask = ((depth < max_threshold) & (depth > min_threshold)).detach().type(depth.dtype)
    count = torch.sum(mask).item()
    return mask, count