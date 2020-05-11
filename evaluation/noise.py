import torch
import random

def create_image_domain_grid(width, height, data_type=torch.float32):        
    v_range = (
        torch.arange(0, height) # [0 - h]
        .view(1, height, 1) # [1, [0 - h], 1]
        .expand(1, height, width) # [1, [0 - h], W]
        .type(data_type)  # [1, H, W]
    )
    u_range = (
        torch.arange(0, width) # [0 - w]
        .view(1, 1, width) # [1, 1, [0 - w]]
        .expand(1, height, width) # [1, H, [0 - w]]
        .type(data_type)  # [1, H, W]
    )
    return torch.stack((u_range, v_range), dim=1)  # [1, 3, H, W]

def disparity_noise(depth, sigma_depth=(1.0/6.0), sigma_space=(1.0/2.0), mean_space=0.5):
    b, c, h, w = depth.size()
    uvgrid = create_image_domain_grid(w, h)
    spatial_distribution = torch.randn_like(uvgrid) * sigma_space + mean_space
    offseted = (uvgrid + spatial_distribution).type(torch.int64)
    offseted[:, 0, :, :] = torch.clamp(offseted[:, 0, :, :], min=0, max=w-1)
    offseted[:, 1, :, :] = torch.clamp(offseted[:, 1, :, :], min=0, max=h-1)
    offsets = offseted[:, 1, :, :] * w + offseted[:, 0, :, :]
    linear_offsets = offsets.reshape(h*w)
    resampled_depth = torch.index_select(depth.reshape(h*w), 0, linear_offsets).reshape(b, c, h, w)
    depth_distribution = torch.randn_like(depth) * sigma_depth * sigma_depth
    baseline = torch.tensor(35130.0, dtype=torch.float32)
    denom = torch.round(baseline / (resampled_depth * 100.0) + depth_distribution + 0.5)
    noisy_depth = baseline / denom / 100.0
    return noisy_depth, resampled_depth

def tof_noise(depth, sigma_fraction=0.1):
    rand = torch.rand_like(depth)  
    sign = torch.ones_like(depth)
    sign[rand < 0.5] = -1.0
    sigma = sigma_fraction * depth
    magnitude = sigma * (1.0 - torch.exp(-0.5 * rand * rand))
    noisy_depth = depth + sign * magnitude
    return noisy_depth

