'''
    PyTorch implementation of https://github.com/google/layered-scene-inference
    accompanying the paper "Layer-structured 3D Scene Inference via View Synthesis", 
    ECCV 2018 https://shubhtuls.github.io/lsi/
'''

import torch

from .projections import *
from .transformations import *
from .masking import *

def depth_distance_weights(depth, mask=None, use_mask=False, max_depth=6.5):
    clamped_depth = torch.clamp(depth, 0.11, 6.4)
    weights = 1.0 / torch.exp(2 * clamped_depth / max_depth)
    if use_mask and mask is None:
        mask, _ = get_mask(depth)
        return weights * mask
    elif mask is not None:
        return weights * mask
    else:
        return weights

def normal_weights(normals):
    z_dir = normals[:, 2, :, :]
    return torch.abs(z_dir).unsqueeze(1)

def fov_weights(coords, horizontal_fov=69.4, vertical_fov=42.5):
    half_horizontal_fov = torch.tensor(horizontal_fov / 2)
    half_vertical_fov = torch.tensor(vertical_fov / 2)
    _, __, h, w = coords.size()
    horizontal_center = w / 2
    vertical_center = h / 2
    u_d = (coords[:, 0, :, :] - horizontal_center) / (w / 2) + 1e-8
    v_d = (coords[:, 1, :, :] - vertical_center) / (h / 2) + 1e-8
    r_d = torch.sqrt(u_d * u_d + v_d * v_d)
    r_u_ud = torch.tan(r_d * torch.tan(half_horizontal_fov)) / torch.tan(half_horizontal_fov)
    r_v_ud = torch.tan(r_d * torch.tan(half_vertical_fov)) / torch.tan(half_vertical_fov)
    r_ud = torch.sqrt(r_u_ud * r_u_ud + r_v_ud * r_v_ud)
    dist = (r_d / r_ud).unsqueeze(1)
    dist = torch.abs((dist - torch.mean(dist)) / torch.std(dist))
    return torch.exp(dist * (dist < 3 * torch.std(dist)).type(dist.dtype))

def weighted_average_splat(depth, weights, epsilon=1e-8):
    zero_weights = (weights <= epsilon).detach().type(depth.dtype)
    return depth / (weights + epsilon * zero_weights)

def splat(values, coords, splatted):
    b, c, h, w = values.size()

    uvs = coords
    u = uvs[:, 0, :, :].unsqueeze(1)
    v = uvs[:, 1, :, :].unsqueeze(1)
    
    u0 = torch.floor(u)
    u1 = u0 + 1
    v0 = torch.floor(v)
    v1 = v0 + 1

    u0_safe = torch.clamp(u0, 0.0, w-1)
    v0_safe = torch.clamp(v0, 0.0, h-1)
    u1_safe = torch.clamp(u1, 0.0, w-1)
    v1_safe = torch.clamp(v1, 0.0, h-1)

    u0_w = (u1 - u) * (u0 == u0_safe).type(values.dtype)
    u1_w = (u - u0) * (u1 == u1_safe).type(values.dtype)
    v0_w = (v1 - v) * (v0 == v0_safe).type(values.dtype)
    v1_w = (v - v0) * (v1 == v1_safe).type(values.dtype)

    top_left_w = u0_w * v0_w
    top_right_w = u1_w * v0_w
    bottom_left_w = u0_w * v1_w
    bottom_right_w = u1_w * v1_w

    weight_threshold = 1e-3
    top_left_w *= (top_left_w >= weight_threshold).type(values.dtype)
    top_right_w *= (top_right_w >= weight_threshold).type(values.dtype)
    bottom_left_w *= (bottom_left_w >= weight_threshold).type(values.dtype)
    bottom_right_w *= (bottom_right_w >= weight_threshold).type(values.dtype)

    for channel in range(c):
        top_left_values = values[:, channel, :, :].unsqueeze(1) * top_left_w
        top_right_values = values[:, channel, :, :].unsqueeze(1) * top_right_w
        bottom_left_values = values[:, channel, :, :].unsqueeze(1) * bottom_left_w
        bottom_right_values = values[:, channel, :, :].unsqueeze(1) * bottom_right_w

        top_left_values = top_left_values.reshape(b, -1)
        top_right_values = top_right_values.reshape(b, -1)
        bottom_left_values = bottom_left_values.reshape(b, -1)
        bottom_right_values = bottom_right_values.reshape(b, -1)

        top_left_indices = (u0_safe + v0_safe * w).reshape(b, -1).type(torch.int64)
        top_right_indices = (u1_safe + v0_safe * w).reshape(b, -1).type(torch.int64)
        bottom_left_indices = (u0_safe + v1_safe * w).reshape(b, -1).type(torch.int64)
        bottom_right_indices = (u1_safe + v1_safe * w).reshape(b, -1).type(torch.int64)
        
        splatted_channel = splatted[:, channel, :, :].unsqueeze(1)
        splatted_channel = splatted_channel.reshape(b, -1)
        splatted_channel.scatter_add_(1, top_left_indices, top_left_values)
        splatted_channel.scatter_add_(1, top_right_indices, top_right_values)
        splatted_channel.scatter_add_(1, bottom_left_indices, bottom_left_values)
        splatted_channel.scatter_add_(1, bottom_right_indices, bottom_right_values)
    splatted = splatted.reshape(b, c, h, w)
