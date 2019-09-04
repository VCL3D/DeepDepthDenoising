import torch

def transform_points(points, rotation, translation):
    b, _, h, w = points.size()  # [B, 3, H, W]
    points3d = points.reshape(b, 3, -1)  # [B, 3, H*W]
    return (
        (rotation @ points3d) # [B, 3, 3] * [B, 3, H*W]
        + translation # [B, 3, 1]
    ).reshape(b, 3, h, w)  # [B, 3, H, W]

def extract_rotation_translation(pose):
    b, _, _ = pose.shape
    return pose[:, :3, :3], pose[:,:3, 3].reshape(b, 3, 1) # rotation, translation