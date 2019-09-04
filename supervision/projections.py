import torch

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
    ones = (
        torch.ones(1, height, width) # [1, H, W] := 1
        .type(data_type)
    )
    return torch.stack((u_range, v_range, ones), dim=1)  # [1, 3, H, W]

def project_points_to_uvs(points, intrinsics):
    b, _, h, w = points.size()  # [B, 3, H, W]
    x_coordinate3d = points[:, 0] #TODO: check if adding small value makes sense to avoid zeros?
    y_coordinate3d = points[:, 1]
    z_coordinate3d = points[:, 2].clamp(min=1e-3)
    x_homogeneous = x_coordinate3d / z_coordinate3d
    y_homogeneous = y_coordinate3d / z_coordinate3d
    ones = z_coordinate3d.new_ones(z_coordinate3d.size())
    homogeneous_coordinates = ( # (x/z, y/z, 1.0)
        torch.stack([x_homogeneous, y_homogeneous, ones], dim=1)  # [B, 3, H, W]
        .reshape(b, 3, -1) # [B, 3, H*W]
    )
    uv_coordinates = intrinsics @ homogeneous_coordinates # [B, 3, H*W]
    return ( # image domain coordinates
        uv_coordinates[:, :2, :] # [B, 2, H*W]
        .reshape(b, 2, h, w) # [B, 2, H, W]
    ) # [B, 2, H, W]

def deproject_depth_to_points(depth, grid, intrinsics_inv):     
    b, _, h, w = depth.size()
    # check https://pytorch.org/docs/stable/torch.html#torch.matmul 
    # need to return a one-dimensional tensor to use the matrix-vector product
    # as a result we reshape to [B, 3, H*W] in order to multiply the intrinsics matrix
    # with a 3x1 vector (u, v, 1)
    current_pixel_coords = ( # convert grid to appropriate dims for matrix multiplication
        grid # [1, 3, H, W] #grid[:,:,:h,:w]
        .expand(b, 3, h, w) # [B, 3, H, W]
        .reshape(b, 3, -1)  # [B, 3, H*W] := [B, 3, UV1]    
    )
    p3d = ( # K_inv * [UV1] * depth
        (intrinsics_inv @ current_pixel_coords) # [B, 3, 3] * [B, 3, UV1]
        .reshape(b, 3, h, w) * # [B, 3, H, W]
        depth
        #.unsqueeze(1) # unsqueeze to tri-channel for element wise product
    ) # [B, 3, H, W]
    return p3d