import torch
import torch.nn.functional as F

def calculate_normals(points , policy = "upright"):
    if policy is "upright":
        points_temp = F.pad(points, (0, 1, 0, 0), mode="replicate")
        dx = points_temp[:, :, :, :-1] - points_temp[:, :, :, 1:]  # NCHW
        points_temp = F.pad(points, (0, 0, 0, 1), mode="replicate")
        dy = points_temp[:, :, :-1, :] - points_temp[:, :, 1:, :]  # NCHW
        normals = torch.cross(dy,dx)
        #mask = (points[:,2,:,:] == 0).float()
        #normals /= torch.sqrt(torch.sum(normals*normals, 1) + mask)        
        #return normals
        return torch.nn.functional.normalize(normals)