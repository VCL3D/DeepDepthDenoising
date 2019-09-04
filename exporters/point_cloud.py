import torch
import os

def save_ply(filename, tensor, scale, color='black' , normals = None):    
    b, _, h, w = tensor.size()
    for n in range(b):
        coords = tensor[n, :, :, :].detach().cpu().numpy()
        x_coords = coords[0, :] * scale
        y_coords = coords[1, :] * scale
        z_coords = coords[2, :] * scale
        if normals is not None:
            norms = normals[n, : , : , :].detach().cpu().numpy()
            nx_coords = norms[0, :]
            ny_coords = norms[1, :]
            nz_coords = norms[2, :]
        with open(filename.replace("#", str(n)), "w") as ply_file:        
            ply_file.write("ply\n")
            ply_file.write("format ascii 1.0\n")
            ply_file.write("element vertex {}\n".format(w * h))
            ply_file.write("property float x\n")
            ply_file.write("property float y\n")
            ply_file.write("property float z\n")
            if normals is not None:
                ply_file.write('property float nx\n')
                ply_file.write('property float ny\n')
                ply_file.write('property float nz\n')
            ply_file.write("property uchar red\n")
            ply_file.write("property uchar green\n")
            ply_file.write("property uchar blue\n")
            ply_file.write("end_header\n")
            
            if normals is None:
                for x in torch.arange(w):
                    for y in torch.arange(h):
                        ply_file.write("{} {} {} {} {} {}\n".format(\
                            x_coords[y, x], y_coords[y, x], z_coords[y, x],\
                            "255" if color=='red' else "0",
                            "255" if color=='green' else "0",
                            "255" if color=='blue' else "0"))
            else:
                for x in torch.arange(w):
                    for y in torch.arange(h):
                        ply_file.write("{} {} {} {} {} {} {} {} {}\n".format(\
                            x_coords[y, x], y_coords[y, x], z_coords[y, x],\
                            nx_coords[y, x], ny_coords[y, x], nz_coords[y, x],\
                            "255" if color=='red' else "0",
                            "255" if color=='green' else "0",
                            "255" if color=='blue' else "0"))