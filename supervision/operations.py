from .projections import *
from .transformations import *
from .ssim import *
from .smoothness import *
from .masking import *
from .normals import *
from .splatting import *

import os

def get_processed_info(batch, model, device, super_list, threshold=3.0):
    processed = ({
        k: { "depth": {}, "color": {}, "p3d": {}, "n3d":{} } for k in batch.keys() 
    })
    for key, proc in processed.items():
        others = [other for other in super_list if other != key]
        for o in others:
            proc[o] = {
                "depth": {},
                "color": {},
                "p3d": {},                        
            }
    for viewpoint in batch:
        for attribure in batch[viewpoint]:
            batch[viewpoint][attribure] = batch[viewpoint][attribure].to(device)
        processed[viewpoint]["depth"]["original"] = batch[viewpoint]["depth"]
        processed[viewpoint]["color"]["original"] = batch[viewpoint]["color"]
        mask, count = get_mask(processed[viewpoint]["depth"]["original"], max_threshold=threshold)
        processed[viewpoint]["depth"]["mask"] = mask
        processed[viewpoint]["depth"]["count"] = count
        processed[viewpoint]["color"]["masked"] = processed[viewpoint]["color"]["original"] * mask
        predicted_depth, activations = model(batch[viewpoint]["depth"], mask)
        processed[viewpoint]["depth"]["prediction"] = predicted_depth
        processed[viewpoint]["activations"] = activations
    return processed

def add_3d_info(inout, info, uv_grid):
    for key in inout.keys():# i.e. src                
        inout[key]["p3d"]["prediction"] = deproject_depth_to_points(\
            inout[key]["depth"]["prediction"], uv_grid, info[key]["intrinsics_inv"])
        others = [k for k in inout.keys() if k != key]
        for other in others:# i.e. tgt
            pose2other = info[other]["extrinsics_inv"] @ info[key]["extrinsics"]
            rot2other, trans2other = extract_rotation_translation(pose2other)
            p3d2other = transform_points(inout[key]["p3d"]["prediction"], rot2other, trans2other / 1000.0)
            uvs2other = project_points_to_uvs(p3d2other, info[other]["intrinsics"])            
            depth2other = p3d2other[:, 2, :, :].unsqueeze(1)           
            inout[key][other]["p3d"]["z"] = depth2other
            inout[key][other]["uvs"] = uvs2other

def add_normal_info(inout):
    for key in inout.keys():# i.e. src
        inout[key]["n3d"]["prediction"] = calculate_normals(inout[key]["p3d"]["prediction"])
        inout[key]["n3d"]["weights"] = normal_weights(inout[key]["n3d"]["prediction"])

def add_forward_rendering_info(inout, uv_grid, use_depth=True, depth_threshold=6.5,\
    use_normals=False, normal_threshold=0.4, fov_w=None):
    for key in inout.keys():# i.e. tgt
        b, _, h, w = inout[key]["depth"]["prediction"].size()
        splatted_weighted_depth = torch.zeros(b, 1, h, w).to(inout[key]["depth"]["prediction"].device)
        splatted_weighted_color = torch.zeros(b, 3, h, w).to(inout[key]["depth"]["prediction"].device)
        splatted_weights = torch.zeros(b, 1, h, w).to(inout[key]["depth"]["prediction"].device)
        others = [k for k in inout.keys() if k != key]
        for other in others: # i.e. src
            inout[other][key]["color"]["weights"] = depth_distance_weights(\
                inout[other][key]["p3d"]["z"], mask=inout[other]["depth"]["mask"],\
                max_depth=depth_threshold)
            if fov_w is not None:
                inout[other][key]["color"]["weights"] *= fov_w
            #TODO: normals usage                        
            splat(inout[other][key]["color"]["weights"] * inout[other][key]["p3d"]["z"],\
                inout[other][key]["uvs"], splatted_weighted_depth)
            splat(inout[other][key]["color"]["weights"] * inout[other]["color"]["original"],\
                inout[other][key]["uvs"], splatted_weighted_color)
            splat(inout[other][key]["color"]["weights"], inout[other][key]["uvs"], splatted_weights)            
        inout[key]["color"]["splatted"] = inout[key]["depth"]["mask"]\
            * weighted_average_splat(splatted_weighted_color, splatted_weights)
        inout[key]["depth"]["splatted"] = inout[key]["depth"]["mask"]\
            * weighted_average_splat(splatted_weighted_depth, splatted_weights)
