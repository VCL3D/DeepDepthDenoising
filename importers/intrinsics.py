import json
import numpy
import torch


def load_intrinsics_repository(filename):    
    #global intrinsics_dict
    with open(filename, 'r') as json_file:
        intrinsics_repository = json.load(json_file)
        intrinsics_dict = dict((intrinsics['Device'], \
            intrinsics['Depth Intrinsics'][0]['1280x720'])\
                for intrinsics in intrinsics_repository)
    return intrinsics_dict

def get_intrinsics(name, intrinsics_dict, scale=1, data_type=torch.float32):
    #global intrinsics_dict
    if intrinsics_dict is not None:
        intrinsics_data = numpy.array(intrinsics_dict[name])
        intrinsics = torch.tensor(intrinsics_data).reshape(3, 3).type(data_type)    
        intrinsics[0, 0] = intrinsics[0, 0] / scale
        intrinsics[0, 2] = intrinsics[0, 2] / scale
        intrinsics[1, 1] = intrinsics[1, 1] / scale
        intrinsics[1, 2] = intrinsics[1, 2] / scale
        intrinsics_inv = intrinsics.inverse()
        return intrinsics, intrinsics_inv
    raise ValueError("Intrinsics repository is empty")
    