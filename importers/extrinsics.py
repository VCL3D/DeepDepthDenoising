import numpy
import torch

def load_extrinsics(filename, data_type=torch.float32):
    data = numpy.loadtxt(filename)
    pose = numpy.zeros([4, 4])
    pose[3, 3] = 1
    pose[:3, :3] = data[:3, :3]
    pose[:3, 3] = data[3, :]
    extrinsics = torch.tensor(pose, dtype=data_type)
    return extrinsics , extrinsics.inverse()