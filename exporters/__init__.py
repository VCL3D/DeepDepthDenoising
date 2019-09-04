from .image import *
from .point_cloud import *

def save_splatted_info(inout, path, super_list):
    def generator(name, type, ext):
        return os.path.join(path, name + "_" + type + "_#" + ext)
    for key in inout.keys():
        save_image(generator(key, "mask",".png"), inout[key]["depth"]["mask"])
        save_image(generator(key, "masked_color",".png"), inout[key]["color"]["masked"])
        save_depth(generator(key, "depth", ".png"), inout[key]["depth"]["original"])
        save_data(generator(key, "prediction", ".exr"), inout[key]["depth"]["prediction"])
        others = [other for other in inout.keys() if other != key]
        for other in others:
            save_image(generator(key, "splatted",".png"), inout[key]["color"]["splatted"])
        
