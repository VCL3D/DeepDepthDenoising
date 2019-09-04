import os
import sys
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import importers
import warnings

#testing
# sys.path.append('E:\\Projects\\vsc\\deep_depth_denoising\\denoise')
# import importers


'''
Dataset importer. We assume that data follows the below structure.
root_path
device_repository.json
	|
	|-----recording_i
	|		|-----Data
	|		|-----Calibration
	|
	|-----recording_i+1
	|		|-----Data
	|		|-----Calibration
	|
'''

class DataLoaderParams:
	def __init__(self
	,root_path 
	,device_list
	,decimation_scale = 2
	,device_repository_path = "."
	,depth_scale = 0.001
	,depth_threshold = 5):
		self.root_path = root_path
		self.device_list = device_list
		self.device_repository_path = device_repository_path
		self.depth_scale = depth_scale
		self.decimation_scale = decimation_scale
		self.depth_threshold = depth_threshold

class DataLoad(Dataset):
	def __init__(self, params):
		super(DataLoad,self).__init__()
		self.params = params
		
		device_repo_path = os.path.join(self.params.device_repository_path,"device_repository.json")
		if not os.path.exists(device_repo_path):
			raise ValueError("{} does not exist, exiting.".format(device_repo_path))
		self.device_repository = importers.intrinsics.load_intrinsics_repository(device_repo_path)

		root_path = self.params.root_path
		

		if not os.path.exists(root_path):
			raise ValueError("{} does not exist, exiting.".format(root_path))

		self.data = {}

		# iterate over each recorded folder
		for recording in os.listdir(root_path):
			abs_recording_path = os.path.join(root_path,recording)
			if not os.path.isdir(abs_recording_path):
				continue
			# path where data supposed to be stored
			data_path = os.path.join(abs_recording_path,"Data")

			if not os.path.exists(data_path):
				warnings.warn("Folder {} does not containt \"Data\" folder".format(abs_recording_path))
				continue
			
			# path to the calibration of that particular recording
			calibration_path = os.path.join(abs_recording_path,"Calibration")
			if not os.path.exists(calibration_path):
				warnings.warn("Folder {} does not containt \"Calibration\" folder".format(calibration_path))
				continue
			
			# data iteration
			for file in os.listdir(data_path):
				full_filename = os.path.join(data_path,file)

				_, ext = os.path.splitext(full_filename)
				if ext != ".png":
					continue
					
				_id,_name,_type,_ = file.split("_")
				unique_name = recording + "-" + str(_id)
				
				# skip names that we do not want to load
				if _name not in self.params.device_list:
					continue

				if unique_name not in self.data:
					self.data[unique_name] = {}
					self.data[unique_name]["calibration"] = calibration_path

				if _name not in self.data[unique_name]:
					self.data[unique_name][_name] = {}

				self.data[unique_name][_name][_type] = full_filename


	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		#get an entry
		key = list(self.data.keys())[idx]
		datum = self.data[key]
		
		datum_out = {}

		for device in self.params.device_list:
			color_img = importers.image.load_image(datum[device]["color"])
			depth_range_mask = (importers.image.load_depth(datum[device]["depth"]) < self.params.depth_threshold).float()
			depth_img = importers.image.load_depth(datum[device]["depth"]) * depth_range_mask
			intrinsics, intrinsics_inv = importers.intrinsics.get_intrinsics(\
				device, self.device_repository, self.params.decimation_scale)
			extrinsics, extrinsics_inv = importers.extrinsics.load_extrinsics(\
				os.path.join(datum["calibration"], device + ".extrinsics"))
			
			datum_out.update({
				device : {
				"color" : color_img.squeeze(0),
				"depth" : depth_img.squeeze(0), 
				"intrinsics" : intrinsics,
				"intrinsics_inv" : intrinsics_inv,
				"extrinsics" : extrinsics,
				"extrinsics_inv" : extrinsics_inv
				}})
		
		return datum_out

	def get_data(self):
		return self.data