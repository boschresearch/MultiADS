# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os
import pdb

import os
import json
import random
import numpy as np
import torch
from torch.utils import data
from PIL import Image

import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


import os
import json
import random
import numpy as np
import torch
from torch.utils import data
from PIL import Image

import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

MVTEC_SPECIE2ID = {"good":0, "bent":1, "bent_lead":1, "bent_wire":1, "manipulated_front":1, "broken":2, "broken_large":2, "broken_small":2, "broken_teeth":2, "color":3, "combined":4, "contamination":5, "metal_contamination":5, "crack":6, "cut":7, "cut_inner_insulation":7, "cut_lead":7, "cut_outer_insulation":7, "fabric":8, "fabric_border":8, "fabric_interior":8, "faulty_imprint":9, "print":9, "glue":10, "glue_strip":10, "hole":11, "missing":12, "missing_wire":12, "missing_cable":12, "poke":13, "poke_insulation":13, "rough":14, "scratch":15, "scratch_head":15, "scratch_neck":15, "squeeze":16, "squeezed_teeth":16, "thread":17, "thread_side":17, "thread_top":17, "liquid":18, "oil":18, "misplaced":19, "cable_swap":19, "flip":19, "fold":19, "split_teeth":19, "damaged_case":20, "defective":20, "gray_stroke":20, "pill_type":20}  
VISA_SPECIE2ID = {'normal': 0, 'damage': 1, 'scratch':2, 'breakage': 3, 'burnt': 4, 'weird wick': 5, 'stuck': 6, 'crack': 7, 'wrong place': 8, 'partical': 9, 'bubble': 10, 'melded': 11, 'hole': 12, 'melt': 13, 'bent':14, 'spot': 15, 'extra': 16, 'chip': 17, 'missing': 18}
MPDD_SPECIE2ID =  {"good":0, 'hole':1, 'scratches':2, 'bend_and_parts_mismatch':3, 'parts_mismatch':4, 'defective_painting':5, 'major_rust':6, 'total_rust':6, 'flattening':7}
MAD_SIM_SPECIE2ID = {"good": 0, "Stains": 1, "Burrs": 2, "Missing": 3}
MAD_REAL_SPECIE2ID = {"good": 0, "Stains": 1, "Missing": 2}
REAL_IAD_SPECIE2ID = {"good":0, 'pit':1, 'deformation':2, 'abrasion':3, 'scratch':4, 'damage':5, 'missing':6, 'foreign':7, 'contamination':8}

class MVTecDataset(data.Dataset):
	"""
	Output:
	  - img: image tensor
	  - img_mask_b: [H,W] float binary mask in {0,1}
	  - img_mask:   [H,W] long mask with values in 0..K-1 (pixel-wise defect id, with good=0)
	  - cls_name: object category (e.g., "transistor")
	  - specie_name / specie_id: for single image, the defect type; for mosaic, returns "mosaic" / -1
	  - anomaly: 0/1 (for mosaic, recomputed based on whether any anomalous pixels exist)
	  - img_path: absolute path to the image
	"""

	def __init__(
		self,
		root,
		transform,
		target_transform,
		target_transform_type,
		aug_rate,
		mode='test',
		k_shot=0,
		save_dir=None,
		obj_name=None,
		specie2id=MVTEC_SPECIE2ID,
	):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
		name = self.root.split('/')[-1]
		meta_info = meta_info[mode]
		if mode == 'train':
			self.cls_names = [obj_name]
			save_dir = os.path.join(save_dir, 'k_shot.txt')
		else:
			if obj_name is None:
				self.cls_names = list(meta_info.keys())
			else:
				self.cls_names = [obj_name]
		for cls_name in self.cls_names:
			if mode == 'train':
				data_tmp = meta_info[cls_name]
				indices = torch.randint(0, len(data_tmp), (k_shot,))
				for i in range(len(indices)):
					self.data_all.append(data_tmp[indices[i]])
					with open(save_dir, "a") as f:
						f.write(data_tmp[indices[i]]['img_path'] + '\n')
			else:
				self.data_all.extend(meta_info[cls_name])
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def get_cls_names(self):
		return self.cls_names

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
															  data['specie_name'], data['anomaly']
		img = Image.open(os.path.join(self.root, img_path))
		if anomaly == 0:
			img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
		else:
			img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
			img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask

		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
				'img_path': os.path.join(self.root, img_path)}

class VisaDataset(data.Dataset):
	def __init__(self, root, transform, target_transform, target_transform_type, specie2id=VISA_SPECIE2ID, mode='test', k_shot=0, save_dir=None, obj_name=None):
		self.root = root
		self.transform = transform
		self.target_transform_b = target_transform
		self.target_transform_type = target_transform_type
		self.specie2id = specie2id

		if self.specie2id is None:
			raise ValueError("specie2id must be provided (fixed mapping).")
		if "normal" not in self.specie2id:
			raise ValueError("specie2id must contain key 'normal'.")
		if self.specie2id["normal"] != 0:
			raise ValueError("specie2id['normal'] must be 0.")

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/meta_wo_md.json', 'r'))
		name = self.root.split('/')[-1]

		self.cls_names = list(meta_info.keys())
		for cls_name in self.cls_names:
			self.data_all.extend(meta_info[cls_name])
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def get_cls_names(self):
		return self.cls_names

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, anomaly, defect_cls = data['img_path'], data['mask_path'], data['cls_name'], data['anomaly'], data['defect_cls']
		img = Image.open(os.path.join(self.root, img_path))
		if anomaly == 0:
			img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
		else:
			img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
			img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask

		gt_b_t = self.target_transform_b(gt_b_pil)
		if torch.is_tensor(gt_b_t) and gt_b_t.ndim == 3 and gt_b_t.shape[0] == 1:
			gt_b_t = gt_b_t.squeeze(0)
		gt_b = (gt_b_t > 0.5).float()	# [H,W] 0/1

		gt_t = self.target_transform_type(gt_pil)
		if torch.is_tensor(gt_t) and gt_t.ndim == 3 and gt_t.shape[0] == 1:
			gt_t = gt_t.squeeze(0)
		gt = gt_t.long()				# [H,W] 0..K-1

		return {
			'img': img,
			'img_mask_b': gt_b,
			'img_mask': gt,
			'cls_name': cls_name,
			'specie_name': specie_name,
			'specie_id': specie_id,
			'anomaly': anomaly,
			'img_path': os.path.join(self.root, img_path),
		}

class VisaDatasetTest(data.Dataset):
	def __init__(self, root, transform=None, target_transform=None, mode='test', k_shot=0, save_dir=None, obj_name=None):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
		meta_info = meta_info[mode]

		if mode == 'train':
			self.cls_names = [obj_name]
			if save_dir is not None:
				save_dir = os.path.join(save_dir, 'k_shot.txt')
		else:
			if obj_name is None:
				self.cls_names = list(meta_info.keys())
			else:
				self.cls_names = [obj_name]

		for cls_name in self.cls_names:
			if mode == 'train':
				data_tmp = meta_info[cls_name]
				indices = torch.randint(0, len(data_tmp), (k_shot,))
				for i in range(len(indices)):
					self.data_all.append(data_tmp[indices[i]])
					if save_dir is not None:
						with open(save_dir, "a") as f:
							f.write(data_tmp[indices[i]]['img_path'] + '\n')
			else:
				self.data_all.extend(meta_info[cls_name])

		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def get_cls_names(self):
		return self.cls_names

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path = data['img_path']
		mask_path = data['mask_path']
		cls_name = data['cls_name']
		anomaly = int(data['anomaly'])

		img = Image.open(os.path.join(self.root, img_path)).convert("RGB")

		# binary ground-truth mask only
		if anomaly == 0:
			gt_pil = Image.fromarray(
				np.zeros((img.size[1], img.size[0]), dtype=np.uint8),
				mode='L'
			)
		else:
			m = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
			gt_pil = Image.fromarray(m.astype(np.uint8) * 255, mode='L')

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			gt = self.target_transform(gt_pil)
		else:
			gt = torch.from_numpy(np.array(gt_pil, dtype=np.uint8))

		if torch.is_tensor(gt) and gt.ndim == 3 and gt.shape[0] == 1:
			gt = gt.squeeze(0)

		# force binary float mask: [H, W], values 0/1
		if torch.is_tensor(gt):
			gt = (gt > 0.5).float()
		else:
			gt = torch.from_numpy((np.array(gt) > 0).astype(np.float32))

		return {
			'img': img,
			'img_mask_b': gt,          # binary mask only
			'cls_name': cls_name,
			'anomaly': anomaly,
			'img_path': os.path.join(self.root, img_path),
		}
	
# class VisaDataset(data.Dataset):
# 	def __init__(self, root, transform, target_transform, mode='test', k_shot=0, save_dir=None, obj_name=None):
# 		self.root = root
# 		self.transform = transform
# 		self.target_transform = target_transform

# 		self.data_all = []
# 		meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
# 		name = self.root.split('/')[-1]
# 		meta_info = meta_info[mode]
# 		if mode == 'train':
# 			self.cls_names = [obj_name]
# 			save_dir = os.path.join(save_dir, 'k_shot.txt')
# 		else:
# 			if obj_name is None:
# 				self.cls_names = list(meta_info.keys())
# 			else:
# 				self.cls_names = [obj_name]
# 		for cls_name in self.cls_names:
# 			if mode == 'train':
# 				data_tmp = meta_info[cls_name]
# 				indices = torch.randint(0, len(data_tmp), (k_shot,))
# 				for i in range(len(indices)):
# 					self.data_all.append(data_tmp[indices[i]])
# 					with open(save_dir, "a") as f:
# 						f.write(data_tmp[indices[i]]['img_path'] + '\n')
# 			else:
# 				self.data_all.extend(meta_info[cls_name])
# 		self.length = len(self.data_all)

# 	def __len__(self):
# 		return self.length

# 	def get_cls_names(self):
# 		return self.cls_names

# 	def __getitem__(self, index):
# 		data = self.data_all[index]
# 		img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
# 															  data['specie_name'], data['anomaly']
# 		img = Image.open(os.path.join(self.root, img_path))
# 		if anomaly == 0:
# 			img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
# 		else:
# 			img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
# 			img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
# 		img = self.transform(img) if self.transform is not None else img
# 		img_mask = self.target_transform(
# 			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
# 		img_mask = [] if img_mask is None else img_mask

# 		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
# 				'img_path': os.path.join(self.root, img_path)}

# class VisaDatasetV2(data.Dataset):
# 	def __init__(self, root, transform, target_transform, k_shot=0, save_dir=None, obj_name=None):
# 		self.root = root
# 		self.transform = transform
# 		self.target_transform = target_transform

# 		self.data_all = []
# 		meta_info = json.load(open(f'{self.root}/meta_wo_md.json', 'r'))
# 		name = self.root.split('/')[-1]

# 		self.cls_names = list(meta_info.keys())
# 		for cls_name in self.cls_names:
# 			self.data_all.extend(meta_info[cls_name])
# 		self.length = len(self.data_all)

# 	def __len__(self):
# 		return self.length

# 	def get_cls_names(self):
# 		return self.cls_names

# 	def __getitem__(self, index):
# 		data = self.data_all[index]
# 		img_path, mask_path, cls_name, anomaly, defect_cls = data['img_path'], data['mask_path'], data['cls_name'], data['anomaly'], data['defect_cls']
# 		img = Image.open(os.path.join(self.root, img_path))
# 		if anomaly == 0:
# 			img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
# 		else:
# 			img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
# 			img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
# 		img = self.transform(img) if self.transform is not None else img
# 		img_mask = self.target_transform(
# 			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
# 		img_mask = [] if img_mask is None else img_mask

# 		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
# 				'img_path': os.path.join(self.root, img_path), 'defect_cls': defect_cls}
	
class MVTecDataset(data.Dataset):
	def __init__(self, root, transform, target_transform, aug_rate, mode='test', k_shot=0, save_dir=None, obj_name=None):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.aug_rate = aug_rate

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
		name = self.root.split('/')[-1]
		meta_info = meta_info[mode]

		if mode == 'train':
			self.cls_names = [obj_name]
			save_dir = os.path.join(save_dir, 'k_shot.txt')
		else:
			self.cls_names = list(meta_info.keys())
		for cls_name in self.cls_names:
			if mode == 'train':
				data_tmp = meta_info[cls_name]
				indices = torch.randint(0, len(data_tmp), (k_shot,))
				for i in range(len(indices)):
					self.data_all.append(data_tmp[indices[i]])
					with open(save_dir, "a") as f:
						f.write(data_tmp[indices[i]]['img_path'] + '\n')
			else:
				self.data_all.extend(meta_info[cls_name])
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def get_cls_names(self):
		return self.cls_names

	def combine_img(self, cls_name):
		img_paths = os.path.join(self.root, cls_name, 'test')
		img_ls = []
		mask_ls = []
		for i in range(4):
			defect = os.listdir(img_paths)
			random_defect = random.choice(defect)
			files = os.listdir(os.path.join(img_paths, random_defect))
			random_file = random.choice(files)
			img_path = os.path.join(img_paths, random_defect, random_file)
			mask_path = os.path.join(self.root, cls_name, 'ground_truth', random_defect, random_file[:3] + '_mask.png')
			img = Image.open(img_path)
			img_ls.append(img)
			if random_defect == 'good':
				img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
			else:
				img_mask = np.array(Image.open(mask_path).convert('L')) > 0
				img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
			mask_ls.append(img_mask)
		# image
		image_width, image_height = img_ls[0].size
		result_image = Image.new("RGB", (2 * image_width, 2 * image_height))
		for i, img in enumerate(img_ls):
			row = i // 2
			col = i % 2
			x = col * image_width
			y = row * image_height
			result_image.paste(img, (x, y))

		# mask
		result_mask = Image.new("L", (2 * image_width, 2 * image_height))
		for i, img in enumerate(mask_ls):
			row = i // 2
			col = i % 2
			x = col * image_width
			y = row * image_height
			result_mask.paste(img, (x, y))

		return result_image, result_mask

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
															  data['specie_name'], data['anomaly']
		random_number = random.random()
		if random_number < self.aug_rate:
			img, img_mask = self.combine_img(cls_name)
		else:
			img = Image.open(os.path.join(self.root, img_path))
			if anomaly == 0:
				img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
			else:
				img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
				img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
		# transforms
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask
		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
				'img_path': os.path.join(self.root, img_path)}


class MPDDDataset(data.Dataset):
	def __init__(self, root, transform, target_transform, target_transform_type, specie2id=MPDD_SPECIE2ID, mode='test', k_shot=0, save_dir=None, obj_name=None):
		self.root = root
		self.transform = transform
		self.target_transform_b = target_transform
		self.target_transform_type = target_transform_type
		self.specie2id = specie2id

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
		name = self.root.split('/')[-1]
		meta_info = meta_info[mode]

		if mode == 'train':
			self.cls_names = [obj_name]
			save_dir = os.path.join(save_dir, 'k_shot.txt')
		else:
			if obj_name is None:
				self.cls_names = list(meta_info.keys())
			else:
				self.cls_names = [obj_name]
		for cls_name in self.cls_names:
			if mode == 'train':
				data_tmp = meta_info[cls_name]
				indices = torch.randint(0, len(data_tmp), (k_shot,))
				for i in range(len(indices)):
					self.data_all.append(data_tmp[indices[i]])
					with open(save_dir, "a") as f:
						f.write(data_tmp[indices[i]]['img_path'] + '\n')
			else:
				self.data_all.extend(meta_info[cls_name])
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def get_cls_names(self):
		return self.cls_names

	def combine_img(self, cls_name):
		img_paths = os.path.join(self.root, cls_name, 'test')
		img_ls = []
		mask_ls = []
		for i in range(4):
			defect = os.listdir(img_paths)
			random_defect = random.choice(defect)
			files = os.listdir(os.path.join(img_paths, random_defect))
			random_file = random.choice(files)
			img_path = os.path.join(img_paths, random_defect, random_file)
			mask_path = os.path.join(self.root, cls_name, 'ground_truth', random_defect, random_file[:3] + '_mask.png')
			img = Image.open(img_path)
			img_ls.append(img)
			if random_defect == 'good':
				img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
			else:
				img_mask = np.array(Image.open(mask_path).convert('L')) > 0
				img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
			mask_ls.append(img_mask)
		# image
		image_width, image_height = img_ls[0].size
		result_image = Image.new("RGB", (2 * image_width, 2 * image_height))
		for i, img in enumerate(img_ls):
			row = i // 2
			col = i % 2
			x = col * image_width
			y = row * image_height
			result_image.paste(img, (x, y))

		# mask
		result_mask = Image.new("L", (2 * image_width, 2 * image_height))
		for i, img in enumerate(mask_ls):
			row = i // 2
			col = i % 2
			x = col * image_width
			y = row * image_height
			result_mask.paste(img, (x, y))

		return result_image, result_mask

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
															  data['specie_name'], data['anomaly']
		random_number = random.random()
		if random_number < self.aug_rate:
			img, img_mask = self.combine_img(cls_name)
		else:
			img = Image.open(os.path.join(self.root, img_path))
			if anomaly == 0:
				img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
			else:
				
				img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
				img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
		# transforms
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask
		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
				'img_path': os.path.join(self.root, img_path)}


class MADDataset(data.Dataset):
	def __init__(self, root, transform, target_transform, target_transform_type, specie2id=None, mode='test', k_shot=0, save_dir=None, obj_name=None, datatype='sim'):
		self.root = root
		self.transform = transform
		self.target_transform_b = target_transform
		self.target_transform_type = target_transform_type
		if datatype not in ["sim", "real"]:
			raise ValueError(f"Invalid data type: {datatype}. Must be 'sim' or 'real'.")

		if datatype == 'sim':
			self.specie2id = MAD_SIM_SPECIE2ID
		else:
			self.specie2id = MAD_REAL_SPECIE2ID

		if self.specie2id is None:
			raise ValueError("specie2id must be provided (fixed mapping).")
		if "good" not in self.specie2id:
			raise ValueError("specie2id must contain key 'good'.")
		if self.specie2id["good"] != 0:
			raise ValueError("specie2id['good'] must be 0.")

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/meta.json', 'r'))[mode]

		if mode == 'train':
			self.cls_names = [obj_name]
			save_dir = os.path.join(save_dir, 'k_shot.txt')
		else:
			self.cls_names = list(meta_info.keys())
		for cls_name in self.cls_names:
			if mode == 'train':
				data_tmp = meta_info[cls_name]
				indices = torch.randint(0, len(data_tmp), (k_shot,))
				for i in range(len(indices)):
					self.data_all.append(data_tmp[indices[i]])
					with open(save_dir, "a") as f:
						f.write(data_tmp[indices[i]]['img_path'] + '\n')
			else:
				self.data_all.extend(meta_info[cls_name])
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def get_cls_names(self):
		return self.cls_names

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, anomaly, defect_cls = data['img_path'], data['mask_path'], data['product_cls'], data['anomaly'], data['defect_cls']
		img = Image.open(os.path.join(self.root, img_path))
		if anomaly == 0:
			img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
		else:
			img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
			img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask
		
		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
				'img_path': os.path.join(self.root, img_path), 'defect_cls': defect_cls}
	


class RealIADDataset_v2(data.Dataset):
	def __init__(self, root, transform, target_transform, target_transform_type, specie2id=REAL_IAD_SPECIE2ID, mode='test', k_shot=0, save_dir=None, obj_name=None):
		self.root = root
		self.transform = transform
		self.target_transform_b = target_transform
		self.target_transform_type = target_transform_type
		self.specie2id = specie2id

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/meta1.json', 'r'))
		name = self.root.split('/')[-1]
		meta_info = meta_info[mode]

		if mode == 'train':
			self.cls_names = [obj_name]
			save_dir = os.path.join(save_dir, 'k_shot.txt')
		else:
			self.cls_names = list(meta_info.keys())
		for cls_name in self.cls_names:
			if mode == 'train':
				data_tmp = meta_info[cls_name]
				indices = torch.randint(0, len(data_tmp), (k_shot,))
				for i in range(len(indices)):
					self.data_all.append(data_tmp[indices[i]])
					with open(save_dir, "a") as f:
						f.write(data_tmp[indices[i]]['img_path'] + '\n')
			else:
				self.data_all.extend(meta_info[cls_name])
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def get_cls_names(self):
		return self.cls_names

	def combine_img(self, cls_name):
		img_paths = os.path.join(self.root, cls_name, 'test')
		img_ls = []
		mask_ls = []
		for i in range(4):
			defect = os.listdir(img_paths)
			random_defect = random.choice(defect)
			files = os.listdir(os.path.join(img_paths, random_defect))
			random_file = random.choice(files)
			img_path = os.path.join(img_paths, random_defect, random_file)
			mask_path = os.path.join(self.root, cls_name, 'ground_truth', random_defect, random_file[:3] + '_mask.png')
			img = Image.open(img_path)
			img_ls.append(img)
			if random_defect == 'good':
				img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
			else:
				img_mask = np.array(Image.open(mask_path).convert('L')) > 0
				img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
			mask_ls.append(img_mask)
		# image
		image_width, image_height = img_ls[0].size
		result_image = Image.new("RGB", (2 * image_width, 2 * image_height))
		for i, img in enumerate(img_ls):
			row = i // 2
			col = i % 2
			x = col * image_width
			y = row * image_height
			result_image.paste(img, (x, y))

		# mask
		result_mask = Image.new("L", (2 * image_width, 2 * image_height))
		for i, img in enumerate(mask_ls):
			row = i // 2
			col = i % 2
			x = col * image_width
			y = row * image_height
			result_mask.paste(img, (x, y))

		return result_image, result_mask

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, anomaly, defect_cls = data['img_path'], data['mask_path'], data['cls_name'], data['anomaly'], data['defect_cls']
															   
		random_number = random.random()
		if random_number < self.aug_rate:
			img, img_mask = self.combine_img(cls_name)
		else:
			img = Image.open(os.path.join(self.root, cls_name, img_path))
			if anomaly == 0:
				img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
			else:
				img_mask = np.array(Image.open(os.path.join(self.root, cls_name, mask_path)).convert('L')) > 0
				img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
		# transforms
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask
		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
				'img_path': os.path.join(self.root, cls_name, img_path), 'defect_cls':defect_cls}
	