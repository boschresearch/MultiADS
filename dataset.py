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
		self.target_transform_b = target_transform
		self.target_transform_type = target_transform_type
		self.aug_rate = aug_rate
		if specie2id is None:
			raise ValueError("You must provide a fixed specie2id mapping (or specie2id_path).")

		self.specie2id = specie2id
		if "good" not in self.specie2id:
			raise ValueError("specie2id must contain key 'good'.")
		if self.specie2id["good"] != 0:
			raise ValueError("specie2id['good'] must be 0.")

		self.num_defect_classes = len(self.specie2id)
		# ===== Load meta data =====
		self.data_all = []
		meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
		meta_info = meta_info[mode]

		if mode == 'train':
			self.cls_names = [obj_name]
			if save_dir is not None:
				save_dir = os.path.join(save_dir, 'k_shot.txt')
		else:
			self.cls_names = list(meta_info.keys())

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

	# def _pil_to_u8_hw(self, pil_img_L):
	#	 """Convert a PIL grayscale image (mode 'L') to a uint8 tensor [H,W]."""
	#	 if self.mask_transform is not None:
	#		 t = self.mask_transform(pil_img_L)  # [1,H,W] uint8
	#		 return t.squeeze(0)
	#	 # Fallback: use the provided target_transform (may be bilinear; not recommended for labels)
	#	 if self.target_transform is None:
	#		 return torch.from_numpy(np.array(pil_img_L, dtype=np.uint8))
	#	 t = self.target_transform(pil_img_L)
	#	 if torch.is_tensor(t):
	#		 if t.ndim == 3 and t.shape[0] == 1:
	#			 t = t.squeeze(0)
	#		 if t.dtype.is_floating_point:
	#			 t = (t * 255.0).round().clamp(0, 255).to(torch.uint8)
	#		 else:
	#			 t = t.to(torch.uint8)
	#		 return t
	#	 raise TypeError("target_transform must return a torch.Tensor when used for masks.")

	def combine_img(self, cls_name):
		"""
		Create a 2x2 mosaic with 4 randomly sampled defect types (all from the same cls_name).
		Returns: (img_pil RGB, gt_b_pil L 0/255, gt_pil L label id 0..K-1)
		"""
		img_root = os.path.join(self.root, cls_name, 'test')

		tiles = []
		for _ in range(4):
			defect_folders = os.listdir(img_root)
			specie = random.choice(defect_folders)  # defect folder name == specie_name

			if specie not in self.specie2id:
				raise KeyError(f"combine_img: specie '{specie}' is not in the fixed specie2id mapping.")

			files = os.listdir(os.path.join(img_root, specie))
			fname = random.choice(files)

			img_path = os.path.join(img_root, specie, fname)
			img = Image.open(img_path).convert("RGB")

			if specie == "good":
				bin_mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
			else:
				mask_path = os.path.join(self.root, cls_name, 'ground_truth', specie, fname[:3] + '_mask.png')
				m = np.array(Image.open(mask_path).convert("L")) > 0
				bin_mask = Image.fromarray(m.astype(np.uint8) * 255, mode="L")

			tiles.append((img, bin_mask, self.specie2id[specie]))

		w, h = tiles[0][0].size
		out_img = Image.new("RGB", (2 * w, 2 * h))

		out_gt_b = np.zeros((2 * h, 2 * w), dtype=np.uint8)  # 0/255
		out_gt = np.zeros((2 * h, 2 * w), dtype=np.uint8)  # label id 0..K-1

		for i, (img, m_pil, sid) in enumerate(tiles):
			row, col = divmod(i, 2)
			x0, y0 = col * w, row * h
			out_img.paste(img, (x0, y0))

			m = np.array(m_pil, dtype=np.uint8)
			m_bin = (m > 0).astype(np.uint8)

			out_gt_b[y0:y0 + h, x0:x0 + w] = np.maximum(
				out_gt_b[y0:y0 + h, x0:x0 + w],
				m_bin * 255
			)

			tile_gt = out_gt[y0:y0 + h, x0:x0 + w]
			tile_gt[m_bin == 1] = int(sid)
			out_gt[y0:y0 + h, x0:x0 + w] = tile_gt

		gt_b_pil = Image.fromarray(out_gt_b, mode="L")
		gt_pil = Image.fromarray(out_gt, mode="L")
		return out_img, gt_b_pil, gt_pil

	def __getitem__(self, index):
		d = self.data_all[index]
		img_path = d['img_path']
		mask_path = d['mask_path']
		cls_name = d['cls_name']
		specie_name = str(d['specie_name'])  # you confirmed this equals the defect folder name
		anomaly = int(d['anomaly'])

		# ===== mosaic augmentation =====
		if random.random() < self.aug_rate:
			img_pil, gt_b_pil, gt_pil = self.combine_img(cls_name)
			specie_name_out = "mosaic"
			specie_id_out = -1
		else:
			img_pil = Image.open(os.path.join(self.root, img_path)).convert("RGB")

			if anomaly == 0:
				specie_name = "good"

			if specie_name not in self.specie2id:
				raise KeyError(f"specie_name '{specie_name}' is not in the fixed specie2id mapping.")

			specie_id = self.specie2id[specie_name]

			if anomaly == 0:
				gt_b_pil = Image.fromarray(
					np.zeros((img_pil.size[1], img_pil.size[0]), dtype=np.uint8), mode='L'
				)
				gt_pil = Image.fromarray(
					np.zeros((img_pil.size[1], img_pil.size[0]), dtype=np.uint8), mode='L'
				)
			else:
				m = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
				gt_b_pil = Image.fromarray(m.astype(np.uint8) * 255, mode='L')

				gt_arr = np.zeros((img_pil.size[1], img_pil.size[0]), dtype=np.uint8)
				gt_arr[m] = int(specie_id)
				gt_pil = Image.fromarray(gt_arr, mode='L')

			specie_name_out = specie_name
			specie_id_out = specie_id

		# ===== transforms =====
		img = self.transform(img_pil) if self.transform is not None else img_pil

		gt_b_u8 = self.target_transform_b(gt_b_pil)   # [H,W] uint8 0..255
		gt_u8 = self.target_transform_type(gt_pil)	   # [H,W] uint8 label id (NEAREST keeps integers)

		gt_b = (gt_b_u8 > 0).to(torch.float32)   # [H,W] float 0/1
		gt = gt_u8.to(torch.int64)			   # [H,W] long 0..K-1

		# Recompute anomaly for mosaic (whether any anomalous pixels exist)
		anomaly_out = int(gt_b.sum().item() > 0)
		return {
			'img': img,
			'img_mask_b': gt_b,
			'img_mask': gt,
			'cls_name': cls_name,
			'specie_name': specie_name_out,
			'specie_id': specie_id_out,
			'anomaly': anomaly_out,
			'img_path': os.path.join(self.root, img_path),
		}

class VisaDataset(data.Dataset):
	def __init__(self, root, transform, target_transform_b, target_transform_type, specie2id, mode='test', k_shot=0, save_dir=None, obj_name=None):
		self.root = root
		self.transform = transform
		self.target_transform_b = target_transform_b
		self.target_transform_type = target_transform_type
		self.specie2id = specie2id

		if self.specie2id is None:
			raise ValueError("specie2id must be provided (fixed mapping).")
		if "normal" not in self.specie2id:
			raise ValueError("specie2id must contain key 'normal'.")
		if self.specie2id["normal"] != 0:
			raise ValueError("specie2id['normal'] must be 0.")

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
		specie_name = str(data['specie_name'])
		anomaly = int(data['anomaly'])

		img = Image.open(os.path.join(self.root, img_path)).convert("RGB")

		# Visa uses "normal" as good
		if anomaly == 0:
			specie_name = "normal"

		if specie_name not in self.specie2id:
			raise KeyError(f"specie_name '{specie_name}' not found in specie2id mapping.")
		specie_id = int(self.specie2id[specie_name])

		# build binary & multi-class masks as PIL
		if anomaly == 0:
			gt_b_pil = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode='L')
			gt_pil = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode='L')
		else:
			m = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
			gt_b_pil = Image.fromarray(m.astype(np.uint8) * 255, mode='L')

			gt_arr = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
			gt_arr[m] = specie_id
			gt_pil = Image.fromarray(gt_arr, mode='L')

		# transforms
		img = self.transform(img) if self.transform is not None else img

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
	

class VisaDatasetV2(data.Dataset):
	def __init__(self, root, transform, target_transform_b, target_transform_type, specie2id=VISA_SPECIE2ID, k_shot=0, save_dir=None, obj_name=None):
		self.root = root
		self.transform = transform
		self.target_transform_b = target_transform_b
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
		img_path = data['img_path']
		mask_path = data['mask_path']
		cls_name = data['cls_name']
		anomaly = int(data['anomaly'])
		defect_cls = data['defect_cls']	# e.g. "scratch", "hole", ... (or list)

		img = Image.open(os.path.join(self.root, img_path)).convert("RGB")

		# defect name -> id (Visa uses "normal" as good)
		if anomaly == 0:
			defect_name = "normal"
		else:
			# defect_cls could be a string or a list; take string directly, or first element
			if isinstance(defect_cls, (list, tuple)):
				if len(defect_cls) == 0:
					raise ValueError("defect_cls is empty for an anomalous sample.")
				defect_name = str(defect_cls[0])
			else:
				defect_name = str(defect_cls)

		if defect_name not in self.specie2id:
			raise KeyError(f"defect '{defect_name}' not found in specie2id mapping.")
		defect_id = int(self.specie2id[defect_name])

		# build masks as PIL
		if anomaly == 0:
			gt_b_pil = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
			gt_pil = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")
		else:
			m = np.array(Image.open(os.path.join(self.root, mask_path)).convert("L")) > 0
			gt_b_pil = Image.fromarray(m.astype(np.uint8) * 255, mode="L")

			gt_arr = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
			gt_arr[m] = defect_id
			gt_pil = Image.fromarray(gt_arr, mode="L")

		# transforms
		img = self.transform(img) if self.transform is not None else img

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
			'anomaly': anomaly,
			'img_path': os.path.join(self.root, img_path),
			'defect_cls': defect_cls,
			'defect_id': defect_id,
		}
	

class MPDDDataset(data.Dataset):
	def __init__(self, root, transform, target_transform_b, target_transform_type, specie2id, aug_rate, mode='test', k_shot=0, save_dir=None, obj_name=None):
		self.root = root
		self.transform = transform
		self.target_transform_b = target_transform_b
		self.target_transform_type = target_transform_type
		self.specie2id = specie2id
		self.aug_rate = aug_rate

		if self.specie2id is None:
			raise ValueError("specie2id must be provided (fixed mapping).")
		if "good" not in self.specie2id:
			raise ValueError("specie2id must contain key 'good'.")
		if self.specie2id["good"] != 0:
			raise ValueError("specie2id['good'] must be 0.")

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
		specie_name = str(data['specie_name'])
		anomaly = int(data['anomaly'])

		img = Image.open(os.path.join(self.root, img_path)).convert("RGB")

		if anomaly == 0:
			specie_name = "good"

		if specie_name not in self.specie2id:
			raise KeyError(f"specie_name '{specie_name}' not found in specie2id mapping.")
		specie_id = int(self.specie2id[specie_name])

		# build binary & multi-class masks as PIL
		if anomaly == 0:
			gt_b_pil = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode='L')
			gt_pil = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode='L')
		else:
			m = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
			gt_b_pil = Image.fromarray(m.astype(np.uint8) * 255, mode='L')

			gt_arr = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
			gt_arr[m] = specie_id
			gt_pil = Image.fromarray(gt_arr, mode='L')

		# transforms
		img = self.transform(img) if self.transform is not None else img

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



class MADDataset(data.Dataset):
	def __init__(self, root, transform, target_transform_b, target_transform_type, specie2id, mode='test', k_shot=0, save_dir=None, obj_name=None):
		self.root = root
		self.transform = transform
		self.target_transform_b = target_transform_b
		self.target_transform_type = target_transform_type
		self.specie2id = specie2id

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
			if save_dir is not None:
				save_dir = os.path.join(save_dir, 'k_shot.txt')
		else:
			self.cls_names = list(meta_info.keys())

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
		cls_name = data['product_cls']
		anomaly = int(data['anomaly'])
		defect_cls = data['defect_cls']	# string or list/tuple

		img = Image.open(os.path.join(self.root, img_path)).convert("RGB")

		# Choose defect name -> id (good=0)
		if anomaly == 0:
			defect_name = "good"
		else:
			if isinstance(defect_cls, (list, tuple)):
				if len(defect_cls) == 0:
					raise ValueError("defect_cls is empty for an anomalous sample.")
				defect_name = str(defect_cls[0])
			else:
				defect_name = str(defect_cls)

		if defect_name not in self.specie2id:
			raise KeyError(f"defect '{defect_name}' not found in specie2id mapping.")
		defect_id = int(self.specie2id[defect_name])

		# build binary & multi-class masks as PIL
		if anomaly == 0:
			gt_b_pil = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode='L')
			gt_pil = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode='L')
		else:
			m = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
			gt_b_pil = Image.fromarray(m.astype(np.uint8) * 255, mode='L')

			gt_arr = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
			gt_arr[m] = defect_id
			gt_pil = Image.fromarray(gt_arr, mode='L')

		# transforms
		img = self.transform(img) if self.transform is not None else img

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
			'anomaly': anomaly,
			'img_path': os.path.join(self.root, img_path),
			'defect_cls': defect_cls,
			'defect_id': defect_id,
		}
	


class RealIADDataset_v2(data.Dataset):
	def __init__(self, root, transform, target_transform_b, target_transform_type, specie2id, aug_rate, mode='test', k_shot=0, save_dir=None, obj_name=None):
		self.root = root
		self.transform = transform
		self.target_transform_b = target_transform_b
		self.target_transform_type = target_transform_type
		self.specie2id = specie2id
		self.aug_rate = aug_rate

		if self.specie2id is None:
			raise ValueError("specie2id must be provided (fixed mapping).")
		if "good" not in self.specie2id:
			raise ValueError("specie2id must contain key 'good'.")
		if self.specie2id["good"] != 0:
			raise ValueError("specie2id['good'] must be 0.")

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/meta1.json', 'r'))
		meta_info = meta_info[mode]

		if mode == 'train':
			self.cls_names = [obj_name]
			if save_dir is not None:
				save_dir = os.path.join(save_dir, 'k_shot.txt')
		else:
			self.cls_names = list(meta_info.keys())

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
		defect_cls = data['defect_cls']	# string or list/tuple

		img = Image.open(os.path.join(self.root, cls_name, img_path)).convert("RGB")

		# Choose defect name -> id (good=0)
		if anomaly == 0:
			defect_name = "good"
		else:
			if isinstance(defect_cls, (list, tuple)):
				if len(defect_cls) == 0:
					raise ValueError("defect_cls is empty for an anomalous sample.")
				defect_name = str(defect_cls[0])
			else:
				defect_name = str(defect_cls)

		if defect_name not in self.specie2id:
			raise KeyError(f"defect '{defect_name}' not found in specie2id mapping.")
		defect_id = int(self.specie2id[defect_name])

		# build binary & multi-class masks as PIL
		if anomaly == 0:
			gt_b_pil = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode='L')
			gt_pil = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode='L')
		else:
			m = np.array(Image.open(os.path.join(self.root, cls_name, mask_path)).convert('L')) > 0
			gt_b_pil = Image.fromarray(m.astype(np.uint8) * 255, mode='L')

			gt_arr = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
			gt_arr[m] = defect_id
			gt_pil = Image.fromarray(gt_arr, mode='L')

		# transforms
		img = self.transform(img) if self.transform is not None else img

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
			'anomaly': anomaly,
			'img_path': os.path.join(self.root, cls_name, img_path),
			'defect_cls': defect_cls,
			'defect_id': defect_id,
		}