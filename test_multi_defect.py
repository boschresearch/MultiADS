# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
import cv2
import json
import torch
import random
import logging
import argparse
import numpy as np
from skimage import measure
from tabulate import tabulate
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from sklearn.metrics import (
	auc,
	roc_auc_score,
	average_precision_score,
	f1_score,
	precision_recall_curve,
	pairwise,
)
from sklearn.preprocessing import label_binarize

import open_clip

from model import LinearLayer
from dataset import (
	VisaDatasetV2,
	MVTecDataset,
	MPDDDataset,
	MADDataset,
	RealIADDataset_v2,
)

from prompts.prompt_ensemble_visa_19cls_single import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_visa
from prompts.prompt_ensemble_mvtec_20cls import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_mvtec
from prompts.new_prompt_ensemble_mpdd import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_mpdd
from prompts.prompt_ensemble_mad_real import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_mad_real
from prompts.prompt_ensemble_mad_sim import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_mad_sim
from prompts.prompt_ensemble_real_IAD import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_real_iad

from tqdm import tqdm


product_type2defect_type_mvtec = {
	'bottle': ['good', 'broken', 'contamination'],
	'cable': ['good', 'bent', 'misplaced', 'combined', 'cut', 'missing', 'poke'],
	'capsule': ['good', 'crack', 'faulty imprint', 'poke', 'scratch', 'squeeze'],
	'carpet': ['good', 'color', 'cut', 'hole', 'contamination', 'thread'],
	'grid': ['good', 'bent', 'broken', 'glue', 'contamination', 'thread'],
	'hazelnut': ['good', 'crack', 'cut', 'hole', 'faulty imprint'],
	'leather': ['good', 'color', 'cut', 'misplaced', 'glue', 'poke'],
	'metal_nut': ['good', 'bent', 'color', 'misplaced', 'scratch'],
	'pill': ['good', 'color', 'combined', 'contamination', 'crack', 'faulty imprint', 'damaged', 'scratch'],
	'screw': ['good', 'fabric', 'scratch', 'thread'],
	'tile': ['good', 'crack', 'glue', 'damaged', 'liquid', 'rough'],
	'toothbrush': ['good', 'damaged'],
	'transistor': ['good', 'bent', 'cut', 'damaged', 'misplaced'],
	'wood': ['good', 'color', 'combined', 'hole', 'liquid', 'scratch'],
	'zipper': ['good', 'broken', 'combined', 'fabric', 'rough', 'misplaced', 'squeeze'],
}


product_type2defect_type_visa = {
	'candle': ['normal', 'damage', 'weird wick', 'partical', 'melded', 'spot', 'extra', 'missing'],
	'capsules': ['normal', 'bubble'],
	'cashew': ['normal', 'scratch', 'breakage', 'burnt', 'stuck', 'hole', 'spot'],
	'chewinggum': ['normal', 'scratch', 'spot', 'missing'],
	'fryum': ['normal', 'scratch', 'breakage', 'burnt', 'stuck', 'spot'],
	'macaroni1': ['normal', 'scratch', 'crack', 'spot', 'chip'],
	'macaroni2': ['normal', 'scratch', 'breakage', 'crack', 'spot', 'chip'],
	'pcb1': ['normal', 'scratch', 'melt', 'bent', 'missing'],
	'pcb2': ['normal', 'scratch', 'melt', 'bent', 'missing'],
	'pcb3': ['normal', 'scratch', 'melt', 'bent', 'missing'],
	'pcb4': ['normal', 'scratch', 'damage', 'extra', 'burnt', 'missing', 'wrong place'],
	'pipe_fryum': ['normal', 'scratch', 'breakage', 'burnt', 'stuck', 'spot'],
}


product_type2defect_type_mpdd = {
	'bracket_black': ['good', 'hole', 'scratch'],
	'bracket_brown': ['good', 'mismatch', 'bent'],
	'bracket_white': ['good', 'defective painting', 'scratch'],
	'connector': ['good', 'mismatch'],
	'metal_plate': ['good', 'rust', 'scratch'],
	'tubes': ['good', 'flattening'],
}


product_type2defect_type_mad_real = {
	'Bear': ['good', 'Stains'],
	'Bird': ['good', 'Missing'],
	'Elephant': ['good', 'Missing'],
	'Parrot': ['good', 'Missing'],
	'Pig': ['good', 'Missing'],
	'Puppy': ['good', 'Stains'],
	'Scorpion': ['good', 'Missing'],
	'Turtle': ['good', 'Stains'],
	'Unicorn': ['good', 'Missing'],
	'Whale': ['good', 'Stains'],
}


product_type2defect_type_mad_sim = {
	'Gorilla': ['good', 'Stains', 'Burrs', 'Missing'],
	'Unicorn': ['good', 'Stains', 'Burrs', 'Missing'],
	'Mallard': ['good', 'Stains', 'Burrs', 'Missing'],
	'Turtle': ['good', 'Stains', 'Burrs', 'Missing'],
	'Whale': ['good', 'Stains', 'Burrs', 'Missing'],
	'Bird': ['good', 'Stains', 'Burrs', 'Missing'],
	'Owl': ['good', 'Stains', 'Burrs', 'Missing'],
	'Sabertooth': ['good', 'Stains', 'Burrs', 'Missing'],
	'Swan': ['good', 'Stains', 'Burrs', 'Missing'],
	'Sheep': ['good', 'Stains', 'Burrs', 'Missing'],
	'Pig': ['good', 'Stains', 'Burrs', 'Missing'],
	'Zalika': ['good', 'Stains', 'Burrs', 'Missing'],
	'Pheonix': ['good', 'Stains', 'Burrs', 'Missing'],
	'Elephant': ['good', 'Stains', 'Burrs', 'Missing'],
	'Parrot': ['good', 'Stains', 'Burrs', 'Missing'],
	'Cat': ['good', 'Stains', 'Burrs', 'Missing'],
	'Scorpion': ['good', 'Stains', 'Burrs', 'Missing'],
	'Obesobeso': ['good', 'Stains', 'Burrs', 'Missing'],
	'Bear': ['good', 'Stains', 'Burrs', 'Missing'],
	'Puppy': ['good', 'Stains', 'Burrs', 'Missing'],
}


product_type2defect_type_real_iad = {
	'switch': ['good', 'missing', 'contamination', 'scratch'],
	'eraser': ['good', 'contamination', 'scratch', 'missing', 'pit'],
	'woodstick': ['good', 'contamination', 'scratch', 'missing', 'pit'],
	'zipper': ['good', 'contamination', 'deformation', 'missing', 'damage'],
	'fire_hood': ['good', 'contamination', 'scratch', 'missing', 'pit'],
	'pcb': ['good', 'contamination', 'scratch', 'missing', 'foreign'],
	'toothbrush': ['good', 'abrasion', 'contamination', 'missing'],
	'plastic_nut': ['good', 'contamination', 'scratch', 'missing', 'pit'],
	'wooden_beads': ['good', 'contamination', 'scratch', 'missing', 'pit'],
	'transistor1': ['good', 'missing', 'contamination', 'deformation'],
	'bottle_cap': ['good', 'contamination', 'scratch', 'missing', 'pit'],
	'u_block': ['good', 'abrasion', 'contamination', 'scratch', 'missing'],
	'sim_card_set': ['good', 'abrasion', 'contamination', 'scratch'],
	'end_cap': ['good', 'contamination', 'scratch', 'missing', 'damage'],
	'usb': ['good', 'contamination', 'deformation', 'scratch', 'missing'],
	'regulator': ['good', 'missing', 'scratch'],
	'plastic_plug': ['good', 'contamination', 'scratch', 'missing', 'pit'],
	'audiojack': ['good', 'contamination', 'deformation', 'scratch', 'missing'],
	'mint': ['good', 'missing', 'contamination', 'foreign'],
	'toy_brick': ['good', 'contamination', 'scratch', 'missing', 'pit'],
	'toy': ['good', 'contamination', 'scratch', 'missing', 'pit'],
	'rolled_strip_base': ['good', 'pit', 'missing', 'contamination'],
	'terminalblock': ['good', 'pit', 'missing', 'contamination'],
	'mounts': ['good', 'missing', 'contamination', 'pit'],
	'button_battery': ['good', 'abrasion', 'contamination', 'scratch', 'pit'],
	'porcelain_doll': ['good', 'abrasion', 'contamination', 'scratch'],
	'phone_battery': ['good', 'contamination', 'scratch', 'damage', 'pit'],
	'usb_adaptor': ['good', 'abrasion', 'contamination', 'scratch', 'pit'],
	'vcpill': ['good', 'contamination', 'scratch', 'missing', 'pit'],
	'tape': ['good', 'missing', 'contamination', 'damage'],
}


def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def normalize(pred, max_value=None, min_value=None):
	if max_value is None or min_value is None:
		den = pred.max() - pred.min()
		if den < 1e-12:
			return np.zeros_like(pred)
		return (pred - pred.min()) / den

	den = max_value - min_value
	if den < 1e-12:
		return np.zeros_like(pred)
	return (pred - min_value) / den


def apply_ad_scoremap(image, scoremap, alpha=0.5):
	np_image = np.asarray(image, dtype=float)
	scoremap = (scoremap * 255).astype(np.uint8)
	scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
	scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
	return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
	binary_amaps = np.zeros_like(amaps, dtype=bool)
	min_th, max_th = amaps.min(), amaps.max()
	if abs(max_th - min_th) < 1e-12:
		return 0.0

	delta = (max_th - min_th) / max_step
	pros, fprs = [], []

	for th in np.arange(min_th, max_th, delta):
		binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
		pro = []

		for binary_amap, mask in zip(binary_amaps, masks):
			for region in measure.regionprops(measure.label(mask)):
				tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
				pro.append(tp_pixels / region.area)

		inverse_masks = 1 - masks
		fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
		fpr = fp_pixels / max(inverse_masks.sum(), 1)

		pros.append(np.array(pro).mean() if len(pro) > 0 else 0.0)
		fprs.append(fpr)

	pros = np.array(pros)
	fprs = np.array(fprs)

	idxes = fprs < expect_fpr
	if idxes.sum() < 2:
		return 0.0

	fprs = fprs[idxes]
	pros = pros[idxes]

	den = fprs.max() - fprs.min()
	if den < 1e-12:
		return 0.0

	fprs = (fprs - fprs.min()) / den
	return auc(fprs, pros)


def build_logger(save_path):
	os.makedirs(save_path, exist_ok=True)
	txt_path = os.path.join(save_path, "log.txt")

	root_logger = logging.getLogger()
	for handler in root_logger.handlers[:]:
		root_logger.removeHandler(handler)
	root_logger.setLevel(logging.WARNING)

	logger = logging.getLogger("test")
	logger.handlers.clear()
	logger.setLevel(logging.INFO)
	logger.propagate = False

	formatter = logging.Formatter(
		"%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
		datefmt="%y-%m-%d %H:%M:%S"
	)

	file_handler = logging.FileHandler(txt_path, mode="a")
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)

	console_handler = logging.StreamHandler()
	console_handler.setFormatter(formatter)
	logger.addHandler(console_handler)

	return logger


def get_dataset_meta(dataset_name):
	if dataset_name == "mvtec":
		defects = [
			"good", "bent", "broken", "color", "combined", "contamination",
			"crack", "cut", "fabric", "faulty imprint", "glue", "hole",
			"missing", "poke", "rough", "scratch", "squeeze", "thread",
			"liquid", "misplaced", "damaged"
		]
		return defects, product_type2defect_type_mvtec

	if dataset_name == "visa":
		defects = [
			"normal", "damage", "scratch", "breakage", "burnt", "weird wick",
			"stuck", "crack", "wrong place", "partical", "bubble", "melded",
			"hole", "melt", "bent", "spot", "extra", "chip", "missing"
		]
		return defects, product_type2defect_type_visa

	if dataset_name == "mpdd":
		defects = [
			"good", "hole", "scratch", "bent", "mismatch",
			"defective painting", "rust", "flattening"
		]
		return defects, product_type2defect_type_mpdd

	if dataset_name == "mad_sim":
		defects = ["good", "Stains", "Missing", "Burrs"]
		return defects, product_type2defect_type_mad_sim

	if dataset_name == "mad_real":
		defects = ["good", "Stains", "Missing"]
		return defects, product_type2defect_type_mad_real

	if dataset_name == "real_iad":
		defects = [
			"good", "pit", "deformation", "abrasion", "scratch",
			"damage", "missing", "foreign", "contamination"
		]
		return defects, product_type2defect_type_real_iad

	raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_dataset(args, preprocess, target_transform_b, target_transform_type):
	if args.dataset == "mvtec":
		return MVTecDataset(
			root=args.data_path,
			transform=preprocess,
			target_transform=target_transform_b,
			target_transform_type=target_transform_type,
			aug_rate=-1,
			mode="test"
		)

	if args.dataset == "visa":
		return VisaDatasetV2(
			root=args.data_path,
			transform=preprocess,
			target_transform_b=target_transform_b,
			target_transform_type=target_transform_type,
		)

	if args.dataset == "mpdd":
		return MPDDDataset(
			root=args.data_path,
			transform=preprocess,
			target_transform=target_transform_b,
			target_transform_type=target_transform_type,
			mode="test"
		)

	if args.dataset == "mad_sim":
		return MADDataset(
			root=args.data_path,
			transform=preprocess,
			target_transform=target_transform_b,
			target_transform_type=target_transform_type,
			mode="test",
			datatype="sim"
		)

	if args.dataset == "mad_real":
		return MADDataset(
			root=args.data_path,
			transform=preprocess,
			target_transform=target_transform_b,
			target_transform_type=target_transform_type,
			mode="test",
			datatype="real"
		)

	if args.dataset == "real_iad":
		return RealIADDataset_v2(
			root=args.data_path,
			transform=preprocess,
			target_transform=target_transform_b,
			target_transform_type=target_transform_type,
			mode="test"
		)

	raise ValueError(f"Unsupported dataset: {args.dataset}")


def build_text_prompts(args, model, obj_list, tokenizer, device):
	with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()), torch.no_grad():
		if args.dataset == "mvtec":
			return encode_text_with_prompt_ensemble_mvtec(model, obj_list, tokenizer, device)
		if args.dataset == "visa":
			return encode_text_with_prompt_ensemble_visa(model, obj_list, tokenizer, device)
		if args.dataset == "mpdd":
			return encode_text_with_prompt_ensemble_mpdd(model, obj_list, tokenizer, device)
		if args.dataset == "mad_sim":
			return encode_text_with_prompt_ensemble_mad_sim(model, obj_list, tokenizer, device)
		if args.dataset == "mad_real":
			return encode_text_with_prompt_ensemble_mad_real(model, obj_list, tokenizer, device)
		if args.dataset == "real_iad":
			return encode_text_with_prompt_ensemble_real_iad(model, obj_list, tokenizer, device)

	raise ValueError(f"Unsupported dataset: {args.dataset}")


def remap_global_mask_to_local(gt_mask, cls_name, defects, p_cls2d_cls):
	"""
	gt_mask:
		[H, W] or [1, H, W], global defect id according to full defect list.
	return:
		[H, W], local class id according to p_cls2d_cls[cls_name].
		local id 0 is normal/good.
	"""
	if torch.is_tensor(gt_mask):
		mask_np = gt_mask.detach().cpu().numpy()
	else:
		mask_np = np.asarray(gt_mask)

	mask_np = np.squeeze(mask_np).astype(np.int64)

	local_defects = p_cls2d_cls[cls_name]
	global_id_to_defect = {i: d for i, d in enumerate(defects)}
	local_defect_to_id = {d: i for i, d in enumerate(local_defects)}

	local_mask = np.zeros_like(mask_np, dtype=np.int64)

	for global_id, defect_name in global_id_to_defect.items():
		if defect_name in local_defect_to_id:
			local_mask[mask_np == global_id] = local_defect_to_id[defect_name]

	return local_mask


def evaluate_multiclass_object(obj, results, p_cls2d_cls):
	table = [obj]

	gt_px = []
	pr_px = []

	for idx in range(len(results["cls_names"])):
		if results["cls_names"][idx] == obj:
			gt_px.append(np.squeeze(np.asarray(results["imgs_masks"][idx])).astype(np.int64))
			pr_px.append(np.asarray(results["anomaly_maps"][idx]))

	gt_px = np.stack(gt_px, axis=0)		# [N, H, W]
	pr_px = np.stack(pr_px, axis=0)		# [N, C, H, W]

	C = pr_px.shape[1]

	gt_px_flat = gt_px.reshape(-1)
	pr_px_flat = np.transpose(pr_px, (0, 2, 3, 1)).reshape(-1, C)

	gt_px_b = label_binarize(gt_px_flat, classes=range(C))
	if C == 2 and gt_px_b.shape[1] == 1:
		gt_px_b = np.hstack((1 - gt_px_b, gt_px_b))

	auroc_px = roc_auc_score(gt_px_b, pr_px_flat, multi_class="ovr")
	ap_px = average_precision_score(gt_px_b, pr_px_flat, average="macro")
	f1_px = f1_score(gt_px_flat, np.argmax(pr_px_flat, axis=-1), average="macro")

	roc_auc_per_class = roc_auc_score(gt_px_b, pr_px_flat, multi_class="ovr", average=None)
	ap_px_per_class = average_precision_score(gt_px_b, pr_px_flat, average=None)
	f1_px_per_class = f1_score(gt_px_flat, np.argmax(pr_px_flat, axis=-1), average=None, labels=list(range(C)))

	table.append(str(np.round(auroc_px * 100, decimals=1)))
	table.append(str(np.round(f1_px * 100, decimals=1)))
	table.append(str(np.round(ap_px * 100, decimals=1)))

	per_class = {}
	for i in range(C):
		defect_name = p_cls2d_cls[obj][i]
		per_class[defect_name] = {
			"auroc": roc_auc_per_class[i],
			"ap": ap_px_per_class[i],
			"f1": f1_px_per_class[i],
		}

	return {
		"obj": obj,
		"table": table,
		"auroc_px": auroc_px,
		"f1_px": f1_px,
		"ap_px": ap_px,
		"per_class": per_class,
	}


def test(args):
	img_size = args.image_size
	features_list = args.features_list
	save_path = args.save_path
	device = "cuda" if torch.cuda.is_available() else "cpu"

	os.makedirs(save_path, exist_ok=True)
	logger = build_logger(save_path)

	for arg in vars(args):
		logger.info(f"{arg}: {getattr(args, arg)}")

	model, _, preprocess = open_clip.create_model_and_transforms(
		args.model,
		img_size,
		pretrained=args.pretrained
	)
	model = model.to(device)
	model.eval()

	tokenizer = open_clip.get_tokenizer(args.model)

	with open(args.config_path.strip(), "r") as f:
		model_configs = json.load(f)

	linearlayer = LinearLayer(
		model_configs["vision_cfg"]["width"],
		model_configs["embed_dim"],
		len(features_list),
		args.model
	).to(device)

	checkpoint = torch.load(args.checkpoint_path.strip(), map_location=device)
	linearlayer.load_state_dict(checkpoint["trainable_linearlayer"])
	linearlayer.eval()

	target_transform_b = transforms.Compose([
		transforms.Resize((img_size, img_size)),
		transforms.CenterCrop(img_size),
		transforms.ToTensor()
	])

	target_transform_type = transforms.Compose([
		transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST),
		transforms.CenterCrop(img_size),
		transforms.PILToTensor(),
		transforms.Lambda(lambda x: x.squeeze(0).long()),
	])

	test_data = build_dataset(args, preprocess, target_transform_b, target_transform_type)

	test_dataloader = torch.utils.data.DataLoader(
		test_data,
		batch_size=args.test_batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=(device == "cuda"),
		persistent_workers=(args.num_workers > 0),
		prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
	)

	obj_list = test_data.get_cls_names()
	defects, p_cls2d_cls = get_dataset_meta(args.dataset)
	defect_to_idx = {d: i for i, d in enumerate(defects)}

	text_prompts = build_text_prompts(args, model, obj_list, tokenizer, device)
	text_prompts = {
		k: F.normalize(v.to(device=device, dtype=torch.float32), dim=0).contiguous()
		for k, v in text_prompts.items()
	}

	results = {
		"cls_names": [],
		"imgs_masks": [],
		"anomaly_maps": [],
	}

	for items in tqdm(test_dataloader, desc="Testing", leave=True):
		images = items["img"].to(device, non_blocking=True)
		cls_names = list(items["cls_name"])
		batch_size = images.shape[0]

		gt_masks_global = items["img_mask"]

		for b in range(batch_size):
			cls_b = cls_names[b]
			local_mask = remap_global_mask_to_local(
				gt_mask=gt_masks_global[b],
				cls_name=cls_b,
				defects=defects,
				p_cls2d_cls=p_cls2d_cls
			)
			results["cls_names"].append(cls_b)
			results["imgs_masks"].append(local_mask)

		with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
			_, patch_tokens = model.encode_image(images, features_list)
			patch_tokens = linearlayer(patch_tokens)

			anomaly_maps_batch = [None] * batch_size

			grouped_indices = {}
			for i, cls in enumerate(cls_names):
				if cls not in grouped_indices:
					grouped_indices[cls] = []
				grouped_indices[cls].append(i)

			for cls, indices in grouped_indices.items():
				idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)

				local_defects = p_cls2d_cls[cls]
				defect_indices = [defect_to_idx[d] for d in local_defects]
				text_features_g = text_prompts[cls][:, defect_indices]
				text_features_g = text_features_g.unsqueeze(0).repeat(len(indices), 1, 1)
				text_features_g = text_features_g.to(device=device, non_blocking=True)

				acc_prob = None

				for layer in range(len(patch_tokens)):
					layer_tokens = patch_tokens[layer].index_select(0, idx_tensor)
					layer_tokens = F.normalize(layer_tokens, dim=-1)

					logits = (layer_tokens @ text_features_g) / args.temperature

					Bg, L, C = logits.shape
					H = int(np.sqrt(L))
					if H * H != L:
						raise ValueError(f"L={L} is not square.")

					logits = F.interpolate(
						logits.permute(0, 2, 1).contiguous().view(Bg, C, H, H),
						size=img_size,
						mode="bilinear",
						align_corners=True
					)

					prob = torch.softmax(logits, dim=1)
					acc_prob = prob if acc_prob is None else acc_prob + prob

				for local_i, global_i in enumerate(indices):
					anomaly_maps_batch[global_i] = acc_prob[local_i].detach().cpu().numpy()

			for b in range(batch_size):
				results["anomaly_maps"].append(anomaly_maps_batch[b])

			if args.visualization:
				img_paths = list(items["img_path"])
				for b in range(batch_size):
					path = img_paths[b]
					cls_name_b = cls_names[b]
					cls_folder = path.split("/")[-2]
					filename = path.split("/")[-1]

					vis = cv2.cvtColor(
						cv2.resize(cv2.imread(path), (img_size, img_size)),
						cv2.COLOR_BGR2RGB
					)

					# visualize total anomaly probability, excluding class 0
					vis_map = anomaly_maps_batch[b][1:, :, :].sum(axis=0)
					mask = normalize(vis_map)

					vis = apply_ad_scoremap(vis, mask)
					vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

					save_vis = os.path.join(save_path, "imgs", cls_name_b, cls_folder)
					os.makedirs(save_vis, exist_ok=True)
					cv2.imwrite(os.path.join(save_vis, filename), vis)

	table_ls = []
	auroc_px_ls = []
	f1_px_ls = []
	ap_px_ls = []

	auroc_px_per_defect_cls_all = {}
	ap_px_per_defect_cls_all = {}
	f1_px_per_defect_cls_all = {}

	for obj in tqdm(obj_list, desc="Evaluating", leave=True):
		ret = evaluate_multiclass_object(
			obj=obj,
			results=results,
			p_cls2d_cls=p_cls2d_cls
		)

		table_ls.append(ret["table"])
		auroc_px_ls.append(ret["auroc_px"])
		f1_px_ls.append(ret["f1_px"])
		ap_px_ls.append(ret["ap_px"])

		for defect_name, values in ret["per_class"].items():
			if defect_name not in auroc_px_per_defect_cls_all:
				auroc_px_per_defect_cls_all[defect_name] = []
				ap_px_per_defect_cls_all[defect_name] = []
				f1_px_per_defect_cls_all[defect_name] = []

			auroc_px_per_defect_cls_all[defect_name].append(values["auroc"])
			ap_px_per_defect_cls_all[defect_name].append(values["ap"])
			f1_px_per_defect_cls_all[defect_name].append(values["f1"])

	per_defect_cls_table = []
	for defect_name in sorted(auroc_px_per_defect_cls_all.keys()):
		auroc_v = np.round(np.mean(auroc_px_per_defect_cls_all[defect_name]) * 100, decimals=6)
		ap_v = np.round(np.mean(ap_px_per_defect_cls_all[defect_name]) * 100, decimals=6)
		f1_v = np.round(np.mean(f1_px_per_defect_cls_all[defect_name]) * 100, decimals=6)
		per_defect_cls_table.append([defect_name, auroc_v, f1_v, ap_v])

	per_defect_results = tabulate(
		per_defect_cls_table,
		headers=["defects", "auroc_px", "f1_px", "ap_px"],
		tablefmt="pipe"
	)
	logger.info("\n%s", per_defect_results)

	table_ls.append([
		"mean",
		str(np.round(np.mean(auroc_px_ls) * 100, decimals=1)),
		str(np.round(np.mean(f1_px_ls) * 100, decimals=1)),
		str(np.round(np.mean(ap_px_ls) * 100, decimals=1)),
	])

	results_table = tabulate(
		table_ls,
		headers=["objects", "auroc_px", "f1_px", "ap_px"],
		tablefmt="pipe"
	)
	logger.info("\n%s", results_table)


if __name__ == "__main__":
	parser = argparse.ArgumentParser("MultiADS Multi-Defect Test", add_help=True)

	parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
	parser.add_argument("--save_path", type=str, default="./results/mvtec_visa_multi_seg/zero_shot/", help="path to save results")
	parser.add_argument("--checkpoint_path", type=str, default="./exps/mvtec/epoch_1.pth", help="path to checkpoint")
	parser.add_argument("--config_path", type=str, default="./open_clip/model_configs/ViT-L-14-336.json", help="model configs")

	parser.add_argument("--dataset", type=str, default="visa", help="test dataset")
	parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")
	parser.add_argument("--pretrained", type=str, default="openai", help="pretrained weight used")
	parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
	parser.add_argument("--image_size", type=int, default=518, help="image size")
	parser.add_argument("--temperature", type=float, default=0.01, help="temperature for contrastive segmentation")

	parser.add_argument("--test_batch_size", type=int, default=1, help="test batch size")
	parser.add_argument("--num_workers", type=int, default=4, help="dataloader workers")
	parser.add_argument("--prefetch_factor", type=int, default=2, help="dataloader prefetch factor")
	parser.add_argument("--seed", type=int, default=42, help="random seed")
	parser.add_argument("--visualization", action="store_true")

	args = parser.parse_args()

	setup_seed(args.seed)
	test(args)