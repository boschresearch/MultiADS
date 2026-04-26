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
from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve, pairwise
from torchvision.transforms import InterpolationMode
from concurrent.futures import ProcessPoolExecutor, as_completed

import open_clip
from few_shot import memory as memory_fs
from domain_adaption import memory as memory_da
from model import LinearLayer
from dataset import VisaDatasetTest, MVTecDataset, MPDDDataset, RealIADDataset_v2, MADDataset

from prompts.prompt_ensemble_visa_19cls_test import (
	encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_visa,
	product_type2defect_type as product_type2defect_type_visa,
)
from prompts.prompt_ensemble_mvtec_20cls import (
	encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_mvtec,
	product_type2defect_type as product_type2defect_type_mvtec,
)
from prompts.new_prompt_ensemble_mpdd import (
	encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_mpdd,
	product_type2defect_type as product_type2defect_type_mpdd,
)
from prompts.prompt_ensemble_mad_real import (
	encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_mad_real,
	product_type2defect_type as product_type2defect_type_mad_real,
)
from prompts.prompt_ensemble_mad_sim import (
	encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_mad_sim,
	product_type2defect_type as product_type2defect_type_mad_sim,
)
from prompts.prompt_ensemble_real_IAD import (
	encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_real_iad,
	product_type2defect_type as product_type2defect_type_real_iad,
)

from tqdm import tqdm


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
	else:
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


def evaluate_one_object(obj, cls_names, imgs_masks, anomaly_maps, gt_sp_list, pr_sp_list):
	table = [obj]

	gt_px = []
	pr_px = []
	gt_sp = []
	pr_sp = []
	pr_sp_tmp = []

	for idx in range(len(cls_names)):
		if cls_names[idx] == obj:
			mask = np.asarray(imgs_masks[idx])
			mask = np.squeeze(mask)
			gt_px.append(mask)
			pr_px.append(anomaly_maps[idx])
			pr_sp_tmp.append(np.max(anomaly_maps[idx]))
			gt_sp.append(gt_sp_list[idx])
			pr_sp.append(pr_sp_list[idx])

	gt_px = np.array(gt_px)
	pr_px = np.array(pr_px)
	gt_sp = np.array(gt_sp)
	pr_sp = np.array(pr_sp)

	pr_sp_tmp = np.array(pr_sp_tmp)
	if len(pr_sp_tmp) > 0:
		den = pr_sp_tmp.max() - pr_sp_tmp.min()
		pr_sp_tmp = (pr_sp_tmp - pr_sp_tmp.min()) / den if den > 1e-12 else np.zeros_like(pr_sp_tmp)
	else:
		pr_sp_tmp = np.zeros_like(pr_sp)

	pr_sp = 0.5 * (pr_sp + pr_sp_tmp)

	auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
	auroc_sp = roc_auc_score(gt_sp, pr_sp)
	ap_sp = average_precision_score(gt_sp, pr_sp)
	ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())

	precisions, recalls, _ = precision_recall_curve(gt_sp, pr_sp)
	f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-12)
	f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])

	precisions, recalls, _ = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
	f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-12)
	f1_px = np.max(f1_scores[np.isfinite(f1_scores)])

	if len(gt_px.shape) == 4:
		gt_px = gt_px.squeeze(1)
	if len(pr_px.shape) == 4:
		pr_px = pr_px.squeeze(1)

	aupro = cal_pro_score(gt_px, pr_px)

	table.append(str(np.round(auroc_px * 100, decimals=1)))
	table.append(str(np.round(f1_px * 100, decimals=1)))
	table.append(str(np.round(ap_px * 100, decimals=1)))
	table.append(str(np.round(aupro * 100, decimals=1)))
	table.append(str(np.round(auroc_sp * 100, decimals=1)))
	table.append(str(np.round(f1_sp * 100, decimals=1)))
	table.append(str(np.round(ap_sp * 100, decimals=1)))

	return {
		"obj": obj,
		"table": table,
		"auroc_sp": auroc_sp,
		"auroc_px": auroc_px,
		"f1_sp": f1_sp,
		"f1_px": f1_px,
		"aupro": aupro,
		"ap_sp": ap_sp,
		"ap_px": ap_px,
	}


def test(args):
	img_size = args.image_size
	features_list = args.features_list
	few_shot_features = args.few_shot_features
	dataset_dir = args.data_path
	save_path = args.save_path
	dataset_name = args.dataset

	os.makedirs(save_path, exist_ok=True)
	device = "cuda" if torch.cuda.is_available() else "cpu"
	txt_path = os.path.join(save_path, 'log.txt')

	# clip
	model, _, preprocess = open_clip.create_model_and_transforms(
		args.model, img_size, pretrained=args.pretrained
	)
	model.to(device)
	model.eval()
	tokenizer = open_clip.get_tokenizer(args.model)

	# logger
	root_logger = logging.getLogger()
	for handler in root_logger.handlers[:]:
		root_logger.removeHandler(handler)
	root_logger.setLevel(logging.WARNING)

	logger = logging.getLogger('test')
	logger.handlers.clear()
	logger.setLevel(logging.INFO)
	logger.propagate = False

	formatter = logging.Formatter(
		'%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
		datefmt='%y-%m-%d %H:%M:%S'
	)

	file_handler = logging.FileHandler(txt_path, mode='a')
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)

	console_handler = logging.StreamHandler()
	console_handler.setFormatter(formatter)
	logger.addHandler(console_handler)

	# record parameters
	for arg in vars(args):
		if args.mode == 'zero_shot' and (arg == 'k_shot' or arg == 'few_shot_features'):
			continue
		logger.info(f'{arg}: {getattr(args, arg)}')

	# seg
	with open(args.config_path, 'r') as f:
		model_configs = json.load(f)

	linearlayer = LinearLayer(
		model_configs['vision_cfg']['width'],
		model_configs['embed_dim'],
		len(features_list),
		args.model
	).to(device)

	checkpoint = torch.load(args.checkpoint_path, map_location=device)
	linearlayer.load_state_dict(checkpoint["trainable_linearlayer"])
	linearlayer.eval()

	# dataset
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

	if args.dataset == 'mvtec':
		test_data = MVTecDataset(
			root=dataset_dir,
			transform=preprocess,
			target_transform=target_transform_b,
			target_transform_type=target_transform_type,
			aug_rate=-1,
			mode='test'
		)
		defects = [
			'good', 'bent', 'broken', 'color', 'combined', 'contamination',
			'crack', 'cut', 'fabric', 'faulty imprint', 'glue', 'hole',
			'missing', 'poke', 'rough', 'scratch', 'squeeze', 'thread',
			'liquid', 'misplaced', 'damaged'
		]
		p_cls2d_cls = product_type2defect_type_mvtec

	elif args.dataset == 'visa':
		test_data = VisaDatasetTest(
			root=dataset_dir,
			transform=preprocess,
			target_transform=target_transform_b,
			mode='test'
		)
		defects = [
			'normal', 'damage', 'scratch', 'breakage', 'burnt', 'weird wick',
			'stuck', 'crack', 'wrong place', 'partical', 'bubble', 'melded',
			'hole', 'melt', 'bent', 'spot', 'extra', 'chip', 'missing',
			'discolor', 'leak'
		]
		p_cls2d_cls = product_type2defect_type_visa

	elif args.dataset == 'mpdd':
		test_data = MPDDDataset(
			root=dataset_dir,
			transform=preprocess,
			target_transform=target_transform_b,
			target_transform_type=target_transform_type,
			aug_rate=-1,
			mode='test'
		)
		defects = ['good', 'hole', 'scratch', 'bent', 'mismatch', 'defective painting', 'rust', 'flattening']
		p_cls2d_cls = product_type2defect_type_mpdd

	elif args.dataset == 'real_iad':
		test_data = RealIADDataset_v2(
			root=dataset_dir,
			transform=preprocess,
			target_transform=target_transform_b,
			target_transform_type=target_transform_type,
			mode='test'
		)
		defects = ['good', 'pit', 'deformation', 'abrasion', 'scratch', 'damage', 'missing', 'foreign', 'contamination']
		p_cls2d_cls = product_type2defect_type_real_iad

	elif args.dataset == 'mad_sim':
		test_data = MADDataset(
			root=dataset_dir,
			transform=preprocess,
			target_transform=target_transform_b,
			target_transform_type=target_transform_type,
			mode='test',
			datatype='sim'
		)
		defects = ['good', 'Stains', 'Missing', 'Burrs']
		p_cls2d_cls = product_type2defect_type_mad_sim

	elif args.dataset == 'mad_real':
		test_data = MADDataset(
			root=dataset_dir,
			transform=preprocess,
			target_transform=target_transform_b,
			target_transform_type=target_transform_type,
			mode='test',
			datatype='real'
		)
		defects = ['good', 'Stains', 'Missing']
		p_cls2d_cls = product_type2defect_type_mad_real

	else:
		raise ValueError(f"Unsupported dataset: {args.dataset}")

	test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
	obj_list = test_data.get_cls_names()

	# few-shot
	if args.mode == 'few_shot':
		mem_features = memory_fs(
			args.model, model, obj_list, dataset_dir, save_path, preprocess,
			target_transform_b, target_transform_type,
			args.k_shot, few_shot_features, dataset_name, device
		)

	if args.mode == 'domain_adaption':
		mem_features = memory_da(
			args.model, model, obj_list, dataset_dir, save_path, preprocess,
			target_transform_b, target_transform_type,
			few_shot_features, dataset_name, device
		)

	# text prompt
	with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()), torch.no_grad():
		if args.dataset == 'mvtec':
			text_prompts = encode_text_with_prompt_ensemble_mvtec(model, obj_list, tokenizer, device)
		elif args.dataset == 'visa':
			text_prompts = encode_text_with_prompt_ensemble_visa(model, obj_list, tokenizer, device)
		elif args.dataset == 'mpdd':
			text_prompts = encode_text_with_prompt_ensemble_mpdd(model, obj_list, tokenizer, device)
		elif args.dataset == 'mad_real':
			text_prompts = encode_text_with_prompt_ensemble_mad_real(model, obj_list, tokenizer, device)
		elif args.dataset == 'mad_sim':
			text_prompts = encode_text_with_prompt_ensemble_mad_sim(model, obj_list, tokenizer, device)
		elif args.dataset == 'real_iad':
			text_prompts = encode_text_with_prompt_ensemble_real_iad(model, obj_list, tokenizer, device)
		else:
			raise ValueError(f"Unsupported dataset: {args.dataset}")

	defect_to_idx = {d: i for i, d in enumerate(defects)}

	results = {
		'cls_names': [],
		'imgs_masks': [],
		'anomaly_maps': [],
		'gt_sp': [],
		'pr_sp': [],
	}

	for items in tqdm(test_dataloader, desc="Testing", leave=True):
		image = items['img'].to(device)
		cls_name = items['cls_name']
		results['cls_names'].append(cls_name[0])

		gt_mask = items['img_mask_b'].clone()
		for i in range(gt_mask.size(0)):
			gt_mask[i][gt_mask[i] > 0.5], gt_mask[i][gt_mask[i] <= 0.5] = 1, 0

		results['imgs_masks'].append(gt_mask.cpu().numpy())
		results['gt_sp'].append(items['anomaly'].item())

		with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
			image_features, patch_tokens = model.encode_image(image, features_list)
			image_features /= image_features.norm(dim=-1, keepdim=True)

			text_features = []
			for cls in cls_name:
				defects_indices = [defect_to_idx[d] for d in p_cls2d_cls[cls]]
				text_features.append(text_prompts[cls][:, defects_indices])

			text_features = torch.stack(text_features, dim=0)

			# image-level
			text_probs = (image_features @ text_features[0] / args.temperature).softmax(dim=-1)
			results['pr_sp'].append(text_probs[0][1:].sum().cpu().item())

			# pixel-level
			patch_tokens = linearlayer(patch_tokens)
			anomaly_maps = []
			for layer in range(len(patch_tokens)):
				patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
				anomaly_map = (patch_tokens[layer] @ text_features / args.temperature)
				B, L, C = anomaly_map.shape
				H = int(np.sqrt(L))

				anomaly_map = F.interpolate(
					anomaly_map.permute(0, 2, 1).view(B, C, H, H),
					size=img_size,
					mode='bilinear',
					align_corners=True
				)
				anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1:, :, :].sum(dim=1)
				anomaly_maps.append(anomaly_map.cpu().numpy())

			anomaly_map = np.sum(anomaly_maps, axis=0)

			# few-shot
			if args.mode == 'few_shot':
				_, patch_tokens_fs = model.encode_image(image, few_shot_features)
				anomaly_maps_few_shot = []
				for idx, p in enumerate(patch_tokens_fs):
					if 'ViT' in args.model:
						p = p[0, 1:, :]
					else:
						p = p[0].view(p.shape[1], -1).permute(1, 0).contiguous()

					cos = pairwise.cosine_similarity(mem_features[cls_name[0]][idx].cpu(), p.cpu())
					height = int(np.sqrt(cos.shape[1]))
					anomaly_map_few_shot = np.min((1 - cos), 0).reshape(1, 1, height, height)
					anomaly_map_few_shot = F.interpolate(
						torch.tensor(anomaly_map_few_shot),
						size=img_size,
						mode='bilinear',
						align_corners=True
					)
					anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())

				anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
				anomaly_map = anomaly_map + anomaly_map_few_shot

			# domain adaption
			if args.mode == 'domain_adaption':
				_, patch_tokens_da = model.encode_image(image, few_shot_features)
				anomaly_maps_da = []
				for idx, p in enumerate(patch_tokens_da):
					if 'ViT' in args.model:
						p = p[0, 1:, :]
					else:
						p = p[0].view(p.shape[1], -1).permute(1, 0).contiguous()

					cos = pairwise.cosine_similarity(mem_features[cls_name[0]][idx].cpu(), p.cpu())
					M, _ = cos.shape
					height = int(np.sqrt(cos.shape[1]))
					distances = 1.0 - cos

					k = max(1, int(np.ceil(M * args.quantile)))
					smallest_kplus1 = np.partition(distances, k, axis=0)[:k+1, :]
					sorted_smallest_kplus1 = np.sort(smallest_kplus1, axis=0)
					smallest_k = sorted_smallest_kplus1[1:k+1, :]
					avg_smallest = smallest_k.mean(axis=0)

					anomaly_map_da = avg_smallest.reshape(1, 1, height, height)
					anomaly_map_da = F.interpolate(
						torch.tensor(anomaly_map_da),
						size=img_size,
						mode='bilinear',
						align_corners=True
					)
					anomaly_maps_da.append(anomaly_map_da[0].cpu().numpy())

				anomaly_map_da = np.sum(anomaly_maps_da, axis=0)
				anomaly_map = anomaly_map + anomaly_map_da

			results['anomaly_maps'].append(anomaly_map)

			# visualization
			if args.visualization:
				path = items['img_path']
				cls = path[0].split('/')[-2]
				filename = path[0].split('/')[-1]
				vis = cv2.cvtColor(cv2.resize(cv2.imread(path[0]), (img_size, img_size)), cv2.COLOR_BGR2RGB)
				mask = normalize(anomaly_map[0])
				vis = apply_ad_scoremap(vis, mask)
				vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
				save_vis = os.path.join(save_path, 'imgs', cls_name[0], cls)
				os.makedirs(save_vis, exist_ok=True)
				cv2.imwrite(os.path.join(save_vis, filename), vis)

	# metrics (multi-cpu)
	table_ls = []
	auroc_sp_ls = []
	auroc_px_ls = []
	f1_sp_ls = []
	f1_px_ls = []
	aupro_ls = []
	ap_sp_ls = []
	ap_px_ls = []

	if args.eval_workers <= 0:
		num_workers = min(len(obj_list), os.cpu_count() or 1)
	else:
		num_workers = min(args.eval_workers, len(obj_list))

	with ProcessPoolExecutor(max_workers=num_workers) as executor:
		futures = {
			executor.submit(
				evaluate_one_object,
				obj,
				results['cls_names'],
				results['imgs_masks'],
				results['anomaly_maps'],
				results['gt_sp'],
				results['pr_sp'],
			): obj
			for obj in obj_list
		}

		for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating objects (CPU)", leave=True):
			ret = future.result()
			table_ls.append(ret["table"])
			auroc_sp_ls.append(ret["auroc_sp"])
			auroc_px_ls.append(ret["auroc_px"])
			f1_sp_ls.append(ret["f1_sp"])
			f1_px_ls.append(ret["f1_px"])
			aupro_ls.append(ret["aupro"])
			ap_sp_ls.append(ret["ap_sp"])
			ap_px_ls.append(ret["ap_px"])

	# keep table order same as obj_list
	table_ls = sorted(table_ls, key=lambda x: obj_list.index(x[0]))

	table_ls.append([
		'mean',
		str(np.round(np.mean(auroc_px_ls) * 100, decimals=1)),
		str(np.round(np.mean(f1_px_ls) * 100, decimals=1)),
		str(np.round(np.mean(ap_px_ls) * 100, decimals=1)),
		str(np.round(np.mean(aupro_ls) * 100, decimals=1)),
		str(np.round(np.mean(auroc_sp_ls) * 100, decimals=1)),
		str(np.round(np.mean(f1_sp_ls) * 100, decimals=1)),
		str(np.round(np.mean(ap_sp_ls) * 100, decimals=1))
	])

	results = tabulate(
		table_ls,
		headers=['objects', 'auroc_px', 'f1_px', 'ap_px', 'aupro', 'auroc_sp', 'f1_sp', 'ap_sp'],
		tablefmt="pipe"
	)
	logger.info("\n%s", results)


if __name__ == '__main__':
	parser = argparse.ArgumentParser("MultiADS", add_help=True)

	# paths
	parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
	parser.add_argument("--save_path", type=str, default='./results/visa/zero_shot/', help='path to save results')
	parser.add_argument("--checkpoint_path", type=str, default='./exps/mvtec/epoch_1.pth', help='path to save results')
	parser.add_argument("--config_path", type=str, default='./open_clip/model_configs/ViT-L-14-336.json', help="model configs")

	# model
	parser.add_argument("--dataset", type=str, default='visa', help="test dataset")
	parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")
	parser.add_argument("--pretrained", type=str, default="openai", help="pretrained weight used")
	parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
	parser.add_argument("--few_shot_features", type=int, nargs="+", default=[6, 12, 18, 24], help="features used for few shot")
	parser.add_argument("--image_size", type=int, default=518, help="image size")
	parser.add_argument("--mode", type=str, default="zero_shot", help="zero shot or few shot or domain adaption")
	parser.add_argument("--temperature", type=float, default=0.01, help="temperature for constrastive learning")

	# few shot
	parser.add_argument("--k_shot", type=int, default=10, help="e.g., 10-shot, 5-shot, 1-shot")

	# domain adaption
	parser.add_argument("--quantile", type=float, default=0.0001, help="percent of the qunatile of nearest neighbour")

	# cpu eval workers
	parser.add_argument("--eval_workers", type=int, default=0, help="number of CPU workers for object-level evaluation; 0 means auto")

	parser.add_argument("--seed", type=int, default=42, help="random seed")
	parser.add_argument('--visualization', action='store_true')

	args = parser.parse_args()

	setup_seed(args.seed)
	test(args)