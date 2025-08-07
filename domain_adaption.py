# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
from dataset import VisaDataset, MVTecDataset, MPDDDataset, MADDataset, RealIADDataset_v2
from tqdm import tqdm 
def memory(model_name, model, obj_list, dataset_dir, save_path, preprocess, transform,
           few_shot_features, dataset_name, device):
    mem_features = {}
    for obj in obj_list:
        if dataset_name == 'mvtec':
            data = MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                aug_rate=-1, mode='test', save_dir=save_path, obj_name=obj)
        elif dataset_name == 'visa':
            data = VisaDataset(root=dataset_dir, transform=preprocess, target_transform=transform, mode='test', obj_name=obj)
        elif dataset_name == 'mpdd':
            data = MPDDDataset(root=dataset_dir, transform=preprocess, target_transform=transform, aug_rate=-1, mode='test', obj_name=obj)
        else:
            raise AssertionError("mvtec, visa, mpdd only")
        # elif dataset_name == 'mpdd':
        #     data = MPDDDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
        #                         aug_rate=-1, mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj)
        # elif dataset_name == 'mad_sim' or dataset_name == 'mad_real':
        #     data = MADDataset(root=dataset_dir, transform=preprocess, target_transform=transform, mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj)
        # elif dataset_name == 'real_iad':
        #     data = RealIADDataset_v2(root=dataset_dir, transform=preprocess, target_transform=transform, aug_rate=-1, mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj)

        dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
        features = []
        for items in tqdm(dataloader):
            image = items['img'].to(device)
            with torch.no_grad():
                image_features, patch_tokens = model.encode_image(image, few_shot_features) # patch_tokens (B, L, D)
                if 'ViT' in model_name:
                    patch_tokens = [p[0, 1:, :].cpu() for p in patch_tokens]
                else:
                    patch_tokens = [p[0].view(p.shape[1], -1).permute(1, 0).contiguous() for p in patch_tokens]
                features.append(patch_tokens)
        mem_features[obj] = [torch.cat(
            [features[j][i] for j in range(len(features))], dim=0) for i in range(len(features[0]))] # mem_features[obj]
    return mem_features