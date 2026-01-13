# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import numpy as np
import random
import os
import json
import argparse
from torch.utils.data import DataLoader
from datetime import datetime
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import logging
from torchvision.transforms import InterpolationMode

import open_clip
from dataset import VisaDatasetV2, MVTecDataset, MPDDDataset, RealIADDataset_v2
from model import LinearLayer
from loss import FocalLoss, BinaryDiceLoss
from prompts.prompt_ensemble_mvtec_20cls import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_mvtec
from prompts.prompt_ensemble_visa_19cls import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_visa
from prompts.new_prompt_ensemble_mpdd import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_mpdd
from prompts.prompt_ensemble_real_IAD_simple import encode_text_with_prompt_ensemble as encode_text_with_prompt_ensemble_real_iad
import re
from tqdm import tqdm
import csv

import segmentation_models_pytorch as smp
from loss import DiceLoss

import pdb

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def search_in_csv(file_path, keyword):
    with open(file_path, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # Check if the first column matches the keyword
            if row[0] == keyword:
                return row[1]
        print("Keyword not found.")
        return None

def train(args):
    # configs
    epochs = args.epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    image_size = args.image_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    txt_path = os.path.join(save_path, 'log.txt')  # log

    # model configs
    features_list = args.features_list
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)

    # clip model
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, image_size, pretrained=args.pretrained)
    model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model)

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('train')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # record parameters
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')

    # transforms
    target_transform_b = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])
    target_transform_type = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
        transforms.CenterCrop(image_size),
        transforms.PILToTensor(),  # uint8 [1,H,W], values 0..K-1
        transforms.Lambda(lambda x: x.squeeze(0).long()),  # [H,W] long
    ])


    # datasets
    assert args.dataset in ['mvtec', 'visa', 'mpdd'] 
    if args.dataset == 'mvtec':
        train_data = MVTecDataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform_b, target_transform_type = target_transform_type, 
                                aug_rate=args.aug_rate)
    elif args.dataset == 'visa':
        train_data = VisaDatasetV2(root=args.train_data_path, transform=preprocess, target_transform_b=target_transform_b,target_transform_type = target_transform_type)
    elif args.dataset == 'mpdd':
        train_data =  MPDDDataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform_b, aug_rate=args.aug_rate)


    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # linear layer
    trainable_layer = LinearLayer(model_configs['vision_cfg']['width'], model_configs['embed_dim'],
                                len(args.features_list), args.model).to(device)

    optimizer = torch.optim.Adam(list(trainable_layer.parameters()), lr=learning_rate, betas=(0.5, 0.999))

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_dice_m = DiceLoss(from_logits=False) #DiceLoss()

    # text prompt
    with torch.amp.autocast('cuda'), torch.no_grad():
        obj_list = train_data.get_cls_names()
        if args.dataset == 'mvtec':
            text_prompts = encode_text_with_prompt_ensemble_mvtec(model, obj_list, tokenizer, device)
        elif args.dataset == 'visa':
            text_prompts = encode_text_with_prompt_ensemble_visa(model, obj_list, tokenizer, device)
        elif args.dataset == 'mpdd':
            text_prompts = encode_text_with_prompt_ensemble_mpdd(model, obj_list, tokenizer, device)
        elif args.dataset == 'real_iad':
            text_prompts = encode_text_with_prompt_ensemble_real_iad(model, obj_list, tokenizer, device)

    for epoch in range(epochs):
        print("EPOCH = ", epoch)
        loss_list = []
        idx = 0
        for items in tqdm(train_dataloader):
            idx += 1
            image = items['img'].to(device)
            paths = items['img_path']
            cls_name = items['cls_name'] 
            img_mask = items["img_mask"].to(device, non_blocking=True).long()        # [B,H,W], values: 0..C-1
            img_mask_b = items["img_mask_b"].to(device, non_blocking=True).float()   # [B,H,W], values: 0/1

            # new GT data
            # if args.dataset == 'mvtec' or args.dataset == 'mpdd':
            #     cls_id = []               
            #     for i in paths:
            #         match = re.search(r'\/([^\/]+)\/[^\/]*$', i) # './data/mvtec/transistor/test/good/004.png', './data/mvtec/carpet/test/hole/002.png', './data/mvtec/metal_nut/test/scratch/004.png',
            #         cls_id.append(int(gt_defect[str(match.group(1))]))
            # elif args.dataset == 'visa':
            #     defect_cls = items['defect_cls']
            #     cls_id = [gt_defect[name] for name in defect_cls]


            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    image_features, patch_tokens = model.encode_image(image, features_list)

                    
                    text_features = []
                    for cls in cls_name:
                        text_features.append(text_prompts[cls])
                        
                    text_features = torch.stack(text_features, dim=0)
                # pixel level
                patch_tokens = trainable_layer(patch_tokens) # [4, 1, 1370]         

                anomaly_maps = []
                for layer in range(len(patch_tokens)):
                    patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = ((patch_tokens[layer] @ text_features) / 0.01)

                    B, L, C = anomaly_map.shape
                    H = int(np.sqrt(L))
                    
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, C, H, H),
                                                size=image_size, mode='bilinear', align_corners=True)
                    anomaly_map = torch.softmax(anomaly_map, dim=1)
                    anomaly_maps.append(anomaly_map)

            
            # gt = items['img_mask'].to(device) # B, H, W
            # gt_b = gt.clone()
            # for i in range(gt.size(0)):
            #     gt[i][gt[i] > 0.5], gt[i][gt[i] <= 0.5] = cls_id[i], 0 #cls_id[i], 0
            #     gt_b[i][gt_b[i] > 0.5], gt_b[i][gt_b[i] <= 0.5] = 1, 0 #cls_id[i], 0

            # gt = gt.long()

            # losses
            loss = 0
            for num in range(len(anomaly_maps)):              
                loss += loss_focal(anomaly_maps[num], img_mask) # a->xyz b->abc 21, 518,518
                loss += loss_dice(torch.sum(anomaly_maps[num][:, 1:, :, :], dim=1), img_mask_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({'trainable_linearlayer': trainable_layer.state_dict()}, ckp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("MultiADS", add_help=True)
    # path
    parser.add_argument("--train_data_path", type=str, default="./data/mvtec", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./exps/mvtec/', help='path to save results')
    parser.add_argument("--config_path", type=str, default='./open_clip/model_configs/ViT-L-14-336.json', help="model configs")
    # model
    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")
    parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")
    parser.add_argument("--pretrained", type=str, default="openai", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    # hyper-parameter
    parser.add_argument("--epoch", type=int, default=10, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--aug_rate", type=float, default=0.2, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    # setup_seed(111)
    setup_seed(args.seed)
    #setup_seed(100)
    train(args) 

