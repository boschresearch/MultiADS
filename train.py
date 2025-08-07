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

    # new ids for ground truth data split into specific defect category
    # gt_defect = {'normal':0, 'bent':1, 'breakage down the middle':2, 'bubble':3, 'burnt':4, 'chip around edge and corner':5, 'chunk of gum missing':6, 'chunk of wax missing':7, 'color spot similar to the object':8, 'corner and edge breakage':2, 'corner missing':9, 'corner or edge breakage':2, 'damaged corner of packaging':10, 'different colour spot':8, 'discolor':8, 'melt':11, 'scratch':12, 'different color spot':8, 'middle breakage':2, 'missing':9, 'scratches':12, 'weird candle wick':13, 'damage':10, 'fryum stuck together':14, 'leak':15, 'similar colour spot':8, 'small chip around edge':5, 'extra wax in candle':7, 'extra':16, 'misshape':17, 'small cracks':18, 'small holes':19, 'foreign particals on candle':20, 'other':21, 'small scratches':13, 'wrong place':22, 'dirt':23, 'stuck together':15, 'wax melded out of the candle':7, 'same colour spot':8}
    # gt_defect = {'normal':0, 'bent':1, 'breakage down the middle':2, 'bubble':3, 'burnt':4, 'chip around edge and corner':5, 'chunk of gum missing':6, 'chunk of wax missing':7, 'color spot similar to the object':8, 'corner and edge breakage':9, 'corner missing':10, 'corner or edge breakage':11, 'damaged corner of packaging':12, 'different colour spot':13, 'discolor':14, 'melt':15, 'scratch':16, 'different color spot':16, 'middle breakage':18, 'missing':19, 'scratches':20, 'weird candle wick':21, 'damage':22, 'fryum stuck together':23, 'leak':24, 'similar colour spot':25, 'small chip around edge':26, 'extra wax in candle':27, 'extra':28, 'misshape':29, 'small cracks':30, 'small holes':31, 'foreign particals on candle':32, 'other':33, 'small scratches':34, 'wrong place':35, 'dirt':36, 'stuck together':37, 'wax melded out of the candle':38, 'same colour spot':39}

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
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])

    # datasets
    assert args.dataset in ['mvtec', 'visa', 'mpdd'] 
    if args.dataset == 'mvtec':
        train_data = MVTecDataset(root=args.train_data_path, transform=preprocess, target_transform=transform,
                                aug_rate=args.aug_rate)
        # gt_defect = {"good":0, "bent":1, "bent_lead":2, "bent_wire":3, "broken":4, "broken_large":5, "broken_small":6, "broken_teeth":7, "color":8, "combined":9, "contamination":10, "metal_contamination":11, "crack":12, "cut":13, "cut_inner_insulation":14, "cut_lead":15, "cut_outer_insulation":16, "fabric":17, "manipulated_front":18, "fabric_border":19, "fabric_interior":20, "faulty_imprint":21, "print":22, "glue":23, "glue_strip":24, "hole":25, "missing":26, "missing_wire":27, "missing_cable":28, "poke":29, "poke_insulation":30, "rough":31, "scratch":32, "scratch_head":33, "scratch_neck":34, "squeeze":35, "squeezed_teeth":36, "thread":37, "thread_side":38, "thread_top":39, "liquid":40, "oil":41, "misplaced":42, "cable_swap":43, "flip":44, "fold":45, "split_teeth":46, "damaged_case":47, "defective":48, "gray_stroke":49, "pill_type":50}
        gt_defect = {"good":0, "bent":1, "bent_lead":1, "bent_wire":1, "manipulated_front":1, "broken":2, "broken_large":2, "broken_small":2, "broken_teeth":2, "color":3, "combined":4, "contamination":5, "metal_contamination":5, "crack":6, "cut":7, "cut_inner_insulation":7, "cut_lead":7, "cut_outer_insulation":7, "fabric":8, "fabric_border":8, "fabric_interior":8, "faulty_imprint":9, "print":9, "glue":10, "glue_strip":10, "hole":11, "missing":12, "missing_wire":12, "missing_cable":12, "poke":13, "poke_insulation":13, "rough":14, "scratch":15, "scratch_head":15, "scratch_neck":15, "squeeze":16, "squeezed_teeth":16, "thread":17, "thread_side":17, "thread_top":17, "liquid":18, "oil":18, "misplaced":19, "cable_swap":19, "flip":19, "fold":19, "split_teeth":19, "damaged_case":20, "defective":20, "gray_stroke":20, "pill_type":20}  
    elif args.dataset == 'visa':
        train_data = VisaDatasetV2(root=args.train_data_path, transform=preprocess, target_transform=transform)
        gt_defect = {'normal': 0, 'damage': 1, 'scratch':2, 'breakage': 3, 'burnt': 4, 'weird wick': 5, 'stuck': 6, 'crack': 7, 'wrong place': 8, 'partical': 9, 'bubble': 10, 'melded': 11, 'hole': 12, 'melt': 13, 'bent':14, 'spot': 15, 'extra': 16, 'chip': 17, 'missing': 18}
    elif args.dataset == 'mpdd':
        train_data =  MPDDDataset(root=args.train_data_path, transform=preprocess, target_transform=transform, aug_rate=args.aug_rate)
        gt_defect =  {"good":0, 'hole':1, 'scratches':2, 'bend_and_parts_mismatch':3, 'parts_mismatch':4, 'defective_painting':5, 'major_rust':6, 'total_rust':6, 'flattening':7}
    elif args.dataset == 'real_iad':
        test_data = RealIADDataset_v2(root=args.train_data_path, transform=preprocess, aug_rate=-1, target_transform=transform, mode='test')
        gt_defect =  {"good":0, 'pit':1, 'deformation':2, 'abrasion':3, 'scratch':4, 'damage':5, 'missing':6, 'foreign':7, 'contamination':8}
        
        


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
    with torch.cuda.amp.autocast(), torch.no_grad():
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

            # new GT data
            if args.dataset == 'mvtec' or args.dataset == 'mpdd':
                cls_id = []               
                for i in paths:
                    match = re.search(r'\/([^\/]+)\/[^\/]*$', i) # './data/mvtec/transistor/test/good/004.png', './data/mvtec/carpet/test/hole/002.png', './data/mvtec/metal_nut/test/scratch/004.png',
                    cls_id.append(int(gt_defect[str(match.group(1))]))
            elif args.dataset == 'visa':
                defect_cls = items['defect_cls']
                cls_id = [gt_defect[name] for name in defect_cls]


            with torch.cuda.amp.autocast():
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

            # losses
            gt = items['img_mask'].to(device) # B, H, W
            gt_b = gt.clone()
            for i in range(gt.size(0)):
                gt[i][gt[i] > 0.5], gt[i][gt[i] <= 0.5] = cls_id[i], 0 #cls_id[i], 0
                gt_b[i][gt_b[i] > 0.5], gt_b[i][gt_b[i] <= 0.5] = 1, 0 #cls_id[i], 0

            gt = gt.long()
            loss = 0
            for num in range(len(anomaly_maps)):              
                loss += loss_focal(anomaly_maps[num], gt) # a->xyz b->abc 21, 518,518
                loss += loss_dice(torch.sum(anomaly_maps[num][:, 1:, :, :], dim=1), gt_b)

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

