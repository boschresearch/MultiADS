# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np

import pdb

product_type2defect_type = {
    'Bear': ['good', 'Stains'],
    'Bird': ['good', 'Missing'],
    'Elephant': ['good', 'Missing'],
    'Parrot': ['good', 'Missing'],
    'Pig': ['good', 'Missing'],
    'Puppy': ['good', 'Stains'],
    'Scorpion': ['good', 'Missing'],
    'Turtle': ['good', 'Stains'],
    'Unicorn': ['good', 'Missing'],
    'Whale': ['good', 'Stains']
}

def encode_text_with_prompt_ensemble(model, objs, tokenizer, device):



    # good = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    # hole = ['{} with a hole defect']
    # scratch = ['{} has a scratch defect', 'flawed  {} with a scratch']
    # mismatch = ['{} with bend and parts mismatch defect', '{} with parts mismatch defect']
    # defective_painting = ['{} with a defective painting defect']
    # rust = ['{} with a rust defect']
    # flattening = ['{} becomes flattened', '{} has a flatten defect']

    good = [
        '{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', 
        '{} without defect', '{} without damage', '{} in pristine condition', 
        '{} with no imperfections', '{} of ideal quality'
    ]

    stain = [
        '{} with a stain defect', 'inconsistent color on {}', '{} with color discrepancies'
    ]

    missing = [
        '{} has a missing defect', 'flawed {} with something missing', '{} has missing components', 
        'missing parts on {}', '{} shows absent pieces', 'certain parts missing from {}'
    ]


    prompt_state = [good, stain, missing]
    
    prompt_templates = ['a bad photo of a {}.','a low resolution photo of the {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.', 'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

    text_prompts = {}
    for obj in objs:
        text_features = []
        for i in range(len(prompt_state)):
            prompted_state = [state.format(obj) for state in prompt_state[i]]
            prompted_sentence = []
            for s in prompted_state:
                for template in prompt_templates:
                    prompted_sentence.append(template.format(s))
            prompted_sentence = tokenizer(prompted_sentence).to(device)
            class_embeddings = model.encode_text(prompted_sentence)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_features.append(class_embedding)

        text_features = torch.stack(text_features, dim=1).to(device)
        text_prompts[obj] = text_features

    return text_prompts