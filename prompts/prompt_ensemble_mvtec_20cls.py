# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np

import pdb

product_type2defect_type = {
    'bottle': ['good', 'broken', 'contamination'],
    'cable': ['good', 'bent', 'misplaced', 'combined', 'cut', 'missing', 'poke'],
    'capsule': ['good', 'crack', 'faulty imprint','poke', 'scratch', 'squeeze'],
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
    'zipper': ['good', 'broken', 'combined', 'fabric', 'rough', 'misplaced', 'squeeze']
}

def encode_text_with_prompt_ensemble(model, objs, tokenizer, device):

    # good = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    # bent = ['{} has a bent defect', '{} has a bent lead defect', '{} has a bent wire defect']
    # broken = ['{} has a broken defect', '{} has a broken large defect', '{} has a broken small defect', '{} has a broken teeth defect']
    # color = ['{} has a color defect']
    # combined = ['{} has a combined defect']
    # contamination = ['{} has a contamination defect', '{} has a metal contamination defect']
    # crack = ['{} has a crack defect']
    # cut = ['{} has a cut defect', '{} has a cut inner insulation defect', '{} has a cut lead defect', '{} has a cut outer insulation defect']
    # fabric = ['{} has a fabric defect', '{} has a manipulated front defect', '{} has a fabric border defect', '{} has a fabric interior defect']
    # faulty_imprint = ['{} has a faulty imprint defect', '{} has a print defect']
    # glue = ['{} has a glue defect', '{} has a glue strip defect']
    # hole = ['{} has a hole defect']
    # missing = ['{} has a missing defect', '{} has a missing wire defect', '{} has a missing cable defect']
    # poke = ['{} has a poke defect', '{} has a poke insulation defect']
    # rough = ['{} has a rough defect']
    # scratch = ['{} has a scratch defect', '{} has a scratch head defect', '{} has a scratch neck defect']
    # squeeze = ['{} has a squeeze defect', '{} has a squeeze teeth defect']
    # thread = ['{} has a thread defect', '{} has a thread side defect', '{} has a thread top defect']
    # liquid = ['{} has a liquid defect', '{} has a oil defect']
    # misplaced  = ['{} has a misplaced defect', '{} has a cable swap defect', '{} has a flip defect', '{} has a fold defect', '{} has a split teeth defect']
    # damaged = ['{} has a damaged case defect', '{} has a defective defect', '{} has a gray stroke defect', '{} has a pill type defect']

    # good = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    # bent = ['{} has a bent defect', 'flawed {} with a bent']
    # broken = ['{} has a broken defect', 'flawed {} with breakage']
    # color = ['{} has a color defect']
    # combined = ['{} has a combined defect']
    # contamination = ['{} has a contamination defect']
    # crack = ['{} has a crack defect']
    # cut = ['{} has a cut defect']
    # fabric = ['{} has a fabric defect', '{} has a fabric border defect', '{} has a fabric interior defect']
    # faulty_imprint = ['{} has a faulty imprint defect', '{} has a print defect']
    # glue = ['{} has a glue defect', '{} has a glue strip defect']
    # hole = ['{} has a hole defect', 'a hole on {}']
    # missing = ['{} has a missing defect', 'flawed {} with something missing']
    # poke = ['{} has a poke defect', '{} has a poke insulation defect']
    # rough = ['{} has a rough defect']
    # scratch = ['{} has a scratch defect', 'flawed  {} with a scratch']
    # squeeze = ['{} has a squeeze defect', 'flawed {} with a squeeze']
    # thread = ['{} has a thread defect', 'flawed {} with a thread']
    # liquid = ['{} has a liquid defect', 'flawed {} with liquid', '{} with oil']
    # misplaced  = ['{} has a misplaced defect', 'flawed {} with misplacing']
    # damaged = ['{} has a damaged defect', 'flawed {} with damage']

    good = [
        '{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', 
        '{} without defect', '{} without damage', '{} in ideal condition', 
        '{} with immaculate quality', '{} without any imperfections'
    ]

    bent = [
        '{} has a bent defect', 'flawed {} with a bent', 'a bend found in {}', 
        '{} with noticeable bending', 'a bent edge on {}', '{} has a slight curve defect'
    ]

    broken = [
        '{} has a broken defect', 'flawed {} with breakage', 'visible breakage on {}', 
        '{} with broken areas', '{} shows signs of breaking', 'cracked or broken spots on {}'
    ]

    color = [
        '{} has a color defect', 'inconsistent color on {}', '{} with color discrepancies', 
        '{} has a noticeable color difference', '{} with irregular coloring', 
        '{} has off-color patches'
    ]

    combined = [
        '{} has a combined defect', 'multiple issues with {}', '{} with mixed defects', 
        '{} showing multiple imperfections', 'several flaws found on {}', 
        '{} with combined defect types'
    ]

    contamination = [
        '{} has a contamination defect', 'foreign particles on {}', 
        '{} is contaminated', '{} contains contaminants', '{} has impurity issues', 
        'traces of contamination on {}'
    ]

    crack = [
        '{} has a crack defect', 'a crack is present on {}', 'cracked area on {}', 
        '{} with noticeable cracking', 'fine cracks found on {}', 
        '{} shows surface cracks'
    ]

    cut = [
        '{} has a cut defect', 'cut marks on {}', '{} with visible cuts', 
        'a cut detected on {}', '{} is sliced or cut', 'surface cut seen on {}'
    ]

    fabric = [
        '{} has a fabric defect', '{} has a fabric border defect', '{} has a fabric interior defect', 
        'fabric quality issues on {}', '{} with textile irregularities', 
        'fabric borders on {} show defects'
    ]

    faulty_imprint = [
        '{} has a faulty imprint defect', '{} has a print defect', 'incorrect printing on {}', 
        'misaligned print on {}', 'printing errors present on {}', '{} has a blurred print defect'
    ]

    glue = [
        '{} has a glue defect', '{} has a glue strip defect', 'excess glue on {}', 
        '{} with uneven glue application', '{} has visible glue spots', 
        'misplaced glue seen on {}'
    ]

    hole = [
        '{} has a hole defect', 'a hole on {}', 'visible hole on {}', 
        '{} with punctures', 'small hole found in {}', 'perforations present on {}'
    ]

    missing = [
        '{} has a missing defect', 'flawed {} with something missing', '{} has missing components', 
        'missing parts on {}', '{} shows absent pieces', 'certain parts missing from {}'
    ]

    poke = [
        '{} has a poke defect', '{} has a poke insulation defect', 'visible poke mark on {}', 
        '{} has puncture marks', 'a poke flaw on {}', 'small poke defect on {}'
    ]

    rough = [
        '{} has a rough defect', 'rough texture on {}', 'uneven surface on {}', 
        '{} is coarser than expected', 'surface roughness seen on {}', 
        'texture defects on {}'
    ]

    scratch = [
        '{} has a scratch defect', 'flawed {} with a scratch', 'visible scratches on {}', 
        '{} with surface scratches', 'minor scratches seen on {}', '{} shows scratch marks'
    ]

    squeeze = [
        '{} has a squeeze defect', 'flawed {} with a squeeze', 'squeezed area on {}', 
        '{} has compression marks', '{} appears squeezed', 'flattened areas on {}'
    ]

    thread = [
        '{} has a thread defect', 'flawed {} with a thread', 'loose threads on {}', 
        '{} has visible threads', 'untrimmed threads on {}', 'threads sticking out on {}'
    ]

    liquid = [
        '{} has a liquid defect', 'flawed {} with liquid', '{} with oil', 
        'liquid marks on {}', '{} with liquid residue', 'stains from liquid on {}'
    ]

    misplaced = [
        '{} has a misplaced defect', 'flawed {} with misplacing', '{} shows misalignment', 
        'misplaced parts on {}', '{} with incorrect positioning', 'positioning defects on {}'
    ]

    damaged = [
        '{} has a damaged defect', 'flawed {} with damage', '{} with visible damage', 
        'damaged areas on {}', 'physical damage seen on {}', 'noticeable wear on {}'
    ]

    
    # prompt_state_all = [good, bent, bent_lead, bent_wire, broken, broken_large, broken_small, broken_teeth, color, combined, contamination, metal_contamination, crack, cut, cut_inner_insulation, cut_lead, cut_outer_insulation, fabric, manipulated_front, fabric_border, fabric_interior, faulty_imprint, print_,glue, glue_strip, hole, missing, missing_wire, missing_cable, poke, poke_insulation, rough, scratch, scratch_head, scratch_neck, squeeze, squeeze_teeth, thread, thread_side, thread_top, liquid, oil, misplaced, cable_swap, flip, fold, split_teeth, damaged_case, defective, gray_stroke, pill_type]
    prompt_state_all = [good, bent, broken, color, combined, contamination, crack, cut, fabric, faulty_imprint, glue, hole, missing, poke, rough, scratch, squeeze, thread, liquid, misplaced, damaged]


    prompt_templates = ['a bad photo of a {}.','a low resolution photo of the {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.', 'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

    text_prompts = {}
    for obj in objs:
        text_features = []
        # if obj == 'bottle':
        #     prompt_state = [good, broken_large, broken_small, contamination]
        # elif obj == 'cable':
        #     prompt_state = [good, bent_wire, cable_swap, combined, cut_inner_insulation, cut_outer_insulation, missing_cable, missing_wire, poke_insulation]
        # elif obj == 'capsule':
        #     prompt_state = [good, crack, faulty_imprint, poke, scratch, squeeze]
        # elif obj == 'carpet':
        #     prompt_state = [good, color, cut, hole, metal_contamination, thread]
        # elif obj == 'grid':
        #     prompt_state = [good, bent, broken, glue, metal_contamination, thread]
        # elif obj == 'hazelnut':
        #     prompt_state = [good, crack, cut, hole, print_]
        # elif obj == 'leather':
        #     prompt_state = [good, color, cut, fold, glue, poke]
        # elif obj == 'metal_nut':
        #     prompt_state = [good, bent, color, flip, scratch]
        # elif obj == 'pill':
        #     prompt_state = [good, color, combined, crack, faulty_imprint, pill_type, scratch]
        # elif obj == 'screw':
        #     prompt_state = [good, manipulated_front, scratch_head, scratch_neck, thread_side, thread_top]
        # elif obj == 'tile':
        #     prompt_state = [good, crack, glue_strip, gray_stroke, oil, rough]
        # elif obj == 'toothbrush':
        #     prompt_state = [good, defective]
        # elif obj == 'transistor':
        #     prompt_state = [good, bent_lead, cut_lead, damaged_case, misplaced]
        # elif obj == 'wood':
        #     prompt_state = [good, color, combined, hole, liquid, scratch]
        # elif obj == 'zipper':
        #     prompt_state = [good, broken_teeth, combined, fabric_border, fabric_interior, rough, split_teeth, squeeze_teeth]
        

        for i in range(len(prompt_state_all)):
            prompted_state = [state.format(obj) for state in prompt_state_all[i]]
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