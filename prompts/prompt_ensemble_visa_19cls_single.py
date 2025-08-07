# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np

def encode_text_with_prompt_ensemble(model, objs, tokenizer, device):
    # normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    # damage = ['{} has a damaged defect', 'flawed {} with damage']
    # scratch = ['{} has a scratch defect', 'flawed  {} with a scratch']
    # breakage = ['{} with a breakage defect', 'broken {}', '{} with broken defect']
    # burnt = ['{} with a burnt defect']
    # weird_wick = ['{} with a weird wick defect']
    # stuck = ['{} with a stuck defect', '{} stuck together']
    # crack = ['{} with a crack defect']
    # wrong_place = ['{} with defect that something on wrong place', '{} has a misplaced defect', 'flawed {} with misplacing']
    # partical = ['{} with particals defect']
    # bubble = ['{} with bubbles defect']
    # melded = ['{} with melded defect']
    # hole = ['{} has a hole defect', 'a hole on {}']
    # melt = ['{} with melt defect']
    # bent = ['{} has a bent defect', 'flawed {} with a bent']
    # spot = ['{} with spot defect']
    # extra = ['{} with extra thing', '{} has a defect with extra thing']
    # chip = ['{} with chip defect', '{} with fragment broken defect'] 
    # missing = ['{} with a missing defect', 'flawed {} with something missing']

    normal = [
        '{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', 
        '{} without defect', '{} without damage', '{} in pristine condition', 
        '{} with no imperfections', '{} with ideal quality'
    ]

    damage = [
        '{} has a damaged defect'
        
    ]

    scratch = [
        '{} has a scratch defect'
    ]

    breakage = [
        '{} with a breakage defect'
    ]

    burnt = [
        '{} with a burnt defect'
    ]

    weird_wick = [
        '{} with a weird wick defect'
    ]

    stuck = [
        '{} with a stuck defect'
    ]

    crack = [
        '{} with a crack defect'
    ]

    wrong_place = [
        '{} with defect that something on wrong place'
    ]

    partical = [
        '{} with particles defect'
    ]

    bubble = [
        '{} with bubbles defect'
    ]

    melded = [
        '{} with melded defect'
    ]

    hole = [
        '{} has a hole defect'
    ]

    melt = [
        '{} with melt defect'
    ]

    bent = [
        '{} has a bent defect'
    ]

    spot = [
        '{} with spot defect'
    ]

    extra = [
        '{} with extra thing'
    ]

    chip = [
        '{} with chip defect'
    ]

    missing = [
        '{} with a missing defect'
    ]

    prompt_state = [normal, damage, scratch, breakage, burnt, weird_wick, stuck, crack, wrong_place, partical, bubble, melded, hole, melt, bent, spot, extra, chip, missing]
    
    # prompt_state = [good, bent, bent_and_melt, breakage_down_the_middle, breakage_color_chip, breakage_color_chip_crack, breakage_color_chip_crack_scratch, breakage_scratch, bubble, bubble_discolor, bubble_discolor_scratch, bubble_discolor_scratch_leak, bubble_discolor_scratch_leak_misshape, burnt, burnt_corner, burnt_color, burnt_dirt, burnt_extra, burnt_samecolor, burnt_scratch_dirt, burnt_scratch, chip_around_edge_and_corner, chip_around_edge_and_corner_cracks, chip_around_edge_and_corner_scratch, chunk_of_gum_missing_corner, chunk_of_gum_missing_corner_crack, chunk_of_wax_missing, chunk_of_wax_missing_damaged_color, chunk_of_wax_missing_color, chunk_of_wax_missing_particals, color_spot_similar_to_the_object, corner_and_edge_breakage, corner_missing, corner_missing_scratch, corner_missing_similar_color, corner_missing_similar_color_cracks, corner_missing_cracks, corner_or_edge_breakage, corner_and_edge_breakage_scratch, damaged_corner_of_packaging, damaged_corner_of_packaging_color, damaged_corner_of_packaging_color_other, damaged_corner_of_packaging_extra_wax, damaged_corner_of_packaging_particals, damaged_corner_of_packaging_weird_candle, different_colour_spot, different_color_spot_particals, burnt_colour, chunk_of_wax_missing_damaged_colour, chunk_of_wax_missing_colour, damaged_corner_of_packaging_colour, damaged_corner_of_packaging_colour_other, colour_colour, colour_similar_colour, colour_holes, colour_wax, weird_colour, weird_color, melt, melt_missing, melt_scratch, melt_scratch_missing, scratch, scratch_damage, scratch_damage_dirt, scratch_damage_extra, scratch_dirt, scratch_extra, scratch_extra_dirt, scratch_extra_wrong, scratch_missing, scratch_missing_extra, different_color_spot, color_particals, color_color, color_similar_color, color_holes, color_wax, middle_breakage, middle_breakage_color, middle_breakage_simialr_color, middle_breakage_cracks, middle_breakage_holes, middle_breakage_scratch, middle_breakage_scratch_crack, missing, missing_damage, missing_damage_extra, missing_dirt, missing_wrong, scratches, scratches_color, chunk_of_gum_missing_scratches, chunk_of_gum_missing_scratches_cracks, scratches_color_cracks, weird_candle_wick, damage, damage_dirt, damage_extra, fryum_stuck_together, similar_color_spot, similar_color_spot_scratches, scratches_colour, scratches_colour_cracks, similar_colour_spot, similar_colour_other, similar_colour_scratches, corner_missing_colour, corner_missing_colour_cracks, middle_breakage_colour, small_chip_around_edge, small_chip_around_edge_scratches, extra_wax_in_candle, extra, extra_dirt, chunk_of_gum_missing_small_cracks, small_cracks, small_cracks_scratches, small_holes, foreign_particals_on_the_candle, foreign_particals_on_the_candle_wax, other, small_scratches, wrong_place_dirt, stuck_together, wax_melded_out_of_the_candle, same_color_spot, same_colour_spot, burnt_same_colour, middle_breakage_same_colour, wrong_place, chunk_of_gum_missing, breakage_color_color_chip_cracks_other, scratch_missing_dirt]

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