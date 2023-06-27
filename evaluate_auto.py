import json
import os
import argparse

import numpy as np

match_list = json.load(open('data/AMT_test/match_list.json', 'r'))
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, choices=['action', 'material'])
parser.add_argument('--match', action='store_true')
parser.add_argument('--mismatch', action='store_true')

def eval_auto(args):
    pred_dir = f'data/AMT_test/{args.task}_pred'
    pred_list = [os.path.join(pred_dir, f) for f in os.listdir(pred_dir)]

    if args.task == 'action':
        # compare action score with target video
        gt_pred = json.load(open(f'data/AMT_test/{args.task}_pred/target_sound_normalize_melspec_{args.task}_preds.json', 'r'))
    else:
        # compare material score with conditional video
        gt_pred = json.load(open(f'data/AMT_test/{args.task}_pred/condition_normalize_melspec_{args.task}_preds.json', 'r'))
    
    if args.match:
        gt_pred = [p for i, p in enumerate(gt_pred) if match_list[i]]
    elif args.mismatch:
        gt_pred = [p for i, p in enumerate(gt_pred) if not match_list[i]]

    gt_pred = np.array(gt_pred)
    for pred_file in pred_list:
        if 'condition' in pred_file or 'target' in pred_file:
            # ignore GT videos
            continue
        pred = json.load(open(pred_file, 'r'))
        if args.match:
            pred = [p for i, p in enumerate(pred) if match_list[i]]
        elif args.mismatch:
            pred = [p for i, p in enumerate(pred) if not match_list[i]]
        
        pred = np.array(pred)
        acc = np.sum(pred == gt_pred) / len(pred)

        # reduce experiment name:
        exp_name = pred_file.split('/')[-1].replace('CondAVTransformer_VNet_randshift_2s_', '').replace(f'_GH_vqgan_normalize_melspec_{args.task}_preds.json', '').replace(f'_GH_vqgan_no_earlystop_normalize_melspec_{args.task}_preds.json', '').replace(f'_normalize_melspec_{args.task}_preds.json', '')

        print(exp_name, f'{acc:.4f}')


if __name__ == '__main__':
    args = parser.parse_args()
    eval_auto(args)

