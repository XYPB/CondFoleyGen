import copy
import csv
import json
import numpy as np
import os
import pickle
import random

import torch
from torch.utils.data.sampler import Sampler

import pdb

class ASMRSampler(Sampler): 
    """
    Total videos: 2794. The sampler ends when last $BATCH_SIZE videos are left. 
    """
    def __init__(self, list_sample, batch_size, rand_per_epoch=True): 
        self.list_sample = list_sample
        self.batch_size = batch_size
        if not rand_per_epoch: 
            random.seed(1234)

        self.N = len(self.list_sample)
        self.sample_class_dict =  self.generate_vid_dict()
        # self.indexes = self.gen_index_batchwise()
        # pdb.set_trace()

    def generate_vid_dict(self): 
        _ = [self.list_sample[i].append(i) for i in range(len(self.list_sample))]
        sample_class_dict = {}
        for i in range(len(self.list_sample)): 
            video_name = self.list_sample[i][0]
            if video_name not in sample_class_dict: 
                sample_class_dict[video_name] = []
            sample_class_dict[video_name].append(self.list_sample[i])
        
        return sample_class_dict

    def gen_index_batchwise(self): 
        indexes = []
        scd_copy = copy.deepcopy(self.sample_class_dict)
        for i in range(self.N // self.batch_size): 
            if len(list(scd_copy.keys())) <= self.batch_size: 
                break 
            batch_vid = random.sample(scd_copy.keys(), self.batch_size)
            for vid in batch_vid: 
                rand_clip = random.choice(scd_copy[vid])
                indexes.append(rand_clip[-1])
                scd_copy[vid].remove(rand_clip)   # removed added element
                # remove dict if empty
                if len(scd_copy[vid]) == 0: 
                    del scd_copy[vid]
        
        # add remain items to indexes
        # for k, v in scd_copy.items(): 
        #     for item in v: 
        #         indexes.append(item[-1])
        return indexes
            
    def __iter__(self): 
        return iter(self.gen_index_batchwise())

    def __len__(self): 
        return self.N


class VoxcelebSampler(Sampler): 
    def __init__(self, list_sample, batch_size, rand_per_epoch=True): 
        self.list_sample = list_sample
        self.batch_size = batch_size
        if not rand_per_epoch: 
            random.seed(1234)
        
        self.N = len(self.list_sample)
        self.sample_class_dict = self.generate_vid_dict()

    def generate_vid_dict(self): 
        _ = [self.sample[i].append(i) for i in range(len(self.list_sample))]
        sample_class_dict = {}
        pdb.set_trace()
        for i in range(len(self.list_sample)): 
            video_name = self.list_sample[i][0]
            if video_name in batch_vid: 
                pdb.set_trace()