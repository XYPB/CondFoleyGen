import os
from glob import glob
import shutil

gen_videos = glob('logs/CondAVTransformer_VNet_randshift_2s_GH_vqgan_no_earlystop_multiple_2s_tolerance/*.mp4')

for i in range(3):
    cond_video = glob(f'logs/CondAVTransformer_VNet_randshift_2s_GH_vqgan_no_earlystop_multiple_2s_tolerance/2sec_full_cond_video_{i}/*.mp4')
    cond_names = [v.split('/')[-1] for v in cond_video]
    dest_dir = f'logs/CondAVTransformer_VNet_randshift_2s_GH_vqgan_no_earlystop_multiple_2s_tolerance/2sec_full_generated_video_{i}'
    for v in gen_videos:
        if v.split('/')[-1] in cond_names:
            shutil.move(v, dest_dir)