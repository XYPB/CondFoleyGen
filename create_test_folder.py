import os
import shutil
import json
import sys
from multiprocessing import Process
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from feature_extraction.demo_utils import extract_melspectrogram

# TEST_FOLDER = 'data/AMT_test/'
TEST_FOLDER = 'data/AMT_test_8s/'
os.makedirs(TEST_FOLDER, exist_ok=True)
match_dict = json.load(open('data/AMT_test_set_match_dict.json', 'r'))


def extract_spec_to_dest(src, st, NT=8, duration=2):
    dest_dir = src + '_normalize_melspec'
    os.makedirs(dest_dir, exist_ok=True)
    v_list = [os.path.join(src, f'{i}.mp4') for i in range(st, 582, NT)]
    for in_path in v_list:
        dest = os.path.join(dest_dir, in_path.split('/')[-1].replace('mp4', 'npy'))
        spec = extract_melspectrogram(in_path, 22050, duration=duration, normalize=True)
        assert spec.shape[0] == 80
        np.save(dest, spec)


def create_spec_dir(src):
    NT = 8
    ps = []
    duration = float(src.split('_')[-1].replace('s', ''))
    for st in range(NT):
        # p = Process(target=extract_spec_to_dest, args=(src, st))
        extract_spec_to_dest(src, st, duration=duration)
        # p.start()
        # ps.append(p)
    # [p.join() for p in ps]


def fname_to_match(fname):
    fname = fname.split('/')[-1]
    target_name = fname.split('_')[0]
    target_time = fname.split('_')[6]
    cond_name = fname.split('_')[11]
    cond_time = fname.split('_')[17]
    target = f'{target_name}_{target_time}'
    cond = f'{cond_name}_{cond_time}'
    return match_dict[target][cond]


def create_gen_key2orig_key(sorted_videos, sorted_orig):
    sorted_orig_short = {vname.split('/')[-1].split('_')[0] + '_' + vname.split('/')[-1].split('_')[6]: i for i, vname in enumerate(sorted_orig)}
    key_map = {}
    for i, gen_name in enumerate(sorted_videos):
        tar_id = gen_name.split('/')[-1].split('_')[0] + '_' + gen_name.split('/')[-1].split('_')[6]
        assert tar_id in list(sorted_orig_short.keys())
        key_map[i] = sorted_orig_short[tar_id]
    return key_map


if __name__ == '__main__':
    input_folder = sys.argv[1]
    if TEST_FOLDER in input_folder:
        create_spec_dir(input_folder)
        exit()
    experiment_name = input_folder.split('/')[-1]

    new_video_folder = os.path.join(TEST_FOLDER, experiment_name)
    os.makedirs(new_video_folder, exist_ok=True)
    videos = []
    orig_video = []
    cond_video = []

    for sub_folder in os.listdir(input_folder):
        sub_input = os.path.join(input_folder, sub_folder)
        if 'generate' in sub_folder and 'sound' not in sub_folder:
            videos += [os.path.join(sub_input, f) for f in os.listdir(sub_input) if '.mp4' in f]
        elif 'orig_video' in sub_folder:
            orig_video += [os.path.join(sub_input, f) for f in os.listdir(sub_input) if '.mp4' in f]
        elif 'cond_video' in sub_folder:
            cond_video += [os.path.join(sub_input, f) for f in os.listdir(sub_input) if '.mp4' in f]

    videos = sorted(videos)
    match_list = []
    type_dict = json.load(open('data/AMT_test_set_type_dict.json'))
    for v in videos:
        match_list.append(fname_to_match(v))
    json.dump(match_list, open(os.path.join(TEST_FOLDER, 'match_list.json'), 'w'))

    orig_video = sorted(orig_video)
    cond_video = sorted(cond_video)
    for i, v in tqdm(enumerate(videos)):
        dest = os.path.join(new_video_folder, f'{i}.mp4')
        shutil.copy(v, dest)
    key_map = create_gen_key2orig_key(videos, orig_video)
    json.dump(key_map, open(os.path.join(TEST_FOLDER, 'key_map.json'), 'w'))

    orig_dest = os.path.join(TEST_FOLDER, 'target_sound')
    cond_dest = os.path.join(TEST_FOLDER, 'condition')
    if not os.path.exists(orig_dest):
        os.makedirs(orig_dest)
        for i, v in tqdm(enumerate(orig_video)):
            dest = os.path.join(orig_dest, f'{i}.mp4')
            # clip = VideoFileClip(v).without_audio()
            # clip.write_videofile(dest)
            shutil.copy(v, dest)
    if not os.path.exists(cond_dest):
        os.makedirs(cond_dest)
        for i, v in tqdm(enumerate(cond_video)):
            dest = os.path.join(cond_dest, f'{i}.mp4')
            shutil.copy(v, dest)





