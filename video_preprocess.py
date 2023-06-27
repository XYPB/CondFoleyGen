import subprocess
import soundfile as sf
import noisereduce as nr
from glob import glob
import os
from tqdm import tqdm
from multiprocessing import Pool
import sys

def pipeline_resize(v):
    dest = v.replace('.mp4', '_resize.mp4')
    if os.path.exists(dest) or '_resize.mp4' in v:
        return
    cmd = f'ffmpeg -v quiet -i {v} -vf scale=640:360 -y {dest}'
    print(cmd)
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    result = result.stdout.decode('utf-8')
    return dest

def pipeline_reencode_mov(v):
    dest = v.replace('.mov', '.mp4')
    if os.path.exists(dest) or '.mp4' in v:
        return
    cmd = f'ffmpeg -v quiet -i {v} -vcodec h264 -acodec aac -y {dest}'
    print(cmd)
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    result = result.stdout.decode('utf-8')
    return dest

def pipeline_extract_audio(v):
    dest = v.replace('.mp4', '.wav')
    if os.path.exists(dest) or '.wav' in v:
        return
    cmd = f'ffmpeg -v quiet -i {v} -f wav -vn -ac 1 -ab 16k -ar 22050 -y {dest}'
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    result = result.stdout.decode('utf-8')
    return dest

def pipeline_resample_video(v):
    dest = v.replace('.mp4', '_15fps.mp4')
    if os.path.exists(dest) or '_15fps.mp4' in v:
        return
    cmd = f'ffmpeg -v quiet -i {v} -filter:v fps=fps=15 -y {dest}'
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    result = result.stdout.decode('utf-8')
    return dest

def pipeline_denoise_audio(a):
    dest = a.replace('.wav', '_denoised.wav')
    if os.path.exists(dest) or '_denoised.wav' in a:
        return
    wav, sr = sf.read(a)
    if len(wav.shape) == 1:
        wav = wav[None, :]
    wav_clean = nr.reduce_noise(y=wav, sr=sr, n_fft=1024, hop_length=1024//4)
    wav_clean = wav_clean.squeeze()
    sf.write(dest, wav_clean, samplerate=sr)
    return dest


def video_pre_process(v):
    if '.mov' in v:
        v = pipeline_reencode_mov(v)
    v = pipeline_resize(v)
    a = pipeline_extract_audio(v)
    v = pipeline_resample_video(v)
    a = pipeline_denoise_audio(a)
    return v, a



if __name__ == '__main__':
    # Define video list for training
    video_path = sys.argv[1]
    video_list = glob(os.path.join(video_path, '*/*.mp4')) + glob(os.path.join(video_path, '*.mp4'))
    print(len(video_list))
    with Pool(8) as p:
        p.map(video_pre_process, video_list)
