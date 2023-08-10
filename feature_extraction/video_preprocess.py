import argparse
import os
from glob import glob
from multiprocessing import Pool
from functools import partial
import soundfile as sf
import noisereduce as nr

def execCmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text

def pipeline(video_path, output_dir, fps='15', sr='22050', W='320', H='240', denoise=False):
    video_name = os.path.basename(video_path)
    audio_name = video_name.replace(".mp4", ".wav")

    audio_output_dir = os.path.join(output_dir, video_name.replace('.mp4', ''), "audio")
    frame_output_dir = os.path.join(output_dir, video_name.replace('.mp4', ''), "frames")

    os.makedirs(audio_output_dir, exist_ok=True)
    os.makedirs(frame_output_dir, exist_ok=True)

    # Audio extraction
    os.system(f"ffmpeg -i {video_path} -loglevel error -f wav -vn -y {os.path.join(audio_output_dir, audio_name)}")
    os.system(f"ffmpeg -i {os.path.join(audio_output_dir, audio_name)} -loglevel error -ac 1 -ab 16k -ar {sr} -y {os.path.join(audio_output_dir, audio_name.replace('.wav', '_resampled.wav'))}")

    # Audio denoise
    if denoise:
        input_path = os.path.join(audio_output_dir, audio_name.replace('.wav', '_resampled.wav'))
        dest = os.path.join(audio_output_dir, audio_name.replace('.wav', '_resampled_denoised.wav'))
        wav, sr = sf.read(input_path)
        if len(wav.shape) == 1:
            wav = wav[None, :]
        wav_clean = nr.reduce_noise(y=wav, sr=sr, n_fft=1024, hop_length=1024//4)
        wav_clean = wav_clean.squeeze()
        sf.write(dest, wav_clean, samplerate=sr)

    frame_dest = os.path.join(frame_output_dir, 'frame%06d.jpg')
    os.system(f'ffmpeg -i {video_path} -loglevel error -filter:v fps=fps={fps} -y {frame_dest}')
    frame_dest = os.path.join(frame_output_dir, 'frame%06d_resize.jpg')
    os.system(f'ffmpeg -i {video_path} -loglevel error -filter:v fps=fps={fps},scale={W}:{H} -y {frame_dest}')

if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument("-i", "--input_dir", default="data/ImpactSet/RawVideos/CountixAV_train")
    paser.add_argument("-o", "--output_dir", default="/datad/duyxxd/impactset-proccess-resize")
    paser.add_argument("-a", '--audio_sample_rate', default='22050')
    paser.add_argument("-v", '--video_fps', default='15')
    paser.add_argument("-w", '--video_width', default='320')
    paser.add_argument('--video_height', default='240')
    paser.add_argument("-n", '--num_worker', type=int, default=8)
    paser.add_argument("--denoise", action='store_true', default=False)
    args = paser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    sr = args.audio_sample_rate
    fps = args.video_fps
    W = args.video_width
    H = args.video_height
    denoise = args.denoise

    video_paths = glob(os.path.join(input_dir, "*/*.mp4"))
    video_paths.sort()

    with Pool(args.num_worker) as p:
        p.map(partial(pipeline, output_dir=output_dir, sr=sr, fps=fps, W=W, H=H, denosie=denoise), video_paths)

