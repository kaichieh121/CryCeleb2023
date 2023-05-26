from pathlib import Path
import shutil
import argparse
from tqdm import tqdm
import torch
import torchaudio
import librosa
import soundfile as sf
import numpy as np

def get_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--dataset_path', default='/home/kcchang3/data/CryCeleb2023/audio_dump')
    parser.add_argument('--des_path', default='/home/kcchang3/data/CryCeleb2023')
    parser.add_argument('--seg_len', type=int, default=10)
    parser.add_argument('--mode', default='sample')
    args = parser.parse_args()
    return args

def change_pitch(audio_file_path, output_file_path):
    # load the audio file
    original_audio, sr = librosa.load(audio_file_path)
    # randomly generate a pitch shift factor in the range of 0.9 to 1.1
    pitch_shift_factor = np.random.uniform(0.9, 1.1)
    # calculate the pitch shift in cents
    pitch_shift_cents = 1200 * np.log2(pitch_shift_factor)
    # perform the pitch shift
    pitch_shifted_audio = librosa.effects.pitch_shift(original_audio, sr, pitch_shift_cents)
    # save the pitch shifted audio
    sf.write(output_file_path, pitch_shifted_audio, sr)

def main(args):
    dataset_path = Path(args.dataset_path)
    mode = args.mode
    des_path = Path(args.des_path)
    des_path = des_path / f'audio_dump_{args.seg_len}s_{mode}'
    des_path.mkdir(parents=True, exist_ok=True)
    

    file_list = sorted(dataset_path.glob('*.wav'))
    i = 0
    idx = 0
    audio = None

    if mode == 'default':
        for f in tqdm(file_list):
            waveform, sample_rate = torchaudio.load(f)
            if audio is None:
                audio = waveform
            else:
                audio = torch.cat((audio, waveform), dim=1)
            i += 1
            if i > args.seg_len:
                i = 0
                torchaudio.save(des_path / f'{idx}.wav', audio, sample_rate)
                idx += 1
                audio = None

    elif mode == 'sample':
        for _ in tqdm(range(args.seg_len*30000)):
            f = np.random.choice(file_list)
            waveform, sample_rate = torchaudio.load(f)
            if audio is None:
                audio = waveform
            else:
                audio = torch.cat((audio, waveform), dim=1)
            i += 1
            if i > args.seg_len:
                i = 0
                torchaudio.save(des_path / f'{idx}.wav', audio, sample_rate)
                idx += 1
                audio = None
    
    elif mode == 'augmentation':
        for _ in tqdm(range(args.seg_len*30000)):
            f = np.random.choice(file_list)
            waveform, sample_rate = torchaudio.load(f)
            if audio is None:
                audio = waveform
            else:
                audio = torch.cat((audio, waveform), dim=1)
            i += 1
            if i > args.seg_len:
                i = 0
                torchaudio.save(des_path / f'{idx}.wav', audio, sample_rate)
                change_pitch(des_path / f'{idx}.wav', des_path / f'{idx}_aug.wav')
                idx += 1
                audio = None
    
    else:
        raise Exception(f'Sorry, {mode} not supported')

if __name__ == "__main__":
    args = get_arguments()
    main(args)