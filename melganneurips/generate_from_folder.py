from mel2wav import MelVocoder

from pathlib import Path
from tqdm import tqdm
import argparse
import librosa
import torch
import torchaudio
from pdb import set_trace as bp
import soundfile as sf
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    vocoder = MelVocoder(args.load_path,github=True)

    args.save_path.mkdir(exist_ok=True, parents=True)
    upsample = torchaudio.transforms.Resample(16000,22050)
    downsample = torchaudio.transforms.Resample(22050,16000)

    for i, fname in tqdm(enumerate(args.folder.glob("*.wav"))):
        wavname = fname.name
        path = os.path.join(args.folder, wavname)
        wav, sample_rate = torchaudio.load(path)
        wav = upsample(wav)
        mel = vocoder(wav)
        recons = vocoder.inverse(mel)
        recons = downsample(recons.cpu()).squeeze().numpy()
        sf.write(args.save_path / wavname, recons, 16000)


if __name__ == "__main__":
    main()
