  
from pathlib import Path
from tqdm import tqdm
import argparse
import librosa
import torch
import torchaudio
from pdb import set_trace as bp
import soundfile as sf
import os
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch.nn as nn
import soundfile as sf
from tqdm import tqdm
import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn
import librosa
import numpy as np
import sox
import torch
from torch.utils.data import Dataset, Sampler, DistributedSampler, DataLoader
from pdb import set_trace as bp
import torchaudio
#from freq_masking import fft, masked_noise
#from multiprocessing import Process, Pool
import torch.nn.functional as F
from parallel_wavegan.utils import download_pretrained_model, load_model




class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=16000,
        n_mel_channels=80,
        mel_fmin=80.0,
        mel_fmax=7600,
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "constant").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec




def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--load_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.save_path.mkdir(exist_ok=True, parents=True)

    f = open( args.folder , "r" )
    lines = f.readlines()

#    for i, fname in tqdm(enumerate(args.folder.glob("*.wav"))):
    for i, fname in tqdm(enumerate(lines)):
        #if i < 23000:
        #    continue
#        wavname = fname.name
        wavname = fname.split("\n")[0]
#        path = os.path.join(args.folder, wavname)
        path = os.path.join("/data/asreeram/deepspeech.pytorch/data/LibriSpeech_dataset/train_clean/wav/", wavname)
        #sound, sample_rate = torchaudio.load(path)
        sound, samplerate = sf.read(path,dtype='float32')
        sound = torch.from_numpy(sound)
        #waveGAN_path = download_pretrained_model("arctic_slt_parallel_wavegan.v1", "waveGAN")
        waveGAN_path = "/data/asreeram/deepspeech.pytorch/deepspeech_pytorch/loader/ParallelWaveGAN/waveGAN/arctic_slt_parallel_wavegan.v1/checkpoint-400000steps.pkl"
        stats_file = "/data/asreeram/deepspeech.pytorch/deepspeech_pytorch/loader/ParallelWaveGAN/waveGAN/arctic_slt_parallel_wavegan.v1/stats.h5"
        mel = Audio2Mel()
        waveGAN = load_model(waveGAN_path, stats=stats_file).cuda()
        x_mel = mel(sound.unsqueeze(0)).squeeze(0).T
        sound = waveGAN.inference(x_mel.cuda(), normalize_before=True).T.view(-1)
        sound = sound.detach().cpu().numpy()
        sf.write(args.save_path / wavname, sound, 16000)


if __name__ == "__main__":
    main()
