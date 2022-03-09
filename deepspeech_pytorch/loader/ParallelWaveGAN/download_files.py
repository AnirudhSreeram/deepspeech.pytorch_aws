# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
from parallel_wavegan.utils import download_pretrained_model, load_model
from parallel_wavegan.bin.preprocess import logmelfilterbank
import torch
import numpy as np
import pdb
from tqdm.auto import tqdm
import gc
from pdb import set_trace as bp
import torch.nn.functional as F
import soundfile as sf
import torchaudio
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from parallel_wavegan.utils import read_hdf5
from parallel_wavegan.utils import write_hdf5
from librosa.filters import mel as librosa_mel_fn


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
        #au = nn.ReflectionPad2d(p)
        #audio = au(audio)
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


class WaveGANTorch(PreprocessorPyTorch):

    """Pytorch Preprocessor Template"""

    def __init__(
        self,
        channels_first: bool = False,
        apply_fit: bool = False,
        apply_predict: bool = True,
        verbose: bool = False,
        
    ) -> None:

        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        cuda_idx = torch.cuda.current_device()                                       
        self._device = torch.device("cuda:{}".format(cuda_idx))
        
        self.channels_first = channels_first
        self.verbose = verbose

        #waveGAN_path = download_pretrained_model("ljspeech_parallel_wavegan.v1", "waveGAN")
        waveGAN_path = download_pretrained_model(
            "arctic_slt_parallel_wavegan.v1", "waveGAN")
        stats_file = "/home1/asreeram/waveGAN/waveGAN/arctic_slt_parallel_wavegan.v1/stats.h5"
        self.waveGAN = load_model(
            waveGAN_path, stats=stats_file).to(self._device)
        #self.waveGAN.eval()
        #self.waveGAN.train()
    

    def __call__(self, x, y = None):
        
        try:
            x_out = x.copy()
        except:
            x_out = x.detach().cpu().numpy().copy()

        for i, x_i in enumerate(tqdm(x, disable=not self.verbose)):
            x_i = torch.tensor(x_i, dtype=torch.float32) #.to(self._device)  
            x_i, _ = self.forward(x_i)
            x_i = x_i.detach().cpu().numpy()
            x_out[i] = x_i
        return x_out, y  

    def forward(self, x, y = None):
        T = x.size()[-1]
        mel = Audio2Mel()
        x_mel = mel(x.unsqueeze(0).cpu()).squeeze(0).T.cuda()
        #x = x.detach().cpu().numpy()
        #x_mel = logmelfilterbank(x, sampling_rate=16000)
        #x_mel = torch.tensor(x_mel).to(self._device)
        x = self.waveGAN.inference(x_mel, normalize_before=True).view(-1)
        return x, y
    #     if x.ndim == 1:
    #         x = x.unsqueeze(0)
    #     T = x.size()[-1]
    #     x = x.detach().cpu().numpy()
    #     x_mel = torch.tensor(logmelfilterbank(
    #         x.squeeze(0), sampling_rate=16000)).T.unsqueeze(0).to(self._device)
    #     n = torch.randn((1, 1, T - 832)).to(self._device)
    #     bp()
    #     x = self.waveGAN(n, x_mel) 
    #     x = downsample(x) #.detach().cpu().numpy()
    #     bp()
    #     sf.write('/home1/asreeram/waveGAN/ben.wav',
    #              x.detach().cpu().numpy().squeeze(0).T, 16000)

    #     if x.ndim == 1:
    #      x = x.unsqueeze(0)
    #     x = downsample(x) #.detach().cpu().numpy()
    #     sf.write('/home1/asreeram/waveGAN/ben.wav',x.detach().cpu().numpy().squeeze(0).T, 16000)
    

if __name__ == "__main__":
    vocoder = WaveGANTorch()



