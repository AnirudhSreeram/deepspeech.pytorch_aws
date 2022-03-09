import argparse
import torchaudio 
import soundfile as sf
from scipy.io.wavfile import write
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', default='input.wav', help='The input audio to inject noise into')
parser.add_argument('--noise-path', default='noise.wav', help='The noise file to mix in')
parser.add_argument('--output-path', default='output.wav', help='The noise file to mix in')
parser.add_argument('--sample-rate', default=16000, help='Sample rate to save output as')
args = parser.parse_args()

data, Fs = torchaudio.load(args.input_path)
l= data.shape[1]
noise = np.random.normal(0,1,l)
noisy = data + noise

write(filename=args.output_path,
      data=noisy.numpy(),
      rate=args.sample_rate)

