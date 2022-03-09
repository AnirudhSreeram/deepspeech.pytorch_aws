import random
import torch
from math import exp
from torch.fft import rfft, irfft
import numpy as np
from scipy.signal import find_peaks
from scipy.signal.windows import hamming
#from audio import loadAudio, saveAudio
from sys import exit
from pdb import set_trace as bp
import librosa
import time

EPS = 1e-9

def get_original_freq(normalized_freq, Fs):
    return torch.DoubleTensor(normalized_freq * Fs)

def get_normalized_freq(original_freq, Fs):
    return original_freq/Fs

def hz2bin(original_freq, fft_len, Fs):
    normalzed_freq = get_normalized_freq(original_freq, Fs)
    bin = np.round(normalzed_freq*fft_len)
    return int( max(min(bin, Fs/2),0) )

def bin2hz(bin, fft_len, Fs):
    return bin/fft_len * Fs

def bark2Hz(z):
    if z < 2: z = (z-.3)/85
    elif z > 20.1: z = (z+4.422)/1.22
    f = 1960*(z+.53)/(26.28-z)
    return f

def hz2bark(f):
    z = (26.81*f)/(1960+f) - .53
    if z < 2: z = z + .15*(2-z)
    elif z > 20.1: z = z + .22*(z-20.1)
    return z

def bin2bark(bin, fft_len, Fs):
    return hz2bark(bin2hz(bin, fft_len, Fs))

def PSD(x, fft_len): #Power spectral density
    if type(x) == np.ndarray:
        x = torch.Tensor(x)
    X = rfft(x, fft_len)
    p = 10*torch.log10( torch.abs(1/fft_len * X)**2 + EPS ) 
    return 96 - max(p) + p

def fft(x, fft_len):
    if type(x) == np.ndarray:
        x = torch.from_numpy(x)
    X = rfft(x, fft_len) #.numpy()
    return X

def ifft(X, fft_len):
    if type(X) == np.ndarray:
        X = torch.from_numpy(X)
    x = irfft(X, fft_len)
    return torch.abs(x)

def ATH(f): #Threshold in quite
    if f == 0:
        return 100
    return 3.64 * (f/1000)**(-.8) - 6.5 * exp(-.6 * (f/1000-3.3)**2 ) + 10e-3 * (f/1000)**4

def get_threshold_in_quiet(F):
    return np.array([ATH(f) for f in F])

def get_freq_vector(fft_len, Fs):
    return np.array([bin2hz(f,fft_len,Fs) for f in range(fft_len//2 + 1)])

def find_maskers(x, fft_len = 512, Fs = 16000):
    F = get_freq_vector(fft_len, Fs) #Map to original freqeuncy
    P = PSD(x, fft_len) #Compute power spectral density
    threshold_in_quiet = get_threshold_in_quiet(F)
#    threshold_in_quiet = torch.DoubleTensor(threshold_in_quiet)
    canidate_maskers, _ = find_peaks(P.cpu().numpy(), height=threshold_in_quiet)
    maskers = []
    masker_PSD = []
    for bin in canidate_maskers:
        high_freq = min( bark2Hz( hz2bark(F[bin]) + .5 ), int(Fs/2))
        low_freq = max(bark2Hz( hz2bark(F[bin]) - .5 ), 0)
        high_bin = hz2bin(high_freq, fft_len, Fs)
        low_bin = hz2bin(low_freq, fft_len, Fs)
        if P[bin] == max(P[ low_bin : high_bin+1]):
            masker_PSD.append(smooth_masker_PSD(P,bin))
            maskers.append(bin)
        
    return torch.DoubleTensor(np.array(maskers)), P[maskers]

def smooth_masker_PSD(P, bin):
    if bin == len(P) - 1:
        return 10*torch.log10( 10**(P[bin-1]/10) + 10**(P[bin]/10) + EPS )
    elif bin == 0:
        return 10*torch.log10( 10**(P[bin]/10) + 10**(P[bin+1]/10) + EPS )
    else:
        return 10*torch.log10( 10**(P[bin-1]/10) + 10**(P[bin]/10) + 10**(P[bin+1]/10) + EPS )

def SF(bi, bj, pi): #Two-slope spread function
    delta = bj - bi
    if delta < 0:
        return 27 * delta
    else:
        return (-27 + .37*max(pi - 40, 0)) * delta

def get_masking_threshold(x, fft_len = 512, Fs = 1600):
    
    maskers, P = find_maskers(x, fft_len, Fs)
    T = np.zeros( shape=(len(maskers), fft_len//2+1) ) #Masking theshold per masker/per fft bin
    F = get_freq_vector(fft_len, Fs)

    def masking_function_single(bi, bj, pi):
        delta = -6.025 - 0.275*bi 
        return pi + delta + SF(bi, bj, pi)

    for idx, bin, pi in zip(range(len(maskers)), maskers, P):
        bi = bin2bark(bin, fft_len, Fs).cpu().numpy()
        Bj = np.array([bin2bark(bin, fft_len, Fs) for bin in range(1,fft_len//2+2)])
        T[idx,:] = np.vectorize( lambda bj: masking_function_single(bi, bj, pi.cpu().numpy()) )(Bj)
    T = torch.DoubleTensor(T)
    ath = get_threshold_in_quiet(F)
    ath = torch.DoubleTensor(ath)
    global_masking_threshold = 10*torch.log10( 10 ** (ath/10) + torch.sum( 10 ** (T/10), axis=0 ) )

    return global_masking_threshold

def masking_threshold_to_filter(masking_thr, smoothing = 10):
    filter = np.sqrt(10 ** ((-masking_thr - 96)/10))
    filter = np.convolve(filter, np.ones(smoothing), 'same') / smoothing
    filter = filter/np.max(filter) 
    return torch.DoubleTensor(filter)

def masked_noise(x, filter_amount = .8 ,fft_len = 512, Fs = 16000, win_length = 256, hop_length = 32):
    x_pad = np.concatenate((x,np.zeros(win_length)))
    x_pad = torch.DoubleTensor(x_pad).cuda()
    n = np.random.randn(len(x_pad))
    i = 0
    n_filtered = np.zeros_like(n)
    n = torch.DoubleTensor(n).cuda()
    n_filtered = torch.DoubleTensor(n_filtered).cuda()
    w = hamming(win_length)
    w = torch.DoubleTensor(w).cuda()
    while i + win_length <= len(x):
        x_i = x_pad[i:i+win_length]*w
        n_i = n[i:i+win_length]*w
        masking_thr = get_masking_threshold(x_i, fft_len, Fs)
        filter = masking_threshold_to_filter(masking_thr)
        N_i = fft(n_i, fft_len).cuda()
        #N_i = torch.from_numpy(N_i)
        N_i_shaped = filter_amount * (N_i.cpu() * filter) + (1-filter_amount) * N_i.cpu()
        n_i_shaped = ifft(N_i_shaped.cpu(), fft_len)
        n_filtered [i:i+win_length] += n_i_shaped[:win_length].cuda() * w
        i += hop_length

    n_filtered = n_filtered[:len(x)] #Tuncate to correct length

    return n_filtered
    

if __name__ == '__main__':
        fn_mp3 = "/data/asreeram/deepspeech.pytorch/312_benign.wav"

        start_time = time.time()
        x_full, Fs = librosa.load(fn_mp3, sr=None)
        print(x_full.shape,Fs)
#    x_full, Fs = loadAudio("/data/asreeram/deepspeech.pytorch/312_benign.wav")
        n_masked = masked_noise(x_full)
        print("--- %s seconds ---" % (time.time() - start_time))



