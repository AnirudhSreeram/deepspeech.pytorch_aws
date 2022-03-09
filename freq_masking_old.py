import random
import torch
from math import exp
from torch.fft import rfft, irfft
import numpy as np
from scipy.signal import find_peaks
from scipy.signal.windows import hamming
#from audio import loadAudio, saveAudio
import librosa
from sys import exit
from torch.multiprocessing import Pool, Process, set_start_method
from pdb import set_trace as bp

#try:
#     set_start_method('spawn')
#except RuntimeError:
#    pass

EPS = 1e-9

def get_original_freq(normalized_freq, Fs):
    return normalized_freq * Fs

def get_normalized_freq(original_freq, Fs):
    return original_freq/Fs

def hz2bin(original_freq, fft_len, Fs):
    normalzed_freq = get_normalized_freq(original_freq, Fs)
    normalzed_freq = normalzed_freq
    #fft_len = torch.DoubleTensor(fft_len)
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
    X = rfft(x, fft_len)#.numpy()
    #p = 10*np.log10( np.abs(1/fft_len * X)**2 + EPS ) 
    p = 10*torch.log10( torch.abs(1/fft_len * X)**2 + EPS )
    return 96 - max(p) + p

def fft(x, fft_len):
    if type(x) == np.ndarray:
        x = torch.from_numpy(x)
    X = rfft(x, fft_len)#.numpy()
    return X

def ifft(X, fft_len):
    if type(X) == np.ndarray:
        X = torch.from_numpy(X)
    x = irfft(X, fft_len)#.numpy()
    #return np.abs(x)
    return torch.abs(x)

def ATH(f): #Threshold in quite
    if f == 0:
        return 100
    return 3.64 * (f/1000)**(-.8) - 6.5 * exp(-.6 * (f/1000-3.3)**2 ) + 10e-3 * (f/1000)**4

def get_threshold_in_quiet(F):
    #return np.array([ATH(f) for f in F])
    return torch.from_numpy(np.array([ATH(f) for f in F]))

def get_freq_vector(fft_len, Fs):
    #return np.array([bin2hz(f,fft_len,Fs) for f in range(fft_len//2 + 1)])
    return torch.from_numpy(np.array([bin2hz(f,fft_len,Fs) for f in range(fft_len//2 + 1)]))

def find_maskers(x, fft_len = 512, Fs = 16000):
    F = get_freq_vector(fft_len, Fs) #Map to original freqeuncy
    P = PSD(x, fft_len) #Compute power spectral density
    threshold_in_quiet = get_threshold_in_quiet(F)
    canidate_maskers, _ = find_peaks(P.cpu().numpy(), height=threshold_in_quiet.cpu().numpy())
    canidate_maskers = torch.from_numpy(canidate_maskers)
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
    return torch.from_numpy( np.array(maskers)), P[maskers]
    #return torch.from_numpy(np.array(maskers)), torch.from_numpy(P[maskers])

#def smooth_masker_PSD(P, bin):
#    if bin == len(P) - 1:
#        return 10*np.log10( 10**(P[bin-1]/10) + 10**(P[bin]/10) + EPS )
#    elif bin == 0:
#        return 10*np.log10( 10**(P[bin]/10) + 10**(P[bin+1]/10) + EPS )
#    else:
#        return 10*np.log10( 10**(P[bin-1]/10) + 10**(P[bin]/10) + 10**(P[bin+1]/10) + EPS )

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
    T = torch.zeros( shape=(len(maskers), fft_len//2+1) ) #Masking theshold per masker/per fft bin
    F = get_freq_vector(fft_len, Fs)

    def masking_function_single(bi, bj, pi):
        delta = -6.025 - 0.275*bi 
        return pi + delta + SF(bi, bj, pi)

    for idx, bin, pi in zip(range(len(maskers)), maskers, P):
        bi = bin2bark(bin, fft_len, Fs)
        Bj = ([bin2bark(bin, fft_len, Fs) for bin in range(1,fft_len//2+2)])
        T[idx,:] = np.vectorize( lambda bj: masking_function_single(bi, bj, pi) )(Bj)
        T[idx,:] = torch.DoubleTensor(T[idx,:]).cuda()

    ath = get_threshold_in_quiet(F)
    global_masking_threshold = 10*torch.log10( 10 ** (ath/10) + torch.sum( 10 ** (T/10), acis=0 ) + 10**(-10)) ## getting a runtime warning here

    return global_masking_threshold


def masking_threshold_to_filter(masking_thr, smoothing = 10):
    filter = np.sqrt(10 ** ((-masking_thr - 96)/10))
    filter = np.convolve(filter, np.ones(smoothing), 'same') / smoothing
    filter = filter/np.max(filter)
    return torch.DoubleTensor(filter).cuda()

def masked_noise(x, filter_amount = .8 ,fft_len = 512, Fs = 1600, win_length = 256, hop_length = 128):
    bp()
    x_pad = np.concatenate((x,np.zeros(win_length)))
    x_pad = torch.DoubleTensor(x_pad).cuda()
    n = np.random.randn(len(x_pad))
    i = 0
    n_filtered = np.zeros_like(n)
    n_filtered = torch.DoubleTensor(n_filtered).cuda()
    n = torch.DoubleTensor(n).cuda()
    w = hamming(win_length)
    w = torch.DoubleTensor(w).cuda()
    while i + win_length <= len(x):
        x_i = x_pad[i:i+win_length]*w
        n_i = n[i:i+win_length]*w
        bp()
        masking_thr = get_masking_threshold(x_i, fft_len, Fs)
        filter = masking_threshold_to_filter(masking_thr.cpu().numpy())
        N_i = fft(n_i, fft_len)
        torch.Tensor(N_i, dtype=torch.cfloat)
        N_i_shaped = filter_amount * (N_i * filter) + (1-filter_amount) * N_i
        n_i_shaped = ifft(N_i_shaped, fft_len)
        n_filtered [i:i+win_length] += n_i_shaped[:win_length] * w
        i += hop_length

    n_filtered = n_filtered[:len(x)] #Tuncate to correct length

    return n_filtered
    
if __name__ == '__main__':
    fn_mp3 = "/data/asreeram/deepspeech.pytorch/312_benign.wav"
    x_full, Fs = librosa.load(fn_mp3, sr=None)
#    x_full, Fs = loadAudio("/data/asreeram/deepspeech.pytorch/312_benign.wav")
    n_masked = masked_noise(x_full)



    #import matplotlib.pyplot as plt
    #from random import randint
#    torch.multiprocessing.set_start_method('spawn')
    #NOISE_LEVEL = .25
    #WIN_LENGTH = 256
    #HOP_LENGTH = 32
    #x_full, Fs = loadAudio("/Users/nick/Desktop/GARD/GARD_code/AudioSamples/ben.wav")

    #start = randint(0, len(x_full)-WIN_LENGTH)
    #x = x_full[start:start+WIN_LENGTH]
    #fft_len = 2*WIN_LENGTH

    #maskers, _ = find_maskers(x, fft_len, Fs)
    #P = PSD(x, fft_len)
    #F = get_freq_vector(fft_len, Fs)
    #threshold = get_masking_threshold(x, fft_len=fft_len, Fs=Fs)

    #filter = masking_threshold_to_filter(threshold)
    
    #n_masked = masked_noise(x_full, win_length=WIN_LENGTH, hop_length=HOP_LENGTH)
    #n_energy_masked = np.sqrt( n_masked.dot(n_masked)/len(n_masked) )
    #x_energy = np.sqrt( x_full.dot(x_full)/len(x_full) )
    #x_noise_masked = x_full + NOISE_LEVEL * n_masked * x_energy/n_energy_masked
    #saveAudio(.5*x_noise_masked/np.max(x_noise_masked), '/Users/nick/Desktop/GARD/Presentation Materials/masked_noise.wav', Fs)
    
    #n_gaussian = np.random.randn(len(x_full))
    #n_energy_gaussian = np.sqrt( n_gaussian.dot(n_gaussian )/len(n_gaussian ) )
    #x_noise_gaussian = x_full + NOISE_LEVEL * n_gaussian * x_energy/n_energy_gaussian
    #saveAudio(.5*x_noise_gaussian/np.max(x_noise_gaussian), '/Users/nick/Desktop/GARD/Presentation Materials/gaussian_noise.wav', Fs)

    #plt.plot(F, P, label='Power spectral density')
    #plt.plot(F[maskers], P[maskers], 'x', c='r', markersize=8, label='Maskers')
    #plt.plot(F, threshold, label='Masking threshold')
    #plt.legend()
    #plt.xlabel('Frequency (Hz)')
    #plt.ylabel('Normalized PSD Amplitude')
    #plt.show()
