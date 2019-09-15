#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains several frontend processing tools for feature extraction
and feature normalization
"""

__version__ = '1.2'
__author__ = 'Omid Sadjadi'
__email__ = 'omid.sadjadi@nist.gov'

import h5py
from subprocess import Popen, PIPE
import numpy as np
from scipy import signal
from scipy.special import expit
import soundfile as sf
import resampy
# from .gammatone import gammatone
from .levinson import levinson

class MFCC:
    def __init__(self, fs=8000, nfft=512, frame_len=0.025, frame_inc=0.010,
                 preemph_coef=0.97, nchannels=24, fl=100, fh=4000, nceps=20,
                 filter_shape='', spectrum_method='fft',
                 win_type='hamm', spectrum_type='mag', compression='log'):
        self.fs = fs
        self.nfft = nfft
        self.frame_len = int(frame_len * fs)
        self.frame_inc = int(frame_inc * fs)
        self.preemph_coef = preemph_coef
        self.nchannels = nchannels
        self.fl = fl
        self.fh = min(fh, self.fs/2)
        self.nceps = nceps
        self.win_type = win_type
        self.spectrum_type = spectrum_type
        self.spectrum_method = spectrum_method
        self.compression = compression
        self.melb = self.melbank(filter_shape)
        self.dctmat = dctmtx(nchannels)[:nceps]

    def melbank(self, filter_shape=None):
        mel_fl = MFCC.hz2mel(self.fl)
        mel_fh = MFCC.hz2mel(self.fh)
        edges = MFCC.mel2hz(np.linspace(mel_fl, mel_fh, self.nchannels + 2))
        fftbins = np.round(edges / self.fs * self.nfft).astype(int)
        diff1 = fftbins[1:-1] - fftbins[0:-2]
        diff2 = fftbins[2:] - fftbins[1:-1]
        m = np.zeros((int(self.nfft/2 + 1), self.nchannels))
        for d in range(self.nchannels):
            m[fftbins[d]:fftbins[d+1], d] = np.arange(diff1[d]) / diff1[d]
            m[fftbins[d+1]:fftbins[d+2], d] = np.arange(diff2[d], 0, -1) / diff2[d]

        if filter_shape == 'hamm':
            m = 0.5 - 0.46 * np.cos(np.pi * m)
        elif filter_shape == 'hann':
            m = 0.5 - 0.5 * np.cos(np.pi * m)
        return m

    def extract(self, speech):
        s = rm_dc_n_dither(speech, self.fs)
        s = preemphasis(s, self.preemph_coef)
        s = enframe(s, self.frame_len, self.frame_inc)
        if self.spectrum_method == 'fft': 
            S = compute_fft_spectrum(s, self.nfft, self.win_type, self.spectrum_type)
        elif self.spectrum_method == 'lp':
            S = compute_lp_spectrum(s, self.nfft, self.win_type, self.spectrum_type)
        else:
            raise ValueError('Spectrum method {} not supported!'.format(self.spectrum_method))
        if self.compression == 'log':
            logE = np.log(S.dot(self.melb))
        elif self.compression == 'plaw':
            logE = (S.dot(self.melb))**(1./15)
        else:
            raise ValueError('Compression type {} not supported!'.format(self.compression))
        mfc = logE.dot(self.dctmat.T).T
        return mfc

    @staticmethod
    def hz2mel(f):
        mel1000Hz = np.log(1 + 1000./700) / 1000
        return np.log(1 + f/700.) / mel1000Hz

    @staticmethod
    def mel2hz(mel):
        mel1000Hz = np.log(1 + 1000./700) / 1000
        return 700 * (np.exp(mel * mel1000Hz) - 1)


class BNF:
    def __init__(self, dnnFilename, context_size=21, nonlinearity='sigmoid',
                 renorm=False):
        self.context_size = context_size
        self.nonlin = nonlinearity
        self.renorm = renorm
        self.dnn = BNF.load_dnn(dnnFilename)

    def extract(self, feat, sad=None):
        """ This routine extracts the BNF from Cepstral features such as MFCCs
        """
        if sad is not None:
            feat_tmp = feat[:, sad]
            M = feat_tmp.mean(1, keepdims=True)
            S = (feat_tmp.std(1, keepdims=True) + 1e-20)
            feat = (feat - M) / S
        else:
            sad = np.ones((feat.shape[1],), dtype=np.bool)
        feat_spliced = splice_feats(feat, w=self.context_size)
        return BNF.extract_bn_features(self.dnn, feat_spliced[:, sad],
                                       nonlin=self.nonlin, renorm=self.renorm)

    @staticmethod
    def extract_bn_features(dnn, fea, nonlin='sigmoid', renorm=False):
        """ This routine computes the bottleneck features using the DNN
            parameters (b, W) and the spliced feature vectors fea. It is
            assumed that the last layer is the bottleneck layer. This can be
            achieved by running the following command:
            nnet3-copy --binary=false --nnet-config='echo output-node name=output input=dnn_bn.renorm |' \
                   --edits='remove-orphans' exp/nnet3/swbd9/final.raw exp/nnet3/swbd/final.txt
        """
        b, W = dnn
        aff = fea
        for bi, wi in zip(b[:-1], W[:-1]):
            aff = wi.dot(aff) + bi
            aff = BNF.squashit(aff, nonlin, renorm)
        aff = W[-1].dot(aff) + b[-1]
        return aff

    @staticmethod
    def squashit(aff, nonlin, renorm=False):
        """ This routine applies Sigmoid and RELU activation functions along with the
            RMS renorm
        """
        if nonlin == 'sigmoid':
            aff = sigmoid(aff)
        elif nonlin == 'relu':
            np.maximum(aff, 0, aff)
        if renorm:
            aff = renorm_rms(aff, axis=0)
        return aff

    @staticmethod
    def load_dnn(dnnFilename):
        """ This routine reads in the DNN parameters (b, W) that are saved in a HDF5
            formatted file (also see nnet3read)
        """
        with h5py.File(dnnFilename, 'r') as h5f:
            dnn_layers = list(h5f.keys())
            W = []
            b = []
            print("reading in the DNN parameters ...")
            for l in range(len(dnn_layers)//2):
                W.append(h5f['w'+str(l)][:])
                print("layer {}: [{}]".format(l, W[l].shape))
                b.append(h5f['b'+str(l)][:])
            print("done.")
        return b, W


class MHEC:
    def __init__(self, fs=8000, frame_len=0.025, frame_inc=0.010,
                 preemph_coef=0.97, nchannels=24, fl=100, fh=4000, nceps=20,
                 win_type='hamm', spectrum_type='mag', compression='plaw'):
        self.fs = fs
        self.frame_len = int(frame_len * fs)
        self.frame_inc = int(frame_inc * fs)
        self.preemph_coef = preemph_coef
        self.nchannels = nchannels
        self.fl = fl
        self.fh = min(fh, fs/2)
        self.nceps = nceps
        self.win_type = win_type
        self.spectrum_type = spectrum_type
        self.compression = compression
        self.dctmat = dctmtx(nchannels)[:nceps]

    def extract(self, speech):
        s = rm_dc_n_dither(speech, self.fs)
        s = preemphasis(s, self.preemph_coef)
        erb_fl = MHEC.hz2erb(self.fl)
        erb_fh = MHEC.hz2erb(self.fh)
        cfs = MHEC.erb2hz(np.linspace(erb_fl, erb_fh, self.nchannels))
        maxDelay = self.frame_len
        s_tmp = np.r_[s[:, np.newaxis], np.zeros((maxDelay, 1))]
        s_c = np.zeros((self.nchannels, s.size))
        for ix in range(self.nchannels):
            s_c[ix] = gammatone(s_tmp, self.fs, cfs[ix], 20.)
        frames = enframe(s_c, self.frame_len, self.frame_inc)
        win = window(self.frame_len, self.win_type)
        if self.spectrum_type == 'pow':
            frames = frames * frames
        elif self.spectrum_type != 'mag':
            raise ValueError('Spectrum type {} not supported!'.format(self.spectrum_type))
        S = frames.dot(win) / self.frame_len
        if self.compression == 'log':
            logE = np.log(S)
        elif self.compression == 'plaw':
            logE = S**(1./15)
        else:
            raise ValueError('Compression type {} not supported!'.format(self.compression))
        mhc = self.dctmat.dot(logE.T)
        return mhc

    @staticmethod
    def hz2erb(f):
        return 9.2939019127295879 * np.log(1 + 4.37e-3 * f)

    @staticmethod
    def erb2hz(erb):
        return (np.exp(erb/9.2939019127295879) - 1)/4.37e-3

def l2norm(x, axis=0):
    """ computes the l2 norm of x along axis
    """
    return np.sqrt(np.sum(x * x, axis=axis, keepdims=True))


def renorm_rms(x, target_rms=1.0, axis=0):
    """ scales the data such that RMS is 1.0
    """
    # scale = sqrt(x^t x / (D * target_rms^2)).
    D = np.sqrt(x.shape[axis])
    x_rms = l2norm(x, axis) / D
    x_rms[x_rms == 0] = 1.
    return target_rms * x / x_rms


def sigmoid(x):
    """ This routine implements Sigmoid nonlinearity
    """
    return expit(x)# 1 / (1 + np.exp(-x))


def audioread(filename):
    data, sr = sf.read(filename)
    return data, sr


def sphread(filename, ch, sph2pipe_loc=''):
    cmd = "{}sph2pipe -f wav -p -c {} {}".format(sph2pipe_loc, ch, filename)
    p = Popen(cmd, stdout=PIPE, shell=True)
    out = p.stdout.read()
    nchannels = np.frombuffer(out, dtype='H', count=1, offset=22)[0]
    samplerate = np.frombuffer(out, dtype='uint32', count=1, offset=24)[0]
    data = np.frombuffer(out, dtype=np.int16, count=-1, offset=44).astype('f8')
    data /= 2**15  # assuming 16-bit
    return data, samplerate


def preemphasis(sig, mu=0.97):
    sig[1:] -= mu * sig[:-1]
    return sig


def resample(sig, sr_old, sr_new):
    if sr_old != sr_new:
            sig = resampy.resample(sig, sr_old, sr_new)
    return sig


def rm_dc_n_dither(sig, fs):
    np.random.seed(7)  # for repeatability
    if max(abs(sig)) <= 1:
        sig = sig * 32768  # assuming 16-bit
    if fs == 16e3:
        alpha = 0.99
    elif fs == 8e3:
        alpha = 0.999
    else:
        raise ValueError('Sampling frequency {} not supported'.format(fs))
    slen = sig.size
    sig = signal.lfilter([1, -1], [1, -alpha], sig)
    dither = np.random.rand(slen) + np.random.rand(slen) - 1
    sig_pow = max(sig.std(), 1e-20)
    return sig + 1.e-6 * sig_pow * dither


def window(frame_len, win_type='rect', periodic=False):
    if win_type == 'rect':
        win = np.arange(frame_len)
    elif win_type == 'hamm':
        win = hamming(frame_len, periodic)
    elif win_type == 'hann':
        win = hanning(frame_len, periodic)
    return win


def compute_fft_spectrum(s, nfft, win_type='hamm', spectrum_type='pow', axis=-1):
    if s.ndim == 1:
        frame_len = s.size
    else:
        nframes, frame_len = s.shape
    win = window(frame_len, win_type, True)
    s = s * win
    S = np.fft.rfft(s, nfft, norm='ortho', axis=axis)
    S = S.real * S.real + S.imag * S.imag
    if spectrum_type == 'mag':
        S = np.sqrt(S)
    elif spectrum_type != 'pow':
        raise ValueError('Spectrum type {} not supported!'.format(spectrum_type))
    return S


def compute_lp_spectrum(s, nfft, win_type='hamm', spectrum_type='pow', order=12, axis=-1):
    if s.ndim == 1:
        frame_len = s.size
    else:
        nframes, frame_len = s.shape
    win = window(frame_len, win_type, True)
    s = s * win
    a, g = lpc(s, order, axis=axis)
    S = np.fft.rfft(a, nfft, norm='ortho', axis=axis)
    S = g**2/(S.real * S.real + S.imag * S.imag)
    if spectrum_type == 'mag':
        S = np.sqrt(S)
    elif spectrum_type != 'pow':
        raise ValueError('Spectrum type {} not supported!'.format(spectrum_type))
    return S


def cmvn(x, varnorm=True):
    y = x - x.mean(1, keepdims=True)
    if varnorm:
        y /= (x.std(1, keepdims=True) + 1e-20)
    return y


def wcmvn(x, w=301, varnorm=True):
    if w < 3 or (w & 1) != 1:
        raise ValueError('Window length should be an odd integer >= 3')
    ndim, nobs = x.shape
    if nobs < w:
        return cmvn(x, varnorm)
    hlen = int((w-1)/2)
    y = np.zeros((ndim, nobs), dtype=x.dtype)
    y[:, :hlen] = x[:, :hlen] - x[:, :w].mean(1, keepdims=True)
    for ix in range(hlen, nobs-hlen):
        y[:, ix] = x[:, ix] - x[:, ix-hlen:ix+hlen+1].mean(1)
    y[:, nobs-hlen:nobs] = x[:, nobs-hlen:nobs] - x[:, nobs-w:].mean(1, keepdims=True)
    if varnorm:
        y[:, :hlen] /= (x[:, :w].std(1, keepdims=True) + 1e-20)
        for ix in range(hlen, nobs-hlen):
            y[:, ix] /= (x[:, ix-hlen:ix+hlen+1].std(1) + 1e-20)
        y[:, nobs-hlen:nobs] /= (x[:, nobs-w:].std(1, keepdims=True) + 1e-20)
    return y


def rastafilt(x):
    """ Based on rastafile.m by Dan Ellis
       rows of x = critical bands, cols of x = frame
       same for y but after filtering
       default filter is single pole at 0.94
    """
    ndim, nobs = x.shape
    numer = np.arange(-2, 3)
    numer = -numer / np.sum(numer * numer)
    denom = [1, -0.94]
    y = np.zeros((ndim, 4))
    z = np.zeros((ndim, 4))
    zi = [0., 0., 0., 0.]
    for ix in range(ndim):
        y[ix, :], z[ix, :] = signal.lfilter(numer, 1, x[ix, :4], zi=zi, axis=-1)
    y = np.zeros((ndim, nobs))
    for ix in range(ndim):
        y[ix, 4:] = signal.lfilter(numer, denom, x[ix, 4:], zi=z[ix, :], axis=-1)[0]
    return y


def deltas(x, w=5):
    """ Based on deltas.m by Dan Ellis
       rows of x = features, cols of x = frame
       same for y but after filtering
       default filter is single pole at 0.94
    """
    if w < 3 or (w & 1) != 1:
        raise ValueError('Window length should be an odd integer >= 3')
    hlen = int(w / 2.)
    win = np.arange(hlen, -(hlen+1), -1)
    win = win / np.sum(win * win)
    xx = np.c_[np.tile(x[:, 0][:, np.newaxis], hlen), x, np.tile(x[:, -1][:, np.newaxis], hlen)]
    d = signal.lfilter(win, 1, xx)
    return d[:, 2*hlen:]


def append_deltas(frames, dwin=5, ddwin=5):
    dframes = deltas(frames, dwin)
    frames = np.r_[frames, dframes]
    if ddwin > 0:
        frames = np.r_[frames, deltas(dframes, ddwin)]
    return frames


def append_shifted_deltas(x, N=7, d=1, P=3, k=7):
    if d < 1:
        raise ValueError('d should be an integer >= 1')
    nobs = x.shape[1]
    x = x[:N]
    w = 2 * d + 1
    dx = deltas(x, w)
    sdc = np.empty((k*N, nobs))
    sdc[:] = np.tile(dx[:, -1], k).reshape(k*N, 1)
    for ix in range(k):
        if ix*P > nobs:
            break
        sdc[ix*N:(ix+1)*N, :nobs-ix*P] = dx[:, ix*P:nobs]
    return sdc


def splice_feats(x, w=9):
    if w < 3 or ((w & 1) != 1):
        raise ValueError('Window length should be an odd integer >= 3')
    hlen = int(w / 2.)
    ndim, nobs = x.shape
    xx = np.c_[np.tile(x[:, 0][:, np.newaxis], hlen), x,
               np.tile(x[:, -1][:, np.newaxis], hlen)]
    y = np.empty((w*ndim, nobs), dtype=x.dtype)
    for ix in range(w):
        y[ix*ndim:(ix+1)*ndim, :] = xx[:, ix:ix+nobs]
    return y


def dctmtx(n):
    m = range(n)
    x, y = np.meshgrid(m, m)
    D = np.sqrt(2./n) * np.cos(np.pi * (2*x + 1) * y / (2*n))
    D[0] /= np.sqrt(2)
    return D


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def enframe(sig, frame_len, frame_inc):
    frames = rolling_window(sig, frame_len)
    if frames.ndim > 2:
        frames = np.rollaxis(frames, 1)
    return frames[::frame_inc]


def hamming(n, periodic=False):
    N = n if periodic else n-1
    w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n+1)/N)
    return w[:-1]


def hanning(n, periodic=False):
    # w = 0.50 - 0.50 * np.cos(2 * np.pi * np.arange(n+1)/(n-1))
    N = n if periodic else n-1
    w = np.sin(np.pi * np.arange(n+1)/N)**2
    return w[:-1]


def read_3col_sad(filename, nobs):
    with open(filename, 'r') as fid:
        lines = fid.read().splitlines()
    sad = np.zeros((nobs,), dtype=np.bool)
    for line in lines:
        fields = line.split()
        be = int(100 * float(fields[1]))
        en = min(int(100 * float(fields[2])), nobs)
        sad[be:en] = True
    return sad

def lpc(x, order, axis=-1):
    """ Compute the Linear Prediction Coefficients.
    """
    n = x.shape[axis]
    if order > n:
        raise ValueError("Input signal must have length >= order")
    r = xcorr_biased(x, axis)
    return levinson(r, order)


def xcorr_biased(x, axis=-1):
    """ Compute autocorrelation of x along the given axis.
    """
    maxlag = x.shape[axis]
    nfft = 2 ** nextpow2(2 * maxlag - 1)
    X = compute_fft_spectrum(x, nfft, win_type='hamm', spectrum_type='pow', axis=axis)
    r = np.fft.irfft(X, norm='ortho')
    return r[..., :maxlag+1] / maxlag


def nextpow2(n):
    return np.frexp(n)[1]
