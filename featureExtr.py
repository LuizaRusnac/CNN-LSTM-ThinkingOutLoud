import numpy as np
import numpy.matlib

"""
This module contains:

freq2bin - tranform frequency to index of spectrum

bin2freq - tranform index of spectrum into frequency

powerBands - ompute the power of the desired bands passed with bands

spectrumChn - computes the spectrum componentes for all channels
"""
def freq2bin(f, fs, nfft):
    """
    This function tranform frequency to index of spectrum

    Input Data:
        f - frequency
        fs - sample frequency
        nfft - number of points of the fft transform

    Output Data:
        The computed index for the frquency f
    """
    return int(f*nfft/fs)

def bin2freq(n, nfft, fs):
    """
    This function tranform index of spectrum into frequency

    Input Data:
        n - spectrum index
        nfft - number of points of fft the transform
        fs - sample frequency

    Output Data:
        The computed frequency for the n index
    """
    return int(n*fs/nfft)

def powerBands(X, bands, band_win=200, fs = 1000, nfft = 1024):
    """
    This function compute the power of the desired bands passed with bands

    Input Data:
        X - the data signal. Dimension: [nr. channels x nr. samples]
        bands - the desired bands to be computed. Dimension: [nr. bands x 2], ex. if bands is [1x2] will be computed the power
            spectrum from bands[0] to bands[1]. The values are the desired FREQUENCIES.
        band_win - the dimension of window on which the bands will be computed. If the user desire to compute the band power 
            over entire signal, band_win will be 0. DEFAULT = 200.
        fs - the frequency sample of the signal. DEFAULT = 1000
        nfft - the desired number of fft transform points. DEFAULT = 1024.

    Output Data:
        xf - the final features computed. Dimension: [nr. channels x nr. features]
    """
    dim = X.shape

    if band_win == 0:
        band_win = dim[1]

    xf = np.zeros((dim[0],dim[1],len(bands)*int(dim[2]/band_win)))

    for rec,file in enumerate(X):
        k = 0
        for band in bands:
            for win2 in range(0,dim[2],band_win):
                bl = freq2bin(band[0],fs,nfft); bh = freq2bin(band[1],fs,nfft)
                fft = np.fft.fft(file[:,win2:win2+band_win], nfft, axis = 1)
                fft = abs(fft[:,bl:bh])*abs(fft[:,bl:bh])
                xf[rec,:,k] = 20*np.log(np.sum(fft,axis=1))
                k=k+1

    return xf

def spectrumChn(x, fs = 1000, freq = None, nfft = None):
    """
    This function computes the spectrum componentes for all channels

    Input Data:
        x - the data signal. Dimension: [nr. channels x nr. samples].
        fs - the sample frequency of the signal. DEFAULT = 1000
        freq - the desired frequencies to be savd as final features. If freq is None, all frequencies from spectrum
            will be saved. If len(freq)=1 will be saved all the frequencies lower that the freq. If len(freq) = 2, 
            will be saved the freqencies from freq[0] to freq[1]. DEFAULT = None
        nfft - number of points for fft spectrum. DEFAULT = None

    Output Data:
        xf - the final features computed. Dimension: [nr. channels x nr. features]
    """

    if nfft is None:
        nfft = x.shape[2]

    if freq is None:
        freq = np.empty((1),dtype = int)
        freq[0] = int(nfft/2)
    else:
        if len(freq)==1:
            freq[0] = int((nfft/fs)*freq[0])
        else:
            if len(freq)==2:
                freq[0] = int((nfft/fs)*freq[0])
                freq[1] = int((nfft/fs)*freq[1])
            else:
                raise ValueError("Too many freq values")

    if len(freq)==1:
        xf = np.zeros((x.shape[0],x.shape[1],freq[0]))
    else:
        xf = np.zeros((x.shape[0],x.shape[1],freq[1]-freq[0]))

    for rec,i in zip(x,range(len(x))):
        fft = np.fft.fft(rec, n=nfft)
        if len(freq)==1:
            fft = np.abs(fft[:,:freq[0]])
        else:
            fft = np.abs(fft[:,freq[0]:freq[1]])
        fft = fft*fft
        fft[fft==0]=0.00001

        xf[i,:,:] = 20*np.log(fft)

    return xf

def chConv(x):
    xc = np.zeros((x.shape[0],x.shape[1],x.shape[1]))
    for i in range(len(x)):
        # aux = x[i,:,:].T - np.mean(x[i,:,:].T, axis=0)
        # xc[i,:,:] = np.dot(aux.T,aux)/(x.shape[2]-1)
        xc[i,:,:] = np.cov(x[i,:,:])
    return xc

def freqBandMean(x, b):
    bx = np.zeros((x.shape[0], x.shape[1], int(x.shape[2]/b)))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            cnt = 0
            for k in range(0, x.shape[2] - b, b):
                bx[i,j,cnt] = np.mean(x[i,j,k:k+b])
                cnt = cnt + 1
    
    return bx


def spectrumChnLSTM(x, fs = 1000, freq = None, nfft = None):
    """
    This function computes the spectrum componentes for all channels

    Input Data:
        x - the data signal. Dimension: [nr. channels x nr. samples].
        fs - the sample frequency of the signal. DEFAULT = 1000
        freq - the desired frequencies to be savd as final features. If freq is None, all frequencies from spectrum
            will be saved. If len(freq)=1 will be saved all the frequencies lower that the freq. If len(freq) = 2, 
            will be saved the freqencies from freq[0] to freq[1]. DEFAULT = None
        nfft - number of points for fft spectrum. DEFAULT = None

    Output Data:
        xf - the final features computed. Dimension: [nr. channels x nr. features]
    """

    if nfft is None:
        nfft = x.shape[2]

    if freq is None:
        freq = np.empty((1),dtype = int)
        freq[0] = int(nfft/2)
    else:
        if len(freq)==1:
            freq[0] = int((nfft/fs)*freq[0])
        else:
            if len(freq)==2:
                freq[0] = int((nfft/fs)*freq[0])
                freq[1] = int((nfft/fs)*freq[1])
            else:
                raise ValueError("Too many freq values")

    if len(freq)==1:
        xf = np.zeros((x.shape[0],x.shape[1],freq[0],x.shape[3]))
    else:
        xf = np.zeros((x.shape[0],x.shape[1],freq[1]-freq[0], x.shape[3]))

    
    for rec,i in zip(x,range(len(x))):
        for frame in range(x[rec].shape[2]):
            fft = np.fft.fft(rec[:,:,frame], n=nfft)
            if len(freq)==1:
                fft = np.abs(fft[:,:freq[0]])
            else:
                fft = np.abs(fft[:,freq[0]:freq[1]])
            fft = fft*fft
            fft[fft==0]=0.00001

            xf[i,:,:,frame] = 20*np.log(fft)

    return xf

def chConvLSTM(x):
    xc = np.zeros((x.shape[0],x.shape[1],x.shape[2],x.shape[2]))
    for i in range(len(x)):
        for j in range(x.shape[1]):
        # aux = x[i,:,:].T - np.mean(x[i,:,:].T, axis=0)
        # xc[i,:,:] = np.dot(aux.T,aux)/(x.shape[2]-1)
            xc[i,j,:,:] = np.cov(x[i,j,:,:])
    return xc
