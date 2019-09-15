#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 19:05:58 2017

@author: Omid Sadjadi <omid.sadjadi@ieee.org>
"""

"""
 Based on gammatone_c.c by Ning Ma <n.ma@dcs.shef.ac.uk>
 A Cython implementation of the 4th order gammatone filter
 Usage: env = gammatone(x, fs, cf, lowcut)
     Inputs:
         -x     input speech signal
         -fs    sampling frequency (Hz)
         -cf    centre frequencies of the Gammatone filterbank (Hz)
         -lowcut- cut-off frequency of the envelope smoothing (low-pass) filter (Hz)
     Output:
         -env   instantaneous smoothed envelope
"""

import numpy as np
cimport numpy as np
import cython
cimport cython
#from libc.math cimport round


cdef extern from "math.h" nogil:
    int round(double x)
    double exp(double x)
    double sin(double x)
    double cos(double x)
    double fabs(double x)
    double sqrt(double x)
    
    

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

"""
Utility functions
"""

cdef inline double erb(double f): return ( 24.7 * ( 4.37e-3 * ( f ) + 1.0 ) )

"""
Main Function
"""
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
@cython.nonecheck(False)
def gammatone(np.ndarray[DTYPE_t, ndim=2] x, int fs, double cf, double lpfcut):
    assert x.dtype == DTYPE
    cdef int nsamples = x.shape[0]
    cdef int nchannels = x.shape[1]
    if (( nsamples > 1 ) and ( nchannels > 1 )) or (nchannels>nsamples):
        raise ValueError("Input must be a column vector.")
    cdef int t, intshift, maxDelay
    cdef double a, tpt, tptbw, gain, alpha
    cdef double p0r, p1r, p2r, p3r, p4r, p0i, p1i, p2i, p3i, p4i
    cdef double a1, a2, a3, a4, a5, u0r, u0i
    cdef double qcos, qsin, oldcs, coscf, sincf
   
    cdef double BW_CORRECTION = 1.0190
    cdef double VERY_SMALL_NUMBER = 1e-200
   
    alpha = exp( -2.0 * np.pi * lpfcut / fs ) #single-pole AR filter coefficient

    """
    initialization for phase alignement
    """
    if  fs == 8000:
        maxDelay = 200 # assuming 25ms frames
    elif fs == 16000:
        maxDelay = 400
    cdef np.ndarray[DTYPE_t, ndim=2] env = np.zeros([nsamples, nchannels], dtype=DTYPE)
    nsamples -= maxDelay
    
    """
     Initialising variables
    """
    tpt = 2.0 * np.pi / fs;
    tptbw = tpt * erb(cf) * BW_CORRECTION
    a = exp( -tptbw );
   
    """
    calculating the response delay for each channel
    """
    intshift  = round(3/tptbw)
    nsamples += intshift
   
    """
    based on integral of impulse response
    """
    gain = ( tptbw*tptbw*tptbw*tptbw ) / 3
   
    """
    Update filter coefficients
    """
    a1, a2, a3, a4, a5 = 4.*a, -6.*a*a, 4.*a*a*a, -a*a*a*a, 4.*a*a
    p0r, p1r, p2r, p3r, p4r = 0., 0., 0., 0., 0.
    p0i, p1i, p2i, p3i, p4i = 0., 0., 0., 0., 0.
   
    """
    exp(a+i*b) = exp(a)*(cos(b)+i*sin(b))
    q = exp(-i*tpt*cf*t) = cos(tpt*cf*t) + i*(-sin(tpt*cf*t))
    qcos = cos(tpt*cf*t)
    qsin = -sin(tpt*cf*t)
    """
    coscf = cos( tpt * cf )
    sincf = sin( tpt * cf )
    qcos = 1; qsin = 0   # t=0 & q = exp(-i*tpt*t*cf)
    for t in range(nsamples):
        """
        Filter part 1 & shift down to d.c.
        """
        p0r = qcos*x[t, 0] + a1*p1r + a2*p2r + a3*p3r + a4*p4r
        p0i = qsin*x[t, 0] + a1*p1i + a2*p2i + a3*p3i + a4*p4i
       
        """
        Clip coefficients to stop them from becoming too close to zero
        """
        if (fabs(p0r) < VERY_SMALL_NUMBER): p0r = 0.
        if (fabs(p0i) < VERY_SMALL_NUMBER): p0i = 0.
       
        """
        Filter part 2
        """
        u0r = p0r + a1*p1r + a5*p2r
        u0i = p0i + a1*p1i + a5*p2i
       
        """
        Update filter results
        """
        p4r, p3r, p2r, p1r = p3r, p2r, p1r, p0r
        p4i, p3i, p2i, p1i = p3i, p2i, p1i, p0i
       
        """
        Instantaneous Hilbert envelope (smoothed squared magnitude)
        env = (abs(u) * gain)^2;
        """
        if ( t > 0 ):
            env[t,0] = (1.0) * sqrt( u0r * u0r + u0i * u0i ) * gain +  alpha * env[t-1,0]
        else:
            env[t,0] = sqrt( u0r * u0r + u0i * u0i ) * gain
       
        """
        The basic idea of saving computational load:
        cos(a+b) = cos(a)*cos(b) - sin(a)*sin(b)
        sin(a+b) = sin(a)*cos(b) + cos(a)*sin(b)
        qcos = cos(tpt*cf*t) = cos(tpt*cf + tpt*cf*(t-1))
        qsin = -sin(tpt*cf*t) = -sin(tpt*cf + tpt*cf*(t-1))
        """
        oldcs = qcos
        qcos = coscf * qcos + sincf * qsin
        qsin = coscf * qsin - sincf * oldcs
    """
    cast envelope values into the output variable
    """
    return env[intshift:nsamples,0]
   
