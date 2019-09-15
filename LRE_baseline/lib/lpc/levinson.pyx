"""
A Cython implementation of the Levinson-Durbin algorithm

Created on Mon Sep 25 15:25:17 2017
@author: Omid Sadjadi <omid.sadjadi@ieee.org>
"""

import numpy as np
cimport numpy as np
import cython
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

"""
Main Function
"""
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
@cython.nonecheck(False)
def levinson(np.ndarray[DTYPE_t, ndim=2] r, int order):
    assert r.dtype == DTYPE
    cdef int nsamples = r.shape[1]
    cdef int nframes = r.shape[0]
    cdef int i, j, fr
    cdef double  acc

    cdef np.ndarray[DTYPE_t, ndim=2] acoeff = np.zeros([nframes, order+1], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] kcoeff = np.zeros([nframes, order], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] err = np.zeros([nframes, 1], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] tmp = np.zeros([order, 1], dtype=DTYPE)

    for fr in range(nframes):
        acoeff[fr, 0] = 1.
        err[fr, 0] = r[fr, 0]
        for i in range(1, order + 1):
            acc = r[fr, i]
            for j in range(1, i):
                acc += acoeff[fr, j] * r[fr, i - j]
            kcoeff[fr, i - 1] = -acc / err[fr, 0]
            acoeff[fr, i] = kcoeff[fr, i - 1]
            tmp[0 : order, 0] = acoeff[fr, 0 : order]
            for j in range(1, i):
                    acoeff[fr, j] += kcoeff[fr, i - 1] * tmp[i - j, 0]
            err[fr, 0] *= (1 - kcoeff[fr, i - 1] * kcoeff[fr, i - 1])

    return acoeff, err
