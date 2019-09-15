#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:31:29 2017

@author: Omid Sadjadi <omid.sadjadi@ieee.org>
"""

import numpy as np
import h5py
import mmap
import re
import os

def nnet3read(dnnFilename, outFilename="", write_to_disk=False):
    """ This is a simple, yet fast, routine that reads in Kaldi NNet3 Weight and Bias
        parameters, and converts them into lists of 64-bit floating point numpy arrays
        and optionally dumps the parameters to disk in HDF5 format.
        
        :param dnnFilename: input DNN file name (it is assumed to be in text format)
        :param outFilename: output hdf5 filename [optional]
        :param write_to_disk: whether the parameters should be dumped to disk [optional]
        
        :type dnnFilename: string
        :type outFilename: string    
        :type write_to_diks: bool
        
        :return: returns the NN weight and bias parameters (optionally dumps to disk)
        :rtype: tuple (b,W) of list of 64-bit floating point numpy arrays
        
        :Example:
            
        >>> b, W = nnet3read('final.txt', 'DNN_1024.h5', write_to_disk=True)
    """
    # nn_elements = ['LinearParams', 'BiasParams']
    with open(dnnFilename, 'r') as f:
        pattern = re.compile(rb'<(\bLinearParams\b|\bBiasParams\b)>\s+\[\s+([-?\d\.\de?\s]+)\]')
        with mmap.mmap(f.fileno(), 0,
                       access=mmap.ACCESS_READ) as m:
            b = []
            W = []
            ix = 0
            for arr in pattern.findall(m):
                if arr[0] == b'BiasParams':
                    b.append(arr[1].split())
                    print("layer{}: [{}x{}]".format(ix, len(b[ix]), len(W[ix])//len(b[ix])))
                    ix += 1
                elif arr[0] == b'LinearParams':
                    W.append(arr[1].split())
                else:
                    raise ValueError('oops... NN element not recognized!')
    
    # converting list of strings into lists of 64-bit floating point numpy arrays and reshaping
    b = [np.array(b[ix], dtype=np.float).reshape(-1, 1) for ix in range(len(b))]
    W = [np.array(W[ix], dtype=np.float).reshape(len(b[ix]), len(W[ix])//len(b[ix])) for ix in range(len(W))]
    
    if write_to_disk:
        # writing the DNN parameters to an HDF5 file
        if not outFilename:
            raise ValueError('oops... output file name not specified!')
        filepath = os.path.dirname(outFilename)
        if filepath and not os.path.exists(filepath):
            os.makedirs(filepath)
        with h5py.File(outFilename, 'w') as h5f:
            for ix in range(len(b)):
                h5f.create_dataset('w'+str(ix), data= W[ix], 
                dtype='f8', compression='gzip', compression_opts=9)
                h5f.create_dataset('b'+str(ix), data= b[ix], 
                dtype='f8', compression='gzip', compression_opts=9)
                
    return b, W

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

def renorm_rms(data, target_rms=1.0, axis=0):
    """ scales the data such that RMS is 1.0
    """
    #scale = sqrt(x^t x / (D * target_rms^2)).
    D = data.shape[axis]
    scale = np.sqrt(np.sum(data * data, axis=axis, keepdims=True)/(D * target_rms * target_rms))
    scale[scale==0] = 1.
    return data / scale

def sigmoid(x):
    """ This routine implements Sigmoid nonlinearity
    """
    return 1 / (1 + np.exp(-x))

def squashit(aff, nonlin, renorm=False):
    """ This routine applies Sigmoid and RELU activation functions along with the 
        RMS renorm
    """
    if nonlin=='sigmoid':
        aff = sigmoid(aff)
    elif nonlin=='relu':
        np.maximum(aff, 0, aff)
    if renorm:
        aff = renorm_rms(aff, axis=0)
    return aff

def extract_bn_features(dnn, fea, nonlin='sigmoid', renorm=False):
    """ This routine computes the bottleneck features using the DNN parameters (b, W)
        and the spliced feature vectors fea. It is assumed that the last layer is 
        the bottleneck layer. This can be achieved by running the following command:
        nnet3-copy --binary=false --nnet-config='echo output-node name=output input=dnn_bn.renorm |' \
                   --edits='remove-orphans' exp/nnet3/swbd9/final.raw exp/nnet3/swbd/final.txt
    """
    b, W = dnn
    aff = fea
    for bi,wi in zip(b[:-1],W[:-1]):
        aff = wi.dot(aff) + bi
        aff = squashit(aff, nonlin, renorm)
    aff = W[-1].dot(aff) + b[-1]
    return aff


if __name__ == '__main__':
    # example that shows how to extract bottleneck features from (say) MFCCs        
    dnn = nnet3read('final.txt', 'DNN_1024.h5', write_to_disk=True)
    
    # we assume mfc is a numpy array of [ndim x nframes] dimesnsion, e.g., [39 x 537]
    # that contains 39-dimensional (say) MFCCs. Features are spliced by stacking over 
    # a 21-frame context
    fea = splice_feats(mfc, w=21)
    
    # now we extract bottleneck features using the DNN parameters and the spliced 
    # features. Here we assume that a RELU ativation function is used, and followed
    # by a renorm nonlinearity to scale the RMS of the vector of activations to 1.0.
    # This kind of nonlinearity is implemented in Kaldi nnet3 as 'relu-renorm-layer'.
    bnf = extract_bn_features(dnn, fea, nonlin='relu', renorm=True)
