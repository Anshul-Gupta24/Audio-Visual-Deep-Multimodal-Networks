#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the main class for the NIST LRE17 baseline system. The system supports
both frontend processing (e.g., feature extraction, normalization) and
backend modeling (e.g., GMM, i-vector extraction, LDA, etc). At this time, we
only provide tools for the extraction of MFCC, SDC, and BN features. For the
BNFs, we provide pre-trained DNN models on SWB1 (318 hours of speech) and on
combined SWB1+Fisher corpora (2000+ hours of speech). 
NOTE1: This system makes extensive use of the multiprocessing module to run the
       various methods in parallel on a single machine. We recommend the users
       to run the system on a machine with a large number of CPUs and lots of
       memory.
NOTE2: This is a highly configurable system that uses a .cfg file to set up the
       various modules required to run an LRE experiment. However, at this time,
       the system does not perform a lot of error checking. It is your
       responsibility to make sure things are set up properly if any changes to
       the system are made.
NOTE3: We do NOT overwrite features, stats, i-vectors, etc. It is your
       responsibility to remove the previously generated data after a change in
       the system configuration.
"""

__version__ = '1.2'
__author__ = 'Omid Sadjadi'
__email__ = 'omid.sadjadi@nist.gov'


import os
import multiprocessing as mp
import numpy as np
import lib.frontend as fe
from lib.gmm_em import GMM
from lib.ivector import TMatrix, Ivector
from lib.utils import h5write, h5read


def unwrap_extract_features(args):
    return LRESystem.extract_features(*args)


def unwrap_extract_stats(args):
    return LRESystem.extract_stats(*args)


def unwrap_extract_ivectors(args):
    return LRESystem.extract_ivectors(*args)


class LRESystem:
    def __init__(self, config):
        self.config = config
        self.audio_dir = config['Paths']['audio_dir']
        self.list_dir  = config['Paths']['list_dir']
        self.annot_dir = config['Paths']['sad_dir']
        self.work_dir  = config['Paths']['exp_dir']
        self.feat_dir  = config['Paths']['feat_dir']
        self.stat_dir  = config['Paths']['stat_dir']
        self.ivec_dir  = config['Paths']['ivec_dir']
        self.nworkers = min(config['Multiprocessing'].getint('num_workers'),
                            mp.cpu_count()-2)
        self.sr = config['Frontend'].getfloat('sample_rate')
        self.feat_type = self.config['Frontend']['feat_type']
        self.ndim = self.config['GMM'].getint('feat_dim')
        self.ncomps = self.config['GMM'].getint('num_gaussians')
        self.tv_dim = self.config['FactorAnalysis'].getint('total_subspace_dim')
        self.feat_extractor = self.feat_extractor_init()
        self.bnf_extractor = self.bnf_extractor_init() if config.has_option('Paths', 'dnnFilename') else None
        self.gmm = self.gmm_trainer_init()
        self.Tmat = self.ivec_extractor_init()
        self.classifier = self.config['Classifier']['classifier_type']

    def feat_extractor_init(self):
        fl = self.config['Frontend'].getfloat('filter_lo_edge')
        fh = self.config['Frontend'].getfloat('filter_hi_edge')
        nceps = self.config['Frontend'].getint('num_cepstral_coefs')
        if self.feat_type == 'MFCC':
            return fe.MFCC(self.sr, nchannels=24, fl=fl, fh=fh, nceps=nceps)
        elif self.feat_type == 'MHEC':
            return fe.MHEC(self.sr, nchannels=32, fl=fl, fh=fh, nceps=nceps)
        elif self.feat_type == 'BNF':
            # Ok, let's force this here
            self.config.set('Frontend', 'filter_lo_edge', '100')
            self.config.set('Frontend', 'filter_hi_edge', '4000')
            self.config.set('Frontend', 'num_cepstral_coefs', '40')
            return fe.MFCC(self.sr, nchannels=40, fl=100., fh=4000., nceps=40)
        else:
            raise ValueError('Feature type {} not recognized!'.format(self.feat_type))
        return None

    def bnf_extractor_init(self):
        """ NOTE: at this time, we only support BNF extraction from 39-D MFCCs with
            with the following parameters: fl=100, fh=4000, nceps=13, i.e.,
            mfcc = fe.MFCC(self.sr, nchannels=24, fl=100., fh=4000., nceps=13)
        """
        dnnFilename = self.config['Paths']['dnnFilename']
        if not dnnFilename:
            raise ValueError('DNN file name must be provided in the config file under [Paths]')
        # return fe.BNF(dnnFilename, context_size=21, nonlinearity='sigmoid', renorm=False)
        return fe.BNF(dnnFilename, context_size=21, nonlinearity='relu', renorm=True)

    def gmm_trainer_init(self):
        dsfactor = self.config['GMM'].getint('feat_subsample_factor')
        gm_niter = self.config['GMM'].getint('num_em_iters')
        return GMM(self.ndim, self.ncomps, dsfactor, gm_niter, self.nworkers)

    def ivec_extractor_init(self):
        tv_niter = self.config['FactorAnalysis'].getint('num_em_iters')
        return TMatrix(self.tv_dim, self.ndim, self.ncomps, tv_niter, min(self.nworkers, 12))

    def extract_features(self, filenames):
        for f in filenames:
            audiofile, basename, ch = f
            outfile = self.feat_dir + basename + '.h5'
            annotfile = self.annot_dir + basename + '.txt'
            if os.path.isfile(outfile):
                continue
            feats = self.extract_feat_and_apply_sad_then_cmvn(self.audio_dir+audiofile, ch, annotfile)
            h5write(outfile,  feats, 'fea')

    def extract_feat_and_apply_sad_then_cmvn(self, audiofile, ch='a', annotfile=''):
        data, sr_orig = fe.audioread(audiofile)
        if data.ndim > 1:
            data = data[:, 1] if ch == 'b' else data[:, 0]
        if sr_orig > self.sr:
            data = fe.resample(data, sr_orig, self.sr)
        fea = self.feat_extractor.extract(data)
        try:
            sad = fe.read_3col_sad(annotfile, fea.shape[1])
        except:
            #print('Warning1: SAD file does not exist: {} {}'.format(audiofile, annotfile))
            sad = np.ones((fea.shape[1],), dtype=np.bool)
        if sad.sum() == 0:
            #print('Warning2: SAD file is empty: {} {}'.format(audiofile, annotfile))
            sad = np.ones((fea.shape[1],), dtype=np.bool)
        if self.feat_type == 'BNF':
            # fea = fe.append_deltas(fea)
            dfea = self.bnf_extractor.extract(fea, sad)
            # dfea = np.r_[dfea, fe.append_deltas(fea[:10], ddwin=0)[:, sad]]
        else:
            fea = fe.rastafilt(fea)
            dfea = np.r_[fea[:, sad], fe.append_shifted_deltas(fea)[:, sad]]
        dfea = fe.cmvn(dfea)
        dfea = fe.wcmvn(dfea, 301, False)
        return dfea

    def extract_stats(self, filenames):
        for f in filenames:
            audiofile, basename, ch = f
            annotfile = self.annot_dir + basename + '.txt'
            outfile = self.stat_dir + basename + '.h5'
            featfile = self.feat_dir + basename + '.h5'
            if os.path.isfile(outfile):
                continue
            if os.path.isfile(featfile):
                feats = h5read(featfile, 'fea')[0]
            else:
                feats = self.extract_feat_and_apply_sad_then_cmvn(self.audio_dir+audiofile, ch, annotfile)
                if feats is None:
                    raise RuntimeError('oh dear... something went wrong with {} {} {}'.
                                       format(audiofile, annotfile, featfile))
            N, F_hat = self.gmm.compute_centered_stats(feats)
            h5write(outfile, [N, F_hat], ['N', 'F'])

    def extract_ivectors(self, filenames):
        ivector = Ivector(self.tv_dim, self.ndim, self.ncomps)
        ivector.initialize(self.work_dir + 'ubm.gmm', self.work_dir + 'tvmat.h5')
        for f in filenames:
            audiofile, basename, ch = f
            annotfile = self.annot_dir + basename + '.txt'
            featfile = self.feat_dir + basename + '.h5'
            statfile = self.stat_dir + basename + '.h5'
            outfile = self.ivec_dir + basename + '.h5'
            if os.path.isfile(outfile):
                continue
            if os.path.isfile(statfile):
                N, F_hat = h5read(statfile, ['N', 'F'])
            else:
                feats = self.extract_feat_and_apply_sad_then_cmvn(self.audio_dir+audiofile, ch, annotfile)
                if feats is None:
                    raise RuntimeError('oh dear... something went wrong with {} {} {}'.
                                       format(audiofile, annotfile, featfile))
                N, F_hat = self.gmm.compute_centered_stats(feats)
            iv = ivector.extract(N, F_hat)
            h5write(outfile, iv, 'ivec')

    def run_parallel_func(self, file_list, function_name, batch_size, nworkers):
        parallel_func = {'features': self.extract_features,
                         'stats': self.extract_stats,
                         'ivectors': self.extract_ivectors}
        if type(file_list) == str:
            filenames = np.genfromtxt(file_list, dtype='str')
            nparts = len(filenames) / batch_size
            filenames_split = np.array_split(filenames, nparts)
        else:
            filenames_split = file_list
        p = mp.Pool(nworkers)
        res = p.map(parallel_func[function_name], filenames_split)
        p.close()
        if res is not None:
            return res
