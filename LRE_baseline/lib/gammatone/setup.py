#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 19:04:33 2017

@author: Omid Sadjadi <omid.sadjadi@ieee.org>
"""

from glob import glob
from os import system
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

module_name = 'gammatone'
    
extensions = [
    Extension(module_name, ['gammatone.pyx'],
        include_dirs = [numpy_include]),
]

setup(
  name   = 'Gammatone filter',
  author = 'Omid Sadjadi <omid.sadjadi@ieee.org>',
  ext_modules = cythonize(extensions),

)

so_filename = glob('./gammatone*linux*')
system('mv {} ../{}.so'.format(so_filename[0], module_name))
