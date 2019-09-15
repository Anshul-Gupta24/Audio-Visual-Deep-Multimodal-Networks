#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:58:58 2017

@author: Omid Sadjadi
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

module_name = 'levinson'

extensions = [
    Extension(module_name, ['levinson.pyx'],
              include_dirs=[numpy_include]),
]

setup(name='Levinson-Durbin recursion algorithm',
      author='Omid Sadjadi <omid.sadjadi@ieee.org>',
      ext_modules=cythonize(extensions),)

so_filename = glob('./build/*linux*/{}*'.format(module_name))
system('mv {} ../{}.so'.format(so_filename[0], module_name))
system('rm -r ./build')
