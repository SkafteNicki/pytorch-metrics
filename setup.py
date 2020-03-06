# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 07:52:54 2020

@author: nsde
"""
import os
from io import open

from setuptools import setup, find_packages

import pytorch_metrics as pm


PATH_ROOT = os.path.dirname(__file__)
def load_requirements(path_dir=PATH_ROOT, comment_char='#'):
    with open(os.path.join(path_dir, 'requirements.txt'), 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[:ln.index(comment_char)]
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs

setup(name='pytorch-metrics',
      version=pm.__version__,
      description=pm.__description__,
      packages=find_packages(),
      python_requires='>=3.6',
      install_requires=load_requirements(PATH_ROOT)
      )