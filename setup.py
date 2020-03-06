# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 07:52:54 2020

@author: nsde
"""
import os
from io import open

from setuptools import setup, find_packages


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
      version='0.1',
      author='Nicki Skafte Detlefsen',
      author_email='nsde@dtu.dk',
      packages=find_packages(),
      python_requires='>=3.6',
      install_requires=load_requirements(PATH_ROOT),

      description='Pytorch-metrics is a simple add on library to pytorch that adds '
                   'many commonly used metrics within deep learning'

      )
