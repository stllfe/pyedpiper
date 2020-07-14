#!/usr/bin/env python

from setuptools import setup, find_packages

from pyedpiper import __version__

setup(name='pyedpiper',
      version=__version__,
      description='Small set of handy tools to compliment your ML pipeline and minimize a boilerplate.',
      author='Oleg Pavlovich',
      author_email='pavlovi4.o@gmail.com',
      url='https://github.com/stllfe/pyedpiper',
      install_requires=[
          'pytorch-lightning',
          'tqdm',
          'numpy',
          'torch',
          'omegaconf',
      ],
      packages=find_packages()
      )
