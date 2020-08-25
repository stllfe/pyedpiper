#!/usr/bin/env python
import os

from setuptools import setup, find_packages

from pyedpiper import __version__


PATH_ROOT = os.path.dirname(__file__)


def load_requirements(path_dir=PATH_ROOT, file_name='requirements.txt', comment_char='#'):
    with open(os.path.join(path_dir, file_name), 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[:ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith('http'):
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


setup(
    name='pyedpiper',
    version=__version__,
    description='Small set of handy tools to compliment your ML pipeline and minimize a boilerplate.',
    author='Oleg Pavlovich',
    author_email='pavlovi4.o@gmail.com',
    url='https://github.com/stllfe/pyedpiper',
    download_url='https://github.com/stllfe/pyedpiper',
    install_requires=load_requirements(),
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    keywords=['deep learning', 'pytorch', 'pytorch lightning', 'AI'],
    python_requires='>=3.6',
    setup_requires=[],
    classifiers=[
        'Environment :: Console',
        'Natural Language :: English',
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        'Development Status :: 4 - Beta',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Information Analysis',
        # Pick your license as you wish
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)