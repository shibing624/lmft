# -*- coding: utf-8 -*-
import sys

from setuptools import setup, find_packages

# Avoids IDE errors, but actual version is read from version.py
__version__ = None
exec(open('lmft/version.py').read())

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required.')

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='lmft',
    version=__version__,
    description='Language Model Fine-tuning Toolkit',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='XuMing',
    author_email='xuming624@qq.com',
    url='https://github.com/shibing624/lmft',
    license="Apache License 2.0",
    zip_safe=False,
    python_requires=">=3.6.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords='LLM,lmft,GPT2,transformers,pytorch,language model',
    install_requires=[
        "loguru",
        "peft",
        "transformers",
        "datasets",
        "tqdm",
    ],
    packages=find_packages(exclude=['tests']),
    package_dir={'lmft': 'lmft'},
    package_data={'lmft': ['*.*', 'data/*.txt']}
)
