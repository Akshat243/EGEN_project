# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:02:43 2019
@author: nghiatp
"""

from setuptools import find_packages
from setuptools import setup

setup(
    name='alexa-prediction',
    version='0.1',
    author='aagraw24',
    author_email='aagraw24@uic.edu',
    install_requires=["numpy", "pandas", "google-cloud-storage", "scikit-learn", "nltk"],
    packages=find_packages(exclude=['data']),
    description='Alexa-prediction',
    url=''
)
