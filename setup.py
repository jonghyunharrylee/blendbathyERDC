#!/usr/bin/env python
"""blendbathyERDC : Bathymetry Blending by combining prior bathy (PBT) and celerity data (cBathy Phase 1)
"""
from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.rst') as history_file:
    history = history_file.read()

setup(name='blendbathyERDC',
      description='blendbathyERDC is a Python package to run nearshore bathymetry estimation',
      long_description=readme + '\n\n' + history,
      author='Jonghyun Harry Lee',
      author_email='jonghyun.harry.lee@hawaii.edu',
      url='https://github.com/jonghyunharrylee/blendbathyERDC/',
      license='New BSD',
      install_requires=['numpy>=1.9.0', 'scipy>=0.18'],
      platforms='Windows, Mac OS-X, Linux',
      packages=find_packages(include=['blendbathyERDC',
                                      'blendbathyERDC.*']),
      include_package_data=True,
      version='0.1.0')
