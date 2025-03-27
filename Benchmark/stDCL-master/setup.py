#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/6 15:41
# @Author  : Zhuohan Yu
# @Site    : 
# @File    : setup.py
# @Software: PyCharm
# @Description:

from setuptools import setup

setup(
    name='stDCL',
    version='1.0.1',
    description='Spatial Transcriptome Heterogeneity Dissection of Brain Regions with Dual Graph Contrastive Learning',
    author='Zhuohan Yu',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author_email="zhuohan20@mails.jlu.edu.cn",
    packages=['stDCL'],
    url='https://github.com/Philyzh8/stDCL',
    license='MIT',
    classifiers=['Operating System :: OS Independent',
                'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Programming Language :: Python :: 3.8'],
)
