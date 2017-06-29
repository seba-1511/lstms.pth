#!/usr/bin/env python

from setuptools import (
        setup as install,
        find_packages,
        )

VERSION = '0.1.0'

install(
        name='lstm',
        packages=['lstm'],
        version=VERSION,
        description='Custom implementation of LSTM variations',
        author='Seb Arnold',
        author_email='smr.arnold@gmail.com',
        url='https://github.com/seba-1511/lstm.pth',
        download_url='https://github.com/seba-1511/lstm.pth/archive/0.1.3.zip',
        license='License :: OSI Approved :: Apache Software License',
        classifiers=[],
        scripts=[]
)
