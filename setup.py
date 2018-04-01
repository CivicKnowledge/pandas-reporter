#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from setuptools import find_packages, setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as f:
    readme = f.read()


classifiers = [
    'Development Status :: 4 - Beta',
    'Environment :: Web Environment',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.4',
    'Topic :: Software Development :: Debuggers',
    'Topic :: Software Development :: Libraries :: Python Modules',
]


setup(
    name='pandasreporter',
    version='0.1.0',
    description='Pandas dataframe and series for us with Census Reporter',
    long_description=readme,
    packages=find_packages(),
    install_requires=[
        'fs >= 2',
        'appurl',
        'rowgenerators'
        'pandas',
        'requests',
        'geoid'
    ],
    author='Eric Busboom',
    author_email='eric@civicknowledge.com',
    url='https://github.com/Metatab/pandas-reporter.git',
    license='MIT',
    classifiers=classifiers,
    entry_points={
        'appurl.urls': [
            "censusreporter: = pandasreporter.censusreporter:CensusReporterURL"
        ],
        'rowgenerators': [
            "CRJSON+ = pandasreporter.censusreporter:CensusReporterSource"

        ]

    },
)
