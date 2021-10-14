# -*- coding: utf-8 -*-
# Author: ShiYan(2229029156@qq.com)
# Brief:

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

with open('LICENSE', 'r', encoding='utf-8') as f:
    license = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    reqs = f.read()

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(
    name='CNN4IE',
    author='ShiYan',
    version='0.1.8',
    license='Apache 2.0',
    description='Chinese Information Extraction',
    long_description=readme,
    long_description_content_type='text/markdown',
    author_email='2229029156@qq.com',
    url='https://github.com/jiangnanboy/CNN4IE',
    python_requires='>=3.6',
    classifiers=[
            'Intended Audience :: Developers',
            'Operating System :: OS Independent',
            'Natural Language :: Chinese (Simplified)',
            'Natural Language :: Chinese (Traditional)',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Topic :: Text Processing :: Linguistic',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    keywords='NLP,IE,Chinese Information Extraction',
    #install_requires=reqs.strip().split('\n'),
    packages=find_packages(exclude=['data']),
    package_dir={'CNN4IE': 'CNN4IE'},
    zip_safe=True,
)
