# setup.py

from setuptools import setup, find_packages

setup(
    name='protica',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'unidecode',
        'scikit-learn',
        'catboost'
    ],
    author='Jose Yezid Castro Rodríguez',
    description='Pipeline de análisis agrícola regional con datos EAM',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
