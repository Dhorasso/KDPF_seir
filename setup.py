# setup.py

from setuptools import setup, find_packages

setup(
    name='KDPF_seir',
    version='1.0',
    description='Stochastic SEIR model with Sequential Monte Carlo',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Dhorasso Temfack',
    author_email='temfackd@tcd.ie',
    url='https://github.com/Dhorasso/KDPF_seir',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'joblib',
        'tqdm',
        'plotnine'
        
        # add any other dependencies
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
