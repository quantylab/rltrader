from setuptools import find_namespace_packages, setup

setup(
    name='rltrader',
    version='3.1',
    description='Quantylab Reinforcement Learning for Stock Trading',
    author='Quantylab',
    author_email='quantylab@gmail.com',
    url='https://github.com/quantylab/rltrader',
    packages=find_namespace_packages(where='src', include=['quantylab.*']),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'mplfinance',
        'tqdm',
        'sklearn',
        'tensorflow==2.7.0',
        'torch==1.10.1',
    ]
)
