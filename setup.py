from setuptools import find_namespace_packages, setup

setup(
    name='rltrader',
    version='3.0',
    description='Quantylab Reinforcement Learning for Stock Trading',
    author='Quantylab',
    author_email='quantylab@gmail.com',
    url='https://github.com/quantylab/rltrader',
    packages=find_namespace_packages(include=['quantylab.*']),
)
