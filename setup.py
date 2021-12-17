from setuptools import find_namespace_packages, setup

setup(
    name='rltrader',
    version='1.0',
    description='Quantylab Reinforcement Learning Stock Trader',
    author='Quantylab',
    author_email='quantylab@gmail.com',
    url='https://github.com/quantylab/rltrader',
    packages=find_namespace_packages(include=['quantylab.*']),
    install_requires=[
        'tensorflow'
    ]
)
