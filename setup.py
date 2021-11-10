from setuptools import find_packages, setup

print(find_packages())

setup(
    name='rltrader',
    version='1.0',
    description='Quantylab Reinforcement Learning Stock Trader',
    author='Quantylab',
    author_email='quantylab@gmail.com',
    url='https://github.com/quantylab/rltrader',
    packages=['quantylab.rltrader'],
    install_requires=[
        'django', 'pywinauto'
    ]
)
