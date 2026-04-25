# Leonardo Barazza, acse-lb1223

from setuptools import setup, find_packages

setup(
    name='CELaRe',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
    ],
    python_requires='>=3.11',
)
