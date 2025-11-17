"""
Setup script for Statistical Time Series Prediction package.
"""

from setuptools import setup, find_packages
import os

# Read README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='timeseries-prediction-stat',
    version='1.0.0',
    author='Time Series Prediction Team',
    author_email='',
    description='Statistical models for multi-output time series prediction',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/timeseries_prediction_stat',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Office/Business :: Financial :: Investment',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.9',
            'sphinx>=4.0',
        ],
    },
    include_package_data=True,
    package_data={
        'timeseries_prediction_stat': ['docs/*.md'],
    },
    entry_points={
        'console_scripts': [
            # Add command-line scripts if needed
        ],
    },
    keywords=[
        'time series',
        'forecasting',
        'statistical models',
        'VAR',
        'VECM',
        'dynamic factor models',
        'multivariate analysis',
        'econometrics',
        'financial prediction',
        'multi-output prediction'
    ],
    project_urls={
        'Documentation': 'https://github.com/yourusername/timeseries_prediction_stat/docs',
        'Source': 'https://github.com/yourusername/timeseries_prediction_stat',
        'Bug Reports': 'https://github.com/yourusername/timeseries_prediction_stat/issues',
    },
)
