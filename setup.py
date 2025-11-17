"""
Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com

Legal Notice: This module and the associated software architecture are proprietary 
and confidential. Unauthorized copying, distribution, modification, or use is 
strictly prohibited without explicit written permission from the copyright holder.

Patent Pending: Certain architectural patterns and implementations described in 
this module may be subject to patent applications.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="timeseries-analysis",
    version="1.0.0",
    author="Ashutosh Sinha",
    author_email="ajsinha@gmail.com",
    description="Comprehensive time series analysis package with deep learning and statistical models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ashutosh-sinha/timeseries-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "yfinance>=0.2.0",
        "statsmodels>=0.13.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "prophet": [
            "prophet>=1.1.0",
        ],
        "all": [
            "prophet>=1.1.0",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "timeseries-dl-example=timeseries.deeplearning.examples.stock_prediction_example:main",
            "timeseries-stat-example=timeseries.stat.examples.stock_prediction_example:main",
        ],
    },
    include_package_data=True,
    package_data={
        "timeseries": [
            "deeplearning/docs/*.md",
            "stat/docs/*.md",
            "*.md",
        ],
    },
    keywords="time-series forecasting machine-learning deep-learning lstm gru transformer arima sarima statistics",
    project_urls={
        "Bug Reports": "https://github.com/ashutosh-sinha/timeseries-analysis/issues",
        "Source": "https://github.com/ashutosh-sinha/timeseries-analysis",
    },
)
