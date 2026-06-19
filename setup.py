"""
CPS Security Project — Package Setup
=====================================
Research-grade intrusion detection for Cyber-Physical Systems
using Federated Learning, Differential Privacy, XAI, and Adversarial Robustness.
"""

from setuptools import setup, find_packages

setup(
    name="cps_security_project",
    version="0.1.0",
    description="CPS Intrusion Detection with Federated Learning, DP, XAI & Adversarial Robustness",
    author="Abhinav Dogra",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "flwr>=1.5.0",
        "opacus>=1.4.0",
        "adversarial-robustness-toolbox>=1.15.0",
        "shap>=0.42.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "pymongo>=4.5.0",
        "motor>=3.3.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "kagglehub>=0.2.0",
        "tqdm>=4.65.0",
        "loguru>=0.7.0",
    ],
)
