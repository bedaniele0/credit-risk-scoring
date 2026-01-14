"""
============================================================================
setup.py - Package Configuration for Credit Risk Scoring
============================================================================
Configuración para instalación del paquete y CLI commands

Autor: Ing. Daniel Varela Perez
Email: bedaniele0@gmail.com
Tel: +52 55 4189 3428
Metodología: DVP-PRO
============================================================================
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="credit-risk-scoring",
    version="1.0.0",
    author="Ing. Daniel Varela Perez",
    author_email="bedaniele0@gmail.com",
    description="Credit Risk Scoring System using LightGBM and Isotonic Calibration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danielvarela/credit-risk-scoring",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
            "ipython>=8.0.0",
            "ipykernel>=6.0.0",
        ],
        "mlops": [
            "mlflow>=2.8.0",
            "optuna>=3.0.0",
            "shap>=0.43.0",
        ],
        "monitoring": [
            "evidently>=0.4.0",
            "prometheus-client>=0.18.0",
            "grafana-api>=1.0.3",
        ],
        "all": [
            # Dev
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
            # MLOps
            "mlflow>=2.8.0",
            "optuna>=3.0.0",
            "shap>=0.43.0",
            # Monitoring
            "evidently>=0.4.0",
            "prometheus-client>=0.18.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "credit-train=models.train_credit:main",
            "credit-predict=models.predict:main",
            "credit-evaluate=models.evaluate:main",
            "credit-dashboard=visualization.dashboard:main",
            "credit-api=api.main:main",
            "credit-monitor=monitoring.monitoring_run:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    zip_safe=False,
    keywords=[
        "credit-risk",
        "scoring",
        "machine-learning",
        "lightgbm",
        "calibration",
        "financial-ml",
        "risk-assessment",
        "default-prediction",
        "mlops",
        "dvp-pro"
    ],
    project_urls={
        "Bug Reports": "https://github.com/danielvarela/credit-risk-scoring/issues",
        "Source": "https://github.com/danielvarela/credit-risk-scoring",
        "Documentation": "https://github.com/danielvarela/credit-risk-scoring/blob/main/README.md",
    },
)
