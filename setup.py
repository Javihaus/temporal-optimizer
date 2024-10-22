import sys
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define version-specific dependencies
if sys.version_info < (3, 8):
    install_requires = [
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0,<1.0.0",
        "imbalanced-learn>=0.8.0,<0.10.0",
    ]
else:
    install_requires = [
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "imbalanced-learn>=0.8.0",
    ]

setup(
    name="hamiltonian_ai",
    version="0.1.0",
    author="Javier Marin",
    author_email="javierl@jmarin.info",
    description="A Hamiltonian-inspired approach for AI optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Javihaus/hamiltonian_ai",
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=["tests"]),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=install_requires,
    extras_require={
        "dev": ["pytest", "pytest-cov", "flake8", "black", "isort"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
    },
    entry_points={
        "console_scripts": [
            "hamiltonian_ai=hamiltonian_ai.cli:main",
        ],
    },
)
