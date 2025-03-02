from setuptools import setup, find_packages

setup(
    name="qsim",
    version="0.1.0",
    description="Hybrid Quantum Circuit Simulator",
    author="QSim Team",
    author_email="info@qsim.example.com",
    packages=find_packages(),
    install_requires=[
        "networkx>=2.6.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "tensornetwork>=0.4.0",
        "psutil>=5.9.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-watch>=4.2.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.10.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
