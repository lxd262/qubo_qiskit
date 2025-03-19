# QUBO Image Reconstruction with Qiskit

This project implements a Quadratic Unconstrained Binary Optimization (QUBO) model using IBM's Qiskit to solve a simple image reconstruction problem.

## Overview

QUBO models are widely used in quantum computing for solving optimization problems. This project demonstrates how to:

1. Formulate an image reconstruction problem as a QUBO problem
2. Implement the QUBO model using Qiskit
3. Solve the problem using quantum algorithms
4. Visualize the reconstructed image

## Requirements

The project requires several Python packages, including Qiskit, NumPy, Matplotlib, and SciKit-Image. You can install them using either conda or pip.

## Environment Setup

### Option 1: Using Conda with environment.yml

```bash
# Create conda environment from the YAML file
conda env create -f environment.yml

# Activate the environment
conda activate qubo_qiskit_env
```

### Option 2: Manual Installation

To install dependencies manually:

```bash
# Create a new conda environment
conda create -n qubo_qiskit_env python=3.10
conda activate qubo_qiskit_env

# Install packages
conda install -c conda-forge qiskit matplotlib numpy pillow scipy scikit-image
pip install qiskit-optimization qiskit-aer
```

## Usage

### Basic Demo

```bash
# Run the basic demo
python demo.py
```

### Advanced Demo

```bash
# Run the advanced demo with custom parameters
python advanced_demo.py --size 8 --missing 0.3 --smoothness 0.5 --quantum
```

For more options:
```bash
python advanced_demo.py --help
```

## Structure

- `qubo.py` - Implementation of the QUBO model
- `image_utils.py` - Utilities for image processing and reconstruction
- `demo.py` - Basic demonstration script
- `advanced_demo.py` - Extended demo with custom parameters
- `environment.yml` - Conda environment specification