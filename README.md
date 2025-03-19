# QUBO Image Reconstruction with Qiskit
This project implements a Quadratic Unconstrained Binary Optimization (QUBO) model using IBM's Qiskit to solve an image reconstruction problem.

## Overview
QUBO models are widely used in quantum computing for solving optimization problems. This project demonstrates how to:
1. Formulate an image reconstruction problem as a QUBO problem
2. Implement the QUBO model using Qiskit
3. Solve the problem using both classical and quantum algorithms
4. Visualize and compare the reconstructed images

The project includes various demo scripts for different use cases, from basic examples to advanced features and larger image processing using tiling techniques.

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
The basic demo creates a small 4x4 sample image, corrupts it by removing 30% of pixels, and reconstructs it using both classical and QAOA quantum methods.

```bash
# Run the basic demo
python demo.py
```

This will generate two images showing the results:
- `classical_reconstruction.png`
- `quantum_reconstruction.png`

### Advanced Demo
The advanced demo offers more customization options and can work with your own images. It supports various parameters to experiment with different settings.

```bash
# Run the advanced demo with custom parameters
python advanced_demo.py --size 8 --missing 0.3 --smoothness 0.5 --quantum
```

For more options:
```bash
python advanced_demo.py --help
```

Options include:
- `--image`: Path to input image (if not provided, a sample image will be created)
- `--size`: Size of the image (it will be resized to size x size)
- `--missing`: Percentage of pixels to remove (0.0 to 1.0)
- `--smoothness`: Weight for the smoothness constraint in QUBO model
- `--quantum`: Run quantum solution (QAOA)
- `--qaoa-layers`: Number of QAOA layers (p parameter)
- `--output-dir`: Directory to save output images

### Quantum Tiling Demo
This demo shows how to handle larger images by decomposing them into smaller tiles that can be processed with quantum algorithms. It's useful for images that would otherwise be too large for direct quantum processing.

```bash
# Run the quantum tiling demo
python quantum_tiling_demo.py
```

This generates a comparison visualization saved as `quantum_tiling_reconstruction.png`.

## Structure
- `qubo.py` - Implementation of the QUBO model and solvers
- `image_utils.py` - Utilities for image processing and reconstruction
- `demo.py` - Basic demonstration script
- `advanced_demo.py` - Extended demo with custom parameters
- `quantum_tiling_demo.py` - Demo for processing larger images with tiling approach
- `environment.yml` - Conda environment specification

## Output Files
The demos generate various visualization files:
- `classical_reconstruction.png` - Results from classical optimization
- `quantum_reconstruction.png` - Results from quantum optimization (QAOA)
- `quantum_tiling_reconstruction.png` - Results from the tiling approach for larger images
- Custom named files in the output directory when using `advanced_demo.py`