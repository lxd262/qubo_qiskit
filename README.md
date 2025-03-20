# QUBO Image Reconstruction with Qiskit
This project implements a Quadratic Unconstrained Binary Optimization (QUBO) model using IBM's Qiskit to solve an image reconstruction problem.

## Overview
QUBO models are widely used in quantum computing for solving optimization problems. This project demonstrates how to:
1. Formulate an image reconstruction problem as a QUBO problem
2. Implement the QUBO model using Qiskit
3. Solve the problem using both classical and quantum algorithms
4. Visualize and compare the reconstructed images

The project includes various demo scripts for different use cases, from basic examples to advanced features and larger image processing using tiling techniques. It also demonstrates memory-efficient approaches for handling larger images that would otherwise be too memory-intensive to process with conventional QUBO methods.

## Key Features
- **Memory-Efficient Processing**: Uses sparse matrices and tiled approaches to handle larger images
- **Multiple Reconstruction Approaches**: Classical solver, quantum solver (QAOA), and tiled reconstruction
- **Side-by-Side Comparisons**: Direct comparison of different methods on the same images
- **Parameter Experimentation**: Ability to test various smoothness weights, missing pixel percentages, and quantum parameters

## Requirements
The project requires several Python packages, including Qiskit, NumPy, Matplotlib, SciKit-Image, and SciPy. You can install them using either conda or pip.

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
The basic demo creates sample images of different sizes (4x4, 6x6, 8x8), corrupts them by removing pixels, and reconstructs them using classical optimization methods.

```bash
# Run the basic demo
python demo.py
```

This will generate images showing the reconstruction results for different image sizes.

### Tiled Demo
The tiled demo demonstrates our memory-efficient approach for handling larger images by processing them in smaller tiles. This approach can successfully reconstruct images up to 12x12 or larger without excessive memory usage.

```bash
# Run the tiled demo
python tiled_demo.py
```

This generates reconstruction visualizations for 4x4, 6x6, 8x8, and 12x12 images, saved in the `results/` directory.

### Comparison Demo
The comparison demo provides a side-by-side comparison of the tiled approach versus the quantum approach (QAOA) for image reconstruction. It also tests different parameter combinations to show how they affect reconstruction quality and performance.

```bash
# Run the comparison demo
python comparison_demo.py
```

This generates a detailed comparison showing:
- Accuracy and runtime performance for each approach
- Effects of different parameters (missing pixel percentage, smoothness weight, QAOA layers)
- Visual comparison of reconstruction results

## Parameters to Experiment With
- **Missing Percentage**: Controls how many pixels are randomly removed (0.0 to 1.0)
- **Smoothness Weight**: Controls the balance between data fidelity and image smoothness
- **Tile Size**: For tiled reconstruction, controls the size of each processed tile
- **QAOA Layers**: For quantum reconstruction, controls the depth of the quantum circuit

## Structure
- `qubo.py` - Implementation of the QUBO model and solvers with sparse matrix support
- `image_utils.py` - Utilities for image processing and tiled reconstruction
- `demo.py` - Basic demonstration script for various image sizes
- `tiled_demo.py` - Demo showing the memory-efficient tiled reconstruction
- `comparison_demo.py` - Side-by-side comparison of tiled vs. quantum approaches
- `environment.yml` - Conda environment specification

## Output Files
The demos generate various visualization files in the `results/` directory:
- `classical_reconstruction_*.png` - Results from classical optimization for different image sizes
- `tiled_reconstruction_*.png` - Results from the tiled reconstruction approach for different image sizes
- `comparison_*.png` - Side-by-side comparison of different reconstruction approaches

## Performance Notes
- The tiled approach achieves comparable accuracy to the quantum approach but runs 1000-10000x faster
- For 4x4 images, both approaches achieve >90% accuracy, but tiled completes in milliseconds vs. minutes for quantum
- Images larger than 6x6 are generally too memory-intensive for direct quantum processing but can be handled efficiently with the tiled approach