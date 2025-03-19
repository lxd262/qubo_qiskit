#!/usr/bin/env python
"""
Demo script comparing tiled and quantum approaches for image reconstruction
side by side using QUBO optimization.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from qubo import QUBOSolver
from image_utils import ImageReconstructor

def run_comparison(size, qubo_solver, img_reconstructor, 
                  tile_size=4, smoothness_weight=0.5, 
                  missing_percentage=0.3, p_qaoa=1):
    """
    Run a comparison between tiled and quantum approaches for the same image.
    
    Args:
        size: Tuple of (height, width) for the image
        qubo_solver: Initialized QUBOSolver instance
        img_reconstructor: Initialized ImageReconstructor instance
        tile_size: Size of tiles for tiled approach
        smoothness_weight: Weight for smoothness constraint
        missing_percentage: Percentage of pixels to remove
        p_qaoa: Number of QAOA layers for quantum approach
        
    Returns:
        Dictionary containing results and metrics
    """
    print(f"\n{'='*50}")
    print(f"Processing image of size {size[0]}x{size[1]}")
    print(f"{'='*50}")
    
    # Create and corrupt the test image
    print("Creating a sample binary image...")
    original_image = img_reconstructor.create_sample_image(size=size)
    corrupted_image, mask = img_reconstructor.corrupt_image(
        original_image, missing_percentage=missing_percentage
    )
    print(f"Corrupted image has {np.sum(~mask)} missing pixels out of {mask.size}")
    
    results = {
        "size": size,
        "original_image": original_image,
        "corrupted_image": corrupted_image,
        "mask": mask
    }
    
    # Tiled approach
    print("\nRunning tiled approach...")
    start_time = time.time()
    tiled_reconstructed = img_reconstructor.reconstruct_image_tiled(
        corrupted_image, mask,
        tile_size=tile_size,
        smoothness_weight=smoothness_weight,
        qubo_solver=qubo_solver
    )
    tiled_time = time.time() - start_time
    
    tiled_metrics = img_reconstructor.evaluate_reconstruction(
        original_image, tiled_reconstructed
    )
    print(f"Tiled approach completed in {tiled_time:.2f} seconds")
    print(f"Tiled reconstruction accuracy: {tiled_metrics['accuracy']:.2%}")
    
    results["tiled"] = {
        "reconstructed": tiled_reconstructed,
        "metrics": tiled_metrics,
        "time": tiled_time
    }
    
    # Quantum approach (if image is small enough)
    if size[0] * size[1] <= 25:  # Only for images up to 5x5
        print("\nRunning quantum approach...")
        start_time = time.time()
        
        # Build full QUBO matrix with reduced smoothness weight for quantum
        Q = img_reconstructor.build_qubo_matrix_for_reconstruction(
            corrupted_image, mask, smoothness_weight=smoothness_weight * 0.5
        )
        
        # Solve with QAOA using fewer shots to reduce memory
        quantum_solution, quantum_objective = qubo_solver.solve_qubo_quantum(
            Q, p=p_qaoa, shots=512  # Reduced from default 1024
        )
        
        # Convert solution to image
        quantum_binary_solution = qubo_solver.format_solution(quantum_solution)
        quantum_reconstructed = img_reconstructor.reconstruct_image(
            quantum_binary_solution, original_image.shape
        )
        
        quantum_time = time.time() - start_time
        quantum_metrics = img_reconstructor.evaluate_reconstruction(
            original_image, quantum_reconstructed
        )
        
        print(f"Quantum approach completed in {quantum_time:.2f} seconds")
        print(f"Quantum reconstruction accuracy: {quantum_metrics['accuracy']:.2%}")
        
        results["quantum"] = {
            "reconstructed": quantum_reconstructed,
            "metrics": quantum_metrics,
            "time": quantum_time,
            "objective": quantum_objective
        }
    else:
        print("\nSkipping quantum approach (image too large)")
        results["quantum"] = None
    
    # Visualize results
    plt.figure(figsize=(20, 5))
    
    # Original and corrupted
    plt.subplot(141)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(142)
    plt.imshow(corrupted_image, cmap='gray')
    plt.title(f"Corrupted Image\n({missing_percentage:.0%} missing)")
    plt.axis('off')
    
    # Tiled reconstruction
    plt.subplot(143)
    plt.imshow(tiled_reconstructed, cmap='gray')
    plt.title(f"Tiled Reconstruction\nAccuracy: {tiled_metrics['accuracy']:.2%}\nTime: {tiled_time:.2f}s")
    plt.axis('off')
    
    # Quantum reconstruction (if available)
    plt.subplot(144)
    if results["quantum"] is not None:
        plt.imshow(quantum_reconstructed, cmap='gray')
        plt.title(f"Quantum Reconstruction\nAccuracy: {quantum_metrics['accuracy']:.2%}\nTime: {quantum_time:.2f}s")
    else:
        plt.text(0.5, 0.5, "Image too large\nfor quantum approach", 
                ha='center', va='center')
    plt.axis('off')
    
    plt.suptitle(f"Reconstruction Comparison ({size[0]}x{size[1]} image)", fontsize=16)
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/comparison_{size[0]}x{size[1]}.png")
    print(f"\nComparison saved as results/comparison_{size[0]}x{size[1]}.png")
    
    return results

def main():
    print("QUBO Image Reconstruction Comparison Demo")
    print("----------------------------------------")
    
    # Initialize solver and reconstructor
    qubo_solver = QUBOSolver(seed=42)
    img_reconstructor = ImageReconstructor()
    
    # We'll use 4x4 images with different parameters
    size = (4, 4)
    
    # Test different parameter combinations
    test_cases = [
        {
            "missing_percentage": 0.3,
            "smoothness_weight": 0.5,
            "p_qaoa": 1,
            "description": "Default parameters"
        },
        {
            "missing_percentage": 0.5,  # More missing pixels
            "smoothness_weight": 0.5,
            "p_qaoa": 1,
            "description": "More missing pixels"
        },
        {
            "missing_percentage": 0.3,
            "smoothness_weight": 1.0,  # Higher smoothness
            "p_qaoa": 1,
            "description": "Higher smoothness"
        },
        {
            "missing_percentage": 0.3,
            "smoothness_weight": 0.5,
            "p_qaoa": 2,  # More QAOA layers
            "description": "More QAOA layers"
        }
    ]
    
    # Run comparisons
    results = {}
    for i, params in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {params['description']}")
        results[f"case_{i}"] = run_comparison(
            size, qubo_solver, img_reconstructor,
            tile_size=4,
            smoothness_weight=params["smoothness_weight"],
            missing_percentage=params["missing_percentage"],
            p_qaoa=params["p_qaoa"]
        )
    
    # Print final comparison table
    print("\n" + "="*100)
    print("Final Comparison Results")
    print("="*100)
    print(f"{'Case':<20} {'Parameters':<40} {'Approach':<10} {'Time (s)':<10} {'Accuracy':<10}")
    print("-"*100)
    
    for case_key, result in results.items():
        case_num = int(case_key.split('_')[1])
        params = test_cases[case_num - 1]
        param_str = f"m={params['missing_percentage']:.1f}, s={params['smoothness_weight']:.1f}, p={params['p_qaoa']}"
        
        # Tiled results
        tiled = result["tiled"]
        print(f"{params['description']:<20} {param_str:<40} {'Tiled':<10} {tiled['time']:<10.2f} {tiled['metrics']['accuracy']:.2%}")
        
        # Quantum results
        quantum = result["quantum"]
        print(f"{'':<20} {'':<40} {'Quantum':<10} {quantum['time']:<10.2f} {quantum['metrics']['accuracy']:.2%}")
        print("-"*100)

if __name__ == "__main__":
    main()