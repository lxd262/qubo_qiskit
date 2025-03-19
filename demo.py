#!/usr/bin/env python
"""
Demo script for image reconstruction using QUBO optimization with Qiskit.
Demonstrates reconstruction for 4x4, 6x6, and 8x8 images.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from qubo import QUBOSolver
from image_utils import ImageReconstructor


def run_demo_for_size(size, qubo_solver, img_reconstructor, smoothness_weight=0.5, missing_percentage=0.3):
    """
    Run the demo for a specific image size
    
    Args:
        size: Tuple of (height, width) for the image
        qubo_solver: Initialized QUBOSolver instance
        img_reconstructor: Initialized ImageReconstructor instance
        smoothness_weight: Weight for smoothness constraint in QUBO
        missing_percentage: Percentage of pixels to remove
        
    Returns:
        Dictionary containing the results
    """
    print(f"\n{'='*50}")
    print(f"Processing image of size {size[0]}x{size[1]}")
    print(f"{'='*50}")
    
    # Create a sample binary image
    print("Creating a sample binary image...")
    original_image = img_reconstructor.create_sample_image(size=size)
    print(f"Image size: {size}")
    
    # Corrupt the image
    print(f"\nCorrupting the image (removing {missing_percentage:.0%} of pixels)...")
    corrupted_image, mask = img_reconstructor.corrupt_image(
        original_image, missing_percentage=missing_percentage
    )
    print(f"Corrupted image has {np.sum(~mask)} missing pixels out of {mask.size}")
    
    # Build the QUBO matrix for the reconstruction problem
    print("\nBuilding QUBO matrix for image reconstruction...")
    start_time = time.time()
    Q = img_reconstructor.build_qubo_matrix_for_reconstruction(
        corrupted_image, mask, smoothness_weight=smoothness_weight
    )
    qubo_time = time.time() - start_time
    print(f"QUBO matrix size: {Q.shape}")
    print(f"QUBO matrix built in {qubo_time:.2f} seconds")
    
    # Solve the QUBO problem using classical optimization first
    print("\nSolving QUBO problem using classical optimization...")
    start_time = time.time()
    classical_solution, classical_objective = qubo_solver.solve_qubo_classical(Q)
    classical_time = time.time() - start_time
    print(f"Classical solution found in {classical_time:.2f} seconds")
    print(f"Classical objective value: {classical_objective:.4f}")
    
    # Reconstruct the image from the classical solution
    classical_binary_solution = qubo_solver.format_solution(classical_solution)
    classical_reconstructed = img_reconstructor.reconstruct_image(
        classical_binary_solution, original_image.shape
    )
    
    # Evaluate the classical reconstruction
    classical_metrics = img_reconstructor.evaluate_reconstruction(
        original_image, classical_reconstructed
    )
    print(f"Classical reconstruction accuracy: {classical_metrics['accuracy']:.2%}")
    print(f"Classical reconstruction MSE: {classical_metrics['mse']:.4f}")
    
    # Now solve with QAOA quantum algorithm (this will be slower)
    print("\nSolving QUBO problem using QAOA quantum algorithm...")
    print("(This may take a while for larger images)")
    
    # For demonstration, we'll use a small number of repetitions
    p_qaoa = 1  # Number of QAOA layers
    quantum_time = None
    quantum_objective = None
    quantum_reconstructed = None
    quantum_metrics = None
    
    # Only run quantum for small sizes due to quantum simulator limitations
    if size[0] <= 6:  # For 4x4 and 6x6 sizes
        start_time = time.time()
        quantum_solution, quantum_objective = qubo_solver.solve_qubo_quantum(Q, p=p_qaoa)
        quantum_time = time.time() - start_time
        
        print(f"Quantum solution found in {quantum_time:.2f} seconds")
        print(f"Quantum objective value: {quantum_objective:.4f}")
        
        # Reconstruct the image from the quantum solution
        quantum_binary_solution = qubo_solver.format_solution(quantum_solution)
        quantum_reconstructed = img_reconstructor.reconstruct_image(
            quantum_binary_solution, original_image.shape
        )
        
        # Evaluate the quantum reconstruction
        quantum_metrics = img_reconstructor.evaluate_reconstruction(
            original_image, quantum_reconstructed
        )
        print(f"Quantum reconstruction accuracy: {quantum_metrics['accuracy']:.2%}")
        print(f"Quantum reconstruction MSE: {quantum_metrics['mse']:.4f}")
    else:
        print("Skipping quantum solution for large image size (8x8) due to computational limitations")
        print("Consider using the quantum_tiling_demo.py for larger images")
    
    # Visualize the results
    print("\nVisualizing the results...")
    
    # Create directories for saving results
    os.makedirs("results", exist_ok=True)
    
    # Classical reconstruction
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(corrupted_image, cmap='gray')
    plt.title("Corrupted Image")
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(classical_reconstructed, cmap='gray')
    plt.title("Classical Reconstruction")
    plt.axis('off')
    plt.tight_layout()
    plt.suptitle(f"Classical QUBO Reconstruction ({size[0]}x{size[1]})", fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.savefig(f"results/classical_reconstruction_{size[0]}x{size[1]}.png")
    
    # Quantum reconstruction
    if quantum_reconstructed is not None:
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(original_image, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(corrupted_image, cmap='gray')
        plt.title("Corrupted Image")
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(quantum_reconstructed, cmap='gray')
        plt.title("Quantum Reconstruction")
        plt.axis('off')
        plt.tight_layout()
        plt.suptitle(f"Quantum QUBO Reconstruction (QAOA) ({size[0]}x{size[1]})", fontsize=16)
        plt.subplots_adjust(top=0.85)
        plt.savefig(f"results/quantum_reconstruction_{size[0]}x{size[1]}.png")
    
    # Return the results
    return {
        "size": size,
        "original_image": original_image,
        "corrupted_image": corrupted_image,
        "classical_reconstructed": classical_reconstructed,
        "classical_metrics": classical_metrics,
        "classical_time": classical_time,
        "classical_objective": classical_objective,
        "quantum_reconstructed": quantum_reconstructed,
        "quantum_metrics": quantum_metrics,
        "quantum_time": quantum_time,
        "quantum_objective": quantum_objective
    }


def main():
    print("QUBO Image Reconstruction Demo with Qiskit")
    print("------------------------------------------")
    print("Demonstrating image reconstruction for 4x4, 6x6, and 8x8 images")
    
    # Initialize the QUBO solver and image reconstructor
    qubo_solver = QUBOSolver(seed=42)
    img_reconstructor = ImageReconstructor()
    
    # Parameters for all demos
    smoothness_weight = 0.5
    missing_percentage = 0.3
    
    # Process different sized images
    results = {}
    
    # 4x4 image (small, easily processable with quantum)
    results["4x4"] = run_demo_for_size((4, 4), qubo_solver, img_reconstructor, 
                                        smoothness_weight, missing_percentage)
    
    # 6x6 image (medium, still processable with quantum but slower)
    results["6x6"] = run_demo_for_size((6, 6), qubo_solver, img_reconstructor, 
                                        smoothness_weight, missing_percentage)
    
    # 8x8 image (larger, may be challenging for quantum simulators)
    results["8x8"] = run_demo_for_size((8, 8), qubo_solver, img_reconstructor, 
                                        smoothness_weight, missing_percentage)
    
    # Compare results across different sizes
    print("\n" + "="*80)
    print("Performance Comparison Across Different Image Sizes")
    print("="*80)
    print(f"{'Size':<10} {'Classical Time':<15} {'Classical Acc':<15} {'Quantum Time':<15} {'Quantum Acc':<15}")
    print(f"{'-'*70}")
    
    for size, result in results.items():
        q_time = result['quantum_time'] if result['quantum_time'] is not None else "N/A"
        q_acc = f"{result['quantum_metrics']['accuracy']:.2%}" if result['quantum_metrics'] is not None else "N/A"
        
        print(f"{size:<10} {result['classical_time']:<15.2f} {result['classical_metrics']['accuracy']:<15.2%} "
              f"{q_time:<15} {q_acc:<15}")
    
    print("\nImages saved in the 'results' directory")
    print("\nDemo completed!")


if __name__ == "__main__":
    main()