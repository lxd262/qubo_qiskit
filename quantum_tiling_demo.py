#!/usr/bin/env python
"""
Demo script for processing larger images using QUBO with tile-based quantum decomposition.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from qubo import QUBOSolver
from image_utils import ImageReconstructor

def main():
    print("QUBO Image Reconstruction with Quantum Tiling Decomposition")
    print("--------------------------------------------------------")
    
    # Initialize the QUBO solver and image reconstructor
    qubo_solver = QUBOSolver(seed=42)
    img_reconstructor = ImageReconstructor()
    
    # Create a sample binary image (8x8 for demonstration)
    print("Creating a sample binary image...")
    image_size = (8, 8)  # Larger image that would normally be too large for QAOA
    original_image = img_reconstructor.create_sample_image(size=image_size)
    print(f"Image size: {image_size}")
    
    # Corrupt the image (remove 30% of pixels)
    print("\nCorrupting the image (removing 30% of pixels)...")
    missing_percentage = 0.3
    corrupted_image, mask = img_reconstructor.corrupt_image(
        original_image, missing_percentage=missing_percentage
    )
    print(f"Corrupted image has {np.sum(~mask)} missing pixels out of {mask.size}")
    
    # Build the QUBO matrix for the reconstruction problem
    print("\nBuilding QUBO matrix for image reconstruction...")
    smoothness_weight = 0.5
    start_time = time.time()
    Q = img_reconstructor.build_qubo_matrix_for_reconstruction(
        corrupted_image, mask, smoothness_weight=smoothness_weight
    )
    print(f"QUBO matrix size: {Q.shape}")
    print(f"QUBO matrix built in {time.time() - start_time:.2f} seconds")
    
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
    
    # Now solve with quantum tiling decomposition
    print("\nSolving QUBO problem using quantum tiling decomposition...")
    print("(Breaking the image into smaller tiles that can be processed with QAOA)")
    
    # For demonstration, we'll use a small number of repetitions
    p_qaoa = 1  # Number of QAOA layers
    tile_size = 4  # Use 4x4 tiles (16 qubits each) that can be processed with QAOA
    
    start_time = time.time()
    quantum_solution, quantum_objective = qubo_solver.solve_qubo_quantum(Q, p=p_qaoa)
    quantum_time = time.time() - start_time
    
    print(f"Quantum tiling solution found in {quantum_time:.2f} seconds")
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
    
    # Compare classical and quantum approaches
    print("\nPerformance Comparison:")
    print(f"{'Method':<20} {'Time (s)':<10} {'Objective':<10} {'Accuracy':<10} {'MSE':<10}")
    print(f"{'-'*60}")
    print(f"{'Classical':<20} {classical_time:<10.2f} {classical_objective:<10.4f} "
          f"{classical_metrics['accuracy']:<10.2%} {classical_metrics['mse']:<10.4f}")
    print(f"{'Quantum Tiling':<20} {quantum_time:<10.2f} {quantum_objective:<10.4f} "
          f"{quantum_metrics['accuracy']:<10.2%} {quantum_metrics['mse']:<10.4f}")
    
    # Visualize the results
    print("\nVisualizing the results...")
    
    # Create a figure with side-by-side comparison
    plt.figure(figsize=(15, 8))
    
    # Original and corrupted images
    plt.subplot(231)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(232)
    plt.imshow(corrupted_image, cmap='gray')
    plt.title("Corrupted Image")
    plt.axis('off')
    
    # Classical reconstruction
    plt.subplot(234)
    plt.imshow(classical_reconstructed, cmap='gray')
    plt.title(f"Classical Reconstruction\nAccuracy: {classical_metrics['accuracy']:.2%}")
    plt.axis('off')
    
    # Quantum tiling reconstruction
    plt.subplot(235)
    plt.imshow(quantum_reconstructed, cmap='gray')
    plt.title(f"Quantum Tiling Reconstruction\nAccuracy: {quantum_metrics['accuracy']:.2%}")
    plt.axis('off')
    
    # Difference between classical and quantum
    diff_image = np.abs(classical_reconstructed - quantum_reconstructed)
    plt.subplot(236)
    plt.imshow(diff_image, cmap='hot')
    plt.title("Difference Between Methods")
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.suptitle("Comparison of Classical vs. Quantum Tiling QUBO Reconstruction", fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.savefig("quantum_tiling_reconstruction.png")
    
    print("\nImage saved as 'quantum_tiling_reconstruction.png'")
    print("\nDemo completed!")


if __name__ == "__main__":
    main()