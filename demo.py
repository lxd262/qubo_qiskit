#!/usr/bin/env python
"""
Demo script for image reconstruction using QUBO optimization with Qiskit.
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from qubo import QUBOSolver
from image_utils import ImageReconstructor


def main():
    print("QUBO Image Reconstruction Demo with Qiskit")
    print("------------------------------------------")
    
    # Initialize the QUBO solver and image reconstructor
    qubo_solver = QUBOSolver(seed=42)
    img_reconstructor = ImageReconstructor()
    
    # Create a sample binary image (4x4 for quantum processing)
    print("Creating a sample binary image...")
    image_size = (4, 4)  # Reduced from 8x8 to 4x4 for quantum processing
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
    
    # Now solve with QAOA quantum algorithm (this will be slower)
    print("\nSolving QUBO problem using QAOA quantum algorithm...")
    print("(This may take a while for larger images)")
    
    # For demonstration, we'll use a small number of repetitions
    p_qaoa = 1  # Number of QAOA layers
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
    
    # Compare classical and quantum approaches
    print("\nPerformance Comparison:")
    print(f"{'Method':<10} {'Time (s)':<10} {'Objective':<10} {'Accuracy':<10} {'MSE':<10}")
    print(f"{'-'*50}")
    print(f"{'Classical':<10} {classical_time:<10.2f} {classical_objective:<10.4f} "
          f"{classical_metrics['accuracy']:<10.2%} {classical_metrics['mse']:<10.4f}")
    print(f"{'Quantum':<10} {quantum_time:<10.2f} {quantum_objective:<10.4f} "
          f"{quantum_metrics['accuracy']:<10.2%} {quantum_metrics['mse']:<10.4f}")
    
    # Visualize the results
    print("\nVisualizing the results...")
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
    plt.suptitle("Classical QUBO Reconstruction", fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.savefig("classical_reconstruction.png")
    
    # Quantum reconstruction
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
    plt.suptitle("Quantum QUBO Reconstruction (QAOA)", fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.savefig("quantum_reconstruction.png")
    
    print("\nImages saved as 'classical_reconstruction.png' and 'quantum_reconstruction.png'")
    print("\nDemo completed!")


if __name__ == "__main__":
    main()