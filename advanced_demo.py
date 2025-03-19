#!/usr/bin/env python
"""
Advanced demo script for image reconstruction using QUBO optimization with Qiskit.
This demo shows how to:
1. Use your own images
2. Apply different levels of corruption
3. Compare reconstruction results with different parameters
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from pathlib import Path

from qubo import QUBOSolver
from image_utils import ImageReconstructor


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Advanced QUBO Image Reconstruction Demo")
    
    parser.add_argument("--image", type=str, default=None,
                       help="Path to input image (if not provided, a sample image will be created)")
    parser.add_argument("--size", type=int, default=8,
                       help="Size of the image (it will be resized to size x size)")
    parser.add_argument("--missing", type=float, default=0.3,
                       help="Percentage of pixels to remove (0.0 to 1.0)")
    parser.add_argument("--smoothness", type=float, default=0.5,
                       help="Weight for the smoothness constraint in QUBO model")
    parser.add_argument("--quantum", action="store_true",
                       help="Run quantum solution (QAOA)")
    parser.add_argument("--qaoa-layers", type=int, default=1,
                       help="Number of QAOA layers (p parameter)")
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Directory to save output images")
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("Advanced QUBO Image Reconstruction Demo with Qiskit")
    print("---------------------------------------------------")
    
    # Initialize the QUBO solver and image reconstructor
    qubo_solver = QUBOSolver(seed=42)
    img_reconstructor = ImageReconstructor()
    
    # Get the image
    if args.image:
        print(f"Loading image from {args.image}...")
        try:
            original_image = img_reconstructor.load_image(
                args.image, 
                size=(args.size, args.size), 
                grayscale=True
            )
            # Binarize the image (convert to 0s and 1s)
            original_image = img_reconstructor.binarize_image(original_image)
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            print("Creating a sample image instead...")
            original_image = img_reconstructor.create_sample_image(size=(args.size, args.size))
    else:
        print("Creating a sample binary image...")
        original_image = img_reconstructor.create_sample_image(size=(args.size, args.size))
    
    print(f"Image size: {original_image.shape}")
    
    # Corrupt the image
    print(f"\nCorrupting the image (removing {args.missing:.0%} of pixels)...")
    corrupted_image, mask = img_reconstructor.corrupt_image(
        original_image, missing_percentage=args.missing
    )
    print(f"Corrupted image has {np.sum(~mask)} missing pixels out of {mask.size}")
    
    # Build the QUBO matrix for the reconstruction problem
    print("\nBuilding QUBO matrix for image reconstruction...")
    print(f"Using smoothness weight: {args.smoothness}")
    start_time = time.time()
    Q = img_reconstructor.build_qubo_matrix_for_reconstruction(
        corrupted_image, mask, smoothness_weight=args.smoothness
    )
    print(f"QUBO matrix size: {Q.shape}")
    print(f"QUBO matrix built in {time.time() - start_time:.2f} seconds")
    
    # Solve the QUBO problem using classical optimization
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
    
    # Run quantum solution if requested
    if args.quantum:
        print(f"\nSolving QUBO problem using QAOA (p={args.qaoa_layers})...")
        start_time = time.time()
        quantum_solution, quantum_objective = qubo_solver.solve_qubo_quantum(
            Q, p=args.qaoa_layers
        )
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
    
    # Visualize and save the results
    print("\nVisualizing the results...")
    
    # Classical reconstruction
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(corrupted_image, cmap='gray')
    plt.title(f"Corrupted Image ({args.missing:.0%} missing)")
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(classical_reconstructed, cmap='gray')
    plt.title(f"Classical Reconstruction (acc: {classical_metrics['accuracy']:.0%})")
    plt.axis('off')
    plt.tight_layout()
    plt.suptitle(f"Classical QUBO Reconstruction (smoothness: {args.smoothness})", fontsize=16)
    plt.subplots_adjust(top=0.85)
    
    # Save the classical result
    classical_filename = output_dir / f"classical_s{args.size}_m{args.missing:.1f}_w{args.smoothness:.1f}.png"
    plt.savefig(classical_filename)
    print(f"Classical reconstruction saved as {classical_filename}")
    
    # Quantum reconstruction if enabled
    if args.quantum:
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(original_image, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(corrupted_image, cmap='gray')
        plt.title(f"Corrupted Image ({args.missing:.0%} missing)")
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(quantum_reconstructed, cmap='gray')
        plt.title(f"QAOA Reconstruction (acc: {quantum_metrics['accuracy']:.0%})")
        plt.axis('off')
        plt.tight_layout()
        plt.suptitle(f"Quantum QUBO (QAOA p={args.qaoa_layers}, w={args.smoothness})", fontsize=16)
        plt.subplots_adjust(top=0.85)
        
        # Save the quantum result
        quantum_filename = output_dir / f"quantum_s{args.size}_m{args.missing:.1f}_p{args.qaoa_layers}_w{args.smoothness:.1f}.png"
        plt.savefig(quantum_filename)
        print(f"Quantum reconstruction saved as {quantum_filename}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()