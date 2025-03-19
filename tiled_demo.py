#!/usr/bin/env python
"""
Demo script showcasing the tiled image reconstruction approach for handling larger images.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from qubo import QUBOSolver
from image_utils import ImageReconstructor

def main():
    print("QUBO Image Reconstruction with Tiling")
    print("------------------------------------")
    
    # Initialize the QUBO solver and image reconstructor
    qubo_solver = QUBOSolver(seed=42)
    img_reconstructor = ImageReconstructor()
    
    # Test with different image sizes
    sizes = [(4, 4), (6, 6), (8, 8), (12, 12)]
    missing_percentage = 0.3
    smoothness_weight = 0.5
    tile_size = 4  # Process in 4x4 tiles
    
    for size in sizes:
        print(f"\nProcessing {size[0]}x{size[1]} image...")
        
        # Create and corrupt the test image
        original_image = img_reconstructor.create_sample_image(size=size)
        corrupted_image, mask = img_reconstructor.corrupt_image(
            original_image, missing_percentage=missing_percentage
        )
        
        # Tiled approach
        print("\nReconstructing using tiled approach...")
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
        
        print(f"Completed in {tiled_time:.2f} seconds")
        print(f"Reconstruction accuracy: {tiled_metrics['accuracy']:.2%}")
        
        # Visualize results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(original_image, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(corrupted_image, cmap='gray')
        plt.title(f"Corrupted Image\n({missing_percentage:.0%} pixels missing)")
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(tiled_reconstructed, cmap='gray')
        plt.title(f"Tiled Reconstruction\nAccuracy: {tiled_metrics['accuracy']:.2%}")
        plt.axis('off')
        
        plt.suptitle(f"Tiled Image Reconstruction ({size[0]}x{size[1]})", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"results/tiled_reconstruction_{size[0]}x{size[1]}.png")
        
        print(f"\nResults saved as results/tiled_reconstruction_{size[0]}x{size[1]}.png")
    
    print("\nDemo completed! All results saved in the 'results' directory.")

if __name__ == "__main__":
    main()