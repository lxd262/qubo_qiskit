"""
Utilities for image processing and reconstruction using QUBO models.
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import mean_squared_error
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt
from skimage import color, transform, filters


class ImageReconstructor:
    """
    A class for image reconstruction tasks using QUBO optimization.
    """
    
    def __init__(self):
        """Initialize the ImageReconstructor class."""
        pass
    
    def load_image(self, path: str, size: Tuple[int, int] = (16, 16), grayscale: bool = True) -> np.ndarray:
        """
        Load and preprocess an image.
        
        Args:
            path: Path to the image file
            size: Desired size of the image (width, height)
            grayscale: Convert image to grayscale if True
            
        Returns:
            Preprocessed image as numpy array
        """
        # Load the image
        img = Image.open(path)
        
        # Resize the image
        img = img.resize(size)
        
        # Convert to grayscale if needed
        if grayscale and img.mode != 'L':
            img = img.convert('L')
            
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize to [0, 1]
        if img_array.max() > 1:
            img_array = img_array / 255.0
            
        return img_array
    
    def create_sample_image(self, size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Create a simple sample binary image for testing.
        
        Args:
            size: Size of the image to create
            
        Returns:
            Binary image as numpy array
        """
        # Create a blank image
        img = np.zeros(size)
        
        # Add a simple pattern (a cross)
        mid_h = size[0] // 2
        mid_w = size[1] // 2
        thickness = max(1, min(size) // 8)
        
        # Horizontal line
        img[mid_h-thickness:mid_h+thickness, :] = 1
        
        # Vertical line
        img[:, mid_w-thickness:mid_w+thickness] = 1
        
        return img
    
    def add_noise(self, image: np.ndarray, noise_level: float = 0.2) -> np.ndarray:
        """
        Add random noise to an image.
        
        Args:
            image: Input image
            noise_level: Level of noise to add (0 to 1)
            
        Returns:
            Noisy image
        """
        # Create random noise
        noise = np.random.random(image.shape) * noise_level
        
        # Add noise and clip values
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 1)
        
        return noisy_image
    
    def corrupt_image(self, image: np.ndarray, missing_percentage: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Corrupt an image by randomly removing pixels.
        
        Args:
            image: Input image
            missing_percentage: Percentage of pixels to remove (0 to 1)
            
        Returns:
            Tuple of (corrupted_image, mask) where mask shows the removed pixels
        """
        # Create a mask of pixels to keep (1 = keep, 0 = remove)
        mask = np.random.random(image.shape) > missing_percentage
        
        # Apply the mask to create the corrupted image
        corrupted_image = image.copy()
        corrupted_image[~mask] = 0
        
        return corrupted_image, mask
    
    def binarize_image(self, image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Convert an image to binary (0 or 1) using a threshold.
        
        Args:
            image: Input image
            threshold: Threshold value for binarization
            
        Returns:
            Binary image
        """
        return (image > threshold).astype(int)
    
    def build_qubo_matrix_for_reconstruction(self, corrupted_image: np.ndarray, 
                                            mask: np.ndarray,
                                            smoothness_weight: float = 0.5) -> np.ndarray:
        """
        Build a QUBO matrix for image reconstruction.
        
        Args:
            corrupted_image: The corrupted image to reconstruct
            mask: Mask indicating which pixels are observed (1) and which are missing (0)
            smoothness_weight: Weight for the smoothness constraint
            
        Returns:
            QUBO matrix (Q) for the reconstruction problem
        """
        # Get dimensions
        height, width = corrupted_image.shape
        n = height * width  # Total number of variables
        
        # Initialize Q matrix
        Q = np.zeros((n, n))
        
        # Data fidelity term: penalty for observed pixels if they deviate from observations
        for i in range(height):
            for j in range(width):
                if mask[i, j]:  # If pixel is observed
                    pixel_idx = i * width + j
                    # If observed pixel is 1, we want to minimize (1-x_i)^2 = 1 - 2*x_i + x_i^2
                    # If observed pixel is 0, we want to minimize x_i^2
                    
                    if corrupted_image[i, j] > 0.5:  # If pixel is 1 in the corrupted image
                        Q[pixel_idx, pixel_idx] += 1  # x_i^2 term
                        # Linear term -2*x_i will be added to the diagonal
                        Q[pixel_idx, pixel_idx] -= 2  
                    else:  # If pixel is 0 in the corrupted image
                        Q[pixel_idx, pixel_idx] += 1  # x_i^2 term
                    
        # Smoothness term: penalty for neighboring pixels with different values
        for i in range(height):
            for j in range(width):
                pixel_idx = i * width + j
                
                # Check neighbors (4-connectivity)
                neighbors = []
                if i > 0:  # Top neighbor
                    neighbors.append((i-1) * width + j)
                if i < height-1:  # Bottom neighbor
                    neighbors.append((i+1) * width + j)
                if j > 0:  # Left neighbor
                    neighbors.append(i * width + j-1)
                if j < width-1:  # Right neighbor
                    neighbors.append(i * width + j+1)
                
                # Add smoothness terms to Q
                for neighbor_idx in neighbors:
                    # For each pair of neighbors (i,j), we want to minimize (x_i - x_j)^2
                    # This expands to x_i^2 - 2*x_i*x_j + x_j^2
                    Q[pixel_idx, pixel_idx] += smoothness_weight  # x_i^2 term
                    Q[neighbor_idx, neighbor_idx] += smoothness_weight  # x_j^2 term
                    Q[pixel_idx, neighbor_idx] -= 2 * smoothness_weight  # -2*x_i*x_j term
        
        return Q
    
    def reconstruct_image(self, solution: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """
        Reconstruct an image from the optimization solution.
        
        Args:
            solution: Binary solution vector
            shape: Shape of the original image (height, width)
            
        Returns:
            Reconstructed image
        """
        # Reshape the solution vector to the image shape
        reconstructed = solution.reshape(shape)
        
        return reconstructed
    
    def visualize_reconstruction(self, original: np.ndarray, corrupted: np.ndarray, 
                                reconstructed: np.ndarray, title: str = "Image Reconstruction") -> None:
        """
        Visualize the original, corrupted, and reconstructed images.
        
        Args:
            original: Original image
            corrupted: Corrupted image
            reconstructed: Reconstructed image
            title: Title for the figure
        """
        # Create a figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original image
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Plot corrupted image
        axes[1].imshow(corrupted, cmap='gray')
        axes[1].set_title("Corrupted Image")
        axes[1].axis('off')
        
        # Plot reconstructed image
        axes[2].imshow(reconstructed, cmap='gray')
        axes[2].set_title("Reconstructed Image")
        axes[2].axis('off')
        
        # Set main title
        fig.suptitle(title, fontsize=16)
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
        
    def evaluate_reconstruction(self, original: np.ndarray, reconstructed: np.ndarray) -> dict:
        """
        Evaluate the quality of image reconstruction.
        
        Args:
            original: Original image
            reconstructed: Reconstructed image
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Calculate Mean Squared Error
        mse = mean_squared_error(original.flatten(), reconstructed.flatten())
        
        # Calculate accuracy for binary images
        if np.array_equal(original, original.astype(bool).astype(int)) and \
           np.array_equal(reconstructed, reconstructed.astype(bool).astype(int)):
            accuracy = np.mean(original == reconstructed)
        else:
            accuracy = None
            
        # Return metrics
        return {
            'mse': mse,
            'accuracy': accuracy
        }