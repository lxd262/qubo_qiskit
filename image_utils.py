"""
Utilities for image processing and reconstruction using QUBO models.
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import mean_squared_error
from typing import Tuple, Optional, Union, List
import matplotlib.pyplot as plt
from skimage import color, transform, filters
from scipy.sparse import csr_matrix, lil_matrix


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
        Build a QUBO matrix for image reconstruction using sparse matrices.
        
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
        
        # Initialize Q matrix as sparse matrix in LIL format (efficient for construction)
        Q = lil_matrix((n, n))
        
        # Data fidelity weights - balance between keeping known pixels and optimizing unknown ones
        data_fidelity_weight = 5.0  # Higher weight for known pixels
        
        # Process each pixel
        pixel_indices = np.arange(n)
        observed_pixels = mask.flatten()
        corrupted_values = corrupted_image.flatten()
        
        # Set data fidelity terms for observed pixels
        observed_idx = pixel_indices[observed_pixels]
        for idx in observed_idx:
            if corrupted_values[idx] > 0.5:
                # For pixels that should be 1: penalize (x_i - 1)^2
                Q[idx, idx] += data_fidelity_weight
                Q[idx, idx] -= 2 * data_fidelity_weight  # Linear term for x_i
            else:
                # For pixels that should be 0: penalize x_i^2
                Q[idx, idx] += data_fidelity_weight
        
        # Smoothness terms with normalization by number of neighbors
        base_smoothness = smoothness_weight / 2.0  # Reduced base weight
        for i in range(height):
            for j in range(width):
                pixel_idx = i * width + j
                neighbors = []
                
                # Collect all valid neighbors
                if i > 0:  # Top
                    neighbors.append((i-1) * width + j)
                if i < height-1:  # Bottom
                    neighbors.append((i+1) * width + j)
                if j > 0:  # Left
                    neighbors.append(i * width + (j-1))
                if j < width-1:  # Right
                    neighbors.append(i * width + (j+1))
                
                if not neighbors:
                    continue
                
                # Normalize smoothness weight by number of neighbors
                local_smoothness = base_smoothness / len(neighbors)
                
                # Add smoothness terms for each neighbor
                for neighbor_idx in neighbors:
                    # (x_i - x_j)^2 = x_i^2 + x_j^2 - 2x_i*x_j
                    Q[pixel_idx, pixel_idx] += local_smoothness
                    Q[neighbor_idx, neighbor_idx] += local_smoothness
                    Q[pixel_idx, neighbor_idx] -= 2 * local_smoothness
        
        # Add small bias towards equal distribution of 0s and 1s for unknown pixels
        unknown_pixels = pixel_indices[~observed_pixels]
        if len(unknown_pixels) > 0:
            balance_weight = 0.1
            for idx in unknown_pixels:
                Q[idx, idx] -= balance_weight  # Small bias towards 1
        
        # Convert to CSR format for efficient computations
        return Q.tocsr()
    
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

    def split_into_tiles(self, image: np.ndarray, mask: np.ndarray, tile_size: int) -> List[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]]:
        """
        Split an image into tiles for processing.
        
        Args:
            image: Input image to split
            mask: Mask indicating known pixels
            tile_size: Size of each tile (tile_size x tile_size)
            
        Returns:
            List of tuples containing (tile_image, tile_mask, (start_row, start_col))
        """
        height, width = image.shape
        tiles = []
        
        # Calculate padding needed
        pad_height = (tile_size - height % tile_size) % tile_size
        pad_width = (tile_size - width % tile_size) % tile_size
        
        # Pad the image and mask if necessary
        if pad_height > 0 or pad_width > 0:
            padded_image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant')
            padded_mask = np.pad(mask, ((0, pad_height), (0, pad_width)), mode='constant')
        else:
            padded_image = image
            padded_mask = mask
        
        # Split into tiles
        for i in range(0, padded_image.shape[0], tile_size):
            for j in range(0, padded_image.shape[1], tile_size):
                tile_image = padded_image[i:i+tile_size, j:j+tile_size]
                tile_mask = padded_mask[i:i+tile_size, j:j+tile_size]
                tiles.append((tile_image, tile_mask, (i, j)))
        
        return tiles

    def merge_tiles(self, tiles: List[Tuple[np.ndarray, Tuple[int, int]]], original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Merge reconstructed tiles back into a full image.
        
        Args:
            tiles: List of tuples containing (tile_image, (start_row, start_col))
            original_shape: Shape of the original image
            
        Returns:
            Merged image
        """
        # Calculate the padded shape
        tile_size = tiles[0][0].shape[0]
        rows = max(pos[0] for _, pos in tiles) + tile_size
        cols = max(pos[1] for _, pos in tiles) + tile_size
        
        # Create empty image
        merged = np.zeros((rows, cols))
        
        # Place tiles in the correct positions
        for tile, (start_row, start_col) in tiles:
            merged[start_row:start_row+tile_size, start_col:start_col+tile_size] = tile
        
        # Crop to original size
        return merged[:original_shape[0], :original_shape[1]]

    def reconstruct_image_tiled(self, corrupted_image: np.ndarray, mask: np.ndarray, 
                              tile_size: int = 4, smoothness_weight: float = 0.5,
                              qubo_solver=None) -> np.ndarray:
        """
        Reconstruct an image using a tiled approach to reduce memory usage.
        
        Args:
            corrupted_image: The corrupted image to reconstruct
            mask: Mask indicating which pixels are observed
            tile_size: Size of tiles to process (must be small enough for QUBO solver)
            smoothness_weight: Weight for the smoothness constraint
            qubo_solver: Instance of QUBOSolver to use (if None, will create one)
            
        Returns:
            Reconstructed image
        """
        if qubo_solver is None:
            from qubo import QUBOSolver
            qubo_solver = QUBOSolver()
        
        # Split image into tiles
        tiles = self.split_into_tiles(corrupted_image, mask, tile_size)
        reconstructed_tiles = []
        
        # Process each tile
        for tile_image, tile_mask, position in tiles:
            # Build QUBO matrix for this tile
            Q = self.build_qubo_matrix_for_reconstruction(tile_image, tile_mask, smoothness_weight)
            
            # Solve QUBO problem for this tile
            solution, _ = qubo_solver.solve_qubo_classical(Q)
            
            # Convert solution to binary array and reshape to tile size
            tile_solution = qubo_solver.format_solution(solution)
            tile_reconstructed = self.reconstruct_image(tile_solution, tile_image.shape)
            
            # Store reconstructed tile and its position
            reconstructed_tiles.append((tile_reconstructed, position))
        
        # Merge tiles back into full image
        reconstructed_image = self.merge_tiles(reconstructed_tiles, corrupted_image.shape)
        
        return reconstructed_image