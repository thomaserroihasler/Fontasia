import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from PIL import Image

def cluster_image(binary_image, horizontal_weight=1.0, vertical_weight=1.0, eps=15, min_samples=50):
    """
    Clusters white pixels in a binary image based on weighted distance using DBSCAN.

    Parameters:
    - binary_image: 2D numpy array of type uint8 (255 for white, 0 for black).
    - horizontal_weight: Weight for horizontal (x-axis) distance.
    - vertical_weight: Weight for vertical (y-axis) distance.
    - eps: Maximum distance between two samples for them to be considered as in the same neighborhood.
    - min_samples: Minimum number of samples in a neighborhood for a point to be considered a core point.

    Returns:
    - labels: 2D numpy array of the same shape as binary_image with cluster labels.
    """
    # Get the coordinates of white pixels
    y_indices, x_indices = np.where(binary_image == 255)
    coords = np.column_stack((x_indices, y_indices))

    if coords.size == 0:
        print("No white pixels found in the binary image.")
        return np.full(binary_image.shape, -1, dtype=int)

    # Apply weighting to coordinates
    scaled_coords = coords.copy().astype(float)
    scaled_coords[:, 0] *= horizontal_weight
    scaled_coords[:, 1] *= vertical_weight

    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    clustering.fit(scaled_coords)

    labels = clustering.labels_

    # Create an empty label image
    label_image = np.full(binary_image.shape, -1, dtype=int)  # -1 for background

    # Assign cluster labels to the corresponding pixels
    label_image[y_indices, x_indices] = labels

    return label_image

def save_clusters_as_images(label_image, output_dir='clusters', padding=10, min_cluster_size=100):
    """
    Saves each cluster in the label_image as a separate image with padding.

    Parameters:
    - label_image: 2D numpy array with cluster labels.
    - output_dir: Directory where cluster images will be saved.
    - padding: Number of pixels to pad around the cluster bounding box.
    - min_cluster_size: Minimum number of pixels required to save a cluster.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get unique cluster labels excluding -1 (background)
    unique_labels = np.unique(label_image)
    unique_labels = unique_labels[unique_labels != -1]  # Exclude background

    print(f"Total clusters to save: {len(unique_labels)}")

    for label in unique_labels:
        # Get the coordinates of the current cluster
        y_coords, x_coords = np.where(label_image == label)
        cluster_size = len(x_coords)

        if cluster_size < min_cluster_size:
            print(f"Skipping Cluster {label}: size {cluster_size} < min_cluster_size ({min_cluster_size})")
            continue  # Skip small clusters

        # Determine the bounding box of the cluster
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        # Add padding, ensuring the indices stay within image bounds
        x_min_padded = max(x_min - padding, 0)
        x_max_padded = min(x_max + padding, label_image.shape[1] - 1)
        y_min_padded = max(y_min - padding, 0)
        y_max_padded = min(y_max + padding, label_image.shape[0] - 1)

        # Extract the sub-image with padding
        cluster_sub_image = label_image[y_min_padded:y_max_padded+1, x_min_padded:x_max_padded+1]

        # Create a binary image for the cluster
        cluster_binary = (cluster_sub_image == label).astype(np.uint8) * 255  # 0 or 255 for binary image

        # Convert to PIL Image for better handling
        cluster_image = Image.fromarray(cluster_binary, mode='L')  # 'L' mode for (8-bit pixels, black and white)

        # Define the filename
        cluster_filename = os.path.join(output_dir, f"cluster_{label}.png")

        # Save the image using PIL
        cluster_image.save(cluster_filename)

        print(f"Saved Cluster {label} as '{cluster_filename}' (Size: {cluster_image.size})")

def visualize_clusters(binary_image, label_image, unique_labels, save_path=None):
    """
    Visualizes the original binary image and the clustered image.

    Parameters:
    - binary_image: Original binary image.
    - label_image: Image with cluster labels.
    - unique_labels: Unique cluster labels.
    - save_path: If provided, saves the visualization to the given path.
    """
    plt.figure(figsize=(14, 7))

    # Original Binary Image
    plt.subplot(1, 2, 1)
    plt.title("Original Binary Image")
    plt.imshow(binary_image, cmap='gray')
    plt.axis('off')

    # Clustered Image
    plt.subplot(1, 2, 2)
    plt.title("Clustered Image")
    # Use a colormap that distinguishes clusters, handling many colors if needed
    num_clusters = len(unique_labels)
    cmap = plt.get_cmap('tab20') if num_clusters <= 20 else plt.get_cmap('nipy_spectral')
    plt.imshow(label_image, cmap=cmap)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved visualization as '{save_path}'")

    plt.show()

def main():
    
    # Load the image in grayscale
    image_path = 'Testing Handwritting.jpg'  # Ensure 'notes.png' is in the same directory as this script
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Unable to load image '{image_path}'. Please check the file path.")
        return

    # Invert the image so that text is white (255) and background is black (0)
    # inverted_image = cv2.bitwise_not(image)

    # Binarize the image (thresholding)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Display the binary image
    plt.figure(figsize=(6, 6))
    plt.title("Binarized Image")
    plt.imshow(binary_image, cmap='gray')
    plt.axis('off')
    plt.show()

    # Apply DBSCAN clustering
    horizontal_weight = 1.0    # Adjust as needed
    vertical_weight = 1.2      # Adjust as needed
    eps = 10                    # Adjust based on letter size and spacing
    min_samples = 20            # Adjust to avoid noise

    clustered_labels = cluster_image(binary_image, horizontal_weight, vertical_weight, eps, min_samples)

    # Counting clusters using numpy.unique
    unique_labels = np.unique(clustered_labels)
    num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print(f"Number of clusters found: {num_clusters}")

    # Visualize the clustering results
    visualize_clusters(binary_image, clustered_labels, unique_labels, save_path='cluster_visualization.png')

    # Save each cluster as a separate image with padding
    save_clusters_as_images(
        clustered_labels,
        output_dir="clusters",
        padding=5,           # Adjust padding as needed
        min_cluster_size=10  # Adjust based on expected letter size
    )

    print("Clustering and saving of letters completed.")

if __name__ == "__main__":
    main()
