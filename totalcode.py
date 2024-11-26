import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from PIL import Image, ImageOps

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

def save_clusters_as_images_with_unique_filenames(label_image, output_dir='clusters', padding=10, min_cluster_size=100):
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
        
        # Step 1: List existing files in the output directory
        existing_files = [f for f in os.listdir(output_dir) if f.startswith("letter_") and f.endswith(".png")]
        
        # Step 2: Extract the indices from existing filenames
        existing_indices = [int(f.split("_")[1].split(".")[0]) for f in existing_files]
        
        # Step 3: Find the lowest unused index
        N = 0
        while N in existing_indices:
            N += 1  # Increment until a free index is found

        # Save the image using PIL
        cluster_filename = os.path.join(output_dir, f"letter_{N}.png")
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

    #plt.show()

def generate_image_width_histogram_and_save_narrow_images(folder_path, output_folder):
    """
    Generate and display a histogram of image widths in a given folder,
    save images with width under 35 to a specified output folder, and
    print their filenames.

    Parameters:
    folder_path (str): Path to the folder containing images.
    output_folder (str): Path to save images with width under 30.
    """
    # List to store image widths
    image_widths = []
    narrow_images = []  # List to store filenames of images with width < 30

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            # Open image and get width
            with Image.open(file_path) as img:
                width = img.width
                image_widths.append(width)
                if width < 30:  # Check for narrow images
                    narrow_images.append(filename)
                    # Save the image to the output folder
                    img.save(os.path.join(output_folder, filename))
        except Exception as e:
            # Skip files that are not images
            print(f"Skipping {filename}: {e}")

    # Print images with width under 30
    if narrow_images:
        print("Images with width under 30 pixels:")
        for image in narrow_images:
            print(f"  - {image}")
    else:
        print("No images found with width under 30 pixels.")

    # Check if any valid widths were found
    if not image_widths:
        print("No valid images found in the folder.")
        return

    # Plot histogram of image widths
    plt.figure(figsize=(10, 6))
    plt.hist(image_widths, bins=50, edgecolor="black", alpha=0.7)
    plt.title("Histogram of Image Widths")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    #plt.show()

def pad_images_to_max_height_square(folder_path, output_folder, pad_color=0):
    """
    Pad all black-and-white bitmap images in a folder to make them square,
    with dimensions equal to the largest height found among all images,
    and save them to an output folder.

    Parameters:
    folder_path (str): Path to the folder containing images.
    output_folder (str): Path to save padded square images.
    pad_color (int): Padding color (0 for black, 255 for white in grayscale).
    """
    # List to store image heights
    heights = []

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the folder to find the largest height
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                if img.mode not in ["1", "L"]:  # Accept binary (1-bit) or grayscale (L)
                    print(f"Skipping {filename}: Not a black-and-white or grayscale image.")
                    continue
                heights.append(img.height)
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    if not heights:
        print("No valid black-and-white images found in the folder.")
        return

    # Find the largest height
    max_height = max(heights)
    print(f"Largest height found: {max_height}")

    # Pad images to make them square with dimensions equal to max_height
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                if img.mode not in ["1", "L"]:
                    continue
                # Convert to grayscale (mode "L") for padding
                if img.mode == "1":
                    img = img.convert("L")

                # Set target dimension to max_height for both width and height
                target_dimension = max_height

                # Calculate padding amounts
                width_padding = target_dimension - img.width
                height_padding = target_dimension - img.height

                left_pad = width_padding // 2
                right_pad = width_padding - left_pad

                top_pad = height_padding // 2
                bottom_pad = height_padding - top_pad

                # Add padding to make the image square
                padded_image = ImageOps.expand(
                    img,
                    border=(left_pad, top_pad, right_pad, bottom_pad),
                    fill=pad_color,
                )

                # Save the result in 1-bit (black and white)
                padded_image = padded_image.convert("1")
                padded_image.save(os.path.join(output_folder, filename))
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"All images padded to {max_height}x{max_height} and saved to {output_folder}.")
import os
import glob
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageOps

def generate_typical_symbols(output_png_dir: str, output_typical_dir: str, number_of_characters: int, pad_color=0):
    """
    Computes typical symbols for each type of symbol using K-Means clustering after applying PCA for dimensionality reduction.
    Saves the typical symbols as PNG images and groups images by cluster into separate folders.

    Parameters:
    - output_png_dir (str): Directory containing PNG images.
    - output_typical_dir (str): Directory where typical symbols (PNG) will be saved.
    - number_of_characters (int): Number of clusters for K-Means.
    - pad_color (int): Padding color (0 for black, 255 for white in grayscale).
    """
    os.makedirs(output_typical_dir, exist_ok=True)

    images = []
    padded_images = []
    image_files = []
    heights = []

    # Collect all PNG images
    all_image_files = glob.glob(os.path.join(output_png_dir, "*.png"))
    print(f"Found {len(all_image_files)} image(s) in '{output_png_dir}':")
    for file in all_image_files:
        print(f" - {file}")

    if not all_image_files:
        print(f"No images found in folder '{output_png_dir}'. Skipping...")
        return  # Exit the function if no images are found

    # First pass: find the maximum height among images
    for file_path in all_image_files:
        try:
            img = Image.open(file_path)
            if img.mode not in ["1", "L"]:
                print(f"Skipping '{file_path}': Not a black-and-white or grayscale image.")
                continue
            heights.append(img.height)
            image_files.append(file_path)  # Keep track of valid image files
        except Exception as e:
            print(f"Error processing '{file_path}': {e}")

    if not heights:
        print("No valid images to process after filtering. Exiting...")
        return

    max_height = max(heights)
    print(f"Maximum image height determined: {max_height}")

    # Second pass: pad images and collect their flattened arrays
    for file_path in image_files:
        try:
            img = Image.open(file_path)
            # Pad image to have the same height and width
            padded_img = ImageOps.pad(img, (max_height, max_height), color=pad_color)
            img_array = np.array(padded_img).flatten()
            images.append(img_array)
            padded_images.append(padded_img)
        except Exception as e:
            print(f"Error processing '{file_path}': {e}")

    num_images = len(images)
    print(f"Number of valid images after padding: {num_images}")

    if num_images == 0:
        print("No images to cluster. Exiting...")
        return

    images_original = np.array(images)  # Keep the original images before scaling
    images_array = images_original.copy()  # Alias for clarity

    # Standardize the data before PCA
    scaler = StandardScaler()
    images_scaled = scaler.fit_transform(images_array)
    print("Data standardized using StandardScaler.")

    # Apply PCA for dimensionality reduction, preserving 95% of the variance
    pca = PCA(n_components=0.95, random_state=42)  # Retain 95% of variance
    images_pca = pca.fit_transform(images_scaled)
    print(f"PCA reduced the data to {images_pca.shape[1]} dimensions, preserving 95% of the variance.")

    # Calculate the number of clusters
    M = number_of_characters
    print(f"Number of clusters (M) set to: {M}")

    try:
        kmeans = KMeans(n_clusters=M, random_state=42)
        kmeans.fit(images_pca)
        labels = kmeans.labels_
        print("K-Means clustering completed.")
    except Exception as e:
        print(f"K-Means clustering failed: {e}")
        return

    # Create a directory to store clusters
    clusters_output_dir = os.path.join(output_typical_dir, "clusters")
    os.makedirs(clusters_output_dir, exist_ok=True)

    # Save each centroid as a typical symbol by computing the mean of original images in each cluster
    for cluster_idx in range(M):
        cluster_dir = os.path.join(clusters_output_dir, f"cluster_{cluster_idx+1}")
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Find indices of images belonging to the current cluster
        indices = [i for i, label in enumerate(labels) if label == cluster_idx]
        print(f"Cluster {cluster_idx+1} has {len(indices)} image(s).")

        if not indices:
            print(f"No images found for cluster {cluster_idx+1}. Skipping...")
            continue

        # Compute the centroid as the mean of original images in the cluster
        centroid = images_original[indices].mean(axis=0)
        centroid_image = centroid.reshape((max_height, max_height))
        
        # Normalize the centroid image to the 0-255 range
        centroid_normalized = cv2.normalize(centroid_image, None, 0, 255, cv2.NORM_MINMAX)
        centroid_uint8 = centroid_normalized.astype(np.uint8)
        
        # Define the path to save the typical PNG symbol
        typical_image_filename = f"typical_cluster_{cluster_idx+1}.png"
        typical_image_path = os.path.join(output_typical_dir, typical_image_filename)
        cv2.imwrite(typical_image_path, centroid_uint8)
        print(f"Saved typical symbol: {typical_image_path}")

        # Save each image in the cluster into the cluster directory
        for idx in indices:
            image_name = os.path.basename(image_files[idx])
            image_path = os.path.join(cluster_dir, image_name)
            # Save the padded image
            padded_images[idx].save(image_path)
            print(f"Saved image '{image_name}' to '{cluster_dir}'")

    print("Typical symbol generation and clustering completed.")



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
    #plt.show()

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

    # Process images in 'clusters' directory
    cluster_dir = 'clusters'
    if not os.path.isdir(cluster_dir):
        print(f"Error: Directory '{cluster_dir}' does not exist.")
        return

    # Process each image in the directory
    for filename in os.listdir(cluster_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(cluster_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Error: Unable to load image '{image_path}'. Please check the file path.")
                return

            # Invert the image so that text is white (255) and background is black (0)
            # inverted_image = cv2.bitwise_not(image)

            # Binarize the image (thresholding)
            _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

            # Apply DBSCAN clustering
            horizontal_weight = 1   # Adjust as needed
            vertical_weight = 0.05      # Adjust as needed
            eps = 1.1         # Adjust based on letter size and spacing
            min_samples = 10            # Adjust to avoid noise

            clustered_labels = cluster_image(binary_image, horizontal_weight, vertical_weight, eps, min_samples)

            # Counting clusters using numpy.unique
            unique_labels = np.unique(clustered_labels)
            num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            print(f"Number of clusters found: {num_clusters}")

            # Save each cluster as a separate image with padding
            save_clusters_as_images_with_unique_filenames(
                clustered_labels,
                output_dir="letters",
                padding=5,           # Adjust padding as needed
                min_cluster_size=10  # Adjust based on expected letter size
            )

    print("Processing images in 'clusters' directory completed.")

    # Generate histogram and save narrow images to 'single_letters'
    folder_path = "letters"  # Replace with your folder path
    output_folder = "single_letters"  # Folder to save narrow images
    generate_image_width_histogram_and_save_narrow_images(folder_path, output_folder)
    
    # Pad images to make them square
    folder_path = "single_letters"  # Replace with your folder path
    output_folder = "padded_letters"  # Folder to save padded square images
    pad_images_to_max_height_square(folder_path, output_folder, pad_color=0)  # Use 0 for black background

    # Generate typical symbols using K-Means clustering
    padded_output_folder = 'padded_letters'          # Folder to save padded images
    typical_symbols_output_folder = 'typical_symbols'  # Folder to save typical symbols

    # Ensure output directories exist
    os.makedirs(padded_output_folder, exist_ok=True)
    os.makedirs(typical_symbols_output_folder, exist_ok=True)

    # Step 2: Generate typical symbols using K-Means clustering
    print("Generating typical symbols using K-Means clustering...")
    generate_typical_symbols(
        output_png_dir=padded_output_folder,
        output_typical_dir=typical_symbols_output_folder,
        number_of_characters=100,
        pad_color=0
    )

    print("Processing completed.")

if __name__ == "__main__":
    main()
