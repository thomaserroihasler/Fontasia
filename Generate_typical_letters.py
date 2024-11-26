import os
import glob
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageOps


def generate_typical_symbols(output_png_dir: str, output_typical_dir: str, number_of_characters: int, pad_color=0):
    """
    Computes typical symbols for each type of symbol using K-Means clustering after padding images to the same size.
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

    images_array = np.array(images)

    # Calculate the number of clusters
    M = number_of_characters
    print(f"Number of clusters (M) set to: {M}")

    try:
        kmeans = KMeans(n_clusters=M, random_state=42)
        kmeans.fit(images_array)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
    except Exception as e:
        print(f"K-Means clustering failed: {e}")
        return

    # Save each centroid as a typical symbol
    first_shape = (max_height, max_height)
    for idx, centroid in enumerate(centroids, start=1):
        typical_image = centroid.reshape(first_shape)
        # Normalize the image to the 0-255 range
        typical_image_normalized = cv2.normalize(typical_image, None, 0, 255, cv2.NORM_MINMAX)
        typical_image_uint8 = typical_image_normalized.astype(np.uint8)

        # Define the path to save the typical PNG symbol
        typical_image_filename = f"typical_cluster_{idx}.png"
        typical_image_path = os.path.join(output_typical_dir, typical_image_filename)
        cv2.imwrite(typical_image_path, typical_image_uint8)
        print(f"Saved typical symbol: {typical_image_path}")

    # Create a directory to store clusters
    clusters_output_dir = os.path.join(output_typical_dir, "clusters")
    os.makedirs(clusters_output_dir, exist_ok=True)

    # Save images in each cluster into separate folders
    for cluster_idx in range(M):
        cluster_dir = os.path.join(clusters_output_dir, f"cluster_{cluster_idx+1}")
        os.makedirs(cluster_dir, exist_ok=True)
        # Find indices of images belonging to the current cluster
        indices = [i for i, label in enumerate(labels) if label == cluster_idx]
        print(f"Cluster {cluster_idx+1} has {len(indices)} image(s).")

        for idx in indices:
            image_name = os.path.basename(image_files[idx])
            image_path = os.path.join(cluster_dir, image_name)
            # Save the padded image
            padded_images[idx].save(image_path)
            print(f"Saved image '{image_name}' to '{cluster_dir}'")

    print("Typical symbol generation and clustering completed.")



def main():
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
        number_of_characters= 70,
        pad_color=0
    )

    print("Processing completed.")


if __name__ == '__main__':
    main()
