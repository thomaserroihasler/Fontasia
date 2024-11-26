import cv2
import numpy as np
import random
import pandas as pd
import os
import string
import glob
from Bitmap_to_Paths import bitmap_to_paths
from Paths_to_SVG import Paths_to_svg
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

# ----------------------------- Configuration -----------------------------

# Specify the EMNIST split you want to use
emnist_split = 'byclass'  # Options: 'balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist'

# Define file paths based on the selected split
base_path = 'data/EMNIST/archive'
train_csv = f'{base_path}/emnist-{emnist_split}-train.csv'
test_csv = f'{base_path}/emnist-{emnist_split}-test.csv'
mapping_file = f'{base_path}/emnist-{emnist_split}-mapping.txt'

# Define output directories
output_base_dir = 'output_emnist_characters'
output_png_dir = os.path.join(output_base_dir, 'png')
output_svg_dir = os.path.join(output_base_dir, 'svg')
output_typical_dir = os.path.join(output_base_dir, 'typical_symbols')  # Directory for typical symbols

# Define categories
categories = ['uppercase', 'lowercase', 'digits', 'other']

# Create category directories for PNG and SVG
for category in categories:
    os.makedirs(os.path.join(output_png_dir, category), exist_ok=True)
    os.makedirs(os.path.join(output_svg_dir, category), exist_ok=True)
    os.makedirs(os.path.join(output_typical_dir, category), exist_ok=True)  # Also create for typical symbols

# ----------------------------- Functions -----------------------------

def load_emnist_csv(train_csv_path: str, test_csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads EMNIST data from train and test CSV files.

    Parameters:
    - train_csv_path (str): Path to the training CSV file.
    - test_csv_path (str): Path to the testing CSV file.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Tuple containing images and labels.
    """
    # Load training data
    print(f"Loading training data from {train_csv_path}...")
    train_df = pd.read_csv(train_csv_path, header=None)
    train_labels = train_df.iloc[:, 0].values
    train_images = train_df.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)

    # Load testing data
    print(f"Loading testing data from {test_csv_path}...")
    test_df = pd.read_csv(test_csv_path, header=None)
    test_labels = test_df.iloc[:, 0].values
    test_images = test_df.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)

    # Combine train and test datasets
    all_images = np.concatenate((train_images, test_images), axis=0)
    all_labels = np.concatenate((train_labels, test_labels), axis=0)

    print(f"Total images loaded: {all_images.shape[0]}")
    return all_images, all_labels

def load_mapping(mapping_path: str) -> dict:
    """
    Loads label to character mapping from a mapping file.

    Parameters:
    - mapping_path (str): Path to the mapping file.

    Returns:
    - dict: Dictionary mapping label integers to characters.
    """
    mapping = {}
    print(f"Loading label mapping from {mapping_path}...")
    with open(mapping_path, 'r') as f:
        for line in f:
            label, char = line.strip().split()
            try:
                # Some characters might be represented as integers (ASCII codes)
                mapping[int(label)] = chr(int(char))
            except ValueError:
                # Handle cases where the character is not represented as an integer
                mapping[int(label)] = char
    return mapping

def sanitize_filename(name: str) -> str:
    """
    Sanitizes the filename by removing or replacing invalid characters.

    Parameters:
    - name (str): The original filename.

    Returns:
    - str: The sanitized filename.
    """
    # Define a list of invalid characters for most file systems
    invalid_chars = r'\/:*?"<>|'
    sanitized = ''.join(['_' if c in invalid_chars else c for c in name])
    return sanitized

def label_to_char(label: int, mapping: dict) -> str:
    """
    Converts a label integer to its corresponding character using the mapping.

    Parameters:
    - label (int): The label integer.
    - mapping (dict): The label to character mapping.

    Returns:
    - str: The corresponding character.
    """
    return mapping.get(label, '?')  # Returns '?' if label not found

def categorize_character(char: str) -> str:
    """
    Categorizes the character into 'uppercase', 'lowercase', 'digits', or 'other'.

    Parameters:
    - char (str): The character to categorize.

    Returns:
    - str: The category of the character.
    """
    if char.isupper():
        return 'uppercase'
    elif char.islower():
        return 'lowercase'
    elif char.isdigit():
        return 'digits'
    else:
        return 'other'

def determine_filename(directory: str, base_name: str, extension: str) -> str:
    """
    Determines the next available filename in the directory following the naming convention.

    Parameters:
    - directory (str): The directory to check for existing files.
    - base_name (str): The base name for the file (e.g., 'label').
    - extension (str): The file extension (e.g., 'png' or 'svg').

    Returns:
    - str: The determined filename.
    """
    # Initial filename
    filename = f"{base_name}.{extension}"
    filepath = os.path.join(directory, filename)
    
    if not os.path.exists(filepath):
        return filename
    
    # If file exists, find the smallest unused natural number starting from 2
    i = 2
    while True:
        filename = f"{base_name}_{i}.{extension}"
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            return filename
        i += 1

def plot_paths(outer_paths: List[List[Tuple[float, float]]], inner_paths: List[List[Tuple[float, float]]], character: str):
    """
    Plots the outer and inner paths using Matplotlib.

    Parameters:
    - outer_paths (List[List[Tuple[float, float]]]): List of outer contours.
    - inner_paths (List[List[Tuple[float, float]]]): List of inner contours (holes).
    - character (str): The character being plotted.
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert Y-axis to match image coordinates

    # Plot outer paths
    for path in outer_paths:
        if len(path) == 0:
            continue  # Skip empty paths
        xs, ys = zip(*path)
        ax.plot(xs, ys, color='black', linewidth=2)
        ax.fill(xs, ys, color='lightgray', alpha=0.7)

    # Plot inner paths
    for path in inner_paths:
        if len(path) == 0:
            continue  # Skip empty paths
        xs, ys = zip(*path)
        ax.plot(xs, ys, color='red', linewidth=2)
        ax.fill(xs, ys, color='white', alpha=1.0)

    plt.title(f"Outer and Inner Paths for '{character}'")
    plt.show()

def generate_typical_symbols(output_png_dir: str, output_typical_dir: str, output_typical_svg_dir: str, M: int = 3):
    """
    Computes typical symbols for each type of symbol using K-Means clustering and saves the results.
    Additionally, generates corresponding SVG files for each typical symbol.

    Parameters:
    - output_png_dir (str): Directory containing PNG images organized by category and symbol.
    - output_typical_dir (str): Directory where typical symbols (PNG) will be saved.
    - output_typical_svg_dir (str): Directory where typical symbols (SVG) will be saved.
    - M (int): Number of clusters per symbol to generate typical examples.
    """
    os.makedirs(output_typical_dir, exist_ok=True)
    os.makedirs(output_typical_svg_dir, exist_ok=True)
    
    # Traverse through each category and symbol subdirectory
    for category in os.listdir(output_png_dir):
        category_path = os.path.join(output_png_dir, category)
        if not os.path.isdir(category_path):
            continue  # Skip if not a directory

        # Create corresponding category directories in typical symbols
        category_typical_png_dir = os.path.join(output_typical_dir, category)
        category_typical_svg_dir = os.path.join(output_typical_svg_dir, category)
        os.makedirs(category_typical_png_dir, exist_ok=True)
        os.makedirs(category_typical_svg_dir, exist_ok=True)

        for symbol in os.listdir(category_path):
            symbol_path = os.path.join(category_path, symbol)
            if not os.path.isdir(symbol_path):
                continue  # Skip if not a directory

            images = []
            image_shapes = []

            # Collect all PNG images for the current symbol
            for file_path in glob.glob(os.path.join(symbol_path, "*.png")):
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img.flatten())
                    image_shapes.append(img.shape)

            num_images = len(images)
            if num_images == 0:
                print(f"No images found in folder '{symbol_path}'. Skipping...")
                continue
            else:
                # Ensure all images have the same shape
                first_shape = image_shapes[0]
                if not all(shape == first_shape for shape in image_shapes):
                    print(f"Images in '{symbol_path}' have varying dimensions. Skipping...")
                    continue

                images_array = np.array(images)
                print(f"Performing K-Means clustering for symbol: '{symbol}' in category: '{category}' with {num_images} images.")

                # Adjust M if there are fewer images than desired clusters
                current_M = min(M, num_images)
                if current_M < M:
                    print(f"Only {num_images} images available for symbol '{symbol}'. Reducing number of clusters to {current_M}.")

                try:
                    kmeans = KMeans(n_clusters=current_M, random_state=42)
                    kmeans.fit(images_array)
                    centroids = kmeans.cluster_centers_
                except Exception as e:
                    print(f"Error performing K-Means for symbol '{symbol}' in category '{category}': {e}")
                    print("Skipping this symbol.")
                    continue

                # Save each centroid as a typical symbol
                for idx, centroid in enumerate(centroids, start=1):
                    typical_image = centroid.reshape(first_shape)
                    # Normalize the image to the 0-255 range
                    typical_image_normalized = cv2.normalize(typical_image, None, 0, 255, cv2.NORM_MINMAX)
                    typical_image_uint8 = typical_image_normalized.astype(np.uint8)

                    # Define the path to save the typical PNG symbol
                    typical_image_filename = f"{symbol}_typical_cluster_{idx}.png"
                    typical_image_path = os.path.join(category_typical_png_dir, typical_image_filename)
                    cv2.imwrite(typical_image_path, typical_image_uint8)
                    print(f"Typical symbol saved for symbol '{symbol}' cluster {idx} at '{typical_image_path}'")

                    # ----- Generate SVG for the typical symbol -----
                    # Define SVG filename
                    typical_svg_filename = f"{symbol}_typical_cluster_{idx}.svg"
                    typical_svg_path = os.path.join(category_typical_svg_dir, typical_svg_filename)

                    try:
                        # Convert bitmap to paths
                        inner_paths, outer_paths = bitmap_to_paths(typical_image_path)
                        #plot_paths(inner_paths[1:],outer_paths[1:],"symbol")  # Exclude the first inner path if it's redundant
                        # Generate SVG
                        Paths_to_svg(
                            file_name=typical_svg_path,
                            outer_paths=inner_paths,
                            inner_paths=outer_paths[1:],
                            canvas_size=(1000, 1000),
                            view_box="0 0 1000 1000"
                        )
                        print(f"SVG file saved for symbol '{symbol}' cluster {idx} at '{typical_svg_path}'")
                    except Exception as e:
                        print(f"Error generating SVG for '{typical_image_path}': {e}")
                        continue
                    # ----- End of SVG Generation -----

    print("Typical symbol generation using K-Means clustering and SVG creation completed.")

def get_image_shape(directory: str) -> Tuple[int, int]:
    """
    Retrieves the shape of the first image in the given directory.

    Parameters:
    - directory (str): Directory containing images.

    Returns:
    - Tuple[int, int]: (height, width) of the image.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    return img.shape
    return (28, 28)  # Default fallback

# ----------------------------- Main Processing -----------------------------

def main():
    # Number of samples per class
    N = 300  # Adjust this value as needed

    # Load EMNIST data
    all_images, all_labels = load_emnist_csv(train_csv, test_csv)

    # Load label to character mapping
    label_mapping = load_mapping(mapping_file)

    # Get unique labels
    unique_labels = np.unique(all_labels)
    print(f"Total unique classes found: {len(unique_labels)}")

    # Group indices by label
    label_to_indices = {label: np.where(all_labels == label)[0] for label in unique_labels}

    # Initialize a dictionary to store sampled indices per label
    sampled_indices_per_label = {}

    # Determine sampling for each label
    for label in unique_labels:
        indices = label_to_indices[label]
        num_available = len(indices)
        if num_available < N:
            print(f"Warning: Class '{label_to_char(label, label_mapping)}' (Label: {label}) has only {num_available} samples. Sampling all available.")
            sampled_indices = indices.tolist()
        else:
            sampled_indices = random.sample(list(indices), N)
        sampled_indices_per_label[label] = sampled_indices
        print(f"Selected {len(sampled_indices)} samples for class '{label_to_char(label, label_mapping)}' (Label: {label})")

    # Total number of images to process
    total_images = sum(len(indices) for indices in sampled_indices_per_label.values())
    print(f"Total images to process: {total_images}")

    # Process each sampled image
    processed_count = 0
    for label, indices in sampled_indices_per_label.items():
        character = label_to_char(label, label_mapping)
        sanitized_character = sanitize_filename(character)
        if sanitized_character == '':
            sanitized_character = f"label_{label}"
        
        # Categorize the character
        category = categorize_character(character)
        # print(f"Character '{character}' categorized as '{category}'")

        # Define paths for the symbol's subdirectories in PNG and SVG directories
        symbol_png_dir = os.path.join(output_png_dir, category, sanitized_character)
        symbol_svg_dir = os.path.join(output_svg_dir, category, sanitized_character)

        # Create symbol subdirectories if they don't exist
        os.makedirs(symbol_png_dir, exist_ok=True)
        os.makedirs(symbol_svg_dir, exist_ok=True)

        for idx in indices:
            processed_count += 1
            print(f"\nProcessing image {processed_count}/{total_images} for class '{character}' (Label: {label})...")

            img_emnist = all_images[idx]
            
            # Determine the next available filenames for PNG and SVG
            png_filename = determine_filename(symbol_png_dir, "label", "png")
            svg_filename = determine_filename(symbol_svg_dir, "label", "svg")

            # Define full file paths
            png_filepath = os.path.join(symbol_png_dir, png_filename)
            svg_filepath = os.path.join(symbol_svg_dir, svg_filename)

            # Threshold the image to make it binary
            _, img_emnist_binary = cv2.threshold(img_emnist, 127, 255, cv2.THRESH_BINARY)

            # Resize the EMNIST image to fit into the canvas (e.g., scale it to 500x500)
            img_emnist_resized = cv2.resize(img_emnist_binary, (500, 500), interpolation=cv2.INTER_NEAREST)

            # ----- Apply Transformations -----
            # 1. Flip the image horizontally
            img_flipped = cv2.flip(img_emnist_resized, 1)  # 1 means horizontal flip

            # 2. Rotate the image 90 degrees anti-clockwise
            img_transformed = cv2.rotate(img_flipped, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # ----- End of Transformations -----

            # Create a 1000x1000 canvas
            canvas_size = 1000
            img_canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

            # Place the transformed EMNIST image onto the center of the canvas
            start_x = (img_canvas.shape[1] - img_transformed.shape[1]) // 2
            start_y = (img_canvas.shape[0] - img_transformed.shape[0]) // 2
            img_canvas[start_y:start_y+img_transformed.shape[0], start_x:start_x+img_transformed.shape[1]] = img_transformed
            
            # Save the transformed image (PNG)
            cv2.imwrite(png_filepath, img_canvas)
            print(f"Transformed image saved to '{png_filepath}'")

            # Extract paths from the bitmap
            try:
                outer_paths, inner_paths = bitmap_to_paths(png_filepath)
            except Exception as e:
                print(f"Error extracting paths from '{png_filepath}': {e}")
                continue

            # Plot the paths (optional, can be commented out if processing many images)
            # plot_paths(outer_paths, inner_paths[1:], character)  # Exclude the first inner path if it's redundant

            # Generate the SVG with updated canvas size and view box, named after the character label
            try:
                Paths_to_svg(
                    file_name=svg_filepath,
                    outer_paths=outer_paths,
                    inner_paths=inner_paths[1:],  # Exclude the first inner path if it's redundant
                    canvas_size=(canvas_size, canvas_size),
                    view_box="0 0 1000 1000"
                )
                print(f"SVG file saved to '{svg_filepath}'")
            except Exception as e:
                print(f"Error generating SVG for '{png_filepath}': {e}")
                continue

    print(f"\nProcessing of {total_images} images completed. All files are saved in '{output_base_dir}' directory.")

    # ----- Generate Typical Symbols Using K-Means -----
    print("\nStarting typical symbol generation with SVG creation...")

    # Define the directory to save typical SVGs
    output_typical_svg_dir = os.path.join(output_base_dir, 'typical_symbols_svg')

    # Generate typical symbols and their SVGs
    generate_typical_symbols(output_png_dir, output_typical_dir, output_typical_svg_dir)
    print(f"Typical symbols and their SVGs are saved in '{output_typical_dir}' and '{output_typical_svg_dir}' directories.")

    print("\nAll processing completed successfully.")

if __name__ == "__main__":
    main()
