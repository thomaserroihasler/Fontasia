import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_cluster_image(cluster_image_path, output_folder, global_letter_counter, padding=10, threshold_ratio=0.3):
    """
    Processes a single cluster image to extract individual letters and save them.

    Parameters:
    - cluster_image_path: Path to the cluster image.
    - output_folder: Directory where individual letters will be saved.
    - global_letter_counter: A mutable integer (list with one element) to keep track of letter numbering.
    - padding: Number of pixels to pad around each letter.
    - threshold_ratio: Ratio to determine the space threshold based on max vertical density.
    """
    # Load the cluster image in grayscale
    image = cv2.imread(cluster_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image '{cluster_image_path}'. Skipping.")
        return

    # Binarize the image (ensure it's binary)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Extract the line image (assuming the entire image is a line)
    line_image = binary_image.copy()
    
    # Calculate center of mass for each column in the line
    cols = line_image.shape[1]
    center_of_mass = []
    
    for x in range(cols):
        column = line_image[:, x]
        non_black_pixels = np.where(column != 0)[0]
        if non_black_pixels.size > 0:
            com = np.mean(non_black_pixels)
        else:
            com = np.nan
        center_of_mass.append(com)
    
   
    # Interpolate missing values in center of mass
    center_of_mass = np.array(center_of_mass)
    nans = np.isnan(center_of_mass)
    not_nans = ~nans
    if not_nans.sum() == 0:
        # If all values are NaN, set COM to center of the image
        center_of_mass[:] = line_image.shape[0] / 2
    else:
        center_of_mass[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), center_of_mass[not_nans])

    # Smooth the center of mass curve
    window_size = 20
    kernel = np.ones(window_size) / window_size
    center_of_mass_smooth = np.convolve(center_of_mass, kernel, mode='same')

    # Calculate vertical density perpendicular to the curve

    vertical_density = []
    for x in range(cols):
        window = line_image[:, x]
        density = np.sum(window != 0)
        vertical_density.append(density)

    vertical_density = np.array(vertical_density)
    window_size = 5
    kernel = np.ones(window_size) / window_size
    vertical_density_smooth = np.convolve(vertical_density, kernel, mode='same')

    # Identify spaces between letters based on smoothed vertical density
    threshold = np.max(vertical_density_smooth) * threshold_ratio  # Adjusted threshold
    spaces = vertical_density_smooth < threshold

    print(vertical_density)
    plt.figure(figsize=(10, 5))
    plt.imshow(line_image, cmap='gray', origin='upper')  # Set origin to 'lower'
    plt.plot(spaces, color='red')
    plt.title('Line Image with Center of Mass Overlay')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()

    space_indices = np.where(spaces)[0]

    # Step 2: Group consecutive indices into sequences
    consecutive_groups = np.split(space_indices, np.where(np.diff(space_indices) > 1)[0] + 1)

    # Step 3: Compute the average index for each group of spaces
    average_indices = [np.mean(group) for group in consecutive_groups]
    # Group columns into letters
    letter_indices = np.where(~spaces)[0]
    height, width = line_image.shape[:2]  # Get dimensions of the image

    # Step 1: Round indices
    split_indices = [round(idx) for idx in average_indices]

    # Step 2: Add start and end of the image as splitting points
    split_indices = [0] + split_indices + [width]

    # Step 3: Split the image
    split_images = []
    for i in range(len(split_indices) - 1):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1]
        sub_image = line_image[:, start_idx:end_idx]  # Slice the image along the x-axis
        split_images.append(sub_image)

    split_images = [img for img in split_images if np.any(img)]  # Keep only non-empty images

    # Define the output directory
    output_dir = "letters_split/"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Example list of split images (Replace this with your actual list)
    split_images = [img for img in split_images if np.any(img)]  # Filter out empty images

    for img in split_images:
        # Step 1: List existing files in the output directory
        existing_files = [f for f in os.listdir(output_dir) if f.startswith("letter_") and f.endswith(".png")]
        
        # Step 2: Extract the indices from existing filenames
        existing_indices = [int(f.split("_")[1].split(".")[0]) for f in existing_files]
        
        # Step 3: Find the lowest unused index
        N = 0
        while N in existing_indices:
            N += 1  # Increment until a free index is found

        # Step 4: Save the image with the computed filename
        cv2.imwrite(os.path.join(output_dir, f"letter_{N}.png"), img)

def main_split_letters(clusters_folder='clusters_complex', output_letters_folder='letters_split', padding=10):
    """
    Processes all cluster images to extract individual letters and save them in the same folder.

    Parameters:
    - clusters_folder: Directory containing cluster images.
    - output_letters_folder: Directory where individual letters will be saved.
    - padding: Number of pixels to pad around each letter.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_letters_folder, exist_ok=True)

    # Initialize a global letter counter
    global_letter_counter = [1]  # Using list to allow mutability in the function

    # Iterate through each cluster image
    for filename in os.listdir(clusters_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            cluster_image_path = os.path.join(clusters_folder, filename)
            print(f"Processing {cluster_image_path}...")
            process_cluster_image(
                cluster_image_path,
                output_folder=output_letters_folder,
                global_letter_counter=global_letter_counter,
                padding=padding,
                threshold_ratio=0.2  # Adjust based on your data
            )
    print(f"All letters have been saved in '{output_letters_folder}' folder.")

if __name__ == "__main__":
    # Define the input and output directories
    clusters_folder = "clusters"      # Adjust based on your clustering output folder
    output_letters_folder = "letters_split"

    # Define padding
    padding = 10  # Adjust as needed

    # Process and split letters
    main_split_letters(
        clusters_folder=clusters_folder,
        output_letters_folder=output_letters_folder,
        padding=padding
    )
