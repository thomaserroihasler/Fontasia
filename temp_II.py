import os
from PIL import Image
import matplotlib.pyplot as plt

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
    plt.show()

# Example usage
folder_path = "letters"  # Replace with your folder path
output_folder = "single_letters"  # Folder to save narrow images
generate_image_width_histogram_and_save_narrow_images(folder_path, output_folder)
