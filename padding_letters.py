import os
from PIL import Image, ImageOps

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

# Example usage
folder_path = "single_letters"  # Replace with your folder path
output_folder = "padded_letters"  # Folder to save padded square images
pad_images_to_max_height_square(folder_path, output_folder, pad_color=0)  # Use 0 for black background
