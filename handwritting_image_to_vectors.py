import os
import cv2
import numpy as np

# Define the output folder for letters
output_folder = "letters"
os.makedirs(output_folder, exist_ok=True)

# Load the image in grayscale
image = cv2.imread('notes.png', cv2.IMREAD_GRAYSCALE)

# Invert the image so that text is black and background is white
inverted_image = cv2.bitwise_not(image)

# Binarize the image (thresholding)
_, binary_image = cv2.threshold(inverted_image, 127, 255, cv2.THRESH_BINARY)

# Compute horizontal projection to find text lines
horizontal_projection = np.sum(binary_image == 0, axis=1)

# Identify indices where there are text pixels
line_indices = np.where(horizontal_projection > 0)[0]

# Group indices into lines
lines = []
if len(line_indices) > 0:
    start_idx = line_indices[0]
    for i in range(1, len(line_indices)):
        if line_indices[i] != line_indices[i - 1] + 1:
            end_idx = line_indices[i - 1]
            lines.append((start_idx, end_idx))
            start_idx = line_indices[i]
    lines.append((start_idx, line_indices[-1]))

# Process each line to extract letters
for idx, (start_row, end_row) in enumerate(lines):
    line_image = binary_image[start_row:end_row + 1, :]

    # Calculate center of mass for each column in the line
    cols = line_image.shape[1]
    center_of_mass = []
    for x in range(cols):
        column = line_image[:, x]
        black_pixels = np.where(column == 0)[0]
        if black_pixels.size > 0:
            com = np.mean(black_pixels)
        else:
            com = np.nan
        center_of_mass.append(com)

    # Interpolate missing values in center of mass
    center_of_mass = np.array(center_of_mass)
    nans = np.isnan(center_of_mass)
    not_nans = ~nans
    center_of_mass[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), center_of_mass[not_nans])

    # Smooth the center of mass curve
    window_size = 5
    kernel = np.ones(window_size) / window_size
    center_of_mass_smooth = np.convolve(center_of_mass, kernel, mode='same')

    # Calculate vertical density perpendicular to the curve
    window_height = 15
    vertical_density = []
    for x in range(cols):
        y = int(center_of_mass_smooth[x])
        y_start = max(0, y - window_height // 2)
        y_end = min(line_image.shape[0], y + window_height // 2)
        window = line_image[y_start:y_end, x]
        density = np.sum(window == 0)
        vertical_density.append(density)

    vertical_density = np.array(vertical_density)

    # Smooth the vertical density curve
    density_window_size = 2  # Define the size of the moving average window
    density_kernel = np.ones(density_window_size) / density_window_size
    vertical_density_smooth = np.convolve(vertical_density, density_kernel, mode='same')

    # Identify spaces between letters based on smoothed vertical density
    threshold = np.max(vertical_density_smooth) * 0.3  # Adjusted threshold
    spaces = vertical_density_smooth < threshold

    # Group columns into letters
    letter_indices = np.where(~spaces)[0]
    letters = []
    if letter_indices.size > 0:
        start_col = letter_indices[0]
        for i in range(1, len(letter_indices)):
            if letter_indices[i] != letter_indices[i - 1] + 1:
                end_col = letter_indices[i - 1]
                letters.append((start_col, end_col))
                start_col = letter_indices[i]
        letters.append((start_col, letter_indices[-1]))

    # Save each letter as an image in the folder
    for letter_idx, (start_col, end_col) in enumerate(letters):
        letter_image = line_image[:, start_col:end_col + 1]
        letter_path = os.path.join(output_folder, f'letter_line{idx}_letter{letter_idx}.png')
        cv2.imwrite(letter_path, letter_image)
