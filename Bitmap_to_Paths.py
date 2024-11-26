import cv2
import svgwrite
import numpy as np
import math
from typing import List, Tuple

import cv2
import numpy as np
from typing import List, Tuple

def bitmap_to_paths(bitmap_path: str) -> Tuple[List[List[Tuple[float, float]]], List[List[Tuple[float, float]]]]:
    """
    Convert a bitmap image to outer and inner paths suitable for SVG creation.

    Parameters:
    - bitmap_path (str): Path to the input bitmap image (e.g., PNG, JPG).

    Returns:
    - Tuple[List[List[Tuple[float, float]]], List[List[Tuple[float, float]]]]:
      A tuple containing two lists:
        - Outer paths: List of outer contours (each contour is a list of (x, y) tuples).
        - Inner paths: List of inner contours (holes) (each contour is a list of (x, y) tuples).
    """
    # Load the image in grayscale
    img = cv2.imread(bitmap_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {bitmap_path}")

    # Threshold the image to binary (invert if necessary)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours with hierarchy using RETR_TREE
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        raise ValueError("No contours found in the image.")

    hierarchy = hierarchy[0]  # Simplify hierarchy

    outer_paths = []
    inner_paths = []

    def get_contour_depth(idx, hier):
        depth = 0
        parent = hier[idx][3]
        while parent != -1:
            depth += 1
            parent = hier[parent][3]
        return depth

    for i, contour in enumerate(contours):
        depth = get_contour_depth(i, hierarchy)
        contour_points = contour.squeeze().tolist()
        if isinstance(contour_points[0], int):
            contour_points = [tuple(contour_points)]
        if depth % 2 == 0:
            # Even depth, outer contour
            outer_paths.append(contour_points)
        else:
            # Odd depth, inner contour (hole)
            contour_points = contour_points[::-1]  # Reverse to ensure counterclockwise
            inner_paths.append(contour_points)

    return inner_paths,outer_paths


