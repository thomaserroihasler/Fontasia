import math
import svgwrite

def Paths_to_svg(file_name, outer_paths, inner_paths, canvas_size, view_box):
    """
    Create an SVG file with specified outer and inner paths for use as a glyph in FontForge.

    Parameters:
    - file_name (str): The name of the SVG file to create.
    - outer_paths (List[List[Tuple[float, float]]]): A list of outer paths, each defined as a list of (x, y) tuples.
      These paths should be defined in clockwise order.
    - inner_paths (List[List[Tuple[float, float]]]): A list of inner paths (holes), each defined as a list of (x, y) tuples.
      These paths should be defined in counterclockwise order.
    - canvas_size (Tuple[int, int], optional): The size of the SVG canvas. Defaults to (1000, 1000).
    - view_box (str, optional): The viewBox attribute for the SVG. Defaults to "0 0 1000 1000".

    Raises:
    - ValueError: If both outer_paths and inner_paths are empty or if outer_paths is empty.
    """
    if not outer_paths and not inner_paths:
        raise ValueError("Both outer_paths and inner_paths are empty. At least one outer path is required.")
    elif not outer_paths:
        raise ValueError("outer_paths is empty. At least one outer path is required.")

    # Create the SVG drawing
    dwg = svgwrite.Drawing(file_name, size=canvas_size, viewBox=view_box)

    # Add all outer paths (filled with black) if any
    if outer_paths:
        for idx, outer in enumerate(outer_paths):
            if not outer:
                print(f"Warning: Outer path {idx + 1} is empty and will be skipped.")
                continue
            polygon = dwg.polygon(points=outer, fill="black", stroke="none")
            dwg.add(polygon)
            print(f"Added outer path {idx + 1} with {len(outer)} points.")

    # Add all inner paths (holes, filled with white) if any
    if inner_paths:
        for idx, inner in enumerate(inner_paths):
            if not inner:
                print(f"Warning: Inner path {idx + 1} is empty and will be skipped.")
                continue
            polygon = dwg.polygon(points=inner, fill="white", stroke="none")
            dwg.add(polygon)
            print(f"Added inner path {idx + 1} with {len(inner)} points.")

    # Save the SVG file
    dwg.save()
    print(f"SVG file '{file_name}' created successfully.")