import cv2
import numpy as np

def generate_report(original_path, highlighted_path):
    # Step 1: Load the images
    original = cv2.imread(original_path)
    highlighted = cv2.imread(highlighted_path)
    
    # Step 2: Check if images have the same dimensions
    if original.shape != highlighted.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Step 3: Compute the absolute difference
    diff = cv2.absdiff(original, highlighted)
    
    # Step 4: Convert to grayscale and threshold to create a binary mask
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    
    # Step 5: Find contours in the mask (each contour is a difference)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 6 & 7: Classify and count differences
    H = original.shape[0]  # Image height
    top_count = 0
    middle_count = 0
    bottom_count = 0
    min_area = 100  # Minimum area to filter out noise (adjust as needed)
    
    for contour in contours:
        if cv2.contourArea(contour) > min_area:  # Ignore small contours
            # Compute the centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cY = int(M["m01"] / M["m00"])  # y-coordinate of centroid
                # Classify into top, middle, or bottom
                if cY < H / 3:
                    top_count += 1
                elif cY < 2 * H / 3:
                    middle_count += 1
                else:
                    bottom_count += 1
    
    # Step 8: Generate the report
    report = []
    if top_count > 0:
        report.append(f"{top_count} differences in the top")
    if middle_count > 0:
        report.append(f"{middle_count} in the middle")
    if bottom_count > 0:
        report.append(f"{bottom_count} in the bottom")
    
    if not report:
        return "No differences found."
    else:
        return "There are " + " and ".join(report) + "."

# Example usage with .webp files
report = generate_report("original.webp", "updated.webp")
print(report)