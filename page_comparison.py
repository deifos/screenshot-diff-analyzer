import cv2
import numpy as np
from typing import List, Tuple

def create_difference_visualization(original: np.ndarray, differences: dict, output_path: str):
    """Create a visual representation of the differences"""
    # Create a copy of the original image for visualization
    visualization = original.copy()
    
    # Color scheme for different severities (BGR format)
    colors = {
        "minor": (0, 255, 0),      # Green
        "significant": (0, 165, 255),  # Orange
        "major": (0, 0, 255)       # Red
    }
    
    # Draw rectangles for each difference
    for section, diffs in differences.items():
        for area, (x, y, w, h) in diffs:
            # Determine severity based on area
            severity = "minor" if area < 1000 else "significant" if area < 5000 else "major"
            color = colors[severity]
            
            # Draw rectangle
            cv2.rectangle(visualization, (x, y), (x + w, y + h), color, 2)
            
            # Add label with severity
            label = f"{severity}"
            cv2.putText(visualization, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1, cv2.LINE_AA)
    
    # Save the visualization
    cv2.imwrite(output_path, visualization)
    return visualization

def analyze_differences(original_path: str, comparison_path: str, threshold: int = 30) -> str:
    # Load images
    original = cv2.imread(original_path)
    comparison = cv2.imread(comparison_path)
    
    if original is None or comparison is None:
        raise ValueError("Failed to load one or both images")
    
    # Ensure same dimensions
    if original.shape != comparison.shape:
        raise ValueError("Images must have the same dimensions")
    
    height, width = original.shape[:2]
    
    # Initialize difference trackers
    differences = []
    
    # 1. Check for color differences
    color_diff = cv2.absdiff(original, comparison)
    color_diff_gray = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)
    _, color_mask = cv2.threshold(color_diff_gray, threshold, 255, cv2.THRESH_BINARY)
    
    # 2. Convert to grayscale for structural/text differences
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    comparison_gray = cv2.cvtColor(comparison, cv2.COLOR_BGR2GRAY)
    
    # 3. Apply adaptive thresholding for better text detection
    original_thresh = cv2.adaptiveThreshold(original_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
    comparison_thresh = cv2.adaptiveThreshold(comparison_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    
    # 4. Find structural differences
    struct_diff = cv2.absdiff(original_thresh, comparison_thresh)
    
    # 5. Combine color and structural differences
    combined_diff = cv2.bitwise_or(color_mask, struct_diff)
    
    # 6. Find contours of differences
    contours, _ = cv2.findContours(combined_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and classify differences
    min_area = 50  # Minimum area to consider as a real difference
    differences_dict = {
        "top": [],
        "middle": [],
        "bottom": []
    }
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            center_y = y + h//2
            
            # Classify position
            if center_y < height//3:
                differences_dict["top"].append((area, (x, y, w, h)))
            elif center_y < 2*height//3:
                differences_dict["middle"].append((area, (x, y, w, h)))
            else:
                differences_dict["bottom"].append((area, (x, y, w, h)))
    
    # Generate detailed report
    report_parts = []
    
    def describe_differences(diffs: List[Tuple[float, tuple]], section: str) -> str:
        if not diffs:
            return ""
        count = len(diffs)
        total_area = sum(area for area, _ in diffs)
        severity = "minor" if total_area < 1000 else "significant" if total_area < 5000 else "major"
        return f"{count} {severity} {'difference' if count == 1 else 'differences'} in the {section}"
    
    if differences_dict["top"]:
        report_parts.append(describe_differences(differences_dict["top"], "top"))
    if differences_dict["middle"]:
        report_parts.append(describe_differences(differences_dict["middle"], "middle"))
    if differences_dict["bottom"]:
        report_parts.append(describe_differences(differences_dict["bottom"], "bottom"))
    
    # Create visualization if differences were found
    if any(differences_dict.values()):
        create_difference_visualization(original, differences_dict, "differences_highlighted.webp")
    
    if not report_parts:
        return "No significant differences found between the pages."
    
    return "Found " + " and ".join(report_parts) + ". Check 'differences_highlighted.webp' for visual details."

if __name__ == "__main__":
    # Example usage
    try:
        result = analyze_differences("drupal-screenshot1-1200.webp", "drupal-screenshot2-1200.webp")
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}") 