import cv2
import numpy as np
from typing import List, Tuple
import os
import argparse
from mistralai.client import MistralClient  # Updated import
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Initialize Mistral AI client (only when needed)
client = None

def merge_nearby_boxes(boxes: List[Tuple[int, int, int, int]], max_distance: int = 50) -> List[Tuple[int, int, int, int]]:
    """Merge boxes that are close to each other"""
    if not boxes:
        return []
    
    # Convert to numpy array for easier manipulation
    boxes = np.array(boxes)
    merged = []
    used = set()

    for i in range(len(boxes)):
        if i in used:
            continue

        current_box = list(boxes[i])
        used.add(i)
        
        merged_with_others = True
        while merged_with_others:
            merged_with_others = False
            
            for j in range(len(boxes)):
                if j in used or i == j:
                    continue

                # Calculate the distance between boxes
                x1, y1, w1, h1 = current_box
                x2, y2, w2, h2 = boxes[j]
                
                # Calculate centers
                center1 = (x1 + w1/2, y1 + h1/2)
                center2 = (x2 + w2/2, y2 + h2/2)
                
                # Calculate Euclidean distance between centers
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                if distance < max_distance:
                    # Merge boxes
                    x = min(x1, x2)
                    y = min(y1, y2)
                    w = max(x1 + w1, x2 + w2) - x
                    h = max(y1 + h1, y2 + h2) - y
                    current_box = [x, y, w, h]
                    used.add(j)
                    merged_with_others = True

        merged.append(tuple(current_box))
    
    return merged

def create_difference_visualization(original: np.ndarray, differences: dict, output_path: str):
    """Create a visual representation of the differences with numbered labels"""
    visualization = original.copy()
    colors = {
        "minor": (0, 255, 0),      # Green
        "significant": (0, 165, 255),  # Orange
        "major": (0, 0, 255)       # Red
    }
    
    diff_number = 1  # To label differences sequentially
    for section, diffs in differences.items():
        for _, (x, y, w, h) in diffs:  # Ignore area since severity is in report
            severity = "minor" if w * h < 1000 else "significant" if w * h < 5000 else "major"
            color = colors[severity]
            cv2.rectangle(visualization, (x, y), (x + w, y + h), color, 2)
            # Add numbered label
            cv2.putText(visualization, str(diff_number), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
            diff_number += 1
    
    cv2.imwrite(output_path, visualization)
    return visualization

def get_difference_description(original_crop: np.ndarray, updated_crop: np.ndarray) -> str:
    """Send cropped images to Pixtral and get a description of the difference"""
    global client
    if client is None:
        client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
        
    try:
        # Convert images to base64
        def image_to_base64(img_array):
            img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

        original_b64 = image_to_base64(original_crop)
        updated_b64 = image_to_base64(updated_crop)

        # Create the message for Pixtral
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Compare the first image (original) to the second image (updated). In one sentence, under 50 words, summarize what has been added, removed, or changed in the updated image, focusing only on significant changes."
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{original_b64}"
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{updated_b64}"
                    }
                ]
            }
        ]

        # Get the chat response using the correct method
        chat_response = client.chat(
            model="pixtral-12b-2409",
            messages=messages,
            max_tokens=300
        )
        
        return chat_response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error analyzing difference: {str(e)}"

def analyze_differences(original_path: str, comparison_path: str, generate_report: bool = False, threshold: int = 50,  max_distance: int = 50) -> str:
    # Load images
    original = cv2.imread(original_path)
    comparison = cv2.imread(comparison_path)
    
    if original is None or comparison is None:
        raise ValueError("Failed to load one or both images")
    
    if original.shape != comparison.shape:
        raise ValueError("Images must have the same dimensions")
    
    height, width = original.shape[:2]
    
    # Difference detection with enhanced sensitivity
    color_diff = cv2.absdiff(original, comparison)
    color_diff_gray = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)
    _, color_mask = cv2.threshold(color_diff_gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Enhanced text difference detection
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    comparison_gray = cv2.cvtColor(comparison, cv2.COLOR_BGR2GRAY)
    
    # Use multiple thresholds for better text detection
    original_thresh1 = cv2.adaptiveThreshold(original_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
    comparison_thresh1 = cv2.adaptiveThreshold(comparison_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
    
    original_thresh2 = cv2.adaptiveThreshold(original_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, 21, 2)
    comparison_thresh2 = cv2.adaptiveThreshold(comparison_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY, 21, 2)
    
    # Combine different thresholding results
    struct_diff1 = cv2.absdiff(original_thresh1, comparison_thresh1)
    struct_diff2 = cv2.absdiff(original_thresh2, comparison_thresh2)
    struct_diff = cv2.bitwise_or(struct_diff1, struct_diff2)
    
    # Combine all differences
    combined_diff = cv2.bitwise_or(color_mask, struct_diff)
    
    # Find initial contours
    contours, _ = cv2.findContours(combined_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get initial bounding boxes with lower minimum area
    min_area = 25  # Reduced minimum area to catch smaller differences
    initial_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            # Add additional check for minimum dimensions
            if w >= 3 and h >= 3:  # Minimum 3x3 pixels to filter out noise
                initial_boxes.append((x, y, w, h))
    
    # Merge nearby boxes
    merged_boxes = merge_nearby_boxes(initial_boxes, max_distance=max_distance)
    
    # Organize differences by section
    differences_dict = {"top": [], "middle": [], "bottom": []}
    crops_dict = {"top": [], "middle": [], "bottom": []}
    
    padding = 20  # Padding for crops
    for x, y, w, h in merged_boxes:
        area = w * h
        center_y = y + h // 2
        
        # Define section
        section = "top" if center_y < height//3 else "middle" if center_y < 2*height//3 else "bottom"
        differences_dict[section].append((area, (x, y, w, h)))
        
        if generate_report:
            # Only create crops if we're generating a detailed report
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(width, x + w + padding)
            y2 = min(height, y + h + padding)
            original_crop = original[y1:y2, x1:x2]
            comparison_crop = comparison[y1:y2, x1:x2]
            crops_dict[section].append((original_crop, comparison_crop))
    
    # Create visualization
    if any(differences_dict.values()):
        create_difference_visualization(original, differences_dict, "differences_highlighted.webp")
    
    if not any(differences_dict.values()):
        return "No significant differences found between the pages."
    
    # Generate basic report
    basic_report = []
    for section in ["top", "middle", "bottom"]:
        if differences_dict[section]:
            count = len(differences_dict[section])
            basic_report.append(f"{count} difference{'s' if count > 1 else ''} in the {section}")
    
    summary = "Found " + ", ".join(basic_report) + ". Check 'differences_highlighted.webp' for visual details."
    
    if not generate_report:
        return summary
    
    # Generate detailed report with LLM descriptions
    detailed_report = []
    diff_number = 1
    
    for section in ["top", "middle", "bottom"]:
        if differences_dict[section]:
            section_diffs = []
            for i, (area, _) in enumerate(differences_dict[section]):
                original_crop, updated_crop = crops_dict[section][i]
                description = get_difference_description(original_crop, updated_crop)
                section_diffs.append(f"Difference {diff_number}: {description}")
                diff_number += 1
            detailed_report.append(f"In the {section} section: " + "; ".join(section_diffs))
    
    return summary + "\n\nDetailed changes: " + " ".join(detailed_report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare two screenshots and highlight differences')
    parser.add_argument('original', help='Path to the original screenshot')
    parser.add_argument('comparison', help='Path to the comparison screenshot')
    parser.add_argument('-r', '--report', action='store_true', help='Generate detailed report using Pixtral')
    parser.add_argument('-t', '--threshold', type=int, default=30, help='Threshold for difference detection (default: 30)')
    parser.add_argument('-d', '--distance', type=int, default=300, help='Maximum distance between differences to be merged (default: 300)')
    
    args = parser.parse_args()
    
    try:
        result = analyze_differences(
            args.original,
            args.comparison,
            generate_report=args.report,
            threshold=args.threshold,
            max_distance=args.distance
        )
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}") 