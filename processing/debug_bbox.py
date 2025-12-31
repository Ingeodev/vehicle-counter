
import cv2
import numpy as np

def find_date_bbox(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error reading image")
        return

    # Focus on top-left quadrant
    h, w = img.shape[:2]
    roi = img[0:100, 0:600] # Estimating header area
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find white text
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find bounding box covering all contours
    if not contours:
        print("No text found")
        return

    all_points = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    print(f"Bounding Box in ROI: x={x}, y={y}, w={w}, h={h}")
    # Expand slightly for safety
    print(f"Suggested Crop: x={x-5}, y={y-5}, w={w+10}, h={h+10}")

    # Visualize (save to check)
    cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite("bbox_check.jpg", roi)

if __name__ == "__main__":
    find_date_bbox("/home/miguel/.gemini/antigravity/brain/2abeda1c-4bac-42a1-8b9c-72f13d8f7580/uploaded_image_1767138727493.png")
