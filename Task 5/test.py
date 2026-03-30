import cv2
import numpy as np

def debug_obstacle_extraction(image_path, thresh_values):
    # 1. Load the map image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not find image at {image_path}")
        return
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    def show_threshold(thresh_val):
        # Create a mask where pixels are strictly darker than the threshold
        # This focuses only on the darkest pixels (the walls).
        mask = (img_gray < thresh_val).astype(np.uint8) * 255
        
        # Combine the mask with the original image to easily see the extracted features
        res = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        
        # Add the threshold value to the image
        text = f"Threshold < {thresh_val}"
        cv2.putText(res, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return res

    # 2. Visualize with different thresholds to find the right one
    vis_1 = show_threshold(100)  # Our previous guess - let's see what it really does
    vis_2 = show_threshold(215)  # Much stricter, might isolate just the black
    vis_3 = show_threshold(200)  # Extreeemely strict, might miss parts of the walls

    # 3. Stack images to easily compare
    combined = np.hstack((vis_1, vis_2, vis_3))
    cv2.imwrite('obstacle_debug.png', combined)
    print("Saved 'obstacle_debug.png'. Please visually inspect this image.")

# Execute the debugging function on your map file
debug_obstacle_extraction('aces_relations.png', [100, 215, 200])