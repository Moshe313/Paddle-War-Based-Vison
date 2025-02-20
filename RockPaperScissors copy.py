import cv2
import numpy as np
import random
import time

# ====================================
# Configuration Dictionary (Customize)
# ====================================
config = {
    "window_name": "Rock Paper Scissors",
    "window_size": (800, 600),            # (width, height) for the window (not the frame itself)
    "player1_name": "Alice",
    "player2_name": "Bob",
    # ROIs are defined as (x1, y1, x2, y2) -- ROI is extracted as frame[y1:y2, x1:x2]
    "roi1_coords": (50, 50, 250, 250),    # Player 1's ROI location
    "roi2_coords": (300, 50, 500, 250),   # Player 2's ROI location
    "countdown_duration": 3,              # Countdown in seconds
    "result_display_time": 5,             # Time in seconds to display the result
    "debug": True                       # Set to True to enable debug windows
}

# ====================================
# Helper Functions
# ====================================
def decide_winner(gesture1, gesture2):
    """
    Determines the winner based on the two gestures.
    Returns "Tie" if both gestures are the same.
    Otherwise, applies the rules:
      - Rock beats Scissors
      - Scissors beats Paper
      - Paper beats Rock
    """
    rules = {"Rock": "Scissors", "Scissors": "Paper", "Paper": "Rock"}
    if gesture1 not in rules or gesture2 not in rules:
        return "Unknown"
    elif gesture1 == gesture2:
        return "Tie"
    elif rules.get(gesture1) == gesture2:
        return f"{config['player1_name']} wins!"
    else:
        return f"{config['player2_name']} wins!"

def get_gesture(roi, debug=False, debug_window_name="Debug ROI"):
    """
    Processes the provided ROI to detect the hand gesture without using fixed thresholds.
    This method uses k-means clustering in the LAB color space to automatically segment the hand.
    It assumes the hand is roughly centered in the ROI and selects the cluster whose centroid
    is closest to the ROI center as the hand region. After obtaining a cleaned binary mask of the hand,
    it uses contour and convexity defect analysis to classify the gesture.
    
    Returns one of "Rock", "Paper", "Scissors", or "Unknown".
    If debug is True, displays intermediate results.
    """
    # Convert ROI to LAB color space for better perceptual clustering.
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    
    # Reshape image into a 2D array of pixel values.
    pixel_values = roi_lab.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Perform k-means clustering with K=2 (hand vs. background).
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()
    
    # Reshape the labels to the original ROI dimensions.
    labels_2d = labels.reshape(roi.shape[:2])
    
    # Compute the centroid of each cluster.
    cluster_centroids = []
    for i in range(K):
        ys, xs = np.where(labels_2d == i)
        if len(xs) > 0:
            centroid = (np.mean(xs), np.mean(ys))
        else:
            centroid = (0, 0)
        cluster_centroids.append(centroid)
    
    # Assume the hand is near the center of the ROI.
    roi_center = (roi.shape[1] / 2, roi.shape[0] / 2)
    distances = [np.linalg.norm(np.array(c) - np.array(roi_center)) for c in cluster_centroids]
    hand_cluster = np.argmin(distances)
    
    # Create a binary mask for the selected cluster.
    mask = np.uint8((labels_2d == hand_cluster) * 255)
    
    # Clean up the mask using morphological operations.
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours in the cleaned mask.
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    gesture = "Unknown"
    debug_roi = roi.copy()
    
    if contours:
        # Assume the largest contour corresponds to the hand.
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(debug_roi, [max_contour], -1, (0, 255, 0), 2)
        
        # Approximate the contour for smoother edges.
        epsilon = 0.0005 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        cv2.drawContours(debug_roi, [approx], -1, (0, 255, 255), 2)
        
        # Find convex hull and convexity defects.
        hull = cv2.convexHull(approx, returnPoints=False)
        if hull is not None and len(hull) > 3:
            defects = cv2.convexityDefects(approx, hull)
            count_defects = 0
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(approx[s][0])
                    end = tuple(approx[e][0])
                    far = tuple(approx[f][0])
                    cv2.circle(debug_roi, far, 4, (0, 0, 255), -1)
                    
                    # Calculate the angle using the cosine rule.
                    a = np.linalg.norm(np.array(end) - np.array(start))
                    b = np.linalg.norm(np.array(far) - np.array(start))
                    c = np.linalg.norm(np.array(end) - np.array(far))
                    if b * c != 0:
                        angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * 57
                        if angle <= 90:
                            count_defects += 1
            # Use heuristics based on the number of convexity defects.
            if count_defects == 0:
                gesture = "Rock"
            elif count_defects == 1:
                gesture = "Scissors"
            elif count_defects in [3, 4, 5, 6]:
                gesture = "Paper"
            else:
                gesture = "Unknown"
    
    if debug:
        cv2.imshow(debug_window_name, debug_roi)
        cv2.imshow(debug_window_name + " Mask", mask_clean)
    
    return gesture

# ====================================
# Main Application
# ====================================
def main():
    # Create a named window and set its size.
    cv2.namedWindow(config["window_name"], cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config["window_name"], config["window_size"][0], config["window_size"][1])

    cap = cv2.VideoCapture(0)
    result_text = None
    result_end_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Extract ROIs for the players using configuration coordinates.
        x1, y1, x2, y2 = config["roi1_coords"]
        roi1 = frame[y1:y2, x1:x2]
        x1_p2, y1_p2, x2_p2, y2_p2 = config["roi2_coords"]
        roi2 = frame[y1_p2:y2_p2, x1_p2:x2_p2]

        # Draw rectangles around the ROIs.
        cv2.rectangle(frame, (config["roi1_coords"][0], config["roi1_coords"][1]),
                             (config["roi1_coords"][2], config["roi1_coords"][3]), (255, 0, 0), 2)
        cv2.rectangle(frame, (config["roi2_coords"][0], config["roi2_coords"][1]),
                             (config["roi2_coords"][2], config["roi2_coords"][3]), (0, 255, 0), 2)

        # In debug mode, get live gesture info and display it.
        if config["debug"]:
            gesture1_live = get_gesture(roi1, debug=True, debug_window_name="Debug ROI 1")
            gesture2_live = get_gesture(roi2, debug=True, debug_window_name="Debug ROI 2")
            cv2.putText(frame, f"{config['player1_name']}: {gesture1_live}", 
                        (config["roi1_coords"][0], config["roi1_coords"][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"{config['player2_name']}: {gesture2_live}", 
                        (config["roi2_coords"][0], config["roi2_coords"][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Overlay result text if active.
        if result_text is not None and time.time() < result_end_time:
            cv2.putText(frame, result_text, (50, frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            result_text = None

        cv2.imshow(config["window_name"], frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('g'):
            # --------------------------
            # Countdown (Non-Blocking)
            # --------------------------
            countdown_start = time.time()
            countdown_duration = config["countdown_duration"]
            while time.time() - countdown_start < countdown_duration:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                elapsed = time.time() - countdown_start
                countdown_number = int(countdown_duration - elapsed) + 1
                cv2.putText(frame, str(countdown_number), 
                            (frame.shape[1] // 2 - 50, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                # Draw ROI rectangles during countdown.
                cv2.rectangle(frame, (config["roi1_coords"][0], config["roi1_coords"][1]),
                                     (config["roi1_coords"][2], config["roi1_coords"][3]), (255, 0, 0), 2)
                cv2.rectangle(frame, (config["roi2_coords"][0], config["roi2_coords"][1]),
                                     (config["roi2_coords"][2], config["roi2_coords"][3]), (0, 255, 0), 2)
                cv2.imshow(config["window_name"], frame)
                cv2.waitKey(30)

            # --------------------------
            # Capture Gestures After Countdown
            # --------------------------
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            x1, y1, x2, y2 = config["roi1_coords"]
            roi1 = frame[y1:y2, x1:x2]
            x1_p2, y1_p2, x2_p2, y2_p2 = config["roi2_coords"]
            roi2 = frame[y1_p2:y2_p2, x1_p2:x2_p2]
            gesture1 = get_gesture(roi1, config["debug"], debug_window_name="Debug ROI 1")
            gesture2 = get_gesture(roi2, config["debug"], debug_window_name="Debug ROI 2")
            winner = decide_winner(gesture1, gesture2)
            result_text = (f"{config['player1_name']}: {gesture1} | "
                           f"{config['player2_name']}: {gesture2} => {winner}")
            print("Result:", result_text)
            result_end_time = time.time() + config["result_display_time"]

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
