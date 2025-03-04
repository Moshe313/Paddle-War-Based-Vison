import cv2
import numpy as np
import time

# ====================================
# Configuration Dictionary (Customize)
# ====================================
config = {
    "window_name": "Rock Paper Scissors",
    "window_size": (800, 600),  # (width, height)
    "player1_name": "Alice",
    "player2_name": "Bob",
    # ROIs are defined as (x1, y1, x2, y2)
    "roi1_coords": (50, 50, 250, 250),
    "roi2_coords": (300, 50, 500, 250),
    "countdown_duration": 3,
    "result_display_time": 5,
    "num_train_frames": 30,     # frames to collect for Gaussian model
    "debug": True
}

# Friend’s Gaussian‐model thresholds
THRESHOLD = 30.0  # Mahalanobis threshold
# Optionally used to do an initial rough threshold (friend's code uses inRange):
lower_skin = np.array([0, 80, 60], dtype=np.uint8)
upper_skin = np.array([20, 150, 255], dtype=np.uint8)

# ====================================
# Gaussian Model Helpers
# ====================================
def fit_gaussian_model(pixels, threshold=THRESHOLD):
    """
    Fit a Gaussian model (mean + inverse covariance) to the pixel data.
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    mean = np.mean(pixels, axis=0)
    cov = np.cov(pixels, rowvar=False)
    # To avoid numerical issues, add a small term on the diagonal:
    inv_cov = np.linalg.inv(cov + 1e-8 * np.eye(3))
    return {"mean": mean, "inv_cov": inv_cov, "threshold": threshold}

def are_pixels_in_distribution(model, hsv_image):
    """
    Return a binary mask indicating which pixels in 'hsv_image' match the Gaussian skin model.
    """
    mean = model["mean"]
    inv_cov = model["inv_cov"]
    threshold = model["threshold"]

    # Flatten (H, W, 3) -> (N, 3)
    reshaped = hsv_image.reshape(-1, 3).astype(np.float64)
    diff = reshaped - mean
    # Mahalanobis distance squared
    md_sq = np.sum(diff @ inv_cov * diff, axis=1)
    # Mark “in‐model” if distance < threshold
    mask = (md_sq < threshold).reshape(hsv_image.shape[:2])
    return mask.astype(np.uint8)

# ====================================
# Gesture Classification (unchanged)
# ====================================
def decide_winner(gesture1, gesture2):
    rules = {"Rock": "Scissors", "Scissors": "Paper", "Paper": "Rock"}
    if gesture1 not in rules or gesture2 not in rules:
        return "Unknown"
    elif gesture1 == gesture2:
        return "Tie"
    elif rules[gesture1] == gesture2:
        return f"{config['player1_name']} wins!"
    else:
        return f"{config['player2_name']} wins!"

def approximate_hand_gesture(mask, roi_bgr=None, debug=False, debug_window_name="Debug ROI"):
    """
    Given a binary mask of the hand region, find the largest contour
    and apply the classic convex hull + defects method to classify the gesture.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    gesture = "Unknown"
    debug_roi = roi_bgr.copy() if roi_bgr is not None else None

    if contours:
        # Assume largest contour is the hand
        max_contour = max(contours, key=cv2.contourArea)
        if debug and debug_roi is not None:
            cv2.drawContours(debug_roi, [max_contour], -1, (0, 255, 0), 2)

        # Approximate
        epsilon = 0.0005 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        if debug and debug_roi is not None:
            cv2.drawContours(debug_roi, [approx], -1, (0, 255, 255), 2)

        # Convex hull + defects
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

                    # Calculate angle
                    a = np.linalg.norm(np.array(end) - np.array(start))
                    b = np.linalg.norm(np.array(far) - np.array(start))
                    c = np.linalg.norm(np.array(end) - np.array(far))
                    if b * c != 0:
                        angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * 57
                        if angle <= 90:
                            count_defects += 1

            # Heuristics: 0->Rock, 1->Scissors, 3/4/5+->Paper
            if count_defects == 0:
                gesture = "Rock"
            elif count_defects == 1:
                gesture = "Scissors"
            elif count_defects in [3, 4, 5, 6]:
                gesture = "Paper"

    if debug and debug_roi is not None:
        cv2.putText(debug_roi, f"Gesture: {gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow(debug_window_name, debug_roi)

    return gesture

# ====================================
# GAUSSIAN MODEL CALIBRATION
# ====================================
def build_gaussian_model_for_roi(cap, roi_coords, num_frames=30, roi_name="ROI"):
    """
    Gather frames from the camera for the specified ROI, build a Gaussian model (skin).
    Use your friend's approach:
      - Optionally do a rough inRange threshold
      - Collect the resulting pixels in HSV
      - Fit the model
    """
    print(f"*** Building Gaussian Model for {roi_name} ***")
    x1, y1, x2, y2 = roi_coords

    # We'll store HSV skin pixels
    hsv_pixels = []

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        roi_bgr = frame[y1:y2, x1:x2]
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        # 1) Optional rough threshold (friend’s code uses lower_skin/upper_skin)
        rough_mask = cv2.inRange(roi_hsv, lower_skin, upper_skin)
        # 2) Morphological cleaning
        rough_mask = cv2.erode(rough_mask, None, iterations=2)
        rough_mask = cv2.dilate(rough_mask, None, iterations=2)

        # 3) Collect those pixels that pass this rough test
        skin_pixels = roi_hsv[rough_mask == 255]
        hsv_pixels.extend(skin_pixels)

        # Visual feedback
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"Training {roi_name}: {i+1}/{num_frames}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(config["window_name"], frame)
        cv2.waitKey(30)

    # Fit the Gaussian model
    if len(hsv_pixels) > 0:
        model = fit_gaussian_model(hsv_pixels, threshold=THRESHOLD)
        print(f"[{roi_name}] Model trained with {len(hsv_pixels)} pixels.\n")
        return model
    else:
        print(f"[{roi_name}] Not enough pixels collected — model is None.")
        return None

# ====================================
# GAUSSIAN MODEL DETECTION
# ====================================
def detect_hand_by_gaussian_model(roi_bgr, model, debug=False, debug_window_name="Debug ROI"):
    """
    Convert ROI to HSV, compute the Mahalanobis distance mask, morphological cleanup,
    then return the final mask for the hand.
    """
    if model is None:
        # If no model, just return empty
        h, w, _ = roi_bgr.shape
        return np.zeros((h, w), dtype=np.uint8)

    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # 1) Probability mask from the Gaussian model
    mask = are_pixels_in_distribution(model, roi_hsv)

    # 2) Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    if debug:
        debug_mask = (mask * 255).astype(np.uint8)
        cv2.imshow(debug_window_name + " Mask", debug_mask)

    return mask

# ====================================
# Main Application (Same Structure)
# ====================================
def main():
    cv2.namedWindow(config["window_name"], cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config["window_name"], config["window_size"][0], config["window_size"][1])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    # Gaussian models for each player
    gauss_model_1 = None
    gauss_model_2 = None

    result_text = None
    result_end_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Draw rectangles around ROIs
        (x1, y1, x2, y2) = config["roi1_coords"]
        (x1_p2, y1_p2, x2_p2, y2_p2) = config["roi2_coords"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.rectangle(frame, (x1_p2, y1_p2), (x2_p2, y2_p2), (0, 255, 0), 2)

        # If Gaussian models exist, do real-time debug detection
        if config["debug"] and gauss_model_1 is not None and gauss_model_2 is not None:
            roi1_bgr = frame[y1:y2, x1:x2]
            roi2_bgr = frame[y1_p2:y2_p2, x1_p2:x2_p2]

            # Player 1
            mask1 = detect_hand_by_gaussian_model(roi1_bgr, gauss_model_1, debug=True, debug_window_name="Debug ROI 1")
            gesture1_live = approximate_hand_gesture(mask1, roi1_bgr, debug=True, debug_window_name="Debug ROI 1")

            # Player 2
            mask2 = detect_hand_by_gaussian_model(roi2_bgr, gauss_model_2, debug=True, debug_window_name="Debug ROI 2")
            gesture2_live = approximate_hand_gesture(mask2, roi2_bgr, debug=True, debug_window_name="Debug ROI 2")

            # Show these "live" gestures on the main frame
            cv2.putText(frame, f"{config['player1_name']}: {gesture1_live}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"{config['player2_name']}: {gesture2_live}",
                        (x1_p2, y1_p2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # If there's an active result, display it
        if result_text is not None and time.time() < result_end_time:
            cv2.putText(frame, result_text, (50, frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            result_text = None

        cv2.imshow(config["window_name"], frame)
        key = cv2.waitKey(10) & 0xFF

        # Press 'c' to calibrate each player's Gaussian model
        if key == ord('c'):
            print("[INFO] Calibrating Player 1's hand with Gaussian model...")
            gauss_model_1 = build_gaussian_model_for_roi(
                cap, config["roi1_coords"], num_frames=config["num_train_frames"],
                roi_name=config["player1_name"]
            )

            print("[INFO] Calibrating Player 2's hand with Gaussian model...")
            gauss_model_2 = build_gaussian_model_for_roi(
                cap, config["roi2_coords"], num_frames=config["num_train_frames"],
                roi_name=config["player2_name"]
            )
            print("[INFO] Calibration complete!\n")

        # Press 'g' to do the countdown capture (only if we have both models)
        if key == ord('g') and gauss_model_1 is not None and gauss_model_2 is not None:
            # Countdown
            countdown_start = time.time()
            countdown_duration = config["countdown_duration"]
            while time.time() - countdown_start < countdown_duration:
                ret2, frame_cdown = cap.read()
                if not ret2:
                    break
                frame_cdown = cv2.flip(frame_cdown, 1)
                elapsed = time.time() - countdown_start
                countdown_number = int(countdown_duration - elapsed) + 1
                cv2.putText(frame_cdown, str(countdown_number),
                            (frame_cdown.shape[1] // 2 - 50, frame_cdown.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                # Draw ROI rectangles
                cv2.rectangle(frame_cdown, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.rectangle(frame_cdown, (x1_p2, y1_p2), (x2_p2, y2_p2), (0, 255, 0), 2)

                cv2.imshow(config["window_name"], frame_cdown)
                cv2.waitKey(30)

            # Capture final gestures after countdown
            ret2, frame_capture = cap.read()
            if not ret2:
                break
            frame_capture = cv2.flip(frame_capture, 1)

            roi1_bgr = frame_capture[y1:y2, x1:x2]
            roi2_bgr = frame_capture[y1_p2:y2_p2, x1_p2:x2_p2]

            # Detect + approximate gesture
            mask1 = detect_hand_by_gaussian_model(roi1_bgr, gauss_model_1, debug=config["debug"], debug_window_name="Debug ROI 1")
            gesture1 = approximate_hand_gesture(mask1, roi1_bgr, debug=config["debug"], debug_window_name="Debug ROI 1")

            mask2 = detect_hand_by_gaussian_model(roi2_bgr, gauss_model_2, debug=config["debug"], debug_window_name="Debug ROI 2")
            gesture2 = approximate_hand_gesture(mask2, roi2_bgr, debug=config["debug"], debug_window_name="Debug ROI 2")

            winner = decide_winner(gesture1, gesture2)
            result_text = (f"{config['player1_name']}: {gesture1} | "
                           f"{config['player2_name']}: {gesture2} => {winner}")
            print("Result:", result_text)
            result_end_time = time.time() + config["result_display_time"]

        # Press 'q' to quit
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ====================================
# Run the Main
# ====================================
if __name__ == "__main__":
    main()
