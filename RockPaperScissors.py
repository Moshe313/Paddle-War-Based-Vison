import cv2
import numpy as np
import random
import time

def decide_winner(gesture1, gesture2):
    """
    Determines the winner based on two gestures.
    Returns "Tie" if both gestures are equal.
    Otherwise, applies the rules:
      - Rock beats Scissors
      - Scissors beats Paper
      - Paper beats Rock
    """
    rules = {"Rock": "Scissors", "Scissors": "Paper", "Paper": "Rock"}
    if gesture1 == gesture2:
        return "Tie"
    elif rules.get(gesture1) == gesture2:
        return "Player 1 wins!"
    else:
        return "Player 2 wins!"

def get_gesture(roi, debug=False, debug_window_name="Debug ROI"):
    """
    Processes the provided ROI to detect the hand gesture.
    Returns one of "Rock", "Paper", "Scissors", or "Unknown".
    If debug is True, overlays the detected gesture on the ROI and shows intermediate windows.
    """
    # Convert ROI to HSV and create a skin mask
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 100, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=4)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Copy ROI for drawing debug information
    debug_roi = roi.copy()
    gesture = "Unknown"

    if contours:
        # Assume the largest contour is the hand
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(debug_roi, [max_contour], -1, (0, 255, 0), 2)

        # Approximate the contour for smoother edges
        epsilon = 0.0005 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        cv2.drawContours(debug_roi, [approx], -1, (0, 255, 255), 2)

        # Find convex hull and convexity defects
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
                    
                    # Calculate the angle using the cosine rule
                    a = np.linalg.norm(np.array(end) - np.array(start))
                    b = np.linalg.norm(np.array(far) - np.array(start))
                    c = np.linalg.norm(np.array(end) - np.array(far))
                    if b * c != 0:
                        angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * 57
                        if angle <= 90:
                            count_defects += 1

            # Heuristics to decide the gesture based on the number of defects
            if count_defects == 0:
                gesture = "Rock"
            elif count_defects == 1:
                gesture = "Scissors"
            elif count_defects in [3, 4, 5, 6]:
                gesture = "Paper"
            else:
                gesture = "Unknown"

    if debug:
        # Overlay the recognized gesture on the debug ROI
        cv2.putText(debug_roi, f"Gesture: {gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow(debug_window_name, debug_roi)
        cv2.imshow(debug_window_name + " Mask", mask)
    return gesture

# Initialize video capture
cap = cv2.VideoCapture(0)
debug = True  # Enable debug mode to see extra debugging information

# Variables to hold the result text and when to stop displaying it
result_text = None
result_end_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)

    # Define two ROIs for the two players
    # Player 1: Top-left ROI; Player 2: Top-right ROI
    roi1 = frame[50:250, 50:250]
    roi2 = frame[50:250, 300:500]

    # Draw rectangles around the ROIs on the main frame
    cv2.rectangle(frame, (50, 50), (250, 250), (255, 0, 0), 2)   # Blue for Player 1
    cv2.rectangle(frame, (300, 50), (500, 250), (0, 255, 0), 2)    # Green for Player 2

    # In debug mode, show live gesture info on debug windows
    if debug:
        gesture1_live = get_gesture(roi1, debug=True, debug_window_name="Debug ROI 1")
        gesture2_live = get_gesture(roi2, debug=True, debug_window_name="Debug ROI 2")
        cv2.putText(frame, f"P1: {gesture1_live}", (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"P2: {gesture2_live}", (300, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # If result_text is set and the display period hasn't expired, overlay it
    if result_text is not None and time.time() < result_end_time:
        cv2.putText(frame, result_text, (50, 300), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
    else:
        result_text = None

    cv2.imshow("Rock Paper Scissors", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('g'):
        # --- Countdown (without freezing the camera) ---
        countdown_start = time.time()
        countdown_duration = 3  # seconds for countdown
        while time.time() - countdown_start < countdown_duration:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            elapsed = time.time() - countdown_start
            countdown_number = int(countdown_duration - elapsed) + 1
            cv2.putText(frame, str(countdown_number), (frame.shape[1] // 2 - 50, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            # Draw the ROI rectangles so players know where to show their hands
            cv2.rectangle(frame, (50, 50), (250, 250), (255, 0, 0), 2)
            cv2.rectangle(frame, (300, 50), (500, 250), (0, 255, 0), 2)
            cv2.imshow("Rock Paper Scissors", frame)
            # Use a short waitKey to allow updates (without freezing for the entire countdown)
            cv2.waitKey(30)

        # --- Capture the gestures after the countdown ---
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        roi1 = frame[50:250, 50:250]
        roi2 = frame[50:250, 300:500]
        gesture1 = get_gesture(roi1, debug)
        gesture2 = get_gesture(roi2, debug)
        winner = decide_winner(gesture1, gesture2)
        result_text = f"P1: {gesture1} | P2: {gesture2} => {winner}"
        result_end_time = time.time() + 5  # Display result for 5 seconds

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
