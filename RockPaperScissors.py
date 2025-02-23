import cv2
import numpy as np
import random
import time
import sys


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
    "skin_lower": np.array([0, 50, 70], dtype=np.uint8),
    "skin_upper": np.array([20, 255, 255], dtype=np.uint8),
    "debug": False                       # Set to True to enable debug windows (shows processed ROI, mask, and H/S/V channels)
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
      - Paper beats Rock or gesture2 not in rules:
    """
    rules = {"Rock": "Scissors", "Scissors": "Paper", "Paper": "Rock"}
    if gesture2 not in rules:
        return f"{config['player1_name']} wins!", 1
    if gesture1 not in rules:
        return f"{config['player2_name']} wins!", 2
    elif gesture1 == gesture2:
        return "Tie! Let's play again!", 0
    elif rules.get(gesture1) == gesture2:
        return f"{config['player1_name']} wins!", 1
    else:
        return f"{config['player2_name']} wins!", 2

def get_gesture(roi, debug=False, debug_window_name="Debug ROI", lower_skin=None, upper_skin=None):
    """
    Processes the provided ROI to detect the hand gesture.
    Returns one of "Rock", "Paper", "Scissors", or "Unknown".
    If debug is True, overlays the detected gesture on the ROI and displays intermediate windows,
    including the H, S, and V channels.
    """
    if lower_skin is None:
        lower_skin = config["skin_lower"]
    if upper_skin is None:
        upper_skin = config["skin_upper"]
    
    # Convert ROI to HSV and create a skin mask.
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Display the H, S, V channels when in debug mode.
    if debug:
        h, s, v = cv2.split(hsv)
        cv2.imshow(debug_window_name + " H", h)
        cv2.imshow(debug_window_name + " S", s)
        cv2.imshow(debug_window_name + " V", v)
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological operations to remove noise.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=4)   # Dilate to fill gaps
    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    # Find contours in the mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Copy ROI for drawing debug information.
    debug_roi = roi.copy()
    gesture = "Unknown"

    if contours:
        # Assume the largest contour is the hand.
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

            # Use heuristics based on the number of defects.
            if count_defects == 0:
                gesture = "Rock"
            elif count_defects in [1]:
                gesture = "Scissors"
            elif count_defects in [3, 4, 5, 6]:
                gesture = "Paper"
            else:
                gesture = "Unknown"

    if debug:
        # Overlay the recognized gesture on the debug ROI.
        cv2.putText(debug_roi, f"Gesture: {gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow(debug_window_name, debug_roi)
        cv2.imshow(debug_window_name + " Mask", mask)
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

    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, str(i),
                    (frame.shape[1] // 2 - 20, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

        cv2.imshow(config["window_name"], frame)
        cv2.waitKey(1000)

    player1, player2 = [], []
    iterations = 200
    while True:
        iterations -= 1
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Get ROIs for players using configuration coordinates.
        config["roi1_coords"] = (frame.shape[1] // 16, frame.shape[0] // 10, 5 * frame.shape[1] // 16,
                                 5 * frame.shape[0] // 10)
        config["roi2_coords"] = (11 * frame.shape[1] // 16, frame.shape[0] // 10, 15 * frame.shape[1] // 16,
                                 5 * frame.shape[0] // 10)
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

        else:
            player1.append(get_gesture(roi1))
            player2.append(get_gesture(roi2))
            if not iterations:
                most_common_word1 = max((w for w in player1 if w != "unknown"), key=player1.count)
                most_common_word2 = max((w for w in player2 if w != "unknown"), key=player2.count)

                cv2.putText(frame, f"{config['player1_name']}: {most_common_word1}",
                            (config["roi1_coords"][0], config["roi1_coords"][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f"{config['player2_name']}: {most_common_word2}",
                            (config["roi2_coords"][0], config["roi2_coords"][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                winner = decide_winner(most_common_word1, most_common_word2)
                cv2.putText(frame, winner[0],
                            (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                cv2.imshow(config["window_name"], frame)
                cv2.waitKey(30)

                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                if not winner[1]:
                    return main()
                return winner[1]

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
