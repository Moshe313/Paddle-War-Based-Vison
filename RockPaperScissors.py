import tkinter as tk                  # Import Tkinter for GUI creation
import threading                      # Import threading module for running operations concurrently
import time                           # Import time module for timing operations (e.g., sleep)
import cv2                            # Import OpenCV for image and video processing
import numpy as np                    # Import NumPy for numerical array processing
from PIL import Image, ImageTk         # Import PIL (Pillow) to convert OpenCV images into Tkinter-compatible images

# -------------------------------------
# Global Configuration
# -------------------------------------
config = {                           # Configuration dictionary holding various settings
    "window_name": "Rock Paper Scissors",   # Title for the game window
    "window_size": (900, 600),                # Dimensions of the window
    "player1_name": "Player1",                # Default name for player 1
    "player2_name": "Player2",                # Default name for player 2
    # (x1, y1, x2, y2) coordinates for region of interest (ROI) for player1 and player2 respectively
    "roi1_coords": (50, 50, 250, 250),
    "roi2_coords": (300, 50, 500, 250),
    "countdown_duration": 3,           # Duration for countdown in seconds
    "result_display_time": 5,          # Display time for the result
    "num_train_frames": 30,            # Number of frames to capture during calibration
    "debug": True,                     # If True, show the debug window and extra info
    "winner": None                     # Placeholder for the eventual game winner
}

THRESHOLD = 40.0                     # Threshold used in Gaussian model computations for gesture detection
lower_skin = np.array([0, 80, 60], dtype=np.uint8)    # Lower bound for skin color in HSV space
upper_skin = np.array([20, 150, 255], dtype=np.uint8)   # Upper bound for skin color in HSV space

models = {"model1": None, "model2": None}   # Dictionary to store Gaussian models for two players (initially None)

# --------------------------------------------------
# Gaussian Model Helpers
# --------------------------------------------------
def fit_gaussian_model(pixels, threshold=THRESHOLD):
    # Convert the list of pixel values into a NumPy array (as float64 for precision)
    pixels = np.asarray(pixels, dtype=np.float64)
    # Compute the mean vector of the pixel values along the axis 0 (color channels)
    mean = np.mean(pixels, axis=0)
    # Compute the covariance matrix of the pixel values; columns represent color channels
    cov = np.cov(pixels, rowvar=False)
    # Compute the inverse covariance matrix (adding a small constant to the diagonal for stability)
    inv_cov = np.linalg.inv(cov + 1e-8 * np.eye(3))
    # Return a dictionary with the Gaussian parameters
    return {"mean": mean, "inv_cov": inv_cov, "threshold": threshold}


def are_pixels_in_distribution(model, hsv_image):
    # If no model is provided, create an empty mask (all zeros) with same height and width as hsv_image
    if model is None:
        h, w, _ = hsv_image.shape
        return np.zeros((h, w), dtype=np.uint8)

    # Extract the Gaussian parameters from the model
    mean = model["mean"]
    inv_cov = model["inv_cov"]
    threshold = model["threshold"]

    # Reshape the image into a 2D array where each row is a pixel (3 color channels)
    reshaped = hsv_image.reshape(-1, 3).astype(np.float64)
    # Compute the difference between each pixel and the mean
    diff = reshaped - mean
    # Compute the squared Mahalanobis distance for each pixel
    md_sq = np.sum(diff @ inv_cov * diff, axis=1)  # "@" is matrix multiplication
    # Build a binary mask: pixels with Mahalanobis distance squared less than threshold are set to True
    mask = (md_sq < threshold).reshape(hsv_image.shape[:2])
    # Convert the Boolean mask to uint8 type (0 or 1) and return
    return mask.astype(np.uint8)


def detect_hand_by_gaussian_model(roi_bgr, model):
    # If there is no trained model, return an empty mask (all zeros) matching ROI dimensions
    if model is None:
        h, w, _ = roi_bgr.shape
        return np.zeros((h, w), dtype=np.uint8)

    # Convert the ROI (Region-Of-Interest) from BGR color space to HSV color space
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    # Compute a probability mask using our Gaussian model helper function
    mask = are_pixels_in_distribution(model, roi_hsv)
    # Create a kernel for morphological operations (a 5x5 matrix of ones)
    kernel = np.ones((5, 5), np.uint8)
    # Apply erosion to remove noise from the mask (shrink white regions)
    mask = cv2.erode(mask, kernel, iterations=1)
    # Apply dilation to restore the eroded parts and smooth the mask
    mask = cv2.dilate(mask, kernel, iterations=2)
    # Return the final binary mask
    return mask


# --------------------------------------------------
# Gesture Classification
# --------------------------------------------------
def approximate_hand_gesture(mask):
    # Find contours (continuous curves or boundaries) in the binary mask using OpenCV's findContours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Unknown"  # If no contours are found, return "Unknown"

    # Select the contour with the maximum area (assumed to be the hand)
    max_contour = max(contours, key=cv2.contourArea)
    # Compute an approximation of the contour's shape to reduce the number of points
    epsilon = 0.0005 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)

    try:
        # Compute the convex hull of the approximated contour (returns indices)
        hull = cv2.convexHull(approx, returnPoints=False)
        if hull is None or len(hull) <= 3:
            return "Unknown"
        # Compute convexity defects, which are the deviations from the convex hull
        defects = cv2.convexityDefects(approx, hull)
        if defects is None:
            return "Unknown"
    except cv2.error:
        # In case of an OpenCV error (bad contour data), return "Unknown"
        return "Unknown"

    count_defects = 0  # Initialize a counter for number of valid defects
    # Loop over each defect; each defect is represented by start, end, farthest point and distance
    for i in range(defects.shape[0]):
        s, e, f, _ = defects[i, 0]
        start = approx[s][0]
        end = approx[e][0]
        far = approx[f][0]
        # Compute distances between the points to use in angle calculation
        a = np.linalg.norm(end - start)
        b = np.linalg.norm(far - start)
        c = np.linalg.norm(end - far)
        if b * c != 0:
            # Calculate the angle using the cosine law and convert radians to degrees (approximately)
            angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * 57
            if angle <= 90:
                count_defects += 1  # Valid defect: angle is small enough

    # Use a simple heuristic to determine gesture based on defects (as per common hand-sign recognition)
    if count_defects == 0:
        return "Rock"
    elif count_defects == 1:
        return "Scissors"
    elif count_defects >= 3:
        return "Paper"
    return "Unknown"


def decide_winner(gesture1, gesture2):
    rules = {"Rock": "Scissors", "Scissors": "Paper", "Paper": "Rock"}  # Standard RPS rules

    # If either gesture is not one of the valid keys, return "Unknown"
    if gesture2 not in rules or gesture1 not in rules:
        return "Unknown"
    elif gesture1 == gesture2:
        return "Tie"  # If both gestures are the same, it's a tie
    elif rules[gesture1] == gesture2:
        return f"{config['player1_name']} wins!"  # Player1 wins if their gesture beats Player2's
    else:
        return f"{config['player2_name']} wins!"  # Otherwise, Player2 wins


# --------------------------------------------------
# The GUI
# --------------------------------------------------
class RPSGameGUI:
    def __init__(self, root):
        self.root = root                              # Save the Tkinter root window in the instance
        self.root.title("Rock-Paper-Scissors Game")   # Set the title of the window

        # Create a label in the root to display video frames
        self.video_label = tk.Label(root)
        self.video_label.grid(row=0, column=0, rowspan=5, padx=10, pady=10)

        # Flags used to control quitting of the game
        self.hard_quit_flag = False
        self.quit_flag = False

        # Create a status label to show messages to the user
        self.status_label = tk.Label(root, text="Status: Waiting", font=("Arial", 12))
        self.status_label.grid(row=4, column=1, padx=10, pady=5)

        # Buttons to trigger calibration for both players
        self.p1_button = tk.Button(root, text=f"Calibrate {config['player1_name']}'s Hand",
                                   command=self.calibrate_p1, width=30, height=2)
        self.p1_button.grid(row=0, column=1, padx=10, pady=5)

        self.p2_button = tk.Button(root, text=f"Calibrate {config['player2_name']}'s Hand",
                                   command=self.calibrate_p2, width=30, height=2)
        self.p2_button.grid(row=1, column=1, padx=10, pady=5)

        # Button to start the Rock-Paper-Scissors game
        self.start_button = tk.Button(root, text="Start Rock-Paper-Scissors",
                                      command=self.start_rps, width=30, height=2)
        self.start_button.grid(row=2, column=1, padx=10, pady=5)

        # Button to quit the game
        self.quit_button = tk.Button(root, text="Quit", command=self.hard_quit_game, width=30, height=2)
        self.quit_button.grid(row=3, column=1, padx=10, pady=5)

        # Initialize video capture from the default camera (index 0)
        self.cap = cv2.VideoCapture(0)
        self.update_video_id = None   # This variable will hold the ID for the scheduled video update callback
        self.running = True           # Flag to control video update loop
        self.calibrating = False      # Flag to indicate calibration state
        # Dictionary to keep track of whether each player has been calibrated
        self.calibrated_players = {"Player1": False, "Player2": False}
        self.calibration_frame = 0    # Counter for the current calibration frame
        self.calibrating_player = 0   # Indicates which player's calibration is running
        # Dictionary for realtime detected gestures of each player
        self.current_detection = {"Player1": "Waiting", "Player2": "Waiting"}
        self.result_text = None       # Will hold the result text (winner announcement)
        self.result_start_time = 0    # Time when the result was first displayed

        # Initialize countdown variables
        self.countdown_value = None   # Current countdown value
        self.countdown_active = False # Flag indicating whether the countdown is active

        # If not quitting, begin updating video from the camera
        if not self.hard_quit_flag and not self.quit_flag:
            self.update_video()

    def update_video(self):
        # Exit if the overall running flag is False
        if not self.running:
            return

        ret, frame = self.cap.read()  # Capture a frame from the camera
        if ret:
            frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect
            (x1, y1, x2, y2) = config["roi1_coords"]  # Extract ROI for player1
            (xx1, yy1, xx2, yy2) = config["roi2_coords"]  # Extract ROI for player2

            # Draw rectangles for both regions of interest on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(frame, (xx1, yy1), (xx2, yy2), (0, 255, 0), 2)

            # If a final result is available, display it for a limited duration
            if self.result_text is not None:
                elapsed = time.time() - self.result_start_time
                if elapsed < config["result_display_time"]:
                    cv2.putText(frame, self.result_text, (50, frame.shape[0] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    self.result_text = None

            # If calibration is in progress, display calibration progress text on the frame
            if self.calibrating:
                txt = f"Calibrating frame {self.calibration_frame}/{config['num_train_frames']}"
                pos = (x1, y1 - 10) if self.calibrating_player == 1 else (xx1, yy1 - 10)
                color = (255, 0, 0) if self.calibrating_player == 1 else (0, 255, 0)
                cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                # If not calibrating, perform real-time gesture detection if calibrated
                if self.calibrated_players["Player1"]:
                    roi1_bgr = frame[y1:y2, x1:x2]
                    mask1 = detect_hand_by_gaussian_model(roi1_bgr, models["model1"])
                    gesture1 = approximate_hand_gesture(mask1)
                    self.current_detection["Player1"] = gesture1

                if self.calibrated_players["Player2"]:
                    roi2_bgr = frame[yy1:yy2, xx1:xx2]
                    mask2 = detect_hand_by_gaussian_model(roi2_bgr, models["model2"])
                    gesture2 = approximate_hand_gesture(mask2)
                    self.current_detection["Player2"] = gesture2

                # Display each player's detected gesture on the frame
                txt1 = f"{config['player1_name']}: {self.current_detection['Player1']}"
                txt2 = f"{config['player2_name']}: {self.current_detection['Player2']}"
                cv2.putText(frame, txt1, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(frame, txt2, (xx1, yy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # If debug mode is enabled and at least one player is calibrated, show a debug window
                if config["debug"] and (self.calibrated_players["Player1"] or self.calibrated_players["Player2"]):
                    self.show_debug_window(frame)

            # If the countdown is active, overlay the countdown number in large red text
            if self.countdown_active:
                cv2.putText(frame, str(self.countdown_value),
                            (frame.shape[1] // 2 - 30, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

            # Convert the processed BGR frame to RGB, then to a PIL Image,
            # and then wrap it in an ImageTk.PhotoImage for display in Tkinter
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.video_label.imgtk = imgtk         # Keep a reference so it isn't garbage-collected
            self.video_label.config(image=imgtk)     # Update the label with the new image

        # Schedule the next call to update_video in 10 milliseconds using Tkinter's after method
        self.update_video_id = self.root.after(10, self.update_video)

    def show_debug_window(self, main_frame):
        # Extract ROI coordinates for both players
        (x1, y1, x2, y2) = config["roi1_coords"]
        (xx1, yy1, xx2, yy2) = config["roi2_coords"]

        # Copy the regions of interest from the main frame for debugging
        roi1_bgr = main_frame[y1:y2, x1:x2].copy()
        mask1 = (np.zeros((roi1_bgr.shape[0], roi1_bgr.shape[1]), dtype=np.uint8)
                 if not self.calibrated_players["Player1"] else
                 detect_hand_by_gaussian_model(roi1_bgr, models["model1"]))

        roi2_bgr = main_frame[yy1:yy2, xx1:xx2].copy()
        mask2 = (np.zeros((roi2_bgr.shape[0], roi2_bgr.shape[1]), dtype=np.uint8)
                 if not self.calibrated_players["Player2"] else
                 detect_hand_by_gaussian_model(roi2_bgr, models["model2"]))

        # Prepare debugging images by drawing contours and mask overlays
        debug_p1_raw  = roi1_bgr.copy()
        debug_p1_mask = cv2.cvtColor(mask1 * 255, cv2.COLOR_GRAY2BGR)
        debug_p1_contour = self.draw_debug_info_for_gesture(roi1_bgr, mask1, 
            f"Detected: {self.current_detection['Player1']}")

        debug_p2_raw  = roi2_bgr.copy()
        debug_p2_mask = cv2.cvtColor(mask2 * 255, cv2.COLOR_GRAY2BGR)
        debug_p2_contour = self.draw_debug_info_for_gesture(roi2_bgr, mask2, 
            f"Detected: {self.current_detection['Player2']}")

        # Stack debugging images horizontally and then vertically to create a combined debug display
        debug_p1_combined = np.hstack((debug_p1_raw, debug_p1_mask, debug_p1_contour))
        debug_p2_combined = np.hstack((debug_p2_raw, debug_p2_mask, debug_p2_contour))
        combined_debug = np.vstack((debug_p1_combined, debug_p2_combined))

        cv2.imshow("Debug Window", combined_debug)  # Show the combined debug window
        cv2.waitKey(1)  # Wait briefly to allow image to render

    def draw_debug_info_for_gesture(self, roi_bgr, mask, text_label):
        debug_roi = roi_bgr.copy()  # Make a copy of the ROI for drawing debug info
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(debug_roi, [max_contour], -1, (0, 255, 0), 2)  # Draw the largest contour in green
            epsilon = 0.0005 * cv2.arcLength(max_contour, True)
            approx = cv2.approxPolyDP(max_contour, epsilon, True)
            cv2.drawContours(debug_roi, [approx], -1, (0, 255, 255), 2)  # Draw the approximated contour in yellow
            try:
                hull = cv2.convexHull(approx, returnPoints=False)
                if hull is not None and len(hull) > 3:
                    defects = cv2.convexityDefects(approx, hull)
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, _ = defects[i, 0]
                            far = tuple(approx[f][0])
                            cv2.circle(debug_roi, far, 4, (0, 0, 255), -1)  # Mark convexity defects with a red circle
            except cv2.error:
                pass  # If error occurs, ignore it
        cv2.putText(debug_roi, text_label, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # Overlay the text label
        return debug_roi  # Return the annotated ROI

    # --------------------------------------------------
    # Calibration + RPS
    # --------------------------------------------------
    def calibrate_p1(self):
        # Start a new daemon thread to perform calibration for player 1
        threading.Thread(target=self._calib_thread, args=(1,), daemon=True).start()

    def calibrate_p2(self):
        # Start a new daemon thread to perform calibration for player 2
        threading.Thread(target=self._calib_thread, args=(2,), daemon=True).start()

    def _calib_thread(self, which_player):
        self.calibrating = True                  # Set flag to indicate calibration is active
        self.calibrating_player = which_player   # Which player's calibration is being done
        self.calibration_frame = 0                # Reset the calibration frame counter
        p_name = config["player1_name"] if which_player == 1 else config["player2_name"]
        self.status_label.config(text=f"Status: Calibrating {p_name}...")  # Update status to show calibration in progress

        pixels = []  # List to collect skin pixel data for training the Gaussian model
        for i in range(config["num_train_frames"]):
            self.calibration_frame = i + 1        # Update current calibration frame
            ret, frame = self.cap.read()           # Capture a frame from the camera
            if ret:
                frame = cv2.flip(frame, 1)         # Flip the frame horizontally
                if which_player == 1:
                    (x1, y1, x2, y2) = config["roi1_coords"]
                    roi = frame[y1:y2, x1:x2]       # Extract region of interest for player 1
                else:
                    (x1, y1, x2, y2) = config["roi2_coords"]
                    roi = frame[y1:y2, x1:x2]       # Extract region of interest for player 2

                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # Convert the ROI to HSV color space
                rough_mask = cv2.inRange(hsv, lower_skin, upper_skin)  # Create a rough mask for skin-color detection
                rough_mask = cv2.erode(rough_mask, None, iterations=2) # Remove noise via erosion
                rough_mask = cv2.dilate(rough_mask, None, iterations=2) # Restore and smooth mask via dilation
                skin_pixels = hsv[rough_mask == 255]       # Extract pixels detected as skin
                pixels.extend(skin_pixels)                 # Append these pixels to the list

            time.sleep(0.05)  # Short delay between frames

        if len(pixels) > 0:
            models[f"model{which_player}"] = fit_gaussian_model(pixels)  # Train the Gaussian model with the collected pixels
        else:
            models[f"model{which_player}"] = None

        # Mark the player as calibrated
        self.calibrated_players[f"Player{which_player}"] = True
        self.status_label.config(text=f"Status: {p_name} Calibration done!")  # Update status message
        self.calibrating = False  # Clear the calibration flag

    def start_rps(self):
        if not all(self.calibrated_players.values()):
            self.status_label.config(text="Status: Both players must calibrate first!")
            return  # Abort if both players are not calibrated
        # Start the RPS game in a separate daemon thread
        threading.Thread(target=self._start_rps_thread, daemon=True).start()

    def _start_rps_thread(self):
        # Countdown phase before capturing gestures
        self.status_label.config(text="Status: Countdown started...")
        self.countdown_value = config["countdown_duration"]
        self.countdown_active = True

        for remaining in range(config["countdown_duration"], 0, -1):
            self.countdown_value = remaining  # Update countdown display value
            time.sleep(1)  # Pause one second per tick

        self.countdown_active = False
        self.status_label.config(text="Status: Capturing final gestures...")

        # Capture a final frame to analyze gestures
        ret, final_frame = self.cap.read()
        if not ret:
            self.status_label.config(text="Status: Camera error capturing final gestures!")
            return

        final_frame = cv2.flip(final_frame, 1)  # Flip the captured frame
        (x1, y1, x2, y2) = config["roi1_coords"]  # Get region coordinates for player 1
        (xx1, yy1, xx2, yy2) = config["roi2_coords"]  # Get region coordinates for player 2

        roi1_bgr = final_frame[y1:y2, x1:x2]  # Extract ROI for player 1
        roi2_bgr = final_frame[yy1:yy2, xx1:xx2]  # Extract ROI for player 2

        # Process each ROI to get a binary mask, then infer the gesture
        mask1 = detect_hand_by_gaussian_model(roi1_bgr, models["model1"])
        gesture1 = approximate_hand_gesture(mask1)

        mask2 = detect_hand_by_gaussian_model(roi2_bgr, models["model2"])
        gesture2 = approximate_hand_gesture(mask2)

        # Decide the winner based on the two detected gestures
        winner_text = decide_winner(gesture1, gesture2)
        config["winner"] = winner_text

        # Build a result string showing both players' gestures and the winner
        self.result_text = f"{config['player1_name']}: {gesture1} | {config['player2_name']}: {gesture2} => {winner_text}"
        self.result_start_time = time.time()  # Record the time when result was displayed
        self.status_label.config(text="Status: " + self.result_text)

        # If a valid winner is determined (not a tie or unknown), then quit the GUI's main loop
        if winner_text not in ["Tie", "Unknown"]:
            self.quit_flag = True
            self.quit_game()
    
    def hard_quit_game(self):
        self.hard_quit_flag = True
        self.quit_game()

    def quit_game(self):
        self.running = False  # Signal the video update loop to stop
        if self.update_video_id is not None:
            self.root.after_cancel(self.update_video_id)  # Cancel any scheduled video update
            self.update_video_id = None
            # Schedule a dummy function after 11ms as a small delay
            def dummy(): pass
            self.root.after(11, dummy)

        self.cap.release()  # Release the camera resource
        self.root.quit()   # Quit the Tkinter main loop

def main(left_player_name="Player1", right_player_name="Player2"):
    # Update configuration with the provided player names
    config["player1_name"] = left_player_name
    config["player2_name"] = right_player_name
    root = tk.Tk()             # Create the Tkinter root window
    app = RPSGameGUI(root)     # Instantiate the game GUI with the root
    root.mainloop()            # Start the Tkinter event loop

    # After the event loop, perform cleanup based on quit flags:
    if app.hard_quit_flag:
        root.destroy()
        app.cap.release()
        cv2.destroyAllWindows()
        return True
    if app.quit_flag:
        time.sleep(3)
        root.destroy()
        app.cap.release()
        cv2.destroyAllWindows()
        return config["winner"]

if __name__ == "__main__":
    winner = main()  # Run main and get the winner value
    print(f"Game over! Winner: {winner}")  # Print the result to the console