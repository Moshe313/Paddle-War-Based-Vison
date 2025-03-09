import tkinter as tk                  # Import Tkinter for GUI creation
import threading                      # Import threading module for running operations concurrently
import time                           # Import time module for timing operations (e.g., sleep)
import cv2                            # Import OpenCV for image and video processing
import numpy as np                    # Import NumPy for numerical array processing
from PIL import Image, ImageTk        # Import PIL (Pillow) to convert OpenCV images into Tkinter-compatible images
from collections import Counter       # Import Counter to count occurrences of elements in a list

# -------------------------------------
# Global Configuration
# -------------------------------------
config = {
    "window_name": "Rock Paper Scissors",   # Title for the game window
    "window_size": (900, 600),                # Dimensions of the window
    "player1_name": "Player1",                # Default name for player 1
    "player2_name": "Player2",                # Default name for player 2
    # (x1, y1, x2, y2) coordinates for ROI for player1 and player2 respectively
    "roi1_coords": (125, 275, 325, 475),
    "roi2_coords": (350, 275, 550, 475),
    "countdown_duration": 3,                  # Countdown duration in seconds
    "result_display_time": 5,                 # How long to display the result (in seconds)
    "num_train_frames": 40,                   # Number of frames for calibration
    "debug": True,                            # If True, show extra debugging info
    "winner": None,                           # Placeholder for the eventual winner text
    "num_capture_frames": 10                  # Number of frames to capture for majority voting
}

THRESHOLD = 30.0  # Threshold for Gaussian model

# These are the original skin thresholds in HSV space
lower_skin = np.array([0, 80, 60], dtype=np.uint8)
upper_skin = np.array([20, 150, 255], dtype=np.uint8)

models = {"model1": None, "model2": None}   # Dictionaries for storing Gaussian models
backgrounds = {"model1": None, "model2": None}

# --------------------------------------------------
# New Helper Functions for Preprocessing and Segmentation
# --------------------------------------------------
def normalize_illumination(roi_bgr):
    """
    Normalize illumination by converting to HSV and equalizing the V channel.
    This helps reduce effects of uneven lighting (e.g., fluorescent lights).
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge((h, s, v_eq))
    normalized_roi = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    return normalized_roi

def segment_skin_hsv(roi_hsv):
    """
    Apply simple thresholding in HSV space for skin segmentation.
    """
    mask = cv2.inRange(roi_hsv, lower_skin, upper_skin)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
    return (mask > 0).astype(np.uint8)

def segment_skin_ycrcb(roi_bgr):
    """
    Convert ROI to YCrCb and threshold the Cr and Cb channels.
    Typical skin thresholds in YCrCb:
       Cr: 133 to 173, Cb: 77 to 127.
    """
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
    return (mask > 0).astype(np.uint8)

def morphological_processing(mask, kernel_size=5):
    """
    Apply morphological opening (erosion then dilation) to remove noise,
    then closing to fill small holes.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_open = cv2.morphologyEx(mask.astype(np.uint8)*255, cv2.MORPH_OPEN, kernel)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)
    _, binary = cv2.threshold(mask_close, 127, 1, cv2.THRESH_BINARY)
    return binary.astype(np.uint8)

def keep_only_largest_contour(mask):
    """
    Takes a binary mask (values 0 and 1) and returns a new mask where only the largest contour 
    is filled completely (no holes) and all other white regions are removed.
    """
    # Convert binary mask (0/1) to 0/255 for contour detection.
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask  # No contours found; return original mask.
    # Find the largest contour by area.
    largest_contour = max(contours, key=cv2.contourArea)
    # Create a blank mask and fill the largest contour.
    new_mask = np.zeros_like(mask_uint8)
    cv2.fillPoly(new_mask, [largest_contour], 255)
    # Convert back to a binary mask (0 and 1).
    new_mask = (new_mask // 255).astype(np.uint8)
    return new_mask

def detect_hand_black_seg_single_scale(roi_bgr, model):
    """
    Performs hand segmentation on a single scale.
    Normalizes illumination, segments in both HSV and YCrCb, combines the masks,
    and applies morphological processing.
    """
    normalized_roi = normalize_illumination(roi_bgr)
    roi_hsv = cv2.cvtColor(normalized_roi, cv2.COLOR_BGR2HSV)
    if model is not None:
        mask1 = are_pixels_in_distribution(model, roi_hsv)
    else:
        mask1 = segment_skin_hsv(roi_hsv)
    mask2 = segment_skin_ycrcb(normalized_roi)
    combined_mask = cv2.bitwise_or(mask1, mask2)
    processed_mask = morphological_processing(combined_mask)
    return processed_mask

def multi_scale_segmentation(roi_bgr, model):
    """
    Perform segmentation at the original scale and at a 2x-upscaled version.
    The upscaled mask is downscaled and combined with the original mask to capture finer details.
    """
    mask_original = detect_hand_black_seg_single_scale(roi_bgr, model)
    # Upscale ROI 2x
    roi_up = cv2.resize(roi_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    mask_up = detect_hand_black_seg_single_scale(roi_up, model)
    # Downscale the upscaled mask back to original ROI size
    mask_up_down = cv2.resize(mask_up, (roi_bgr.shape[1], roi_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    combined = cv2.bitwise_or(mask_original, mask_up_down)
    return combined

def detect_hand_black_seg(roi_bgr, model):
    """
    Main segmentation function that uses multi-scale segmentation and then filters
    the result to keep only the largest contour filled completely.
    """
    mask = multi_scale_segmentation(roi_bgr, model)
    mask_clean = keep_only_largest_contour(mask)
    return mask_clean

# --------------------------------------------------
# Gaussian Model Helpers and Gesture Detection
# --------------------------------------------------
def fit_gaussian_model(pixels):
    pixels = np.asarray(pixels, dtype=np.float64)
    mean = np.mean(pixels, axis=0)
    cov = np.cov(pixels, rowvar=False)
    inv_cov = np.linalg.inv(cov + 1e-8 * np.eye(3))
    diff = pixels - mean
    md_sq = np.sum(diff @ inv_cov * diff, axis=1)
    threshold = THRESHOLD  # A fixed value; can be made adaptive if needed.
    return {"mean": mean, "inv_cov": inv_cov, "threshold": threshold}

def are_pixels_in_distribution(model, hsv_image):
    if model is None:
        h, w, _ = hsv_image.shape
        return np.zeros((h, w), dtype=np.uint8)
    mean = model["mean"]
    inv_cov = model["inv_cov"]
    threshold = model["threshold"]
    reshaped = hsv_image.reshape(-1, 3).astype(np.float64)
    diff = reshaped - mean
    md_sq = np.sum(diff @ inv_cov * diff, axis=1)
    mask = (md_sq < threshold).reshape(hsv_image.shape[:2])
    return mask.astype(np.uint8)

def fill_small_black_holes(mask, kernel_size=3):
    """
    Applies morphological closing to fill small black holes within the mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return filled_mask

def approximate_hand_gesture(mask):
    """
    Approximate the hand gesture based on convexity defects.
    The epsilon for contour approximation has been lowered to capture more details.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Unknown"
    max_contour = max(contours, key=cv2.contourArea)
    # Lower epsilon factor for more detailed contour approximation.
    epsilon = 0.0003 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    try:
        hull = cv2.convexHull(approx, returnPoints=False)
        if hull is None or len(hull) <= 3:
            return "Unknown"
        defects = cv2.convexityDefects(approx, hull)
        if defects is None:
            return "Unknown"
    except cv2.error:
        return "Unknown"
    count_defects = 0
    for i in range(defects.shape[0]):
        s, e, f, _ = defects[i, 0]
        start = approx[s][0]
        end = approx[e][0]
        far = approx[f][0]
        a = np.linalg.norm(end - start)
        b = np.linalg.norm(far - start)
        c = np.linalg.norm(end - far)
        if b * c != 0:
            angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * 57
            if angle <= 90:
                count_defects += 1
    if count_defects == 0:
        return "Rock"
    elif count_defects == 1:
        return "Scissors"
    elif count_defects >= 3:
        return "Paper"
    return "Unknown"

def decide_winner(gesture1, gesture2):
    rules = {"Rock": "Scissors", "Scissors": "Paper", "Paper": "Rock"}
    if gesture2 not in rules or gesture1 not in rules:
        return "Unknown"
    elif gesture1 == gesture2:
        return "Tie"
    elif rules[gesture1] == gesture2:
        return f"{config['player1_name']} wins!"
    else:
        return f"{config['player2_name']} wins!"

def aggregate_gesture(gesture_list):
    """
    Returns the majority gesture from a list of detected gestures.
    """
    if not gesture_list:
        return "Unknown"
    # Ignore "unknown" gestures
    gesture_list = [g for g in gesture_list if g != "Unknown"]
    # return the gesture with the highest count
    return Counter(gesture_list).most_common(1)[0][0]

# --------------------------------------------------
# The RPS Game GUI Class
# --------------------------------------------------
class RPSGameGUI:
    def __init__(self, root):
        self.root = root                              # Tkinter root or Toplevel
        self.root.title("Rock-Paper-Scissors Game")
        # Video label for camera feed
        self.video_label = tk.Label(root)
        self.video_label.grid(row=0, column=0, rowspan=5, padx=10, pady=10)
        # Flags for quitting
        self.hard_quit_flag = False
        self.quit_flag = False
        # Status label
        self.status_label = tk.Label(root, text="Status: Waiting", font=("Arial", 12))
        self.status_label.grid(row=4, column=1, padx=10, pady=5)
        # Calibration buttons for both players
        self.p1_button = tk.Button(root, text=f"Calibrate {config['player1_name']}'s Hand",
                                   command=self.calibrate_p1, width=30, height=2)
        self.p1_button.grid(row=0, column=1, padx=10, pady=5)
        self.p2_button = tk.Button(root, text=f"Calibrate {config['player2_name']}'s Hand",
                                   command=self.calibrate_p2, width=30, height=2)
        self.p2_button.grid(row=1, column=1, padx=10, pady=5)
        # Button to start RPS game
        self.start_button = tk.Button(root, text="Start Rock-Paper-Scissors",
                                      command=self.start_rps, width=30, height=2)
        self.start_button.grid(row=2, column=1, padx=10, pady=5)
        # Quit button
        self.quit_button = tk.Button(root, text="Quit", command=self.hard_quit_game, width=30, height=2)
        self.quit_button.grid(row=3, column=1, padx=10, pady=5)
        # Initialize camera capture
        self.cap = cv2.VideoCapture(0)
        self.update_video_id = None
        self.running = True
        self.calibrating = False
        self.calibrated_players = {"Player1": False, "Player2": False}
        self.calibration_frame = 0
        self.calibrating_player = 0
        self.current_detection = {"Player1": "Waiting", "Player2": "Waiting"}
        self.result_text = None
        self.result_start_time = 0
        self.countdown_value = None
        self.countdown_active = False

        # For temporal smoothing of ROIs (using exponential moving average)
        self.smoothed_roi1 = None
        self.smoothed_roi2 = None
        self.smoothing_alpha = 0.5  # Weight for new ROI

        if not self.hard_quit_flag and not self.quit_flag:
            self.update_video()

    def update_video(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            (x1, y1, x2, y2) = config["roi1_coords"]
            (xx1, yy1, xx2, yy2) = config["roi2_coords"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(frame, (xx1, yy1), (xx2, yy2), (0, 255, 0), 2)

            # Display game result if available.
            if self.result_text is not None:
                elapsed = time.time() - self.result_start_time
                if elapsed < config["result_display_time"]:
                    cv2.putText(frame, self.result_text, (50, frame.shape[0] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    self.result_text = None

            if self.calibrating:
                if self.calibrating_player == 1:
                    roi_x, roi_y, roi_x2, roi_y2 = config["roi1_coords"]
                    color = (255, 0, 0)
                else:
                    roi_x, roi_y, roi_x2, roi_y2 = config["roi2_coords"]
                    color = (0, 255, 0)
                roi_width = roi_x2 - roi_x
                progress = self.calibration_frame / config["num_train_frames"]
                filled_width = int(progress * roi_width)
                # Define progress bar rectangle above the ROI.
                progress_bar_top_left = (roi_x, roi_y - 30)
                progress_bar_bottom_right = (roi_x + roi_width, roi_y - 10)
                # Draw progress bar outline.
                cv2.rectangle(frame, progress_bar_top_left, progress_bar_bottom_right, color, 2)
                # Draw filled portion.
                cv2.rectangle(frame, progress_bar_top_left, (roi_x + filled_width, roi_y - 10), color, thickness=-1)
                # Put percentage text in the center.
                percentage_text = f"{int(progress * 100)}%"
                text_size, _ = cv2.getTextSize(percentage_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                text_x = roi_x + (roi_width - text_size[0]) // 2
                text_y = roi_y - 15 + text_size[1] // 2
                cv2.putText(frame, percentage_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)
                                                                                                      
            else:
                # Apply temporal smoothing for each player's ROI.
                if self.calibrated_players["Player1"]:
                    roi1_bgr = frame[y1:y2, x1:x2]
                    if self.smoothed_roi1 is None:
                        self.smoothed_roi1 = roi1_bgr.astype("float")
                    else:
                        cv2.accumulateWeighted(roi1_bgr, self.smoothed_roi1, self.smoothing_alpha)
                    roi1_smoothed = cv2.convertScaleAbs(self.smoothed_roi1)
                    mask1 = detect_hand_black_seg(roi1_smoothed, models["model1"])
                    gesture1 = approximate_hand_gesture(mask1)
                    self.current_detection["Player1"] = gesture1

                if self.calibrated_players["Player2"]:
                    roi2_bgr = frame[yy1:yy2, xx1:xx2]
                    if self.smoothed_roi2 is None:
                        self.smoothed_roi2 = roi2_bgr.astype("float")
                    else:
                        cv2.accumulateWeighted(roi2_bgr, self.smoothed_roi2, self.smoothing_alpha)
                    roi2_smoothed = cv2.convertScaleAbs(self.smoothed_roi2)
                    mask2 = detect_hand_black_seg(roi2_smoothed, models["model2"])
                    gesture2 = approximate_hand_gesture(mask2)
                    self.current_detection["Player2"] = gesture2

                txt1 = f"{config['player1_name']}: {self.current_detection['Player1']}"
                txt2 = f"{config['player2_name']}: {self.current_detection['Player2']}"
                if config["debug"]:
                    cv2.putText(frame, txt1, (x1, y2 -220),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(frame, txt2, (xx1, yy2 - 220),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if config["debug"] and (self.calibrated_players["Player1"] or self.calibrated_players["Player2"]):
                    self.show_debug_window(frame)

            if self.countdown_active:
                cv2.putText(frame, str(self.countdown_value),
                            (frame.shape[1] // 2 - 30, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
        self.update_video_id = self.root.after(10, self.update_video)

    def show_debug_window(self, main_frame):
        (x1, y1, x2, y2) = config["roi1_coords"]
        (xx1, yy1, xx2, yy2) = config["roi2_coords"]
        roi1_bgr = main_frame[y1:y2, x1:x2].copy()
        mask1 = (np.zeros((roi1_bgr.shape[0], roi1_bgr.shape[1]), dtype=np.uint8)
                 if not self.calibrated_players["Player1"] else
                 detect_hand_black_seg(roi1_bgr, models["model1"]))
        roi2_bgr = main_frame[yy1:yy2, xx1:xx2].copy()
        mask2 = (np.zeros((roi2_bgr.shape[0], roi2_bgr.shape[1]), dtype=np.uint8)
                 if not self.calibrated_players["Player2"] else
                 detect_hand_black_seg(roi2_bgr, models["model2"]))
        debug_p1_raw  = roi1_bgr.copy()
        debug_p1_mask = cv2.cvtColor(mask1 * 255, cv2.COLOR_GRAY2BGR)
        debug_p1_contour = self.draw_debug_info_for_gesture(roi1_bgr, mask1, 
            f"Detected: {self.current_detection['Player1']}")
        debug_p2_raw  = roi2_bgr.copy()
        debug_p2_mask = cv2.cvtColor(mask2 * 255, cv2.COLOR_GRAY2BGR)
        debug_p2_contour = self.draw_debug_info_for_gesture(roi2_bgr, mask2, 
            f"Detected: {self.current_detection['Player2']}")
        debug_p1_combined = np.hstack((debug_p1_raw, debug_p1_mask, debug_p1_contour))
        debug_p2_combined = np.hstack((debug_p2_raw, debug_p2_mask, debug_p2_contour))
        combined_debug = np.vstack((debug_p1_combined, debug_p2_combined))
        cv2.imshow("Debug Window", combined_debug)
        cv2.waitKey(1)

    def draw_debug_info_for_gesture(self, roi_bgr, mask, text_label):
        debug_roi = roi_bgr.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(debug_roi, [max_contour], -1, (0, 255, 0), 2)
            # Use the refined epsilon here too.
            epsilon = 0.0003 * cv2.arcLength(max_contour, True)
            approx = cv2.approxPolyDP(max_contour, epsilon, True)
            cv2.drawContours(debug_roi, [approx], -1, (0, 255, 255), 2)
            try:
                hull = cv2.convexHull(approx, returnPoints=False)
                if hull is not None and len(hull) > 3:
                    defects = cv2.convexityDefects(approx, hull)
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, _ = defects[i, 0]
                            far = tuple(approx[f][0])
                            cv2.circle(debug_roi, far, 4, (0, 0, 255), -1)
            except cv2.error:
                pass
        cv2.putText(debug_roi, text_label, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return debug_roi

    def calibrate_p1(self):
        threading.Thread(target=self._calib_thread, args=(1,), daemon=True).start()

    def calibrate_p2(self):
        threading.Thread(target=self._calib_thread, args=(2,), daemon=True).start()

    def _calib_thread(self, which_player):
        self.calibrating = True
        self.calibrating_player = which_player
        self.calibration_frame = 0
        p_name = config["player1_name"] if which_player == 1 else config["player2_name"]
        self.status_label.config(text=f"Status: Calibrating {p_name}...")
        pixels = []
        calib_rois = []  # To store each ROI (in HSV) from calibration frames
        for i in range(config["num_train_frames"]):
            self.calibration_frame = i + 1
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                if which_player == 1:
                    (x1, y1, x2, y2) = config["roi1_coords"]
                else:
                    (x1, y1, x2, y2) = config["roi2_coords"]
                roi = frame[y1:y2, x1:x2]
                # Normalize illumination in calibration too.
                roi_norm = normalize_illumination(roi)
                hsv = cv2.cvtColor(roi_norm, cv2.COLOR_BGR2HSV)
                calib_rois.append(hsv)  # Save the ROI for background computation
                rough_mask = cv2.inRange(hsv, lower_skin, upper_skin)
                rough_mask = cv2.erode(rough_mask, None, iterations=1)
                rough_mask = cv2.dilate(rough_mask, None, iterations=1)
                skin_pixels = hsv[rough_mask == 255]
                pixels.extend(skin_pixels)
            time.sleep(0.05)
        if len(pixels) > 0:
            models[f"model{which_player}"] = fit_gaussian_model(pixels)
            calib_stack = np.stack(calib_rois, axis=0)  # shape: (num_frames, height, width, channels)
            std = np.std(calib_stack, axis=0)  # per-pixel standard deviation
            threshold_std = 5.0  # Threshold to decide if a pixel is static
            background_mask = np.mean(std, axis=2) < threshold_std
            bg_image = np.median(calib_stack, axis=0).astype(np.uint8)
            backgrounds[f"model{which_player}"] = {"bg_image": bg_image, "bg_mask": background_mask}
        else:
            models[f"model{which_player}"] = None
        self.calibrated_players[f"Player{which_player}"] = True
        self.status_label.config(text=f"Status: {p_name} Calibration done!")
        self.calibrating = False

    def start_rps(self):
        if not all(self.calibrated_players.values()):
            self.status_label.config(text="Status: Both players must calibrate first!")
            return
        threading.Thread(target=self._start_rps_thread, daemon=True).start()

    def _start_rps_thread(self):
        self.status_label.config(text="Status: Countdown started...")
        self.countdown_value = config["countdown_duration"]
        self.countdown_active = True
        for remaining in range(config["countdown_duration"], 0, -1):
            self.countdown_value = remaining
            time.sleep(1)
        self.countdown_active = False
        self.status_label.config(text="Status: Capturing final gestures...")
        gesture_results_p1 = []
        gesture_results_p2 = []
        num_capture_frames = config["num_capture_frames"]
        for _ in range(num_capture_frames):
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                (x1, y1, x2, y2) = config["roi1_coords"]
                (xx1, yy1, xx2, yy2) = config["roi2_coords"]
                roi1_bgr = frame[y1:y2, x1:x2]
                roi2_bgr = frame[yy1:yy2, xx1:xx2]
                mask1 = detect_hand_black_seg(roi1_bgr, models["model1"])
                gesture1 = approximate_hand_gesture(mask1)
                mask2 = detect_hand_black_seg(roi2_bgr, models["model2"])
                gesture2 = approximate_hand_gesture(mask2)
                gesture_results_p1.append(gesture1)
                gesture_results_p2.append(gesture2)
            time.sleep(0.1)  # Delay between frames
        final_gesture_p1 = aggregate_gesture(gesture_results_p1)
        final_gesture_p2 = aggregate_gesture(gesture_results_p2)
        winner_text = decide_winner(final_gesture_p1, final_gesture_p2)
        config["winner"] = winner_text
        self.result_text = (f"{config['player1_name']}: {final_gesture_p1} | "
                            f"{config['player2_name']}: {final_gesture_p2} => {winner_text}")
        self.result_start_time = time.time()
        self.status_label.config(text="Status: " + self.result_text)
        if winner_text not in ["Tie", "Unknown"]:
            self.quit_flag = True
            # Prepare the final result text with each player's gesture and the winner declaration.
            self.result_text = (f"{config['player1_name']}: {final_gesture_p1} | "
                                f"{config['player2_name']}: {final_gesture_p2} => {winner_text}")
            # Create a full-window overlay label for a dramatic winner announcement.
            winner_label = tk.Label(self.root, text=self.result_text,
                                    font=("Helvetica", 30, "bold"),
                                    fg="gold", bg="black")
            winner_label.place(relx=0.5, rely=0.5, anchor="center")
            
            # Define a simple pulsating animation for the text.
            def pulsate(scale=1.0, direction=1):
                new_size = int(30 * scale)
                winner_label.config(font=("Helvetica", new_size, "bold"))
                if scale >= 1.2:
                    direction = -1
                elif scale <= 1.0:
                    direction = 1
                scale += 0.02 * direction
                self.root.after(50, lambda: pulsate(scale, direction))
            pulsate()
            
            # Delay for 3 seconds to let the animation play before quitting.
            self.root.after(3000, self.quit_game)

            #self.quit_game()

    def hard_quit_game(self):
        self.hard_quit_flag = True
        self.quit_game()

    def quit_game(self):
        self.running = False
        if self.update_video_id is not None:
            self.root.after_cancel(self.update_video_id)
            self.update_video_id = None
            self.root.after(11, lambda: None)
        self.cap.release()
        self.root.quit()

def main(left_player_name="Player1", right_player_name="Player2", parent=None):
    config["player1_name"] = left_player_name
    config["player2_name"] = right_player_name
    if parent is None:
        root = tk.Tk()
    else:
        root = tk.Toplevel(parent)
    app = RPSGameGUI(root)
    root.mainloop()
    root.destroy()
    app.cap.release()
    cv2.destroyAllWindows()
    if app.hard_quit_flag:
        return True
    elif app.quit_flag:
        return config["winner"]

if __name__ == "__main__":
    winner = main()
    print(f"Game over! Winner: {winner}")
