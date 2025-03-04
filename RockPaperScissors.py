import tkinter as tk
import threading
import time
import cv2
import numpy as np
from PIL import Image, ImageTk

# -------------------------------------
# Global Configuration
# -------------------------------------
config = {
    "window_name": "Rock Paper Scissors",
    "window_size": (900, 600),
    "player1_name": "Player1",
    "player2_name": "Player2",
    # (x1, y1, x2, y2)
    "roi1_coords": (50, 50, 250, 250),
    "roi2_coords": (300, 50, 500, 250),
    "countdown_duration": 3,
    "result_display_time": 5,
    "num_train_frames": 30,
    "debug": True,
    "winner": None
}

THRESHOLD = 40.0
lower_skin = np.array([0, 80, 60], dtype=np.uint8)
upper_skin = np.array([20, 150, 255], dtype=np.uint8)

models = {"model1": None, "model2": None}


# --------------------------------------------------
# Gaussian Model Helpers
# --------------------------------------------------
def fit_gaussian_model(pixels, threshold=THRESHOLD):
    """
    Fit a Gaussian model (mean + inv_cov) to the pixel data.
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    mean = np.mean(pixels, axis=0)
    cov = np.cov(pixels, rowvar=False)
    inv_cov = np.linalg.inv(cov + 1e-8 * np.eye(3))
    return {"mean": mean, "inv_cov": inv_cov, "threshold": threshold}


def are_pixels_in_distribution(model, hsv_image):
    """
    Return a (binary) mask of pixels in the Gaussian distribution.
    """
    if model is None:
        h, w, _ = hsv_image.shape
        return np.zeros((h, w), dtype=np.uint8)

    mean = model["mean"]
    inv_cov = model["inv_cov"]
    threshold = model["threshold"]

    reshaped = hsv_image.reshape(-1, 3).astype(np.float64)
    diff = reshaped - mean
    md_sq = np.sum(diff @ inv_cov * diff, axis=1)  # Mahalanobis distance^2
    mask = (md_sq < threshold).reshape(hsv_image.shape[:2])

    return mask.astype(np.uint8)


def detect_hand_by_gaussian_model(roi_bgr, model):
    """
    Mirror the old 'detect_hand_by_gaussian_model' approach:
      1) Convert ROI to HSV
      2) Probability mask (Mahalanobis)
      3) Morphological cleanup
      4) Return final binary mask
    """
    if model is None:
        h, w, _ = roi_bgr.shape
        return np.zeros((h, w), dtype=np.uint8)

    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # 1) Probability mask from the Gaussian model
    mask = are_pixels_in_distribution(model, roi_hsv)

    # 2) Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    return mask


# --------------------------------------------------
# Gesture Classification
# --------------------------------------------------
def approximate_hand_gesture(mask):
    """
    Determine a gesture string: "Rock", "Paper", "Scissors", or "Unknown".
    Also used for debug drawing in the new code.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Unknown"

    max_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.0005 * cv2.arcLength(max_contour, True)
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
        end   = approx[e][0]
        far   = approx[f][0]
        a = np.linalg.norm(end - start)
        b = np.linalg.norm(far - start)
        c = np.linalg.norm(end - far)
        if b * c != 0:
            angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * 57
            if angle <= 90:
                count_defects += 1

    # Same heuristic from the old code:
    # - 0 defects -> Rock
    # - 1 defect  -> Scissors
    # - 3+ defects -> Paper
    if count_defects == 0:
        return "Rock"
    elif count_defects == 1:
        return "Scissors"
    elif count_defects >= 3:
        return "Paper"
    return "Unknown"


def decide_winner(gesture1, gesture2):
    """
    RPS logic from the old code: if one player's gesture is invalid, the other wins by default.
    """
    rules = {"Rock": "Scissors", "Scissors": "Paper", "Paper": "Rock"}

    if gesture2 not in rules or gesture1 not in rules:
        return "Unknown"  # P2 invalid => P1 auto-wins

    elif gesture1 == gesture2:
        return "Tie"
    elif rules[gesture1] == gesture2:
        return f"{config['player1_name']} wins!"
    else:
        return f"{config['player2_name']} wins!"


# --------------------------------------------------
# The GUI
# --------------------------------------------------
class RPSGameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Rock-Paper-Scissors Game")

        self.video_label = tk.Label(root)
        self.video_label.grid(row=0, column=0, rowspan=5, padx=10, pady=10)
       
      

        self.quit_flag = False
        self.status_label = tk.Label(root, text="Status: Waiting", font=("Arial", 12))
        self.status_label.grid(row=4, column=1, padx=10, pady=5)

        self.p1_button = tk.Button(root, text=f"Calibrate {config['player1_name']}'s Hand",
                                   command=self.calibrate_p1, width=30, height=2)
        self.p1_button.grid(row=0, column=1, padx=10, pady=5)

        self.p2_button = tk.Button(root, text=f"Calibrate {config['player2_name']}'s Hand",
                                   command=self.calibrate_p2, width=30, height=2)
        self.p2_button.grid(row=1, column=1, padx=10, pady=5)

        self.start_button = tk.Button(root, text="Start Rock-Paper-Scissors",
                                      command=self.start_rps, width=30, height=2)
        self.start_button.grid(row=2, column=1, padx=10, pady=5)

        self.quit_button = tk.Button(root, text="Quit", command=self.quit_game, width=30, height=2)
        self.quit_button.grid(row=3, column=1, padx=10, pady=5)

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

        self.update_video()

    def update_video(self):
        """
        Periodically called to grab a frame, display it,
        and (if calibrated) do live detection + debugging.
        """
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            (x1, y1, x2, y2) = config["roi1_coords"]
            (xx1, yy1, xx2, yy2) = config["roi2_coords"]

            # Draw ROI rectangles
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(frame, (xx1, yy1), (xx2, yy2), (0, 255, 0), 2)

            # If there's a result to show, show it for 'result_display_time' seconds
            if self.result_text is not None:
                elapsed = time.time() - self.result_start_time
                if elapsed < config["result_display_time"]:
                    cv2.putText(frame, self.result_text, (50, frame.shape[0] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    self.result_text = None

            # Show calibration progress if calibrating
            if self.calibrating:
                if self.calibrating_player == 1:
                    txt = f"Calibrating frame {self.calibration_frame}/{config['num_train_frames']}"
                    cv2.putText(frame, txt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                elif self.calibrating_player == 2:
                    txt = f"Calibrating frame {self.calibration_frame}/{config['num_train_frames']}"
                    cv2.putText(frame, txt, (xx1, yy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            else:
                # Real-time detection for both players (if calibrated)
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

                # Write the live gestures above the ROIs
                txt1 = f"{config['player1_name']}: {self.current_detection['Player1']}"
                txt2 = f"{config['player2_name']}: {self.current_detection['Player2']}"
                cv2.putText(frame, txt1, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(frame, txt2, (xx1, yy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # If debug, build a combined debug window for each player’s ROI
                if config["debug"] and (self.calibrated_players["Player1"] or self.calibrated_players["Player2"]):
                    self.show_debug_window(frame)

            # Convert to RGB for Tk
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        self.update_video_id = self.root.after(10, self.update_video)

    def show_debug_window(self, main_frame):
        """
        Creates a single debug window with 3 “subframes” for each hand:
          1) Raw ROI
          2) Final morphological mask
          3) ROI with drawn contour/hull/defects
        Then for the second player, the same 3 subframes in the same window,
        stacked or side by side.
        """
        (x1, y1, x2, y2) = config["roi1_coords"]
        (xx1, yy1, xx2, yy2) = config["roi2_coords"]

        # Player1 ROI
        roi1_bgr = main_frame[y1:y2, x1:x2].copy()
        mask1 = (np.zeros((roi1_bgr.shape[0], roi1_bgr.shape[1]), dtype=np.uint8)
                 if not self.calibrated_players["Player1"] else
                 detect_hand_by_gaussian_model(roi1_bgr, models["model1"]))

        # Player2 ROI
        roi2_bgr = main_frame[yy1:yy2, xx1:xx2].copy()
        mask2 = (np.zeros((roi2_bgr.shape[0], roi2_bgr.shape[1]), dtype=np.uint8)
                 if not self.calibrated_players["Player2"] else
                 detect_hand_by_gaussian_model(roi2_bgr, models["model2"]))

        # Build 3 debug frames for Player1
        debug_p1_raw  = roi1_bgr.copy()
        debug_p1_mask = cv2.cvtColor(mask1 * 255, cv2.COLOR_GRAY2BGR)
        debug_p1_contour = self.draw_debug_info_for_gesture(roi1_bgr, mask1, 
            f"Detected: {self.current_detection['Player1']}")

        # Build 3 debug frames for Player2
        debug_p2_raw  = roi2_bgr.copy()
        debug_p2_mask = cv2.cvtColor(mask2 * 255, cv2.COLOR_GRAY2BGR)
        debug_p2_contour = self.draw_debug_info_for_gesture(roi2_bgr, mask2, 
            f"Detected: {self.current_detection['Player2']}")

        # Horizontally stack Player1’s 3 frames, then Player2’s 3 frames
        debug_p1_combined = np.hstack((debug_p1_raw, debug_p1_mask, debug_p1_contour))
        debug_p2_combined = np.hstack((debug_p2_raw, debug_p2_mask, debug_p2_contour))

        # Finally, vertically stack them => total shape: 2 rows x 3 columns
        combined_debug = np.vstack((debug_p1_combined, debug_p2_combined))

        cv2.imshow("Debug Window", combined_debug)
        cv2.waitKey(1)

    def draw_debug_info_for_gesture(self, roi_bgr, mask, text_label):
        """
        Draw the largest contour, polygon approximation, hull/defects
        on top of the ROI. Similar to the 'previous version' debug steps.
        """
        debug_roi = roi_bgr.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(debug_roi, [max_contour], -1, (0, 255, 0), 2)

            epsilon = 0.0005 * cv2.arcLength(max_contour, True)
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

    # --------------------------------------------------
    # Calibration + RPS
    # --------------------------------------------------
    def calibrate_p1(self):
        threading.Thread(target=self._calib_thread, args=(1,), daemon=True).start()

    def calibrate_p2(self):
        threading.Thread(target=self._calib_thread, args=(2,), daemon=True).start()

    def _calib_thread(self, which_player):
        """
        Gathers frames from the camera for the specified ROI,
        builds the Gaussian model using inRange as a rough filter.
        """
        self.calibrating = True
        self.calibrating_player = which_player
        self.calibration_frame = 0
        p_name = config["player1_name"] if which_player == 1 else config["player2_name"]
        self.status_label.config(text=f"Status: Calibrating {p_name}...")

        pixels = []
        for i in range(config["num_train_frames"]):
            self.calibration_frame = i + 1
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                if which_player == 1:
                    (x1, y1, x2, y2) = config["roi1_coords"]
                    roi = frame[y1:y2, x1:x2]
                else:
                    (x1, y1, x2, y2) = config["roi2_coords"]
                    roi = frame[y1:y2, x1:x2]

                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # Rough threshold to pick up likely skin pixels
                rough_mask = cv2.inRange(hsv, lower_skin, upper_skin)
                # morphological cleaning just for the calibration mask
                rough_mask = cv2.erode(rough_mask, None, iterations=2)
                rough_mask = cv2.dilate(rough_mask, None, iterations=2)

                # collect the HSV pixels that pass the rough threshold
                skin_pixels = hsv[rough_mask == 255]
                pixels.extend(skin_pixels)

            time.sleep(0.05)

        # Fit Gaussian model
        if len(pixels) > 0:
            models[f"model{which_player}"] = fit_gaussian_model(pixels)
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
        """
        Implements the countdown and final capture (like the old code).
        """
        self.status_label.config(text="Status: Countdown started...")
        countdown_time = config["countdown_duration"]

        for remaining in range(countdown_time, 0, -1):
            end_time_for_number = time.time() + 1
            while time.time() < end_time_for_number:
                time.sleep(0.01)  # Slight sleep for smoother update

                ret, frame_count = self.cap.read()
                if ret:
                    frame_count = cv2.flip(frame_count, 1)

                    # Draw Countdown Number
                    cv2.putText(frame_count, str(remaining),
                                (frame_count.shape[1] // 2 - 30, frame_count.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

                    # Draw ROI Rectangles
                    (x1, y1, x2, y2) = config["roi1_coords"]
                    (xx1, yy1, xx2, yy2) = config["roi2_coords"]
                    cv2.rectangle(frame_count, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.rectangle(frame_count, (xx1, yy1), (xx2, yy2), (0, 255, 0), 2)

                    # Update the Tkinter window
                    rgb = cv2.cvtColor(frame_count, cv2.COLOR_BGR2RGB)
                    imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
                    self.video_label.imgtk = imgtk
                    self.video_label.config(image=imgtk)

                    # Ensure Tkinter processes UI updates smoothly
                    self.root.update_idletasks()
                    self.root.update()

        # Countdown done, capture final frame
        self.status_label.config(text="Status: Capturing final gestures...")
        ret, final_frame = self.cap.read()
        if not ret:
            self.status_label.config(text="Status: Camera error capturing final gestures!")
            return

        final_frame = cv2.flip(final_frame, 1)

        (x1, y1, x2, y2) = config["roi1_coords"]
        (xx1, yy1, xx2, yy2) = config["roi2_coords"]

        roi1_bgr = final_frame[y1:y2, x1:x2]
        roi2_bgr = final_frame[yy1:yy2, xx1:xx2]

        # Use the same detection + gesture classification
        mask1 = detect_hand_by_gaussian_model(roi1_bgr, models["model1"])
        gesture1 = approximate_hand_gesture(mask1)

        mask2 = detect_hand_by_gaussian_model(roi2_bgr, models["model2"])
        gesture2 = approximate_hand_gesture(mask2)

        winner_text = decide_winner(gesture1, gesture2)
        config["winner"] = winner_text

        self.result_text = f"{config['player1_name']}: {gesture1} | " \
                           f"{config['player2_name']}: {gesture2} => {winner_text}"
        self.result_start_time = time.time()
        self.status_label.config(text="Status: " + self.result_text)

        if winner_text not in ["Tie", "Unknown"]:
            self.quit_flag = True  # Set the flag to indicate the game should quit
            self.root.quit()  # Ensure the main loop exits

        
    def quit_game(self):
        self.running = False

        # Cancel scheduled update_video callback, if any
        if self.update_video_id is not None:
            self.root.after_cancel(self.update_video_id)
            self.update_video_id = None

        self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()

def main(left_player_name="Player1", right_player_name="Player2"):
    """
    The main entry point, returns the final winner as in the previous code.
    """
    config["player1_name"] = left_player_name
    config["player2_name"] = right_player_name
    root = tk.Tk()
    app = RPSGameGUI(root)
    root.mainloop()

    if app.quit_flag:  # Check if the game should quit
        # Kill the GUI window, release the camera, and close all OpenCV windows
        # Delay to preserve the final frame for a few seconds
        time.sleep(3)
        root.destroy()
        app.cap.release()
        cv2.destroyAllWindows()
        return config["winner"]


if __name__ == "__main__":
    winner = main()
    print(f"Game over! Winner: {winner}")
