import tkinter as tk
import threading
import time
import cv2
import numpy as np
from PIL import Image, ImageTk

config = {
    "window_name": "Rock Paper Scissors",
    "window_size": (900, 600),
    "player1_name": "Player1",
    "player2_name": "Player2",
    "roi1_coords": (50, 50, 250, 250),
    "roi2_coords": (300, 50, 500, 250),
    "countdown_duration": 3,
    "result_display_time": 5,
    "num_train_frames": 30,
    "debug": True,
    "winner": None
}

THRESHOLD = 60.0
lower_skin = np.array([0, 80, 60], dtype=np.uint8)
upper_skin = np.array([20, 150, 255], dtype=np.uint8)

models = {"model1": None, "model2": None}

# --------------------------------------------------
# Gaussian Model Helpers
# --------------------------------------------------
def fit_gaussian_model(pixels, threshold=THRESHOLD):
    pixels = np.asarray(pixels, dtype=np.float64)
    mean = np.mean(pixels, axis=0)
    cov = np.cov(pixels, rowvar=False)
    inv_cov = np.linalg.inv(cov + 1e-8 * np.eye(3))
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

# --------------------------------------------------
# Gesture Classification
# --------------------------------------------------
def approximate_hand_gesture(mask):
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
        s, e, f, d = defects[i, 0]
        a = np.linalg.norm(approx[e][0] - approx[s][0])
        b = np.linalg.norm(approx[f][0] - approx[s][0])
        c = np.linalg.norm(approx[e][0] - approx[f][0])
        if b * c != 0:
            angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * 57
            if angle <= 90:
                count_defects += 1

    return ["Rock", "Scissors", "Paper"][min(count_defects, 2)]

def decide_winner(g1, g2):
    rules = {"Rock": "Scissors", "Scissors": "Paper", "Paper": "Rock"}
    if g1 == g2:
        return "Tie"
    elif g1 not in rules or g2 not in rules:
        return "Unknown"
    elif rules[g1] == g2:
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
        self.running = True
        self.calibrating = False
        self.calibrated_players = {"Player1": False, "Player2": False}
        self.calibration_frame = 0
        self.current_detection = {"Player1": "Waiting", "Player2": "Waiting"}
        self.update_video()

    def update_video(self):
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

            # Initialize empty masks
            mask1 = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            mask2 = np.zeros((yy2 - yy1, xx2 - xx1), dtype=np.uint8)

            roi1, roi2 = None, None  # Default values in case they are not processed

            # Show real-time calibration progress if calibrating
            if self.calibrating:
                if self.calibrating_player == 1:
                    text1 = f"Calibrating frame {self.calibration_frame}/{config['num_train_frames']}"
                    cv2.putText(frame, text1, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                elif self.calibrating_player == 2:
                    text2 = f"Calibrating frame {self.calibration_frame}/{config['num_train_frames']}"
                    cv2.putText(frame, text2, (xx1, yy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # Perform real-time hand detection for calibrated players
                if self.calibrated_players["Player1"]:
                    roi1 = frame[y1:y2, x1:x2]
                    hsv1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
                    mask1 = are_pixels_in_distribution(models["model1"], hsv1)
                    self.current_detection["Player1"] = approximate_hand_gesture(mask1)

                if self.calibrated_players["Player2"]:
                    roi2 = frame[yy1:yy2, xx1:xx2]
                    hsv2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
                    mask2 = are_pixels_in_distribution(models["model2"], hsv2)
                    self.current_detection["Player2"] = approximate_hand_gesture(mask2)

                # Display the detected gestures above ROIs
                text1 = f"{config['player1_name']}: {self.current_detection['Player1']}"
                text2 = f"{config['player2_name']}: {self.current_detection['Player2']}"
                cv2.putText(frame, text1, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(frame, text2, (xx1, yy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Show debug information only if the player has been calibrated
            if config["debug"] and (self.calibrated_players["Player1"] or self.calibrated_players["Player2"]):
                self.show_debug_window(mask1, mask2, roi1, roi2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        self.root.after(10, self.update_video)


    def show_debug_window(self, mask1, mask2, roi1, roi2):
        # If the player is not calibrated, show a black frame instead of None
        if roi1 is None:
            roi1 = np.zeros((mask1.shape[0], mask1.shape[1], 3), dtype=np.uint8)
        if roi2 is None:
            roi2 = np.zeros((mask2.shape[0], mask2.shape[1], 3), dtype=np.uint8)

        debug_frame1 = cv2.bitwise_and(roi1, roi1, mask=mask1)
        debug_frame2 = cv2.bitwise_and(roi2, roi2, mask=mask2)

        cv2.putText(debug_frame1, f"Detected: {self.current_detection['Player1']}", (5, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_frame2, f"Detected: {self.current_detection['Player2']}", (5, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Concatenate debug frames for both players
        combined_debug = np.hstack((debug_frame1, debug_frame2))

        cv2.imshow("Debug Window", combined_debug)
        cv2.waitKey(1)


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
        for i in range(config["num_train_frames"]):
            self.calibration_frame = i + 1
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                roi = frame[config["roi1_coords"][1]:config["roi1_coords"][3], 
                            config["roi1_coords"][0]:config["roi1_coords"][2]] if which_player == 1 else \
                      frame[config["roi2_coords"][1]:config["roi2_coords"][3], 
                            config["roi2_coords"][0]:config["roi2_coords"][2]]
                
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower_skin, upper_skin)
                pixels.extend(hsv[mask > 0])

            time.sleep(0.05)

        models[f"model{which_player}"] = fit_gaussian_model(pixels)
        self.calibrated_players[f"Player{which_player}"] = True
        self.status_label.config(text=f"Status: {p_name} Calibration done!")
        self.calibrating = False

    def start_rps(self):
        if not all(self.calibrated_players.values()):
            self.status_label.config(text="Status: Both players must calibrate first!")
            return

        self.status_label.config(text="Status: Playing RPS...")
        threading.Thread(target=self._start_rps_thread, daemon=True).start()

    def quit_game(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()




def main(left_player_name="Player1", right_player_name="Player2"):
    config["player1_name"] = left_player_name
    config["player2_name"] = right_player_name
    root = tk.Tk()
    app = RPSGameGUI(root)
    root.mainloop()
    
    return config["winner"]

if __name__ == "__main__":
    winner = main()
    print(f"Game over! Winner: {winner}")

    

