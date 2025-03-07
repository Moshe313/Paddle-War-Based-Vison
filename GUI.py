import tkinter as tk                     # For GUI operations
from tkinter import messagebox           # For error messages
import threading                         # For threading support
import pygame                            # For game functionalities
import cv2                               # For camera capture
import numpy as np                       # For numerical operations
from PIL import Image, ImageTk           # For converting images for Tkinter

from Game import game                    # Import the Pong game function
from Image_processing import keys_detection  # For keys detection
import RockPaperScissors as rps          # Import the RPS game module

# Global state variables.
keys = {
    "player left": False,
    "player right": False,
    "opponent left": False,
    "opponent right": False,
    "starts": "left",
    "speed_limit": 20
}
show_screen = {
    "show_screen": False,
    "game_width": 550,
    "pic": False,
    "video_prepared": False
}
update_camera_feeds_id = None  # ID for the camera update callback.
destroy_flag = False           # True when the GUI is being destroyed.
cap = None                     # Global camera capture object.
left_cam_label = None          # Left camera feed label.
right_cam_label = None         # Right camera feed label.

def update_camera_feeds():
    """
    Captures frames from the camera, splits them into left/right halves,
    converts them into a Tkinter-compatible image, and updates the corresponding labels.
    Schedules itself every 30 ms.
    """
    global cap, left_cam_label, right_cam_label, update_camera_feeds_id
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Mirror the frame.
            height, width, _ = frame.shape
            half_w = width // 2
            left_half = frame[:, :half_w]
            right_half = frame[:, half_w:]
            left_img = cv2.cvtColor(left_half, cv2.COLOR_BGR2RGB)
            right_img = cv2.cvtColor(right_half, cv2.COLOR_BGR2RGB)
            left_img_pil = Image.fromarray(left_img)
            right_img_pil = Image.fromarray(right_img)
            left_img_tk = ImageTk.PhotoImage(image=left_img_pil)
            right_img_tk = ImageTk.PhotoImage(image=right_img_pil)
            left_cam_label.configure(image=left_img_tk)
            right_cam_label.configure(image=right_img_tk)
            left_cam_label.image = left_img_tk  # Keep reference.
            right_cam_label.image = right_img_tk
    update_camera_feeds_id = root.after(30, update_camera_feeds)

def take_pictures():
    """
    Captures one frame, flips it, splits into left/right halves, concatenates with hand images,
    and saves the resulting images.
    """
    ret, frame = cap.read()
    if not ret:
        return
    frame = cv2.flip(frame, 1)
    mid_x = frame.shape[1] // 2
    left_frame = frame[:, :mid_x]
    right_frame = frame[:, mid_x:]
    left_hand = cv2.imread('left_hand.png')
    right_hand = cv2.imread('right_hand.png')
    target_height = right_frame.shape[0]
    l_scaling_factor = target_height / left_hand.shape[0]
    r_scaling_factor = target_height / right_hand.shape[0]
    l_new_width = int(left_hand.shape[1] * l_scaling_factor)
    r_new_width = int(right_hand.shape[1] * r_scaling_factor)
    l_resized = cv2.resize(left_hand, (l_new_width, target_height), interpolation=cv2.INTER_LINEAR)
    r_resized = cv2.resize(right_hand, (r_new_width, target_height), interpolation=cv2.INTER_LINEAR)
    l_combined = np.hstack((l_resized, left_frame))
    r_combined = np.hstack((right_frame, r_resized))
    cv2.imwrite('left_frame.jpg', l_combined)
    print("Left frame saved as 'left_frame.jpg'")
    cv2.imwrite('right_frame.jpg', r_combined)
    print("Right frame saved as 'right_frame.jpg'")
    show_screen["pic"] = False

def start_game(player_name, opponent_name):
    """
    Runs the RPS game to decide who starts and then runs the Pong game.
    The main GUI root is passed to rps.main so that the RPS game appears as a child window.
    """
    global destroy_flag
    winner = rps.main(player_name, opponent_name, parent=root)
    if winner == True:
        destroy_flag = True
        return
    # Start keys detection in a background thread.
    detection_thread = threading.Thread(target=keys_detection, args=(keys, show_screen))
    detection_thread.daemon = True
    detection_thread.start()
    starting = "left" if winner == f"{player_name} wins!" else "right"
    global update_camera_feeds_id
    if update_camera_feeds_id is not None:
        root.after_cancel(update_camera_feeds_id)
        update_camera_feeds_id = None
    result = game("player.png", "opponent.png", keys, show_screen, player_name, opponent_name, starting_player=starting)
    return result

def on_start():
    """
    Called when "Start Game" is pressed.
    Cancels camera updates, releases the camera, hides the main window,
    runs the game, then calls game_finished to rebuild and show the GUI.
    """
    global cap, update_camera_feeds_id
    player_name = player_name_entry.get()
    opponent_name = opponent_name_entry.get()
    if not player_name or not opponent_name:
        messagebox.showerror("Error", "Please enter both player and opponent names.")
        return
    if update_camera_feeds_id is not None:
        root.after_cancel(update_camera_feeds_id)
        update_camera_feeds_id = None
        root.after(35, lambda: None)
    if cap is not None and cap.isOpened():
        cap.release()
    cap = None
    root.withdraw()  # Hide main window.
    result = start_game(player_name, opponent_name)
    game_finished(result)

def game_finished(result):
    """
    Called when the game finishes.
    Schedules a rebuild of the GUI and then deiconifies the main window.
    """
    root.after(200, rebuild_and_show)

def rebuild_and_show():
    global destroy_flag
    # Close game window.
    pygame.quit()

    if not destroy_flag:
        restart_gui()
        root.deiconify()

def reset_camera_and_gui():
    """
    Reinitializes the camera and restarts camera feed updates.
    """
    global cap, update_camera_feeds_id
    if cap is None:
        cap_temp = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap_temp.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap_temp.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap_temp.isOpened():
            messagebox.showerror("Error", "Cannot open camera.")
        else:
            cap = cap_temp
    if update_camera_feeds_id is None:
        update_camera_feeds()

def on_quit():
    """
    Called when "Quit" is pressed.
    Cancels camera updates, releases the camera, and destroys the main window.
    """
    global cap, update_camera_feeds_id, destroy_flag
    if update_camera_feeds_id is not None:
        root.after_cancel(update_camera_feeds_id)
        update_camera_feeds_id = None
        root.after(35, lambda: None)
    if cap is not None and cap.isOpened():
        cap.release()
    cap = None
    destroy_flag = True
    root.destroy()

def restart_gui():
    """
    Clears all widgets from the main window, rebuilds the GUI layout,
    reinitializes the camera, and restarts camera feed updates.
    """
    global left_cam_label, right_cam_label, player_name_entry, opponent_name_entry, cap
    for widget in root.winfo_children():
        widget.destroy()
    root.title("Tennis Game + Camera Feeds")
    # Create left, center, and right frames.
    left_frame = tk.Frame(root, bg="lightgreen")
    center_frame = tk.Frame(root, bg="lightgreen")
    right_frame = tk.Frame(root, bg="lightgreen")
    left_frame.grid(row=0, column=0, sticky="ns")
    center_frame.grid(row=0, column=1, sticky="nsew")
    right_frame.grid(row=0, column=2, sticky="ns")
    root.columnconfigure(1, weight=1)
    # Left: Camera feed.
    left_cam_label = tk.Label(left_frame, text="Left Camera Feed", bg="lightgreen")
    left_cam_label.pack(padx=10, pady=10)
    # Center: Control panel.
    tk.Label(center_frame, text="Left Player's Name:", bg="white").pack(pady=10)
    player_name_entry = tk.Entry(center_frame)
    player_name_entry.pack(pady=5)
    tk.Label(center_frame, text="Right Player's Name:", bg="white").pack(pady=10)
    opponent_name_entry = tk.Entry(center_frame)
    opponent_name_entry.pack(pady=5)
    take_pics_button = tk.Button(center_frame, text="Take Players Pictures", command=take_pictures)
    take_pics_button.pack(pady=10)
    start_button = tk.Button(center_frame, text="Start Game", command=on_start)
    start_button.pack(pady=10)
    quit_button = tk.Button(center_frame, text="Quit", command=on_quit)
    quit_button.pack(pady=10)
    # Right: Camera feed.
    right_cam_label = tk.Label(right_frame, text="Right Camera Feed", bg="lightgreen")
    right_cam_label.pack(padx=10, pady=10)
    # Initialize camera if needed.
    if cap is None:
        cap_temp = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap_temp.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap_temp.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap_temp.isOpened():
            messagebox.showerror("Error", "Cannot open camera.")
        else:
            cap = cap_temp
    if not destroy_flag:
        update_camera_feeds()

def main():
    """
    Main entry point: creates the root window, builds the GUI, and starts the Tkinter event loop.
    """
    global root, quit_flag
    root = tk.Tk()
    restart_gui()
    while (not destroy_flag):
     root.mainloop()
    print("Exiting...")

if __name__ == "__main__":
    main()
