import tkinter as tk
from tkinter import messagebox
import threading
import pygame
import cv2
import numpy as np
from PIL import Image, ImageTk  # To fix color issues in Tkinter

from Game import game
from Image_processing import keys_detection
import RockPaperScissors as rps

# Global Variables
keys = {"player left": False, "player right": False, "opponent left": False, "opponent right": False, "starts": "left"}
show_screen = {"show_screen": False, "game_width": 550, "pic": False, "video_prepared": False}
update_camera_feeds_id = None
update_video_id = None


# We'll open one camera capture shared across the GUI
cap = None

# Tkinter labels for the camera feeds
left_cam_label = None
right_cam_label = None

def update_camera_feeds():
    """
    Continuously read frames from `cap`, split them into left and right halves,
    convert to a Tkinter-compatible image, and display them on left_cam_label / right_cam_label.
    """
    global cap, left_cam_label, right_cam_label

    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Mirror horizontally

            # Split into left / right
            height, width, _ = frame.shape
            half_w = width // 2
            left_half = frame[:, :half_w]
            right_half = frame[:, half_w:]

            # Convert to normal RGB (no bluish tone) using PIL
            left_half_rgb = cv2.cvtColor(left_half, cv2.COLOR_BGR2RGB)
            right_half_rgb = cv2.cvtColor(right_half, cv2.COLOR_BGR2RGB)

            # Create ImageTk objects
            left_img_pil = Image.fromarray(left_half_rgb)
            right_img_pil = Image.fromarray(right_half_rgb)

            left_img_tk = ImageTk.PhotoImage(image=left_img_pil)
            right_img_tk = ImageTk.PhotoImage(image=right_img_pil)

            # Update labels
            left_cam_label.configure(image=left_img_tk)
            right_cam_label.configure(image=right_img_tk)

            # Keep references so they don't get GC'ed
            left_cam_label.image = left_img_tk
            right_cam_label.image = right_img_tk

    update_camera_feeds_id = root.after(30, update_camera_feeds)



def take_pictures():
    """
    1) Open the camera with DirectShow (avoids MSMF on Windows).
    2) Attempt 1280x720 resolution.
    3) Capture exactly one frame, flip horizontally.
    4) Split into left_frame and right_frame.
    5) Read left_hand.png / right_hand.png; scale them to match height.
    6) np.hstack(...) => left_frame.jpg and right_frame.jpg
    7) Release camera and return (no continuous capture).
    """
    # Open camera 
    ret, frame = cap.read()
    if not ret:
        return

    # Mirror the frame horizontally
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
    l_resized_hand = cv2.resize(left_hand, (l_new_width, target_height), interpolation=cv2.INTER_LINEAR)
    r_resized_hand = cv2.resize(right_hand, (r_new_width, target_height), interpolation=cv2.INTER_LINEAR)
    # Concatenate to form combined images for saving
    l_combined_image = np.hstack((l_resized_hand, left_frame))
    r_combined_image = np.hstack((right_frame, r_resized_hand))
    cv2.imwrite('left_frame.jpg', l_combined_image)
    print("Left frame saved as 'left_frame.jpg'")
    cv2.imwrite('right_frame.jpg', r_combined_image)
    print("Right frame saved as 'right_frame.jpg'")
    show_screen["pic"] = False
   



def start_game(player_name, opponent_name):
    """
    Called after the GUI is destroyed to run the RPS + Pong game.
    """
    # Run rock-paper-scissors to determine the player who starts
    winner = rps.main(player_name, opponent_name)

    # Start detection thread (keys_detection updates 'keys' in real-time)
    detection_thread = threading.Thread(target=keys_detection, args=(keys, show_screen))
    detection_thread.daemon = True
    detection_thread.start()

    if (winner == f"{player_name} wins!"):
        winner = "left"
    else:
        winner = "right"

    # Run the Pong game
    game("player.png", "opponent.png", keys, show_screen, player_name, opponent_name, starting_player= winner)

    # Close the game window
    pygame.quit()

    # Restart the GUI
    #restart_gui()

def on_start():

    player_name = player_name_entry.get()
    opponent_name = opponent_name_entry.get()
    if not player_name or not opponent_name:
        messagebox.showerror("Error", "Please enter both player and opponent names.")
        return
    
   
    global cap, update_camera_feeds_id

    # Cancel the camera update callback
    if update_camera_feeds_id is not None:
        root.after_cancel(update_camera_feeds_id)

    # Release the camera
    if cap is not None and cap.isOpened():
        cap.release()
    cap = None

    root.destroy()


    # Proceed with the RPS + Pong flow
    start_game(player_name, opponent_name)

def on_quit():

    global cap, update_camera_feeds_id

    # Cancel the camera update callback
    if update_camera_feeds_id is not None:
        root.after_cancel(update_camera_feeds_id)

    # Release the camera
    if cap is not None and cap.isOpened():
        cap.release()
    cap = None

    root.destroy()



def restart_gui():
    global root, cap
    global left_cam_label, right_cam_label
    global player_name_entry, opponent_name_entry

    root = tk.Tk()
    root.title("Tennis Game + Camera Feeds")

    # 3-column layout
    left_frame = tk.Frame(root, bg="lightblue")
    center_frame = tk.Frame(root, bg="white")
    right_frame = tk.Frame(root, bg="lightblue")

    left_frame.grid(row=0, column=0, sticky="ns")
    center_frame.grid(row=0, column=1, sticky="nsew")
    right_frame.grid(row=0, column=2, sticky="ns")

    # Expand center frame on resize
    root.columnconfigure(1, weight=1)

    # LEFT CAMERA
    left_cam_label = tk.Label(left_frame, text="Left Camera Feed", bg="lightblue")
    left_cam_label.pack(padx=10, pady=10)

    # CENTER MENU
    tk.Label(center_frame, text="Left Player's Name:", bg="white").pack(pady=10)
    player_name_entry = tk.Entry(center_frame)
    player_name_entry.pack(pady=5)

    tk.Label(center_frame, text="Right Player's Name:", bg="white").pack(pady=10)
    opponent_name_entry = tk.Entry(center_frame)
    opponent_name_entry.pack(pady=5)

    # Button: Take pictures
    take_pics_button = tk.Button(center_frame, text="Take Players Pictures", command=take_pictures)
    take_pics_button.pack(pady=10)

    # Start & Quit buttons
    start_button = tk.Button(center_frame, text="Start Game", command=on_start)
    start_button.pack(pady=10)

    quit_button = tk.Button(center_frame, text="Quit", command=on_quit)
    quit_button.pack(pady=10)

    # RIGHT CAMERA
    right_cam_label = tk.Label(right_frame, text="Right Camera Feed", bg="lightblue")
    right_cam_label.pack(padx=10, pady=10)

    # Open camera if needed
    if cap is None:
        cap_temp = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap_temp.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap_temp.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap_temp.isOpened():
            messagebox.showerror("Error", "Cannot open camera.")
        else:
            cap = cap_temp

    # Start updating camera feeds
    update_camera_feeds()

    

def main():
    # Start the GUI
    while True:
        # Destroy everything before restarting
        restart_gui()
        root.mainloop()


if __name__ == "__main__":
    main()