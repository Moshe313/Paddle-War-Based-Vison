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


root = None  # Add this line to define root as a global variable
cap = None
left_cam_label = None
right_cam_label = None
player_name_entry = None
opponent_name_entry = None

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
    def run_rps():
        # Schedule the RPS game to run on the main thread.
        winner = rps.main(root, player_name, opponent_name)
        # Determine starting side based on the winner...
        starting_side = "left" if winner == f"{player_name} wins!" else "right"
        root.after(500, lambda: start_countdown(3, starting_side, player_name, opponent_name))
        def start_countdown(n, starting_side, player_name, opponent_name):
            if n > 0:
                countdown_label.config(text=str(n))  # Update countdown text
                root.after(1000, start_countdown, n - 1, starting_side, player_name, opponent_name)  # Wait 1 second
            else:
                countdown_label.config(text="GO!")  # Show "GO!" for a moment
                root.after(500, lambda: start_pong(starting_side, player_name, opponent_name))  # Start Pong game after 0.5 seconds

    def start_pong(starting_side, player_name, opponent_name):
        countdown_label.pack_forget()  # Hide countdown label

        # Start key detection in a separate thread
        detection_thread = threading.Thread(target=keys_detection, args=(keys, show_screen))
        detection_thread.daemon = True
        detection_thread.start()

        # Start the Pong game
        game("player.png", "opponent.png", keys, show_screen, player_name, opponent_name, starting_player=starting_side)

        #pygame.quit() - elon
        root.after(100, restart_gui)  # Return to menu after game ends

    # Add a label for the countdown in the center of the GUI
    countdown_label = tk.Label(root, text="", font=("Arial", 50), fg="red", bg="white")
    countdown_label.place(relx=0.5, rely=0.5, anchor="center")

    # Run Rock-Paper-Scissors first in a separate thread
    threading.Thread(target=run_rps, daemon=True).start()

def on_start():
    player_name = player_name_entry.get()
    opponent_name = opponent_name_entry.get()
    
    if not player_name or not opponent_name:
        messagebox.showerror("Error", "Please enter both player and opponent names.")
        return

    # Stop camera updates
    global cap, update_camera_feeds_id
    if update_camera_feeds_id is not None:
        root.after_cancel(update_camera_feeds_id)

    if cap is not None and cap.isOpened():
        cap.release()
    cap = None

    # Run start_game() in a separate thread to prevent UI freeze
    threading.Thread(target=start_game, args=(player_name, opponent_name), daemon=True).start()


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

    # Hide any existing game windows (prevents crashes)
    if root is not None:
        for widget in root.winfo_children():
            widget.destroy()
    else:
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
    global root
    root = tk.Tk()  # Initialize root before calling restart_gui
    root.withdraw()  # Hide the root window initially

    # Start the GUI
    while True:
        # Destroy everything before restarting
        restart_gui()
        root.deiconify()  # Show the root window
        root.mainloop()
        if root.winfo_exists():
            root.withdraw()  # Hide the root window after mainloop ends


if __name__ == "__main__":
    main()