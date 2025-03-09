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
    if cap is not None and cap.isOpened():  # Check if the camera is initialized and opened
        ret, frame = cap.read()  # Capture a frame from the camera
        if ret:
            frame = cv2.flip(frame, 1)  # Mirror the frame horizontally
            height, width, _ = frame.shape  # Get the dimensions of the frame
            half_w = width // 2  # Calculate the midpoint of the width
            left_half = frame[:, :half_w]  # Extract the left half of the frame
            right_half = frame[:, half_w:]  # Extract the right half of the frame
            left_img = cv2.cvtColor(left_half, cv2.COLOR_BGR2RGB)  # Convert left half to RGB
            right_img = cv2.cvtColor(right_half, cv2.COLOR_BGR2RGB)  # Convert right half to RGB
            left_img_pil = Image.fromarray(left_img)  # Convert left half to PIL image
            right_img_pil = Image.fromarray(right_img)  # Convert right half to PIL image
            left_img_tk = ImageTk.PhotoImage(image=left_img_pil)  # Convert left PIL image to Tkinter image
            right_img_tk = ImageTk.PhotoImage(image=right_img_pil)  # Convert right PIL image to Tkinter image
            left_cam_label.configure(image=left_img_tk)  # Update the left camera label with the new image
            right_cam_label.configure(image=right_img_tk)  # Update the right camera label with the new image
            left_cam_label.image = left_img_tk  # Keep a reference to avoid garbage collection
            right_cam_label.image = right_img_tk  # Keep a reference to avoid garbage collection
    update_camera_feeds_id = root.after(30, update_camera_feeds)  # Schedule the function to run again after 30 ms

def take_pictures():
    """
    Captures one frame, flips it, splits into left/right halves, concatenates with hand images,
    and saves the resulting images.
    """
    ret, frame = cap.read()  # Capture a frame from the camera
    if not ret:
        return
    frame = cv2.flip(frame, 1)  # Mirror the frame horizontally
    mid_x = frame.shape[1] // 2  # Calculate the midpoint of the width
    left_frame = frame[:, :mid_x]  # Extract the left half of the frame
    right_frame = frame[:, mid_x:]  # Extract the right half of the frame
    left_hand = cv2.imread('left_hand.png')  # Read the left hand image
    right_hand = cv2.imread('right_hand.png')  # Read the right hand image
    target_height = right_frame.shape[0]  # Get the height of the right frame
    l_scaling_factor = target_height / left_hand.shape[0]  # Calculate the scaling factor for the left hand image
    r_scaling_factor = target_height / right_hand.shape[0]  # Calculate the scaling factor for the right hand image
    l_new_width = int(left_hand.shape[1] * l_scaling_factor)  # Calculate the new width for the left hand image
    r_new_width = int(right_hand.shape[1] * r_scaling_factor)  # Calculate the new width for the right hand image
    l_resized = cv2.resize(left_hand, (l_new_width, target_height), interpolation=cv2.INTER_LINEAR)  # Resize the left hand image
    r_resized = cv2.resize(right_hand, (r_new_width, target_height), interpolation=cv2.INTER_LINEAR)  # Resize the right hand image
    l_combined = np.hstack((l_resized, left_frame))  # Concatenate the left hand image with the left frame
    r_combined = np.hstack((right_frame, r_resized))  # Concatenate the right hand image with the right frame
    cv2.imwrite('left_frame.jpg', l_combined)  # Save the combined left image
    print("Left frame saved as 'left_frame.jpg'")  # Print a message indicating the left image was saved
    cv2.imwrite('right_frame.jpg', r_combined)  # Save the combined right image
    print("Right frame saved as 'right_frame.jpg'")  # Print a message indicating the right image was saved
    show_screen["pic"] = False  # Set the pic flag to False

def start_game(player_name, opponent_name):
    """
    Runs the RPS game to decide who starts and then runs the Pong game.
    The main GUI root is passed to rps.main so that the RPS game appears as a child window.
    """
    global destroy_flag
    winner = rps.main(player_name, opponent_name, parent=root)  # Run the Rock-Paper-Scissors game
    print("GUI: RPS winner:", winner)  # Print the RPS winner
    if winner == True:
        destroy_flag = True  # Set the destroy flag to True if the game was quit
        return
    # Start keys detection in a background thread.
    detection_thread = threading.Thread(target=keys_detection, args=(keys, show_screen))  # Create a new thread for keys detection
    detection_thread.daemon = True  # Set the thread as a daemon
    detection_thread.start()  # Start the thread
    starting = "left" if winner == f"{player_name} wins!" else "right"  # Determine the starting player based on the RPS result
    global update_camera_feeds_id
    if update_camera_feeds_id is not None:
        root.after_cancel(update_camera_feeds_id)  # Cancel the camera update callback if it exists
        update_camera_feeds_id = None
    result = game("player.png", "opponent.png", keys, show_screen, player_name, opponent_name, starting_player=starting)  # Run the Pong game
    return result

def on_start():
    """
    Called when "Start Game" is pressed.
    Cancels camera updates, releases the camera, hides the main window,
    runs the game, then calls game_finished to rebuild and show the GUI.
    """
    global cap, update_camera_feeds_id
    player_name = player_name_entry.get()  # Get the player's name from the entry widget
    opponent_name = opponent_name_entry.get()  # Get the opponent's name from the entry widget
    if not player_name or not opponent_name:  # If either name is empty
        messagebox.showerror("Error", "Please enter both player and opponent names.")  # Show an error message
        return
    if update_camera_feeds_id is not None:
        root.after_cancel(update_camera_feeds_id)  # Cancel the camera update callback if it exists
        update_camera_feeds_id = None
        root.after(35, lambda: None)  # Wait for 35 ms to ensure the callback is canceled
    if cap is not None and cap.isOpened():
        cap.release()  # Release the camera if it is opened
    cap = None
    root.withdraw()  # Hide the main window
    result = start_game(player_name, opponent_name)  # Start the game
    game_finished(result)  # Call game_finished to rebuild and show the GUI

def game_finished(result):
    """
    Called when the game finishes.
    Schedules a rebuild of the GUI and then deiconifies the main window.
    """
    root.after(200, rebuild_and_show)  # Schedule rebuild_and_show to run after 200 ms

def rebuild_and_show():
    global destroy_flag
    # Close game window.
    pygame.quit()  # Quit the Pygame window

    if not destroy_flag:
        restart_gui()  # Rebuild the GUI
        root.deiconify()  # Show the main window

def reset_camera_and_gui():
    """
    Reinitializes the camera and restarts camera feed updates.
    """
    global cap, update_camera_feeds_id
    if cap is None:
        cap_temp = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Initialize the camera
        cap_temp.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set the camera frame width
        cap_temp.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set the camera frame height
        if not cap_temp.isOpened():
            messagebox.showerror("Error", "Cannot open camera.")  # Show an error message if the camera cannot be opened
        else:
            cap = cap_temp
    if update_camera_feeds_id is None:
        update_camera_feeds()  # Start the camera feed updates

def on_quit():
    """
    Called when "Quit" is pressed.
    Cancels camera updates, releases the camera, and destroys the main window.
    """
    global cap, update_camera_feeds_id, destroy_flag
    if update_camera_feeds_id is not None:
        root.after_cancel(update_camera_feeds_id)  # Cancel the camera update callback if it exists
        update_camera_feeds_id = None
        root.after(35, lambda: None)  # Wait for 35 ms to ensure the callback is canceled
    if cap is not None and cap.isOpened():
        cap.release()  # Release the camera if it is opened
    cap = None
    destroy_flag = True  # Set the destroy flag to True
    root.destroy()  # Destroy the main window

def restart_gui():
    """
    Clears all widgets from the main window, rebuilds the GUI layout,
    reinitializes the camera, and restarts camera feed updates.
    """
    global left_cam_label, right_cam_label, player_name_entry, opponent_name_entry, cap
    for widget in root.winfo_children():
        widget.destroy()  # Destroy all existing widgets
    root.title("Tennis Game + Camera Feeds")  # Set the window title
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
        cap_temp = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Initialize the camera
        cap_temp.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set the camera frame width
        cap_temp.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set the camera frame height
        if not cap_temp.isOpened():
            messagebox.showerror("Error", "Cannot open camera.")  # Show an error message if the camera cannot be opened
        else:
            cap = cap_temp
    if not destroy_flag:
        update_camera_feeds()  # Start the camera feed updates

def main():
    """
    Main entry point: creates the root window, builds the GUI, and starts the Tkinter event loop.
    """
    global root, quit_flag
    root = tk.Tk()  # Create the main Tkinter window
    restart_gui()  # Build the GUI layout
    while (not destroy_flag):
        root.mainloop()  # Start the Tkinter event loop
    print("Exiting...")  # Print a message when exiting

if __name__ == "__main__":
    main()  # Run the main function if this script is executed directly