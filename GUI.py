import tkinter as tk  # Import the Tkinter module for GUI
from tkinter import messagebox  # Import messagebox for displaying error messages
import threading  # Import threading for running tasks in parallel
import pygame  # Import pygame for game functionalities
import cv2  # Import OpenCV for camera functionalities
import numpy as np  # Import numpy for numerical operations
from PIL import Image, ImageTk  # Import PIL for image processing and displaying in Tkinter

from Game import game  # Import the game module
from Image_processing import keys_detection  # Import keys_detection from Image_processing module
import RockPaperScissors as rps  # Import RockPaperScissors module

# Global Variables
keys = {"player left": False, "player right": False, "opponent left": False, "opponent right": False, "starts": "left", "speed_limit": 20}  # Dictionary to store key states
show_screen = {"show_screen": False, "game_width": 550, "pic": False, "video_prepared": False}  # Dictionary to store screen states
update_camera_feeds_id = None  # Variable to store the ID of the camera feed update callback
update_video_id = None  # Variable to store the ID of the video update callback
destroy_flag = False  # Flag to indicate if the GUI has to be destroyed
return_game_status = True  # Flag to indicate if the game was quit or not

# We'll open one camera capture shared across the GUI
cap = None  # Variable to store the camera capture object

# Tkinter labels for the camera feeds
left_cam_label = None  # Label for the left camera feed
right_cam_label = None  # Label for the right camera feed

def update_camera_feeds():
    """
    Continuously read frames from `cap`, split them into left and right halves,
    convert to a Tkinter-compatible image, and display them on left_cam_label / right_cam_label.
    """
    global cap, left_cam_label, right_cam_label  # Use global variables

    if cap is not None and cap.isOpened():  # Check if the camera is opened
        ret, frame = cap.read()  # Read a frame from the camera
        if ret:  # If the frame is read successfully
            frame = cv2.flip(frame, 1)  # Mirror the frame horizontally

            # Split into left / right
            height, width, _ = frame.shape  # Get the dimensions of the frame
            half_w = width // 2  # Calculate the width of each half
            left_half = frame[:, :half_w]  # Get the left half of the frame
            right_half = frame[:, half_w:]  # Get the right half of the frame

            # Convert to normal RGB (no bluish tone) using PIL
            left_half_rgb = cv2.cvtColor(left_half, cv2.COLOR_BGR2RGB)  # Convert the left half to RGB
            right_half_rgb = cv2.cvtColor(right_half, cv2.COLOR_BGR2RGB)  # Convert the right half to RGB

            # Create ImageTk objects
            left_img_pil = Image.fromarray(left_half_rgb)  # Create a PIL image from the left half
            right_img_pil = Image.fromarray(right_half_rgb)  # Create a PIL image from the right half

            left_img_tk = ImageTk.PhotoImage(image=left_img_pil)  # Create an ImageTk object from the left PIL image
            right_img_tk = ImageTk.PhotoImage(image=right_img_pil)  # Create an ImageTk object from the right PIL image

            # Update labels
            left_cam_label.configure(image=left_img_tk)  # Update the left camera label with the new image
            right_cam_label.configure(image=right_img_tk)  # Update the right camera label with the new image

            # Keep references so they don't get GC'ed
            left_cam_label.image = left_img_tk  # Keep a reference to the left image
            right_cam_label.image = right_img_tk  # Keep a reference to the right image

    update_camera_feeds_id = root.after(30, update_camera_feeds)  # Schedule the next update in 30 milliseconds

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
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:  # If the frame is not read successfully
        return  # Return without doing anything

    # Mirror the frame horizontally
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    mid_x = frame.shape[1] // 2  # Calculate the midpoint of the frame
    left_frame = frame[:, :mid_x]  # Get the left half of the frame
    right_frame = frame[:, mid_x:]  # Get the right half of the frame
    
    left_hand = cv2.imread('left_hand.png')  # Read the left hand image
    right_hand = cv2.imread('right_hand.png')  # Read the right hand image
    target_height = right_frame.shape[0]  # Get the height of the right frame
    l_scaling_factor = target_height / left_hand.shape[0]  # Calculate the scaling factor for the left hand image
    r_scaling_factor = target_height / right_hand.shape[0]  # Calculate the scaling factor for the right hand image
    l_new_width = int(left_hand.shape[1] * l_scaling_factor)  # Calculate the new width of the left hand image
    r_new_width = int(right_hand.shape[1] * r_scaling_factor)  # Calculate the new width of the right hand image
    l_resized_hand = cv2.resize(left_hand, (l_new_width, target_height), interpolation=cv2.INTER_LINEAR)  # Resize the left hand image
    r_resized_hand = cv2.resize(right_hand, (r_new_width, target_height), interpolation=cv2.INTER_LINEAR)  # Resize the right hand image
    # Concatenate to form combined images for saving
    l_combined_image = np.hstack((l_resized_hand, left_frame))  # Concatenate the left hand image and the left frame
    r_combined_image = np.hstack((right_frame, r_resized_hand))  # Concatenate the right frame and the right hand image
    cv2.imwrite('left_frame.jpg', l_combined_image)  # Save the combined left image
    print("Left frame saved as 'left_frame.jpg'")  # Print a message indicating that the left image has been saved
    cv2.imwrite('right_frame.jpg', r_combined_image)  # Save the combined right image
    print("Right frame saved as 'right_frame.jpg'")  # Print a message indicating that the right image has been saved
    show_screen["pic"] = False  # Set the pic flag to False

def start_game(player_name, opponent_name):
    """
    Called after the GUI is destroyed to run the RPS + Pong game.
    """
    # Run rock-paper-scissors to determine the player who starts
    winner = rps.main(player_name, opponent_name)  # Run the Rock-Paper-Scissors game

    if winner == True:  # If the game was quit
        return False  # Return False

    # Start detection thread (keys_detection updates 'keys' in real-time)
    detection_thread = threading.Thread(target=keys_detection, args=(keys, show_screen))  # Create a new thread for keys detection
    detection_thread.daemon = True  # Set the thread as a daemon thread
    detection_thread.start()  # Start the thread

    if (winner == f"{player_name} wins!"):  # If the player wins
        winner = "left"  # Set the winner to left
    else:  # If the opponent wins
        winner = "right"  # Set the winner to right

    # Run the Pong game
    game("player.png", "opponent.png", keys, show_screen, player_name, opponent_name, starting_player= winner)  # Run the Pong game

    # Close the game window
    pygame.quit()  # Quit pygame

    # Restart the GUI
    #restart_gui()  # Restart the GUI (commented out)

def on_start():
    """
    Called when the "Start Game" button is pressed.
    """
    global return_game_status  # Use global variables
    player_name = player_name_entry.get()  # Get the player's name from the entry widget
    opponent_name = opponent_name_entry.get()  # Get the opponent's name from the entry widget
    if not player_name or not opponent_name:  # If either name is empty
        messagebox.showerror("Error", "Please enter both player and opponent names.")  # Show an error message
        return  # Return without doing anything
    
    global cap, update_camera_feeds_id  # Use global variables

    # Cancel the camera update callback
    if update_camera_feeds_id is not None:  # If the update callback is set
        root.after_cancel(update_camera_feeds_id)  # Cancel the update callback
        # Make sure update_camera_feeds() doesn't run again, waiting for the next 35ms (root)
        update_camera_feeds_id = None  # Set the update callback ID to None
        # Root wait 35ms:
        def dummy():
            pass
        root.after(35, dummy)
        

    # Release the camera
    if cap is not None and cap.isOpened():  # If the camera is opened
        cap.release()  # Release the camera
    cap = None  # Set the camera variable to None

    root.destroy()  # Destroy the root window

    # Proceed with the RPS + Pong flow
    return_game_status = start_game(player_name, opponent_name)  # Start the game

def on_quit():
    """
    Called when the "Quit" button is pressed.
    """
    global cap, update_camera_feeds_id, destroy_flag  # Use global variables

    # Cancel the camera update callback
    if update_camera_feeds_id is not None:  # If the update callback is set
        root.after_cancel(update_camera_feeds_id)  # Cancel the update callback
        # Make sure update_camera_feeds() doesn't run again, waiting for the next 35ms (root)
        update_camera_feeds_id = None  # Set the update callback ID to None
        # Root wait 35ms:
        def dummy():
            pass
        root.after(35, dummy)

    # Release the camera
    if cap is not None and cap.isOpened():  # If the camera is opened
        cap.release()  # Release the camera
    cap = None  # Set the camera variable to None

    destroy_flag = True  # Set the destroy flag to True
    root.destroy()  # Destroy the root window

def restart_gui():
    """
    Initializes and restarts the GUI.
    """
    global root, cap, destroy_flag  # Use global variables
    global left_cam_label, right_cam_label  # Use global variables
    global player_name_entry, opponent_name_entry, return_game_status  # Use global variables

    root = tk.Tk()  # Create a new Tkinter root window
    root.title("Tennis Game + Camera Feeds")  # Set the title of the window

    # 3-column layout
    left_frame = tk.Frame(root, bg="lightgreen")  # Create a frame for the left column
    center_frame = tk.Frame(root, bg="lightgreen")  # Create a frame for the center column
    right_frame = tk.Frame(root, bg="lightgreen")  # Create a frame for the right column

    left_frame.grid(row=0, column=0, sticky="ns")  # Place the left frame in the grid
    center_frame.grid(row=0, column=1, sticky="nsew")  # Place the center frame in the grid
    right_frame.grid(row=0, column=2, sticky="ns")  # Place the right frame in the grid

    # Expand center frame on resize
    root.columnconfigure(1, weight=1)  # Allow the center frame to expand when the window is resized

    # LEFT CAMERA
    left_cam_label = tk.Label(left_frame, text="Left Camera Feed", bg="lightgreen")  # Create a label for the left camera feed
    left_cam_label.pack(padx=10, pady=10)  # Pack the label with padding

    # CENTER MENU
    tk.Label(center_frame, text="Left Player's Name:", bg="white").pack(pady=10)  # Create and pack a label for the left player's name
    player_name_entry = tk.Entry(center_frame)  # Create an entry widget for the left player's name
    player_name_entry.pack(pady=5)  # Pack the entry widget with padding

    tk.Label(center_frame, text="Right Player's Name:", bg="white").pack(pady=10)  # Create and pack a label for the right player's name
    opponent_name_entry = tk.Entry(center_frame)  # Create an entry widget for the right player's name
    opponent_name_entry.pack(pady=5)  # Pack the entry widget with padding

    # Button: Take pictures
    take_pics_button = tk.Button(center_frame, text="Take Players Pictures", command=take_pictures)  # Create a button to take pictures
    take_pics_button.pack(pady=10)  # Pack the button with padding

    # Start & Quit buttons
    start_button = tk.Button(center_frame, text="Start Game", command=on_start)  # Create a button to start the game
    start_button.pack(pady=10)  # Pack the button with padding

    quit_button = tk.Button(center_frame, text="Quit", command=on_quit)  # Create a button to quit the game
    quit_button.pack(pady=10)  # Pack the button with padding

    # RIGHT CAMERA
    right_cam_label = tk.Label(right_frame, text="Right Camera Feed", bg="lightgreen")  # Create a label for the right camera feed
    right_cam_label.pack(padx=10, pady=10)  # Pack the label with padding

    # Open camera if needed
    if cap is None:  # If the camera is not opened
        cap_temp = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Open the camera with DirectShow
        cap_temp.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set the camera resolution to 1280x720
        cap_temp.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set the camera resolution to 1280x720

        if not cap_temp.isOpened():  # If the camera is not opened successfully
            messagebox.showerror("Error", "Cannot open camera.")  # Show an error message
        else:  # If the camera is opened successfully
            cap = cap_temp  # Set the camera variable to the opened camera

    # Start updating camera feeds if not quit
    if not destroy_flag and return_game_status != False:  # If the GUI is not quit
        update_camera_feeds()  # Start updating the camera feeds


def main():
    """
    Main function to start the GUI.
    """
    global root, destroy_flag, return_game_status  # Use global variables
    # Start the GUI
    while not destroy_flag:  # Infinite loop to restart the GUI
        restart_gui()  # Restart the GUI
        root.mainloop()  # Start the Tkinter main loop
        if return_game_status == False:  # If the game was quit
            on_quit()  # Quit the game if the game was quit
            
    


if __name__ == "__main__":  # If this script is run directly
    main()  # Call the main function