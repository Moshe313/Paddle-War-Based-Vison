import tkinter as tk
from tkinter import messagebox
from tkinter import PhotoImage
from Game import game
import threading
from Image_processing import keys_detection

# Global Variables
keys = {"player left": False, "player right": False, "opponent left": False, "opponent right": False}
show_screen = {"show_screen": False, "game_width": 550, "pic": False, "video_prepared": False}


def start_game(player_name, opponent_name):
    # Start detection thread
    detection_thread = threading.Thread(target=keys_detection, args=(keys, show_screen))
    detection_thread.daemon = True
    detection_thread.start()

    # Run the game
    game("player.png", "opponent.png", keys, show_screen, player_name, opponent_name)

    # Restart the GUI after the game ends
    restart_gui()


def on_start():
    player_name = player_name_entry.get()
    opponent_name = opponent_name_entry.get()
    if not player_name or not opponent_name:
        messagebox.showerror("Error", "Please enter both player and opponent names.")
        return
    root.destroy()
    start_game(player_name, opponent_name)

def on_quit():
    root.destroy()

def restart_gui():
    global root, player_name_entry, opponent_name_entry, start_button, quit_button, background_label
    root = tk.Tk()
    root.title("Tennis Game")

    # Load the background image
    background_image = PhotoImage(file="GUI_BG.png")
    background_label = tk.Label(root, image=background_image)
    background_label.place(relwidth=1, relheight=1)

    # Create and place the labels and entry widgets
    tk.Label(root, text="Left Player's Name:", bg="white").grid(row=0, column=0, padx=10, pady=10)
    player_name_entry = tk.Entry(root)
    player_name_entry.grid(row=0, column=1, padx=10, pady=10)

    tk.Label(root, text="Right Player's Name:", bg="white").grid(row=1, column=0, padx=10, pady=10)
    opponent_name_entry = tk.Entry(root)
    opponent_name_entry.grid(row=1, column=1, padx=10, pady=10)

    # Create and place the start button
    start_button = tk.Button(root, text="Start Game", command=on_start)
    start_button.grid(row=2, column=0, columnspan=2, pady=10)

    # Create and place the quit button
    quit_button = tk.Button(root, text="Quit", command=on_quit)
    quit_button.grid(row=3, column=0, columnspan=2, pady=10)

    # Run the GUI event loop
    root.mainloop()

# Initialize the GUI
restart_gui()
