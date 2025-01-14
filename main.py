from Game import game
import threading
from Image_processing import keys_detection

# Global Variables
keys = {"player left": False, "player right": False, "opponent left": False, "opponent right": False}
show_screen = {"show_screen": False, "game_width": 550}

# Prompt for player and opponent names
player_name = input("Enter the player's name: ")
opponent_name = input("Enter the opponent's name: ")


# Start detection thread
detection_thread = threading.Thread(target=keys_detection, args=(keys, show_screen))
detection_thread.daemon = True
detection_thread.start()

# Run the game
game("player.png", "opponent.png", keys, show_screen, str(player_name), str(opponent_name))
