from Game import game
import threading
from Image_processing import keys_detection

# Global Variables
keys = {"player left": False, "player right": False, "opponent left": False, "opponent right": False}
show_screen = {"show_screen": False, "game_width": 576}

# Start detection thread
detection_thread = threading.Thread(target=keys_detection, args=(keys, show_screen))
detection_thread.daemon = True
detection_thread.start()

# Run the game
game("player.png", "opponent.png", keys, show_screen)
