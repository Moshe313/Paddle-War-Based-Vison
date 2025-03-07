import pygame
import sys
import random
import os
import cv2  # needed for converting camera frames
from screeninfo import get_monitors
import threading
import math

# ---------------------
# Helper Functions
# ---------------------

def update_display(SCREEN, game_surface, cam_width, game_width, game_height, show_screen):
    """
    Update the full display by drawing left/right camera panels and the center game surface.
    """
    # Clear the main screen with black.
    SCREEN.fill((0, 0, 0))
    
    # LEFT CAMERA PANEL:
    if "left_cam" in show_screen and show_screen["left_cam"] is not None:
        left_frame = show_screen["left_cam"]
        # Convert the frame from BGR (OpenCV) to RGB (pygame).
        left_frame_rgb = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
        left_surface = pygame.image.frombuffer(
            left_frame_rgb.tobytes(), 
            (left_frame_rgb.shape[1], left_frame_rgb.shape[0]), 
            "RGB"
        )
        left_surface = pygame.transform.scale(left_surface, (cam_width, game_height))
        SCREEN.blit(left_surface, (0, 0))
    else:
        pygame.draw.rect(SCREEN, (100, 100, 100), (0, 0, cam_width, game_height))

    # CENTER GAME AREA:
    SCREEN.blit(game_surface, (cam_width, 0))

    # RIGHT CAMERA PANEL:
    if "right_cam" in show_screen and show_screen["right_cam"] is not None:
        right_frame = show_screen["right_cam"]
        right_frame_rgb = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
        right_surface = pygame.image.frombuffer(
            right_frame_rgb.tobytes(), 
            (right_frame_rgb.shape[1], right_frame_rgb.shape[0]), 
            "RGB"
        )
        right_surface = pygame.transform.scale(right_surface, (cam_width, game_height))
        SCREEN.blit(right_surface, (cam_width + game_width, 0))
    else:
        pygame.draw.rect(SCREEN, (100, 100, 100), (cam_width + game_width, 0, cam_width, game_height))

    pygame.display.update()


def draw_scoreboard(surface, player_score, opponent_score, player_name, opponent_name, game_width, scoreboard_height, font, bg_color):
    """
    Draws a nice scoreboard at the top of the game surface.
    """
    scoreboard_rect = pygame.Rect(0, 0, game_width, scoreboard_height)
    pygame.draw.rect(surface, bg_color, scoreboard_rect)  
    pygame.draw.rect(surface, (255, 255, 255), scoreboard_rect, 3)
    player_text = font.render(f"{player_name}: {player_score}", True, "white")
    opponent_text = font.render(f"{opponent_name}: {opponent_score}", True, "white")
    surface.blit(player_text, (10, scoreboard_rect.centery - player_text.get_height() // 2))
    surface.blit(opponent_text, (game_width - opponent_text.get_width() - 10, scoreboard_rect.centery - opponent_text.get_height() // 2))


def countdown(SCREEN, game_surface, cam_width, game_width, game_height, show_screen, background, countdown_font, go_text, keys, player, opponent, ball, player_image, opponent_image, player_score, opponent_score, player_name, opponent_name, FONT, scoreboard_height):
    """
    Displays a countdown from 3 to 1 and then "GO!" before starting the game.
    During the countdown, players can move and the ball remains stationary.
    """
    while not show_screen.get("show_screen", False):
        update_display(SCREEN, game_surface, cam_width, game_width, game_height, show_screen)
        pygame.time.delay(50)

    for num in [3, 2, 1]:
        start_time = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start_time < 1000:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            if keys["player right"] and player.right < game_width:
                player.right += 8
            if keys["player left"] and player.left > 0:
                player.left -= 8
            if keys["opponent right"] and opponent.right < game_width:
                opponent.right += 8
            if keys["opponent left"] and opponent.left > 0:
                opponent.left -= 8

            game_surface.blit(background, (0, 0))
            draw_scoreboard(game_surface, player_score, opponent_score, player_name, opponent_name, game_width, scoreboard_height, FONT, (30,144,255))
            pygame.draw.circle(game_surface, "white", ball.center, 10)
            game_surface.blit(player_image, player.topleft)
            game_surface.blit(opponent_image, opponent.topleft)
            num_text = countdown_font.render(str(num), True, "black")
            game_surface.blit(num_text, (game_width/2 - num_text.get_width()/2, game_height/2 - num_text.get_height()/2))
            update_display(SCREEN, game_surface, cam_width, game_width, game_height, show_screen)
            pygame.time.delay(33)
    start_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start_time < 500:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        if keys["player right"] and player.right < game_width:
            player.right += 8
        if keys["player left"] and player.left > 0:
            player.left -= 8
        if keys["opponent right"] and opponent.right < game_width:
            opponent.right += 8
        if keys["opponent left"] and opponent.left > 0:
            opponent.left -= 8

        game_surface.blit(background, (0, 0))
        draw_scoreboard(game_surface, player_score, opponent_score, player_name, opponent_name, game_width, scoreboard_height, FONT, (30,144,255))
        pygame.draw.circle(game_surface, "white", ball.center, 10)
        game_surface.blit(player_image, player.topleft)
        game_surface.blit(opponent_image, opponent.topleft)
        game_surface.blit(go_text, (game_width/2 - go_text.get_width()/2, game_height/2 - go_text.get_height()/2))
        update_display(SCREEN, game_surface, cam_width, game_width, game_height, show_screen)
        pygame.time.delay(33)

# ---------------------
# Main Game Function
# ---------------------

def game(player_image_path, opponent_image_path, keys, show_screen, 
         player_name, opponent_name, starting_player="left"):
    """
    Main game function that initializes the window, assets, powerup system,
    runs the countdown, and then starts the main game loop.
    """
    starting_speed = 0.8
    pygame.init()
    proportion = 0.6
    game_width = int(720 * proportion)
    game_height = int(1280 * proportion)
    cam_width = game_width
    total_width = game_width + 2 * cam_width

    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{int((screen_width - total_width)/2)},{int((screen_height - game_height)/2)}"
    show_screen["game_width"] = game_width

    SCREEN = pygame.display.set_mode((total_width, game_height))
    pygame.display.set_caption("Hand Movement Game")

    background = pygame.transform.scale(pygame.image.load("background.png").convert(), (game_width, game_height))
    game_surface = pygame.Surface((game_width, game_height))
    game_surface.blit(background, (0, 0))

    # Fonts & Clock
    FONT = pygame.font.SysFont("Consolas", int(game_width / 20))
    LARGE_FONT = pygame.font.SysFont("Consolas", int(game_width / 15))
    countdown_font = pygame.font.SysFont("Consolas", int(game_width / 5))
    go_text = pygame.font.SysFont("Consolas", int(game_width / 3)).render("GO!", True, "red")
    CLOCK = pygame.time.Clock()
    scoreboard_height = int(game_width / 6)

    # Load paddle images.
    or_player_image = pygame.image.load("left_frame.jpg").convert_alpha()
    or_opponent_image = pygame.image.load("right_frame.jpg").convert_alpha()
    players_width = int(200 * proportion)
    players_height = int(100 * proportion)
    default_paddle_width = players_width
    default_paddle_height = players_height

    # Set initial paddle sizes.
    player_width = default_paddle_width
    opponent_width = default_paddle_width

    # Create paddle rectangles.
    player = pygame.Rect(0, 0, player_width, default_paddle_height)
    player.center = (game_width/2, game_height - 100)
    opponent = pygame.Rect(0, 0, opponent_width, default_paddle_height)
    opponent.center = (game_width/2, 100)

    # Initialize ball.
    default_ball_radius = 10
    ball_radius = default_ball_radius
    ball = pygame.Rect(0, 0, default_ball_radius*2, default_ball_radius*2)

    # Determine server & initial serve.
    server = "player" if starting_player == "left" else "opponent"
    if server == "player":
        ball.centerx = player.centerx
        ball.centery = player.top - 10
        x_speed = random.choice([starting_speed, -starting_speed])
        y_speed = -starting_speed
    else:
        ball.centerx = opponent.centerx
        ball.centery = opponent.bottom + 10
        x_speed = random.choice([starting_speed, -starting_speed])
        y_speed = starting_speed

    # Collision flags and last hit tracking.
    player_colliding = False
    opponent_colliding = False
    last_hit = None  # "player" or "opponent"

    # Variables for powerup effects (end times in ms).
    effect_duration = 5000  # 5 seconds
    ball_powerup_end = 0
    player_powerup_end = 0
    opponent_powerup_end = 0

    # Powerup bubble system.
    powerups = []  # List of dicts with keys: type, pos, radius, spawn_time
    powerup_spawn_interval = 7000  # spawn every 7 seconds
    last_powerup_spawn_time = pygame.time.get_ticks()
    powerup_lifetime = 10000  # a powerup stays on court for 10 seconds if not collected

    # For enhanced scoreboard effect.
    last_score_time = 0

    # ------------------------------
    # CALL THE COUNTDOWN BEFORE THE GAME STARTS
    # ------------------------------
    countdown(SCREEN, game_surface, cam_width, game_width, game_height, show_screen,
              background, countdown_font, go_text, keys, player, opponent, ball, 
              pygame.transform.scale(or_player_image, (player_width, default_paddle_height)),
              pygame.transform.scale(or_opponent_image, (opponent_width, default_paddle_height)),
              0, 0, player_name, opponent_name, FONT, scoreboard_height)

    # ------------------------------
    # MAIN GAME LOOP STARTS AFTER COUNTDOWN FINISHES
    # ------------------------------
    while True:
        current_time = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Update paddle positions.
        if keys["player right"] and player.right < game_width:
            player.right += 8
        if keys["player left"] and player.left > 0:
            player.left -= 8
        if keys["opponent right"] and opponent.right < game_width:
            opponent.right += 8
        if keys["opponent left"] and opponent.left > 0:
            opponent.left -= 8

        # Update scoreboard highlight if a score just happened.
        if current_time - last_score_time < 500:
            sb_color = (0,191,255)  # Highlight color
        else:
            sb_color = (30,144,255)

        # Ball boundary collisions.
        if ball.x <= 0:
            x_speed = abs(x_speed)
        if ball.x >= game_width:
            x_speed = -abs(x_speed)

        # Scoring: if ball goes out at the top, player scores.
        if ball.y <= 0:
            player_score = 1  if 'player_score' not in locals() else player_score + 1
            last_score_time = current_time
            server = "player"
            ball.centerx = player.centerx
            ball.centery = player.top - 10
            x_speed = random.choice([starting_speed, -starting_speed])
            y_speed = -starting_speed

        # If ball goes out at the bottom, opponent scores.
        if ball.y >= game_height:
            opponent_score = 1 if 'opponent_score' not in locals() else opponent_score + 1
            last_score_time = current_time
            server = "opponent"
            ball.centerx = opponent.centerx
            ball.centery = opponent.bottom + 10
            x_speed = random.choice([starting_speed, -starting_speed])
            y_speed = starting_speed

        # -------------------------------
        # Paddle Collision Detection (One-Hit-Per-Contact)
        # For player's paddle:
        if player.colliderect(ball) and not player_colliding:
            last_hit = "player"
            current_speed = math.sqrt(x_speed**2 + y_speed**2)
            new_speed = current_speed * 1.05
            offset = (ball.centerx - player.centerx) / (player.width / 2)
            max_angle = math.radians(60)
            angle = offset * max_angle
            min_vertical = new_speed * 0.6
            if new_speed * abs(math.cos(angle)) < min_vertical:
                angle = math.copysign(math.acos(0.6), angle)
            x_speed = new_speed * math.sin(angle)
            y_speed = -new_speed * math.cos(angle)
            player_colliding = True
        if not player.colliderect(ball):
            player_colliding = False

        # For opponent's paddle:
        if opponent.colliderect(ball) and not opponent_colliding:
            last_hit = "opponent"
            current_speed = math.sqrt(x_speed**2 + y_speed**2)
            new_speed = current_speed * 1.05
            offset = (ball.centerx - opponent.centerx) / (opponent.width / 2)
            max_angle = math.radians(60)
            angle = offset * max_angle
            min_vertical = new_speed * 0.6
            if new_speed * abs(math.cos(angle)) < min_vertical:
                angle = math.copysign(math.acos(0.6), angle)
            x_speed = new_speed * math.sin(angle)
            y_speed = new_speed * math.cos(angle)
            opponent_colliding = True
        if not opponent.colliderect(ball):
            opponent_colliding = False

        # Enforce maximum speed limit.
        current_speed = math.sqrt(x_speed**2 + y_speed**2)
        if current_speed > keys["speed_limit"]:
            factor = keys["speed_limit"] / current_speed
            x_speed *= factor
            y_speed *= factor

        # -------------------------------
        # Powerup Bubble Spawning
        if current_time - last_powerup_spawn_time > powerup_spawn_interval:
            pu_type = random.choice(["ball_size", "large_paddle", "small_opponent_paddle"])
            pu_radius = 15
            pu_x = random.randint(pu_radius+10, game_width - pu_radius-10)
            pu_y = random.randint(scoreboard_height+20, game_height - pu_radius-20)
            powerups.append({"type": pu_type, "pos": (pu_x, pu_y), "radius": pu_radius, "spawn_time": current_time})
            last_powerup_spawn_time = current_time

        # Remove expired powerups.
        powerups = [pu for pu in powerups if current_time - pu["spawn_time"] < powerup_lifetime]

        # Check for powerup collision with ball.
        for pu in powerups[:]:
            pu_x, pu_y = pu["pos"]
            dist = math.hypot(ball.centerx - pu_x, ball.centery - pu_y)
            if dist < ball_radius + pu["radius"]:
                # Award the powerup to the player who last hit the ball.
                collector = last_hit
                other = "player" if collector == "opponent" else "opponent"
                if pu["type"] == "ball_size":
                    ball_radius = int(default_ball_radius * 1.5)
                    ball_powerup_end = current_time + effect_duration
                elif pu["type"] == "large_paddle":
                    if collector == "player":
                        player_width = int(default_paddle_width * 1.5)
                        player_powerup_end = current_time + effect_duration
                    else:
                        opponent_width = int(default_paddle_width * 1.5)
                        opponent_powerup_end = current_time + effect_duration
                elif pu["type"] == "small_opponent_paddle":
                    if collector == "player":
                        opponent_width = int(default_paddle_width * 0.7)
                        opponent_powerup_end = current_time + effect_duration
                    else:
                        player_width = int(default_paddle_width * 0.7)
                        player_powerup_end = current_time + effect_duration
                powerups.remove(pu)

        # Check and revert expired powerup effects.
        if current_time > ball_powerup_end:
            ball_radius = default_ball_radius
        if current_time > player_powerup_end:
            player_width = default_paddle_width
        if current_time > opponent_powerup_end:
            opponent_width = default_paddle_width

        # Update paddle rectangles and images according to current sizes.
        player.width = player_width
        # Keep player's center constant.
        player.centerx = max(player.width//2, min(player.centerx, game_width - player.width//2))
        opponent.width = opponent_width
        opponent.centerx = max(opponent.width//2, min(opponent.centerx, game_width - opponent.width//2))
        player_image = pygame.transform.scale(or_player_image, (player_width, default_paddle_height))
        opponent_image = pygame.transform.scale(or_opponent_image, (opponent_width, default_paddle_height))

        # Redraw game surface.
        game_surface.blit(background, (0, 0))
        # Draw powerup bubbles.
        for pu in powerups:
            if pu["type"] == "ball_size":
                color = (255, 0, 0)  # red
            elif pu["type"] == "large_paddle":
                color = (0, 255, 0)  # green
            elif pu["type"] == "small_opponent_paddle":
                color = (0, 0, 255)  # blue
            pygame.draw.circle(game_surface, color, pu["pos"], pu["radius"])
        # Draw the ball.
        pygame.draw.circle(game_surface, "white", ball.center, ball_radius)
        # Draw paddles.
        game_surface.blit(player_image, player.topleft)
        game_surface.blit(opponent_image, opponent.topleft)
        # Draw the scoreboard.
        draw_scoreboard(game_surface, 
                        player_score if 'player_score' in locals() else 0, 
                        opponent_score if 'opponent_score' in locals() else 0, 
                        player_name, opponent_name, game_width, scoreboard_height, FONT, sb_color)

        # Update ball position.
        ball.x += x_speed * 2
        ball.y += y_speed * 2

        update_display(SCREEN, game_surface, cam_width, game_width, game_height, show_screen)
        CLOCK.tick(100)

        # Check win condition (first to 3 points).
        if ('player_score' in locals() and player_score >= 3) or ('opponent_score' in locals() and opponent_score >= 3):
            game_surface.blit(background, (0, 0))
            draw_scoreboard(game_surface, player_score, opponent_score, player_name, opponent_name, game_width, scoreboard_height, FONT, (30,144,255))
            overlay = pygame.Surface((game_width, game_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            game_surface.blit(overlay, (0, 0))
            if player_score >= 3:
                winner_text = LARGE_FONT.render(player_name + " wins!", True, "white")
                winner_image = player_image
            else:
                winner_text = LARGE_FONT.render(opponent_name + " wins!", True, "white")
                winner_image = opponent_image
            game_surface.blit(winner_image, (game_width/2 - winner_image.get_width()/2, game_height/2 - winner_image.get_height()/2))
            game_surface.blit(winner_text, (game_width/2 - winner_text.get_width()/2, game_height/4 - winner_text.get_height()/2))
            update_display(SCREEN, game_surface, cam_width, game_width, game_height, show_screen)
            pygame.time.delay(5000)
            break


def main():
    keys = {"player left": False, "player right": False, 
            "opponent left": False, "opponent right": False, "starts": "left", "speed_limit": 2.5}
    show_screen = {"show_screen": False, "game_width": 550, "pic": False, "video_prepared": False}
    
    from Image_processing import keys_detection
    detection_thread = threading.Thread(target=keys_detection, args=(keys, show_screen))
    detection_thread.daemon = True  
    detection_thread.start()  
    
    game("player.png", "opponent.png", keys, show_screen, "player_name", "opponent_name", starting_player="left")


if __name__ == "__main__":
    main()
