import pygame
import sys
import random
import os
import cv2  # needed for converting camera frames
from screeninfo import get_monitors

def update_display(SCREEN, game_surface, cam_width, game_width, game_height, show_screen):
    # Clear main screen
    SCREEN.fill((0, 0, 0))
    # Left camera panel
    if "left_cam" in show_screen and show_screen["left_cam"] is not None:
        left_frame = show_screen["left_cam"]
        left_frame_rgb = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
        left_surface = pygame.image.frombuffer(left_frame_rgb.tobytes(), (left_frame_rgb.shape[1], left_frame_rgb.shape[0]), "RGB")
        left_surface = pygame.transform.scale(left_surface, (cam_width, game_height))
        SCREEN.blit(left_surface, (0, 0))
    else:
        pygame.draw.rect(SCREEN, (100, 100, 100), (0, 0, cam_width, game_height))
    # Center game area
    SCREEN.blit(game_surface, (cam_width, 0))
    # Right camera panel
    if "right_cam" in show_screen and show_screen["right_cam"] is not None:
        right_frame = show_screen["right_cam"]
        right_frame_rgb = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
        right_surface = pygame.image.frombuffer(right_frame_rgb.tobytes(), (right_frame_rgb.shape[1], right_frame_rgb.shape[0]), "RGB")
        right_surface = pygame.transform.scale(right_surface, (cam_width, game_height))
        SCREEN.blit(right_surface, (cam_width + game_width, 0))
    else:
        pygame.draw.rect(SCREEN, (100, 100, 100), (cam_width + game_width, 0, cam_width, game_height))
    pygame.display.update()

def game(player_image_path, opponent_image_path, keys, show_screen, player_name, opponent_name):
    starting_speed = 0.8
    # Initialize Pygame and define sizes
    pygame.init()
    proportion = 0.6
    game_width = int(720 * proportion)
    game_height = int(1280 * proportion)
    # Define camera panel width (for left and right feeds)
    cam_width = int(game_width)
    total_width = game_width + 2 * cam_width

    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{int((screen_width - total_width)/2)},{int((screen_height - game_height)/2)}"
    # Update shared state (if needed elsewhere)
    show_screen["game_width"] = game_width

    # Create main display with three panels: left cam, game, right cam
    SCREEN = pygame.display.set_mode((total_width, game_height))
    pygame.display.set_caption("Hand Movement Game")

    # Load and scale the background for the game area only
    background = pygame.transform.scale(pygame.image.load("background.png").convert(), (game_width, game_height))
    # Create a separate surface for the game area (center panel)
    game_surface = pygame.Surface((game_width, game_height))
    game_surface.blit(background, (0, 0))

    FONT = pygame.font.SysFont("Consolas", int(game_width / 20))
    LARGE_FONT = pygame.font.SysFont("Consolas", int(game_width / 15))
    CLOCK = pygame.time.Clock()

    # Display initial text (on the game area)
    pic_text = LARGE_FONT.render("Be ready for picture!", True, "black")
    game_surface.blit(pic_text, (game_width/2 - pic_text.get_width()/2, game_height/2 - pic_text.get_height()/2))
    update_display(SCREEN, game_surface, cam_width, game_width, game_height, show_screen)
    pygame.display.update()
    pygame.time.delay(1000)

    # Countdown before picture capture
    for num in [3, 2, 1]:
        game_surface.blit(background, (0, 0))
        # Wait until the camera thread is showing video
        while not show_screen["show_screen"]:
            pass
        num_text = LARGE_FONT.render(str(num), True, "black")
        game_surface.blit(num_text, (game_width/2 - num_text.get_width()/2, game_height/2 - num_text.get_height()/2))
        update_display(SCREEN, game_surface, cam_width, game_width, game_height, show_screen)
        pygame.time.delay(500)
    game_surface.blit(background, (0, 0))
    pygame.display.update()
    pygame.time.delay(500)
    say_cheese = LARGE_FONT.render("Say Cheese!", True, "black")
    game_surface.blit(say_cheese, (game_width/2 - say_cheese.get_width()/2, game_height/2 - say_cheese.get_height()/2))
    update_display(SCREEN, game_surface, cam_width, game_width, game_height, show_screen)
    pygame.time.delay(1000)

    # Signal picture capture (the camera thread will save frames and reset show_screen["pic"])
    show_screen["pic"] = True
    while show_screen["pic"]:
        update_display(SCREEN, game_surface, cam_width, game_width, game_height, show_screen)
        CLOCK.tick(30)

    # Load the saved camera images for paddles (as before)
    or_player_image = pygame.image.load("left_frame.jpg").convert_alpha()
    or_opponent_image = pygame.image.load("right_frame.jpg").convert_alpha()
    players_width, players_height = int(200 * proportion), int(100 * proportion)
    player_image = pygame.transform.scale(or_player_image, (players_width, players_height))
    player_image_for_win = pygame.transform.scale(
        or_player_image, (int(600 * proportion), int(600 * proportion * (or_player_image.get_height() / or_player_image.get_width())))
    )
    opponent_image = pygame.transform.scale(or_opponent_image, (players_width, players_height))
    opponent_image_for_win = pygame.transform.scale(
        or_opponent_image, (int(600 * proportion), int(600 * proportion * (or_opponent_image.get_height() / or_opponent_image.get_width())))
    )

    # Create paddle rectangles on the game_surface (coordinates relative to game_surface)
    player = pygame.Rect(0, 0, players_width, players_height)
    player.center = (game_width / 2, game_height - 100)
    opponent = pygame.Rect(0, 0, players_width, players_height)
    opponent.center = (game_width / 2, 100)

    # Score and ball initialization
    player_score, opponent_score = 0, 0
    ball = pygame.Rect(0, 0, 20, 20)
    ball.center = (game_width / 2, game_height / 2)
    x_speed, y_speed = starting_speed, starting_speed

    go_text = LARGE_FONT.render("GO!", True, "black")
    for num in [3, 2, 1]:
        game_surface.blit(background, (0, 0))
        while not show_screen["show_screen"]:
            pass
        num_text = LARGE_FONT.render(str(num), True, "black")
        game_surface.blit(num_text, (game_width/2 - num_text.get_width()/2, game_height/2 - num_text.get_height()/2))
        update_display(SCREEN, game_surface, cam_width, game_width, game_height, show_screen)
        pygame.time.delay(500)
    game_surface.blit(background, (0, 0))
    game_surface.blit(go_text, (game_width/2 - go_text.get_width()/2, game_height/2 - go_text.get_height()/2))
    update_display(SCREEN, game_surface, cam_width, game_width, game_height, show_screen)
    pygame.time.delay(500)

    # Main game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Update paddle positions based on the keys updated by the detection thread
        if keys["player right"]:
            if player.right < game_width:
                player.right += 8
        if keys["player left"]:
            if player.left > 0:
                player.left -= 8
        if keys["opponent right"]:
            if opponent.right < game_width:
                opponent.right += 8
        if keys["opponent left"]:
            if opponent.left > 0:
                opponent.left -= 8

        # Ball boundary logic
        if ball.x >= game_width:
            x_speed = -abs(x_speed)
        if ball.x <= 0:
            x_speed = abs(x_speed)
        if ball.y <= 0:
            player_score += 1
            x_speed = 1.1 * starting_speed
            y_speed = 1.1 * starting_speed
            ball.center = (game_width / 2, game_height / 2)
            x_speed, y_speed = random.choice([x_speed, -x_speed]), random.choice([y_speed, -y_speed])
        if ball.y >= game_height:
            opponent_score += 1
            x_speed = 1.1 * starting_speed
            y_speed = 1.1 * starting_speed
            ball.center = (game_width / 2, game_height / 2)
            x_speed, y_speed = random.choice([x_speed, -x_speed]), random.choice([y_speed, -y_speed])
        if player_score == 3 or opponent_score == 3:
            game_surface.blit(background, (0, 0))
            player_score_text = FONT.render(str(player_score), True, "black")
            opponent_score_text = FONT.render(str(opponent_score), True, "black")
            game_surface.blit(opponent_score_text, (game_width / 2 + 50 * proportion, 50 * proportion))
            game_surface.blit(player_score_text, (game_width / 2 - 80 * proportion, 50 * proportion))
            update_display(SCREEN, game_surface, cam_width, game_width, game_height, show_screen)
            CLOCK.tick(100)
            if player_score == 3:
                winner_text = LARGE_FONT.render(player_name + " win!", True, "black")
                game_surface.blit(player_image_for_win, (game_width / 2 - player_image_for_win.get_width() / 2, game_height / 2 - player_image_for_win.get_height() / 2))
            else:
                winner_text = LARGE_FONT.render(opponent_name + " win!", True, "black")
                game_surface.blit(opponent_image_for_win, (game_width / 2 - opponent_image_for_win.get_width() / 2, game_height / 2 - opponent_image_for_win.get_height() / 2))
            game_surface.blit(winner_text, (game_width / 2 - winner_text.get_width() / 2, game_height / 4 - winner_text.get_height() / 2))
            update_display(SCREEN, game_surface, cam_width, game_width, game_height, show_screen)
            pygame.time.delay(3000)
            break

        # Paddle collision detection
        if (player.y - ball.height) <= ball.y <= player.bottom and ball.x in range(player.left - ball.height, player.right + ball.height):
            y_speed = -abs(y_speed)
            
        if opponent.y - ball.height <= ball.y <= opponent.bottom and ball.x in range(opponent.left - ball.height, opponent.right + ball.height):
            y_speed = abs(y_speed)
        
        # Update scores and ball movement
        player_score_text = FONT.render(str(player_score), True, "black")
        opponent_score_text = FONT.render(str(opponent_score), True, "black")
        ball.x += x_speed * 2
        ball.y += y_speed * 2

        game_surface.blit(background, (0, 0))
        game_surface.blit(player_image, player.topleft)
        game_surface.blit(opponent_image, opponent.topleft)
        pygame.draw.circle(game_surface, "white", ball.center, 10)
        game_surface.blit(opponent_score_text, (game_width / 2 + 50 * proportion, 50 * proportion))
        game_surface.blit(player_score_text, (game_width / 2 - 80 * proportion, 50 * proportion))

        update_display(SCREEN, game_surface, cam_width, game_width, game_height, show_screen)
        CLOCK.tick(100)
