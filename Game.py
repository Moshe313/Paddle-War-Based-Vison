import pygame
import sys
import random
import os
from screeninfo import get_monitors


def game(player_image_path, opponent_image_path, keys, show_screen, player_name, opponent_name):
    # Initialize Pygame
    pygame.init()

    # Define screen size and background
    proportion = 0.6
    WIDTH, HEIGHT = 720 * proportion, 1280 * proportion   #720,1280

    monitor = get_monitors()[0]  # Use the primary monitor
    screen_width = monitor.width
    screen_height = monitor.height
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{int((screen_width - WIDTH)/2)},{int((screen_height - HEIGHT)/2)}"

    show_screen["game_width"] = WIDTH
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))  # This must be done before convert_alpha()
    background = pygame.transform.scale(pygame.image.load("background.png").convert(), (WIDTH, HEIGHT))
    SCREEN.blit(background, (0, 0))
    pygame.display.set_caption("Hand Movement Game")

    # Load the player and opponent images
    player_image = pygame.image.load(player_image_path).convert_alpha()
    opponent_image = pygame.image.load(opponent_image_path).convert_alpha()
    players_width, players_height = 200 * proportion, 100 * proportion

    # Resize the images to the appropriate size
    player_image = pygame.transform.scale(player_image, (players_width, players_height))
    opponent_image = pygame.transform.scale(opponent_image, (players_width, players_height))

    # Load the player and opponent images
    player = pygame.Rect(0, 0, players_width, players_height)
    player.center = (WIDTH / 2, HEIGHT - 100)
    opponent = pygame.Rect(0, 0, players_width, players_height)
    opponent.center = (WIDTH / 2, 100)

    # Score reset
    player_score, opponent_score = 0, 0

    # Ball
    ball = pygame.Rect(0, 0, 20, 20)
    ball.center = (WIDTH / 2, HEIGHT / 2)
    x_speed, y_speed = 0.8, 0.8

    FONT = pygame.font.SysFont("Consolas", int(WIDTH / 20))
    LARGE_FONT = pygame.font.SysFont("Consolas", int(WIDTH / 10))
    CLOCK = pygame.time.Clock()
    go_text = LARGE_FONT.render("GO!", True, "black")
    counting_text = [LARGE_FONT.render(str(i), True, "black") for i in range(3, 0, -1)]
    for num in counting_text:
        SCREEN.blit(background, (0, 0))
        while not show_screen["show_screen"]:
            pass
        SCREEN.blit(num, (WIDTH / 2 - num.get_width() / 2, HEIGHT / 2 - num.get_height() / 2))
        pygame.display.update()
        pygame.time.delay(500)
    SCREEN.blit(background, (0, 0))
    SCREEN.blit(go_text, (WIDTH / 2 - go_text.get_width() / 2, HEIGHT / 2 - go_text.get_height() / 2))
    pygame.display.update()
    pygame.time.delay(500)  # 500 milliseconds

    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Update player position based on hand movement
        if keys["player right"]:
            if player.right < WIDTH:
                player.right += 8
        if keys["player left"]:
            if player.left > 0:
                player.left -= 8

        # Opponent:
        if keys["opponent right"]:
            if opponent.right < WIDTH:
                opponent.right += 8
        if keys["opponent left"]:
            if opponent.left > 0:
                opponent.left -= 8

        # Right and left limits
        if ball.x >= WIDTH:
            x_speed = -1 * abs(x_speed)
        if ball.x <= 0:
            x_speed = abs(x_speed)

        # Raise score while the ball didn't touch the player paddle and start over
        if ball.y <= 0:
            player_score += 1
            x_speed *= 1.1
            y_speed *= 1.1
            ball.center = (WIDTH / 2, HEIGHT / 2)
            x_speed, y_speed = random.choice([x_speed, -1 * x_speed]), random.choice([y_speed, -1 * y_speed])
        if ball.y >= HEIGHT:
            opponent_score += 1
            x_speed *= 1.1
            y_speed *= 1.1
            ball.center = (WIDTH / 2, HEIGHT / 2)
            x_speed, y_speed = random.choice([x_speed, -1 * x_speed]), random.choice([y_speed, -1 * y_speed])
        if player_score == 3 or opponent_score == 3:
            if player_score == 3: 
               winner_text = LARGE_FONT.render(player_name +" win!", True, "black")
            else:
                winner_text = LARGE_FONT.render(opponent_name + " win!", True, "black")
            SCREEN.blit(winner_text, (WIDTH / 2 - winner_text.get_width() / 2, HEIGHT / 2 - winner_text.get_height() / 2))
            pygame.display.update()
            pygame.time.delay(3000)
            break
        # Change the ball direction when touch the player paddle
        if player.y - ball.height <= ball.y <= player.bottom and ball.x in range(player.left - ball.height,
                                                                                 player.right + ball.height):
            y_speed = -1 * abs(y_speed)
        if opponent.y - ball.height <= ball.y <= opponent.bottom and ball.x in range(opponent.left - ball.height,
                                                                                     opponent.right + ball.height):
            y_speed = 1 * abs(y_speed)

        # Convert the score to pygame string
        player_score_text = FONT.render(str(player_score), True, "black")
        opponent_score_text = FONT.render(str(opponent_score), True, "black")

        # Advance the ball
        ball.x += x_speed * 2
        ball.y += y_speed * 2

        # Delete the old ball and paddles from screen
        SCREEN.blit(background, (0, 0))
    
        # Draw the images in the new positions
        SCREEN.blit(player_image, player.topleft)
        SCREEN.blit(opponent_image, opponent.topleft)
        pygame.draw.circle(SCREEN, "white", ball.center, 10)

        # Show thw players score
        SCREEN.blit(player_score_text, (WIDTH / 2 + 50 * proportion, 50 * proportion))
        SCREEN.blit(opponent_score_text, (WIDTH / 2 - 80 * proportion, 50 * proportion))

        pygame.display.update()
        CLOCK.tick(100)


