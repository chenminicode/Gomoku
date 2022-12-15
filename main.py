import pygame

# Intialize the pygame
pygame.init()

# create the screen
screen = pygame.display.set_mode((840, 840))

# Caption and Icon
pygame.display.set_caption("Gomuku AI")

def draw_board():
    for i in range(1, 21):
        pygame.draw.rect(screen, 'black', pygame.Rect(20, 40*i, 800, 1))

# Game Loop
running = True
while running:

    # RGB = Red, Green, Blue
    screen.fill((247, 242, 230))
    
    draw_board()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()
