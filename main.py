import pygame
import numpy as np
import torch
from torch import nn

# Load Gomoku AI
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, (7, 7), padding='same'), 
            nn.ReLU(),
            nn.Conv2d(64, 128, (7, 7), padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 64, (7, 7), padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 1, (1, 1), padding='same'),
            nn.Flatten(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        logits = self.model(x)
        return logits

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))


def init_chess():
    '''Init chess piece position'''

    return torch.Tensor(np.zeros((1, 20, 20)))


def draw_board():
    screen.fill((247, 242, 180))
    for i in range(20):
        pygame.draw.rect(screen, 'black', pygame.Rect(20, 40*i+20, 760, 2))
        pygame.draw.rect(screen, 'black', pygame.Rect(40*i+20, 20, 2, 760))


def draw_last_move(last_move):
    pygame.draw.circle(screen, (0, 255, 0), (last_move[0] * 40 + 20, last_move[1] * 40 + 20), 5)


def draw_chess(chess, last_move):
    chess_np = chess.view(20, 20).detach().numpy()
    for x in range(20):
        for y in range(20):
            # let value equal -1 is black player
            if chess_np[x, y] == -1:
                pygame.draw.circle(screen, (0, 0, 0), (x * 40 + 20, y * 40 + 20), 19)
            # let value equal 1 is white player
            elif chess_np[x, y] == 1:
                pygame.draw.circle(screen, (255, 255, 255), (x * 40 + 20, y * 40 + 20), 19)
    draw_last_move(last_move)


def which_move(move):
    '''for simpificity, player is always black'''
    if move % 2 == 0:
        return 'player'
    else:
        return 'AI'


def AI_move(chess, move):
    predict_move = model(chess).view(20, 20).detach().numpy()
    ind = np.unravel_index(np.argmax(predict_move, axis=None), predict_move.shape)
    chess[0, ind[0], ind[1]] = 1
    pygame.time.wait(1000)
    move += 1
    return ind, move


def player_move(chess, move):
    chess_x, chess_y, left_click = get_mouse_stat()
    if left_click and valid_move(chess, chess_x, chess_y):
        chess[0, chess_x, chess_y] = -1
        move += 1
        return (chess_x, chess_y), move
    else:
        return last_move, move


def valid_move(chess, chess_x, chess_y):
    '''Check if player move is valid'''

    chess_np = chess.view(20, 20).detach().numpy()
    if chess_np[chess_x, chess_y] == 0:
        return True
    else:
        return False


def get_mouse_stat():
    # Get mouse position
    x, y = pygame.mouse.get_pos()
    chess_x = round((x - 20) / 40)
    chess_y = round((y - 20) / 40)

    # Get the state of the mouse buttons
    left_click, _, _ = pygame.mouse.get_pressed()

    return chess_x, chess_y, left_click


def new_game():
    chess = init_chess()
    move = 0
    last_move = (9, 9)
    return chess, move, last_move

# Intialize pygame
pygame.init()
screen = pygame.display.set_mode((800, 800))
pygame.display.set_caption("Gomuku AI")

# Game Loop
chess, move, last_move = new_game()

running = True

while running:
    draw_board()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                chess, move, last_move = new_game()
            elif event.key == pygame.K_q:
                running = False

        # Get mouse position
        chess_x, chess_y, left_click = get_mouse_stat()
        
        # move piece
        next_move = which_move(move)
        if next_move == 'player':
            last_move, move = player_move(chess, move)
        elif next_move == 'AI':
            last_move, move = AI_move(chess, move)

    draw_chess(chess, last_move)

    pygame.display.update()
