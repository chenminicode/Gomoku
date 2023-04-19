# Overview

This Gomoku AI is based on a Korea ML engineer's work on Github, but I'm can't find the original link when I finished it.

The difference to the previous work is:

- use pytorch rather than tensorflow
- use less training data, cause I don't have much RAM
- use pygame as GAME GUI rather than HTML

# Install
Install `numpy`, `pytorch`, `pygame` library, then run:

```python
python main.py
```

press `r` to restart game, `q` to quit game.

# Training

Training process can be find on below kaggle notebook, training output is `model.pth`.

https://www.kaggle.com/code/metseq/gomoku-ai

# Limits

I have not write the win condition, so the game will go on if you do not restart or quit.
