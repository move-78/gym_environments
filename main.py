import sys
import gym_pymunk
import gym
import time
import pygame as pg
import numpy as np


env = gym.make("BallReacher-v0")

while True:
    env.render()
    action = np.random.uniform(-1, 1, size=(2, 1))
    length = np.linalg.norm(action)
    action = action / length
    env.step(action)
    for event in pg.event.get():
        if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
            print("Quitting requested. Now exiting the game.")
            pg.quit()
            sys.exit(0)
