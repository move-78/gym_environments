import pygame as pg
import sys
from webcolors import name_to_rgb
from pymunk.pygame_util import DrawOptions


CLOCK_TICK_RATE = 120


colors = {
    'grey': name_to_rgb('grey')
}


class Window:
    WIDTH = 600
    HEIGHT = 600

    def __init__(self, space):
        pg.init()
        self.space = space
        self.screen = pg.display.set_mode((Window.WIDTH, Window.HEIGHT))
        pg.display.flip()
        self.clock = pg.time.Clock()
        self.draw_options = DrawOptions(self.screen)

    def update(self):
        self.screen.fill(color=colors['grey'])
        self.space.debug_draw(self.draw_options)
        pg.display.update()
        self.clock.tick(CLOCK_TICK_RATE)
