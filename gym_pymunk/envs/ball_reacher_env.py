import numpy as np
import gym
import pymunk as pm
from gym_pymunk.envs.window import Window


DT = (1.0/60)
WALL_LENGTH = 500
WALL_WIDTH = 25
OFFSET = 50

WINDOW_WIDTH = Window.WIDTH
WINDOW_HEIGHT = Window.HEIGHT

FORCE_MULTIPLIER = 100

ACTION_TO_VECTOR = {
    0: np.array([0, 0]),
    1: np.array([0, 1]),
    2: np.array([0, -1]),
    3: np.array([1, 0]),
    4: np.array([-1, 0])
}

MAX_VELOCITY = 100


def limit_velocity(body, gravity, damping, dt):
    pm.Body.update_velocity(body, gravity, damping, dt)
    body_vel_magnitude = body.velocity.length
    scale = 1
    if body_vel_magnitude > MAX_VELOCITY:
        scale = MAX_VELOCITY / body_vel_magnitude
    body.velocity = body.velocity * scale


class BallReacherEnv(gym.Env):

    def __init__(self):
        self._window = None
        self._space = pm.Space()
        #self._space.gravity = (0, 500)
        self.agent = pm.Body(body_type=pm.Body.DYNAMIC)
        self.agent.position = (WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2)
        self.agent_shape = pm.Circle(self.agent, 25)
        self.agent_shape.mass = 1

        self.agent.velocity_func = limit_velocity
        self._space.add(self.agent, self.agent_shape)

        wall_e = pm.Poly.create_box(self._space.static_body, (WALL_WIDTH, WALL_LENGTH))
        wall_e.body.position = (OFFSET, OFFSET + WALL_LENGTH / 2)

        self._space.add(wall_e)

        wall_w = pm.Poly.create_box(self._space.static_body, (WALL_WIDTH, WALL_LENGTH))
        wall_w.body.position = (OFFSET + WALL_LENGTH, OFFSET + WALL_LENGTH / 2)

        self._space.add(wall_w)

        wall_n = pm.Poly.create_box(self._space.static_body, (WALL_LENGTH + OFFSET / 2, WALL_WIDTH))
        wall_n.body.position = (OFFSET + WALL_LENGTH / 2, OFFSET)

        self._space.add(wall_n)

        wall_s = pm.Poly.create_box(self._space.static_body, (WALL_LENGTH + OFFSET / 2, WALL_WIDTH))
        wall_s.body.position = (OFFSET + WALL_LENGTH / 2, WALL_LENGTH + OFFSET)
        wall_s.elasticity = 0.8
        wall_s.friction = 0.8
        self._space.add(wall_s)

    def reset(self):
        pass

    def render(self, mode='human'):
        if self._window is None:
            self._window = Window(space=self._space)
        self._window.update()

    def step(self, action):
        vector = tuple(action * FORCE_MULTIPLIER)
        self.agent.apply_force_at_local_point(force=vector)
        self._space.step(DT)
