import pathlib
import numpy as np
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from enum import Enum


class TileType(Enum):
    GROUND = 'g'
    AGENT = 'a'
    WALL = 'w'
    START = 's'
    END = 'e'


class Direction:
    id_to_next_dir_step = {
        0: np.array([-1, 0]),
        1: np.array([0, -1]),
        2: np.array([1, 0]),
        3: np.array([0, 1])
    }

    dir_to_id = {
        'n': 0,
        'e': 1,
        's': 2,
        'w': 3
    }

    id_to_facing_dir = {
        0: np.array([[0.1, 0.1], [0.5, 0.95], [0.9, 0.1]]),
        1: np.array([[0.1, 0.1], [0.1, 0.9], [0.95, 0.5]]),
        2: np.array([[0.1, 0.9], [0.9, 0.9], [0.5, 0.05]]),
        3: np.array([[0.05, 0.5], [0.9, 0.9], [0.9, 0.1]])
    }


class Action(Enum):
    FORWARD = 0
    TURN_LEFT = -1
    TURN_RIGHT = 1


class Tile:
    def __init__(self, tile_type):
        self.tile_type = tile_type

    def __str__(self):
        return TileType(self.tile_type).value


def load_gridworld(gridworld_layout_name):
    grid = []
    file_path = pathlib.Path(__file__).parent.parent / "gridworld_env_layouts" / gridworld_layout_name
    with open(file_path) as f:
        for line in f.readlines():
            temp = []
            cells = line.strip("\n").split(" ")
            for c in cells:
                temp.append(Tile(TileType(c)))
            grid.append(temp)
    return np.array(grid, dtype=np.object)


class GridworldEnv(gym.Env):
    
    metadata = {'render.modes': ['human']}

    def __init__(self, gridworld_layout_name="7x7-simple.txt"):
        self.render_on = True
        self.grid = load_gridworld(gridworld_layout_name)
        self._add_outer_wall()

        self.start_state = self._get_start_tile_location()
        self.current_state = self.start_state.copy()

        self._get_agent_coords_by_current_state()
        self.ax = _, self.ax = plt.subplots(1)
        self.ax.set_aspect("equal")
        self.ax.set_yticks([])
        self.ax.set_xticks([])

    def _get_agent_coords_by_current_state(self):
        self.agent_coords = Direction.id_to_facing_dir[self.current_state[2]] + self.current_state[:2]

    def _get_start_tile_location(self):
        for i, row in enumerate(self.grid):
            for j, col in enumerate(row):
                if col.tile_type == TileType.START:
                    return np.array([i, j, np.random.randint(0, 3)])

    def render(self, mode='human', close=False):
        grid_size = self.grid.shape

        x_lines = np.linspace(1, grid_size[0], grid_size[0])
        y_lines = np.linspace(1, grid_size[1], grid_size[0])

        for xl in x_lines:
            self.ax.axvline(xl)

        for yl in y_lines:
            self.ax.axhline(yl)
        self.ax.set_xlim(xmin=0, xmax=grid_size[0])
        self.ax.set_ylim(ymin=0, ymax=grid_size[1])

        agent = patches.Polygon(self.agent_coords, closed=True, fill=True)
        self.ax.add_patch(agent)

        plt.show(block=False)

    def reset(self):
        pass

    def step(self, action):
        new_state = self.current_state[:2]
        if action == Action.FORWARD.value:
            direction = self.current_state[2]
            next_possible_state = (self.current_state[:2] + Direction.id_to_next_dir_step[direction])
            if self.grid[next_possible_state[0]][next_possible_state[1]].tile_type != TileType.WALL:
                new_state = next_possible_state
        else:
            direction = (self.current_state[2] + action) % len(Direction.dir_to_id)
        self.current_state = np.concatenate((new_state, np.array([direction])))
        if self.render_on:
            self._get_agent_coords_by_current_state()
        return self.current_state, 0, False, None

    def __str__(self):
        string = ""
        for row in self.grid:
            line = ""
            for col in row:
                line += str(col) + " "
            line += "\n"
            string += line
        return string

    def _create_h_walls(self):
        return np.array([Tile(TileType.WALL) for _ in range(self.grid.shape[1])])

    def _create_v_walls(self):
        return np.array([Tile(TileType.WALL) for _ in range(self.grid.shape[0])])

    def _add_outer_wall(self):
        self.grid = np.r_[[self._create_h_walls()], self.grid]
        self.grid = np.r_[self.grid, [self._create_h_walls()]]

        self.grid = np.c_[self._create_v_walls(), self.grid]
        self.grid = np.c_[self.grid, self._create_v_walls()]
