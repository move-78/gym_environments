import numpy as np
import pymunk as pm
from gym_pymunk.utils.window import Window
from gym_pymunk.envs.pymunk_env import PyMunkGoalEnv
from gym_pymunk.utils.pymunk_helper import create_mocap_circle

DT = (1.0 / 60)
WALL_LENGTH = 500
WALL_WIDTH = 25
OFFSET = 50

WINDOW_WIDTH = Window.WIDTH
WINDOW_HEIGHT = Window.HEIGHT

FORCE_MULTIPLIER = 100
MAX_VELOCITY = 100
AGENT_RADIUS = 25
AGENT_MASS = 1

MIN_END_GOAL_THRESHOLD = AGENT_RADIUS * 1.15

END_GOAL_MOCAP_RADIUS = 15
END_GOAL_MOCAP_COLOR = (255, 255, 0, 1)

SUBGOAL_MOCAP_RADIUS = 15
SUBGOAL_MOCAP_COLOR = (255, 0, 255, 1)


def limit_velocity(body, gravity, damping, dt):
    pm.Body.update_velocity(body, gravity, damping, dt)
    body_vel_magnitude = body.velocity.length
    scale = 1
    if body_vel_magnitude > MAX_VELOCITY:
        scale = MAX_VELOCITY / body_vel_magnitude
    body.velocity = body.velocity * scale


class BallReacherEnv(PyMunkGoalEnv):
    def __init__(self):
        name = "BallReacherPymunkEnv"
        reward_type = "sparse"
        goal_space_train = [[OFFSET + WALL_WIDTH + AGENT_RADIUS * 1.5,
                             OFFSET + WALL_LENGTH - AGENT_RADIUS * 1.5 - WALL_WIDTH],
                            [OFFSET + WALL_WIDTH + AGENT_RADIUS * 1.5,
                             OFFSET + WALL_LENGTH - AGENT_RADIUS * 1.5 - WALL_WIDTH]]

        goal_space_test = [[OFFSET + WALL_WIDTH + AGENT_RADIUS * 1.5,
                            OFFSET + WALL_LENGTH - AGENT_RADIUS * 1.5 - WALL_WIDTH],
                           [OFFSET + WALL_WIDTH + AGENT_RADIUS * 1.5,
                            OFFSET + WALL_LENGTH - AGENT_RADIUS * 1.5 - WALL_WIDTH]]

        end_goal_thresholds = np.array([MIN_END_GOAL_THRESHOLD, MIN_END_GOAL_THRESHOLD])

        initial_state_space = [[OFFSET + WALL_WIDTH + AGENT_RADIUS * 1.5,
                                OFFSET + WALL_LENGTH - AGENT_RADIUS * 1.5 - WALL_WIDTH],
                               [OFFSET + WALL_WIDTH + AGENT_RADIUS * 1.5,
                                OFFSET + WALL_LENGTH - AGENT_RADIUS * 1.5 - WALL_WIDTH]]

        super().__init__(name, reward_type, goal_space_train, goal_space_test, end_goal_thresholds, initial_state_space)

        self.metadata = {'render.modes': ['human']}
        self._window = None
        self._space = pm.Space()

        self._n_subgoals = 1
        self.goal = self._sample_goal()

        self._setup_mocap_goals()
        self._setup_agent()
        self._setup_walls()

    def _setup_mocap_goals(self):
        self._end_goal_mocap = create_mocap_circle(self._space, self.goal, END_GOAL_MOCAP_RADIUS, END_GOAL_MOCAP_COLOR)
        self._space.add(self._end_goal_mocap)

    def _setup_agent(self):
        self.agent = pm.Body(body_type=pm.Body.DYNAMIC)
        self.agent.position = (WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2)
        self.agent_shape = pm.Circle(self.agent, AGENT_RADIUS)
        self.agent_shape.mass = AGENT_MASS

        self.agent.velocity_func = limit_velocity
        self._space.add(self.agent, self.agent_shape)

    def _setup_walls(self):
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
        self._space.add(wall_s)

    def _reset_agent_pos_within_initial_state_space(self):
        pos = np.zeros(shape=len(self.initial_state_space))
        for i, dim in enumerate(self.initial_state_space):
            assert len(dim) == 2, f"The dimension has to be 2 and not {len(dim)}."
            pos[i] = np.random.uniform(low=dim[0], high=dim[1])
        self.agent.position = tuple(pos)

    def reset(self):
        self._reset_agent_pos_within_initial_state_space()

    def render(self, mode='human'):
        if self._window is None:
            self._window = Window(space=self._space)
        self._window.update()

    def step(self, action):
        action = np.array([0, 1])
        vector = tuple(action * FORCE_MULTIPLIER)
        self.agent.apply_force_at_local_point(force=vector)
        self._space.step(DT)

    def project_state_to_end_goal(self):
        return self._get_state()

    def project_state_to_subgoal(self):
        return self._get_state()

    def _reset_sim(self):
        pass

    def _obs2goal(self):
        pass

    def _obs2subgoal(self):
        pass

    def add_noise(self, vec, history, noise_coefficient):
        pass

    def create_graph(self):
        pass

    def display_end_goal(self, end_goal):
        pass

    def display_subgoals(self, subgoals):
        pass

    def _render_callback(self):
        pass

    # Abstract methods
    def compute_reward(self, achieved_goal, desired_goal, info):
        pass

    def _get_obs(self):
        pass

    def goal_achieved(self):
        pass

    def _is_success(self, achieved_goal, desired_goal):
        pass

    def _set_action(self, action):
        pass

    def _get_state(self):
        pass
