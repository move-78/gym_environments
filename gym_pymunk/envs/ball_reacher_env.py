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

MIN_END_GOAL_THRESHOLD = 0.05

END_GOAL_MOCAP_RADIUS = 15
END_GOAL_MOCAP_COLOR = (255, 255, 0, 1)

SUBGOAL_MOCAP_RADIUS = 15
SUBGOAL_MOCAP_COLOR = (255, 0, 255, 1)

MIN_RANGE_ENV = OFFSET + WALL_WIDTH / 2
MAX_RANGE_ENV = WALL_LENGTH + OFFSET - WALL_WIDTH / 2

TARGET_MIN_RANGE_ENV = -1
TARGET_MAX_RANGE_ENV = 1

MIN_DISTANCE_TO_GOAL = 0.4


def limit_velocity(body, gravity, damping, dt):
    pm.Body.update_velocity(body, gravity, damping, dt)
    body_vel_magnitude = body.velocity.length
    scale = 1
    if body_vel_magnitude > MAX_VELOCITY:
        scale = MAX_VELOCITY / body_vel_magnitude
    body.velocity = body.velocity * scale


def _goal_to_render_coordinates(goal):
    return np.interp(goal, (TARGET_MIN_RANGE_ENV, TARGET_MAX_RANGE_ENV), (MIN_RANGE_ENV, MAX_RANGE_ENV))


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

        self._setup_walls()
        self._setup_agent()
        self._setup_mocap_goals()

        self._step_counter = 0

    def _setup_mocap_goals(self):
        goal_render = _goal_to_render_coordinates(self.goal)
        self._end_goal_mocap, self._end_goal_mocap_body = create_mocap_circle(self._space, goal_render,
                                                                              END_GOAL_MOCAP_RADIUS,
                                                                              END_GOAL_MOCAP_COLOR)
        self._space.add(self._end_goal_mocap, self._end_goal_mocap_body)

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
        return self._get_state()

    def reset(self):
        self.goal = self._sample_goal()
        obs = self._reset_sim(next_goal=self.goal)
        self.display_end_goal(self.goal)
        return obs

    def render(self, mode='human'):
        if self._window is None:
            self._window = Window(space=self._space)
        self._render_callback()
        self._window.update()

    def step(self, action):
        self._set_action(action)
        self._space.step(DT)

    def project_state_to_end_goal(self):
        return self._get_state()

    def project_state_to_subgoal(self):
        return self._get_state()

    def _reset_sim(self, next_goal):
        self._step_counter = 0
        pos = self._reset_agent_pos_within_initial_state_space()
        while True:
            if np.linalg.norm(next_goal - pos[:2]) > MIN_DISTANCE_TO_GOAL:
                break
            else:
                pos = self._reset_agent_pos_within_initial_state_space()
        return self._get_obs()

    def _obs2goal(self, state):
        # This returns the first 2 values of the state (i.e. the x & y positions of the ball)
        # That corresponds to the end goal space (which is the 2-dimensional x & y position of the goal)
        return state[:2]

    def _obs2subgoal(self):
        pass

    def add_noise(self, vec, history, noise_coefficient):
        pass

    def create_graph(self):
        pass

    def display_end_goal(self, end_goal):
        end_goal = _goal_to_render_coordinates(end_goal)
        self._end_goal_mocap.body.position = tuple(end_goal)

    def display_subgoals(self, subgoals):
        pass

    def _render_callback(self):
        pass

    def _sample_goal(self) -> np.array:
        goal = super()._sample_goal()
        return np.interp(goal, (MIN_RANGE_ENV, MAX_RANGE_ENV),
                         (TARGET_MIN_RANGE_ENV, TARGET_MAX_RANGE_ENV))

    # Abstract methods
    def compute_reward(self, achieved_goal, desired_goal, info):
        individual_differences = achieved_goal - desired_goal
        d = np.linalg.norm(individual_differences, axis=-1)

        if self.reward_type == 'sparse':
            reward = -1 * np.any(np.abs(individual_differences) > np.array([MIN_END_GOAL_THRESHOLD] * d.size),
                                 axis=-1).astype(np.float32)
            return reward
        else:
            return -1 * d

    def _get_obs(self):
        obs = self._get_state()
        noisy_obs = obs.copy()
        # noisy_obs = add_noise(obs.copy(), self.obs_history, self.obs_noise_coefficient)

        achieved_goal = self._obs2goal(noisy_obs)

        obs = {
            'observation': noisy_obs,
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
            'non_noisy_obs': obs.copy()
        }
        return obs

    def goal_achieved(self):
        pass

    def _is_success(self, achieved_goal, desired_goal):
        d = np.abs(achieved_goal - desired_goal)
        return np.all(d < MIN_END_GOAL_THRESHOLD, axis=-1).astype(np.float32)

    def _set_action(self, action):
        vector = tuple(action * FORCE_MULTIPLIER)
        #self.agent.apply_force_at_local_point(force=vector)

    def _get_state(self):
        normalized_pos = self._get_normalized_pos()
        normalized_vel = self._get_normalized_vel()
        return np.concatenate((normalized_pos, normalized_vel))

    def _get_normalized_pos(self):
        return np.interp(self.agent.position, (MIN_RANGE_ENV, MAX_RANGE_ENV),
                         (TARGET_MIN_RANGE_ENV, TARGET_MAX_RANGE_ENV))

    def _get_normalized_vel(self):
        return np.interp(self.agent.velocity, (-MAX_VELOCITY, MAX_VELOCITY),
                         (TARGET_MIN_RANGE_ENV, TARGET_MAX_RANGE_ENV))