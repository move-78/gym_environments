import numpy as np
import gym
from abc import ABC, abstractmethod


class PyMunkGoalEnv(gym.GoalEnv, ABC):

    def __init__(self, name, reward_type, goal_space_train, goal_space_test, end_goal_thresholds, initial_state_space):
        self.name = name
        self.reward_type = reward_type
        self.goal_space_train = goal_space_train
        self.goal_space_test = goal_space_test
        self.end_goal_thresholds = end_goal_thresholds
        self.initial_state_space = initial_state_space

    @abstractmethod
    def compute_reward(self, achieved_goal, desired_goal, info):
        pass

    def _sample_goal(self) -> np.array:
        end_goal = np.zeros((len(self.goal_space_test)))
        for i, dim in enumerate(self.goal_space_test):
            assert len(dim) == 2, f"Each dimension of the goal space should have 2 values (lower and upper range), " \
                                  f"and not {len(dim)}."
            end_goal[i] = np.random.uniform(low=dim[0], high=dim[1])
        return end_goal

    @abstractmethod
    def _get_obs(self):
        pass

    @abstractmethod
    def goal_achieved(self):
        pass

    @abstractmethod
    def _is_success(self, achieved_goal, desired_goal):
        pass

    @abstractmethod
    def _set_action(self, action):
        pass

    @abstractmethod
    def _get_state(self):
        pass

    def _render_callback(self):
        pass
