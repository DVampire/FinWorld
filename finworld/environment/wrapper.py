from gym import Wrapper, spaces
import random
import numpy as np
import gym
from gym.envs.registration import register
from typing import Any

register(id = "EnvironmentAgentTrading", entry_point = "finworld.environment.wrapper:EnvironmentAgentTradingWrapper")
register(id = "EnvironmentSequenceTrading", entry_point = "finworld.environment.wrapper:EnvironmentSequenceTradingWrapper")
register(id = "EnvironmentPatchTrading", entry_point = "finworld.environment.wrapper:EnvironmentPatchTradingWrapper")
register(id = "EnvironmentPatchPortfolio", entry_point = "finworld.environment.wrapper:EnvironmentPatchPortfolioWrapper")

class EnvironmentAgentTradingWrapper(Wrapper):
    def __init__(self,
                 env: Any,
                 transition_shape = None,
                 seed=42):
        super().__init__(env)
        self.seed = seed

        self.env = env

        random.seed(seed)
        np.random.seed(seed)

        self.action_labels = env.action_labels

        state_shape = transition_shape["states"]["shape"][1:]
        state_type = transition_shape["states"]["type"]

        self.action_space = spaces.Discrete(
            n = env.action_dim,
        )

        self.observation_space = spaces.Dict({
            'states': spaces.Box(low=-np.inf, high=np.inf, shape=state_shape, dtype=state_type),
        })

        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def reset(self):
        state, info = self.env.reset()
        return state, info

    def step(self, action):
        next_state, reward, done, truncted, info = self.env.step(action)
        return next_state, reward, done, truncted, info

class EnvironmentSequenceTradingWrapper(Wrapper):
    def __init__(self,
                 env: Any,
                 transition_shape = None,
                 seed=42):
        super().__init__(env)
        self.seed = seed

        self.env = env

        random.seed(seed)
        np.random.seed(seed)

        self.action_labels = env.action_labels

        self.action_space = spaces.Discrete(
            n = env.action_dim,
        )

        observation_transition_shape = {
            key: value for key, value in transition_shape.items() if value["obs"] is True
        }
        self.observation_space = spaces.Dict({
            key: spaces.Box(low=transition_shape[key]["low"],
                            high=transition_shape[key]["high"],
                            shape=transition_shape[key]["shape"][1:],
                            dtype=transition_shape[key]["type"]
                            )
            for key in observation_transition_shape
        })

        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def reset(self):
        state, info = self.env.reset()
        return state, info

    def step(self, action):
        next_state, reward, done, truncted, info = self.env.step(action)
        return next_state, reward, done, truncted, info

class EnvironmentPatchTradingWrapper(Wrapper):
    def __init__(self,
                 env: Any,
                 transition_shape = None,
                 seed=42):
        super().__init__(env)
        self.seed = seed

        self.env = env

        random.seed(seed)
        np.random.seed(seed)

        self.action_labels = env.action_labels

        self.action_space = spaces.Discrete(
            n = env.action_dim,
        )

        observation_transition_shape = {
            key: value for key, value in transition_shape.items() if value["obs"] is True
        }
        self.observation_space = spaces.Dict({
            key: spaces.Box(low=transition_shape[key]["low"],
                            high=transition_shape[key]["high"],
                            shape=transition_shape[key]["shape"][1:],
                            dtype=transition_shape[key]["type"]
                            )
            for key in observation_transition_shape
        })

        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def reset(self):
        state, info = self.env.reset()
        return state, info

    def step(self, action):
        next_state, reward, done, truncted, info = self.env.step(action)
        return next_state, reward, done, truncted, info


class EnvironmentPatchPortfolioWrapper(Wrapper):
    def __init__(self,
                 env: Any,
                 transition_shape = None,
                 seed=42):
        super().__init__(env)
        self.seed = seed

        self.env = env

        random.seed(seed)
        np.random.seed(seed)

        self.action_space = spaces.Discrete(
            n = env.action_dim,
        )

        self.action_space = spaces.Box(
            low=.0,
            high=1.0,
            shape=(env.action_dim,),
            dtype=np.float32
        )

        observation_transition_shape = {
            key: value for key, value in transition_shape.items() if value["obs"] is True
        }
        self.observation_space = spaces.Dict({
            key: spaces.Box(low=transition_shape[key]["low"],
                            high=transition_shape[key]["high"],
                            shape=transition_shape[key]["shape"][1:],
                            dtype=transition_shape[key]["type"]
                            )
            for key in observation_transition_shape
        })

        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def reset(self):
        state, info = self.env.reset()
        return state, info

    def step(self, action):
        next_state, reward, done, truncted, info = self.env.step(action)
        return next_state, reward, done, truncted, info


def make_env(env_id, env_params):
    def thunk():
        env = gym.make(env_id, **env_params)
        return env
    return thunk