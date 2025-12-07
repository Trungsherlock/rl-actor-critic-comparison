
import math, random, argparse, os
import numpy as np
import matplotlib.pyplot as plt
# from networks.policy_network import PolicyNetwork
# from networks.value_network import ValueNetwork


class MountainCarEnv:
    MIN_POS = -1.2
    MAX_POS = 0.5
    MIN_VEL = -0.07
    MAX_VEL = 0.07
    GOAL_X = MAX_POS

    def __init__(self, max_steps=1000, gamma=1.0, seed=None):
        self.max_steps = max_steps
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.state = None
        self.t = 0

    def reset(self):
        x0 = self.rng.uniform(-0.6, -0.4)
        self.state = np.array([x0, 0.0], dtype=np.float32)
        self.t = 0
        return self.state.copy()

    def step(self, a): 
        x, v = float(self.state[0]), float(self.state[1])
        v_next = v + 0.001 * a - 0.0025 * math.cos(3.0 * x)
        v_next = float(np.clip(v_next, self.MIN_VEL, self.MAX_VEL))
        x_next = x + v_next
        x_next = float(np.clip(x_next, self.MIN_POS, self.MAX_POS))

        if x_next == self.MIN_POS or x_next == self.MAX_POS:
            v_next = 0.0

        self.state[:] = (x_next, v_next)
        self.t += 1

        done = (x_next == self.GOAL_X) or (self.t >= self.max_steps)
        reward = 0.0 if (x_next == self.GOAL_X) else -1.0
        return self.state.copy(), reward, done