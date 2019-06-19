import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces


class GridworldSharingGym(gym.Env):

    def __init__(self, headless=True, gridworld_size=7, max_steps=20000, step_reward=1, window_size=5):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-10000000, high=100000000, dtype=np.float, shape=(window_size, window_size, 2))
        self.headless = headless
        self.gridworld_size = gridworld_size
        self.window_size = window_size
        self.max_steps = max_steps
        self.step_reward = step_reward

        self.reset()

    def reset(self):
        self.agent_position = [int(self.gridworld_size/2), int(self.gridworld_size/2)]
        self.enemy_positions = [[int(self.gridworld_size/2), int(self.gridworld_size/2)]]
        self.coin_positions = [[x,y] for y in range(1, self.gridworld_size, 2) for x in range(1, self.gridworld_size, 2)]
        self.coins_collected = 0
        self.enemy_coins_collected = 0
        self.steps = 0
        self.agent_can_start = np.random.choice([0, 1])
        self.get_observation()

        return self.observation

    def get_observation(self):
        self.observation = np.zeros((self.gridworld_size, self.gridworld_size))

        for coin in self.coin_positions:
            self.observation[coin[0], coin[1]] = -1

        self.observation[max(0, min(self.agent_position[0], self.gridworld_size-1)), max(0, min(self.agent_position[1], self.gridworld_size-1))] = 5*(self.coins_collected + 1)

        closest_enemy = None
        closest_distance = np.inf
        for pos in self.enemy_positions:
            distance = np.linalg.norm(np.array(pos)-np.array(self.agent_position))
            if distance < closest_distance:
                closest_distance = distance
                closest_enemy = pos

        self.observation[max(0, min(closest_enemy[0], self.gridworld_size-1)), max(0, min(closest_enemy[1], self.gridworld_size-1))] = 5*(self.enemy_coins_collected + 1)

        if closest_enemy:
            window = [range(self.agent_position[0] - int(self.window_size / 2),
                            self.agent_position[0] + int(self.window_size / 2) + 1),
                      range(self.agent_position[1] - int(self.window_size / 2),
                            self.agent_position[1] + int(self.window_size / 2) + 1)]
            agent_observation = self.observation.take(window[0], axis=0, mode='wrap')

            agent_observation = agent_observation.take(window[1], axis=1, mode='wrap')

            enemy_window = [range(closest_enemy[0] - int(self.window_size/2), closest_enemy[0] + int(self.window_size/2) + 1), range(closest_enemy[1] - int(self.window_size/2), closest_enemy[1] + int(self.window_size/2) + 1)]
            enemy_observation = self.observation.take(enemy_window[0], axis=0, mode='wrap')
            enemy_observation = enemy_observation.take(enemy_window[1], axis=1, mode='wrap')

        else:
            enemy_observation = np.zeros((self.window_size, self.window_size))

        self.observation = np.dstack([agent_observation, enemy_observation])
        return self.observation

    def plot_env(self):
        env = np.zeros((self.gridworld_size, self.gridworld_size))
        env[max(0, min(self.agent_position[0], self.gridworld_size-1)), max(0, min(self.agent_position[1], self.gridworld_size-1))] = 1

        for pos in self.coin_positions:
            env[max(0, min(pos[0], self.gridworld_size-1)), max(0, min(pos[1], self.gridworld_size-1))] = -1
        for pos in self.enemy_positions:
            env[max(0, min(pos[0], self.gridworld_size-1)), max(0, min(pos[1], self.gridworld_size-1))] = -2
        if not self.headless:
            plt.matshow(env, 1, cmap='gray')
            plt.draw()
            plt.pause(0.01)

    def step(self, action_num):
        coin_collected = False
        enemey_coin_collected = False

        if (self.agent_can_start or self.steps > 0):
            #move agent
            if action_num == 0:
                self.agent_position = [(self.agent_position[0] + 1) % self.gridworld_size, self.agent_position[1]]
            elif action_num == 1:
                self.agent_position = [(self.agent_position[0] - 1) % self.gridworld_size, self.agent_position[1]]
            elif action_num == 2:
                self.agent_position = [self.agent_position[0], (self.agent_position[1] + 1) % self.gridworld_size]
            elif action_num == 3:
                self.agent_position = [self.agent_position[0], (self.agent_position[1] - 1) % self.gridworld_size]

            if self.agent_position in self.coin_positions:
                coin_collected = True
                self.coins_collected += 1
                self.coin_positions = [coin_pos for coin_pos in self.coin_positions if coin_pos != self.agent_position]

        if ((not self.agent_can_start) or self.steps > 0):
            #move enemies
            for i, pos in enumerate(self.enemy_positions):
                if self.steps % 7 == 0:
                    action = 1
                else:
                    action = 2
                if action == 0:
                    pos = [(pos[0] + 1) % self.gridworld_size, pos[1]]
                elif action == 1:
                    pos = [(pos[0] - 1) % self.gridworld_size, pos[1]]
                elif action == 2:
                    pos = [pos[0], (pos[1] + 1) % self.gridworld_size]
                elif action == 3:
                    pos = [pos[0], (pos[1] - 1) % self.gridworld_size]
                self.enemy_positions[i] = pos

                if pos in self.coin_positions:
                    enemey_coin_collected = True
                    self.enemy_coins_collected += 1
                    self.coin_positions = [coin_pos for coin_pos in self.coin_positions if
                                           coin_pos != pos]

        # increase counter
        self.steps += 1

        # plot
        if not self.headless:
            self.plot_env()

        total_rewards_collected = np.sum([(1.1 - x * 0.1) for x in range(self.coins_collected)])
        total_enemy_rewards_collected = np.sum([(1.1 - x * 0.1) for x in range(self.enemy_coins_collected)])

        reward = coin_collected * (1.1 - self.coins_collected * 0.1)
        enemy_reward = enemey_coin_collected * (1.1 - self.enemy_coins_collected * 0.1)

        info = {'got_killed': 0, 'num_killed': 0, 'coins_collected': coin_collected,
                'enemy_coins_collected': enemey_coin_collected, 'enemy_reward': enemy_reward,
                'total_enemy_reward': total_enemy_rewards_collected, 'total_reward': total_rewards_collected}

        restart = len(self.coin_positions) == 0 or self.steps >= self.max_steps

        self.observation = self.get_observation()

        if restart:
            self.reset()

        return self.observation, reward, restart, info

    def render(self, mode='human', close=False):
        pass



if __name__ == "__main__":
    env = GridworldSharingGym(headless=False, gridworld_size=7)

    while True:
        action = np.random.choice(range(4))
        env.step(action)
