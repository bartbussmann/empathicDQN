import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces


class GridworldCoexistenceGym(gym.Env):

    def __init__(self, headless=True, gridworld_size=11, max_steps=20000, kill_reward=0, step_reward=1, window_size=5):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-10000000, high=100000000, dtype=np.float, shape=(window_size, window_size, 2))
        self.headless = headless
        self.gridworld_size = gridworld_size
        self.window_size = window_size
        self.max_steps = max_steps
        self.kill_reward = kill_reward
        self.step_reward = step_reward
        self.reset()

    def reset(self):
        self.agent_position = [int(self.gridworld_size/2), int(self.gridworld_size/2)]
        self.enemy_positions = [[0,0]]
        self.coin_positions = []
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

        window = [range(self.agent_position[0] - int(self.window_size / 2),
                        self.agent_position[0] + int(self.window_size / 2) + 1),
                  range(self.agent_position[1] - int(self.window_size / 2),
                        self.agent_position[1] + int(self.window_size / 2) + 1)]
        agent_observation = self.observation.take(window[0], axis=0, mode='wrap')
        agent_observation = agent_observation.take(window[1], axis=1, mode='wrap')

        closest_enemy = None
        closest_distance = np.inf
        for pos in self.enemy_positions:
            self.observation[max(0, min(pos[0], self.gridworld_size-1)), max(0, min(pos[1], self.gridworld_size-1))] = 5*(self.enemy_coins_collected + 1)
            distance = np.linalg.norm(np.array(pos)-np.array(self.agent_position))
            if distance < closest_distance:
                closest_distance = distance
                closest_enemy = pos

        if closest_enemy:
            self.observation[max(0, min(closest_enemy[0], self.gridworld_size - 1)), max(0, min(closest_enemy[1], self.gridworld_size - 1))] = \
                5 * (self.enemy_coins_collected + 1)


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
        self.observation = self.get_observation()

        env = np.zeros((self.gridworld_size, self.gridworld_size))

        env[max(0, min(self.agent_position[0], self.gridworld_size-1)), max(0, min(self.agent_position[1], self.gridworld_size-1))] = 1
        for pos in self.enemy_positions:
            env[max(0, min(pos[0], self.gridworld_size-1)), max(0, min(pos[1], self.gridworld_size-1))] = -1

        if not self.headless:
            plt.matshow(env, 1, cmap='gray')
            plt.draw()
            plt.pause(0.01)

    def step(self, action_num):

        #increase counter
        self.steps += 1

        #move agent
        if action_num == 0:
            self.agent_position = [(self.agent_position[0] + 1) % self.gridworld_size, self.agent_position[1]]
        elif action_num == 1:
            self.agent_position = [(self.agent_position[0] - 1) % self.gridworld_size, self.agent_position[1]]
        elif action_num == 2:
            self.agent_position = [self.agent_position[0], (self.agent_position[1] + 1) % self.gridworld_size]
        elif action_num == 3:
            self.agent_position = [self.agent_position[0], (self.agent_position[1] - 1) % self.gridworld_size]

        #move enemies
        for i, pos in enumerate(self.enemy_positions):
            if self.steps % 7 == 0:
                action = np.random.choice(range(5))
            else:
                action = np.random.choice(range(5))
            if action == 0:
                pos = [(pos[0] + 1) % self.gridworld_size, pos[1]]
            elif action == 1:
                pos = [(pos[0] - 1) % self.gridworld_size, pos[1]]
            elif action == 2:
                pos = [pos[0], (pos[1] + 1) % self.gridworld_size]
            elif action == 3:
                pos = [pos[0], (pos[1] - 1) % self.gridworld_size]
            self.enemy_positions[i] = pos

        # update deads
        dead = (self.agent_position in [[x[0] + 1, x[1]] for x in self.enemy_positions]) or (
                self.agent_position in [[x[0], x[1] + 1] for x in self.enemy_positions]) or (
                self.agent_position in [[x[0], x[1]] for x in self.enemy_positions] and action_num not in [0,2])

        if dead:
            num_enemies_killed = 0
        else:
            num_enemies_killed = self.enemies_to_be_killed()

        if not self.headless:
            self.plot_env()

        reward = self.step_reward + self.kill_reward*num_enemies_killed

        info = {'got_killed': dead, 'num_killed': num_enemies_killed, 'coins_collected': 0, 'enemy_coins_collected': 0, 'enemy_reward': 0, 'total_reward': 1, 'total_enemy_reward': 1}


        self.observation = self.get_observation()

        if num_enemies_killed > 0:
            self.observation[:, :, 1] = np.zeros((self.window_size,self.window_size))

        if dead:
            self.observation[:, :, 0] = np.zeros((self.window_size, self.window_size))

        self.kill_enemies()

        restart = dead or self.steps > self.max_steps
        if restart:
            self.reset()

        return self.observation, reward, restart, info


    def kill_enemies(self):
        enemies_to_be_killed = []
        for enemy_pos in self.enemy_positions:
            if enemy_pos == [self.agent_position[0] +1, self.agent_position[1]] or enemy_pos == [self.agent_position[0], self.agent_position[1] + 1]:
                enemies_to_be_killed.append(enemy_pos)
        self.enemy_positions = [x for x in self.enemy_positions if x not in enemies_to_be_killed]
        return len(enemies_to_be_killed)

    def enemies_to_be_killed(self):
        enemies_to_be_killed = []
        for enemy_pos in self.enemy_positions:
            if enemy_pos == [self.agent_position[0] +1, self.agent_position[1]] or enemy_pos == [self.agent_position[0], self.agent_position[1] + 1]:
                enemies_to_be_killed.append(enemy_pos)
        return len(enemies_to_be_killed)

    def render(self, mode='human', close=False):
        pass


if __name__ == "__main__":
    env = GridworldCoexistenceGym(headless=False, gridworld_size=7, window_size=5)

    while True:
        action = np.random.choice(range(5))
        env.step(action)
