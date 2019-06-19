import itertools
import numpy as np
import random
import tensorflow as tf
from collections import namedtuple

from GridworldSharingGym import GridworldSharingGym
from GridworldCoexistenceGym import GridworldCoexistenceGym


EXPERIMENT = "Coexistence" # "Sharing"
SELFISHNESS = 0.5
VALID_ACTIONS = [0, 1, 2, 3, 4]
MEMORY_LENGTH = 1


class Estimator():
    """Q-Value Estimator neural network.

    This network is used for the Q-Network, Target Network, and Empathy Network.
    """

    def __init__(self, scope="estimator"):
        self.scope = scope
        with tf.variable_scope(scope):
            self._build_model()

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our input
        # Our input are MEMORY_LENGTH frames of shape 5, 5 each
        self.X_pl = tf.placeholder(shape=[None, 5, 5, MEMORY_LENGTH], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.X_pl)/100
        batch_size = tf.shape(self.X_pl)[0]

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(X)
        fc1 = tf.contrib.layers.fully_connected(flattened, 128)
        fc2 = tf.contrib.layers.fully_connected(fc1, 128)

        self.predictions = tf.contrib.layers.fully_connected(fc2, len(VALID_ACTIONS), activation_fn=None)

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer parameters
        self.optimizer = tf.train.AdamOptimizer()
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, sess, s):
        return sess.run(self.predictions, {self.X_pl: s})

    def update(self, sess, s, a, y):
        feed_dict = {self.X_pl: s, self.y_pl: y, self.actions_pl: a}
        global_step, _, loss = sess.run([tf.contrib.framework.get_global_step(), self.train_op, self.loss], feed_dict)
        return loss

def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        print(f'q_values: {q_values}')
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def empathic_deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    empathic_estimator,
                    num_episodes,
                    replay_memory_size=500000,
                    replay_memory_init_size=50000,
                    update_target_estimator_every=10000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    batch_size=32,
                    selfishness=0.5):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        sess: Tensorflow Session object
        env: OpenAI environment
        q_estimator: Estimator object used for the q values
        target_estimator: Estimator object used for the targets
        empathic_estimator: Estimator object used for the empathic values
        num_episodes: Number of episodes to run for
        replay_memory_size: Size of the replay memory
        replay_memory_init_size: Number of random experiences to sampel when initializing
          the reply memory.
        update_target_estimator_every: Copy parameters from the Q estimator to the
          target estimator every N steps
        discount_factor: Gamma discount factor
        epsilon_start: Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value
        epsilon_end: The final minimum value of epsilon after decaying is done
        epsilon_decay_steps: Number of steps to decay epsilon over
        batch_size: Size of batches to sample from the replay memory
        selfishness: Parameter to weigh own rewards and others' rewards

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    replay_memory = []

    coins_collected = [0]
    enemy_coins_collected = [0]
    enemy_rewards = [0]
    got_killed = [0]
    enemies_killed = [0]
    equalities = []
    rewards = [0]
    episode_steps = []

    total_t = sess.run(tf.contrib.framework.get_global_step())

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(empathic_estimator, len(VALID_ACTIONS))

    # Populate the replay memory with initial experience
    print("Populating replay memory...")
    total_state = env.reset()
    state = total_state[:,:,0]
    enemy_state = total_state[:,:,1]

    state = np.stack([state] * MEMORY_LENGTH, axis=2)
    enemy_state = np.stack([enemy_state] * MEMORY_LENGTH, axis=2)

    total_state = np.stack([state, enemy_state], axis=2)

    for i in range(replay_memory_init_size):
        action_probs = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps-1)])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        next_total_state, reward, done, info = env.step(VALID_ACTIONS[action])

        next_state = next_total_state[:, :, 0]
        next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)

        next_enemy_state = next_total_state[:, :, 1]
        next_enemy_state = np.append(enemy_state[:, :, 1:], np.expand_dims(next_enemy_state, 2), axis=2)

        next_total_state = np.stack([next_state, next_enemy_state], axis=2)

        replay_memory.append(Transition(total_state, action, reward, next_total_state, done))
        if done:
            total_state = env.reset()
            state = total_state[:, :, 0]
            enemy_state = total_state[:, :, 1]

            state = np.stack([state] * MEMORY_LENGTH, axis=2)
            enemy_state = np.stack([enemy_state] * MEMORY_LENGTH, axis=2)

            total_state = np.stack([state, enemy_state], axis=2)
        else:
            total_state = next_total_state
            state = next_state
            enemy_state = next_enemy_state


    for i_episode in range(num_episodes):

        # Reset the environment
        total_state = env.reset()
        state = total_state[:, :, 0]
        enemy_state = total_state[:, :, 1]

        state = np.stack([state] * MEMORY_LENGTH, axis=2)
        enemy_state = np.stack([enemy_state] * MEMORY_LENGTH, axis=2)

        total_state = np.stack([state, enemy_state], axis=2)

        loss = None

        # One step in the environment
        for t in itertools.count():

            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

            # Maybe update the target estimator
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, q_estimator, target_estimator)
                print("\nCopied model parameters to target network.")

            # Take a step
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_total_state, reward, done, info = env.step(VALID_ACTIONS[action])

            next_state = next_total_state[:, :, 0]
            next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)

            next_enemy_state = next_total_state[:, :, 1]
            next_enemy_state = np.append(enemy_state[:, :, 1:], np.expand_dims(next_enemy_state, 2), axis=2)

            next_total_state = np.stack([next_state, next_enemy_state], axis=2)

            coins_collected[-1] += info['coins_collected']
            enemy_coins_collected[-1] += info['enemy_coins_collected']
            enemy_rewards[-1] += info['enemy_reward']
            got_killed[-1] += info['got_killed']
            enemies_killed[-1] += info['num_killed']
            rewards[-1] += reward

            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # Save transition to replay memory
            replay_memory.append(Transition(total_state, action, reward, next_total_state, done))

            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            # Calculate q values and targets (Double DQN)
            q_values_next = q_estimator.predict(sess, next_states_batch[:,:,:,0,:])
            best_actions = np.argmax(q_values_next, axis=1)

            q_values_next_target = target_estimator.predict(sess, next_states_batch[:,:,:,0,:])
            q_values_next_enemy_target = target_estimator.predict(sess, next_states_batch[:,:,:,1,:])

            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                            (discount_factor * q_values_next_target[np.arange(batch_size), best_actions])

            targets_enemy = discount_factor * np.max(q_values_next_enemy_target, axis=1)

            targets_empathy = selfishness*targets_batch + (1-selfishness)*targets_enemy

            # Perform gradient descent update on both the Q-network and Empathy-network
            states_batch = np.array(states_batch)
            q_estimator.update(sess, states_batch[:,:,:,0,:], action_batch, targets_batch)
            empathic_estimator.update(sess, states_batch[:,:,:,0,:], action_batch, targets_empathy)

            if done:
                print(f'Episode: {i_episode}    Reward: {rewards[-1]}')
                equality = (2*min(info['total_enemy_reward'], info['total_reward'])) / \
                           (info['total_enemy_reward'] + info['total_reward'])
                equalities.append(equality)
                episode_steps.append(t)
                coins_collected.append(0)
                enemy_coins_collected.append(0)
                enemy_rewards.append(0)
                got_killed.append(0)
                enemies_killed.append(0)
                rewards.append(0)

                total_state = env.reset()
                state = total_state[:, :, 0]
                enemy_state = total_state[:, :, 1]

                state = np.stack([state] * MEMORY_LENGTH, axis=2)
                enemy_state = np.stack([enemy_state] * MEMORY_LENGTH, axis=2)

                total_state = np.stack([state, enemy_state], axis=2)
                break
            else:
                total_state = next_total_state
                state = next_state
                enemy_state = next_enemy_state

            total_t += 1


if __name__ == "__main__":
    max_steps = 500

    tf.reset_default_graph()

    # Create a global step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create the Q-Network, Target Network, and Empathic Network
    q_estimator = Estimator(scope="q")
    target_estimator = Estimator(scope="target_q")
    empathic_estimator = Estimator(scope="empathic")

    if EXPERIMENT == "Coexistence":
        env = GridworldCoexistenceGym(headless=True,
                                      step_reward=1,
                                      kill_reward=0,
                                      max_steps=500,
                                      gridworld_size=7)
    elif EXPERIMENT == "Sharing":
        env = GridworldSharingGym(headless=True,
                                  step_reward=0,
                                  max_steps=500,
                                  gridworld_size=7)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        empathic_deep_q_learning(sess,
                                 env,
                                 q_estimator=q_estimator,
                                 target_estimator=target_estimator,
                                 empathic_estimator=empathic_estimator,
                                 num_episodes=100000,
                                 replay_memory_size=500000,
                                 replay_memory_init_size=20000,
                                 update_target_estimator_every=10000,
                                 epsilon_start=1.0,
                                 epsilon_end=0.01,
                                 epsilon_decay_steps=1000000,
                                 discount_factor=0.99,
                                 batch_size=32,
                                 selfishness=SELFISHNESS)