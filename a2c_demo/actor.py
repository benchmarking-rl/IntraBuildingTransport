#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import parl
import six
from collections import defaultdict
from parl.algorithms import A3C
from parl.utils.rl_utils import calc_gae

import sys
sys.path.append('./')
from intrabuildingtransport.env import IntraBuildingEnv
from utils import mansion_state_preprocessing, obs_dim, act_dim
from utils import action_idx_to_action_batch

from model import RLDispatcherModel, ElevatorAgent
from copy import deepcopy

class CompuEnv(object):
    """ vector of envs to support vector reset and vector step.
    `vector_step` api will automatically reset envs which are done.
    """

    def __init__(self, env):
        """
        Args:       envs: List of env
        """
        assert isinstance(env, IntraBuildingEnv)
        self.env = env
        self._mansion = env._mansion 
        self.ele_num = self._mansion.attribute.ElevatorNumber

        self._current_reward = None
        self._current_origin_reward = None
        self._num_steps = None
        self._total_steps = None
        self._episode_rewards = []
        self._episode_origin_rewards = []
        self._episode_lengths = []
        self._num_episodes = 0
        self._num_returned = 0

    def reset(self):
        """
        Returns:    List of all elevators' obs
        """
        state = self.env.reset()
        if self._total_steps is None:
            self._total_steps = sum(self._episode_lengths)

        if self._current_reward is not None:
            self._episode_rewards.append(self._current_reward)
            self._episode_origin_rewards.append(self._current_origin_reward)
            self._episode_lengths.append(self._num_steps)
            self._num_episodes += 1

        self._current_reward = 0
        self._current_origin_reward = 0
        self._num_steps = 0
        return state

    def step(self, action):
        """
        Args:       actions: List or array of action
        Returns:       obs_batch: List of next obs of envs
                    reward_batch: List of return reward of envs 
                    done_batch: List of done of envs 
        """
        # reset data after every 3600 s
        if self._total_steps % 3600 == 0:
            if self._current_reward is not None:
                self._episode_rewards.append(self._current_reward)
                self._episode_origin_rewards.append(self._current_origin_reward)
                self._episode_lengths.append(self._num_steps)
                self._num_episodes += 1
            print('origin_reward: %f, shaping_reward: %f, total_steps: %d' % \
                        (self._current_origin_reward, self._current_reward, self._total_steps))
            self._current_reward = 0
            self._current_origin_reward = 0
            self._num_steps = 0

        state, reward, done, info = self.env.step(action)
        origin_reward = reward 
        reward = - (info['time_consume'] + 0.01 * info['energy_consume'] +
                    100 * info['given_up_persons']) * 1.0e-3 / self.ele_num

        self._current_reward += reward
        self._current_origin_reward += origin_reward
        self._num_steps += 1
        self._total_steps += 1

        return (state, reward, done, info)

    def next_episode_results(self):
        for i in range(self._num_returned, len(self._episode_rewards)):
            yield (self._episode_rewards[i], self._episode_lengths[i], self._episode_origin_rewards[i])
        self._num_returned = len(self._episode_rewards)



class MultiVectorEnv(object):
    """ vector of envs to support vector reset and vector step.
    `vector_step` api will automatically reset envs which are done.
    """

    def __init__(self, envs, ele_num, act_dim):
        """
        Args:       envs: List of env
        """
        self.envs = envs
        self.envs_num = len(envs)
        self.ele_num = ele_num
        self.act_dim = act_dim

    def reset(self):
        """
        Returns:    List of all elevators' obs
        """
        reset_obs_batch = []
        for env in self.envs:
            obs = env.reset()
            obs_array = mansion_state_preprocessing(obs)
            reset_obs_batch.extend(obs_array)
        return reset_obs_batch

    def step(self, actions):
        """
        Args:       actions: List or array of action
        Returns:       obs_batch: List of next obs of envs
                    reward_batch: List of return reward of envs 
                    done_batch: List of done of envs 
        """
        obs_batch, reward_batch, done_batch = [], [], []
        multi_actions = np.array_split(actions, int(len(actions) / self.ele_num))

        assert len(multi_actions[0]) == self.ele_num
        actions = action_idx_to_action_batch(multi_actions, self.act_dim)

        for env_id in six.moves.range(self.envs_num):
            obs, reward, done, info = self.envs[env_id].step(actions[env_id])
            obs_array = mansion_state_preprocessing(obs)

            if done:
                obs = self.envs[env_id].reset()
                obs_array = mansion_state_preprocessing(obs)

            obs_batch.extend(obs_array)
            reward_batch.extend([reward for i in range(self.ele_num)])
            done_batch.extend([done for i in range(self.ele_num)])
        return obs_batch, reward_batch, done_batch


@parl.remote_class
class Actor(object):
    def __init__(self, config):
        self.config = config

        self.envs = []
        for _ in six.moves.range(config['env_num']):
            env = IntraBuildingEnv("config.ini")
            env = CompuEnv(env)
            self.envs.append(env)
        
        self._mansion_attr = env._mansion.attribute
        self._obs_dim = obs_dim(self._mansion_attr)
        self._act_dim = act_dim(self._mansion_attr)

        self.config['obs_shape'] = self._obs_dim
        self.config['act_dim'] = self._act_dim

        self.ele_num = self._mansion_attr.ElevatorNumber
        self.max_floor = self._mansion_attr.NumberOfFloor 
        self.config['ele_num'] = self.ele_num
        self.config['max_floor'] = self.max_floor

        self.vector_env = MultiVectorEnv(self.envs, self.ele_num, self._act_dim)

        self.obs_batch = self.vector_env.reset()

        model = RLDispatcherModel(self._act_dim)
        algorithm = A3C(model, hyperparas=config)
        self.agent = ElevatorAgent(algorithm, config)

    def sample(self):  
        sample_data = defaultdict(list)

        env_sample_data = {}
        for env_id in six.moves.range(self.config['env_num'] * self.ele_num):
            env_sample_data[env_id] = defaultdict(list)

        for i in six.moves.range(self.config['sample_batch_steps']):
            actions_batch, values_batch = self.agent.sample(
                np.stack(self.obs_batch))
            
            next_obs_batch, reward_batch, done_batch = \
                    self.vector_env.step(actions_batch)

            for env_id in six.moves.range(self.config['env_num'] * self.ele_num):
                env_sample_data[env_id]['obs'].append(self.obs_batch[env_id])
                env_sample_data[env_id]['actions'].append(
                    actions_batch[env_id])
                env_sample_data[env_id]['rewards'].append(reward_batch[env_id])
                env_sample_data[env_id]['dones'].append(done_batch[env_id])
                env_sample_data[env_id]['values'].append(values_batch[env_id])

                # Calculate advantages when the episode is done or reach max sample steps.
                if done_batch[
                        env_id] or i == self.config['sample_batch_steps'] - 1:
                    next_value = 0
                    if not done_batch[env_id]:
                        next_obs = np.expand_dims(next_obs_batch[env_id], 0)
                        next_value = self.agent.value(next_obs)

                    values = env_sample_data[env_id]['values']
                    rewards = env_sample_data[env_id]['rewards']
                    advantages = calc_gae(rewards, values, next_value,
                                          self.config['gamma'],
                                          self.config['lambda'])
                    target_values = advantages + values

                    sample_data['obs'].extend(env_sample_data[env_id]['obs'])
                    sample_data['actions'].extend(
                        env_sample_data[env_id]['actions'])
                    sample_data['advantages'].extend(advantages)
                    sample_data['target_values'].extend(target_values)

                    env_sample_data[env_id] = defaultdict(list)

            self.obs_batch = deepcopy(next_obs_batch)

        # size of sample_data: env_num * sample_batch_steps
        for key in sample_data:
            sample_data[key] = np.stack(sample_data[key])

        return sample_data

    def get_metrics(self):
        metrics = defaultdict(list)
        for env in self.envs:
            monitor  = env
            if monitor is not None:
                for episode_rewards, episode_steps, episode_origin_rewards in monitor.next_episode_results():
                    metrics['episode_rewards'].append(episode_rewards)
                    metrics['episode_origin_rewards'].append(episode_origin_rewards)
                    metrics['episode_steps'].append(episode_steps)
        return metrics

    def set_params(self, params):
        self.agent.set_params(params)


if __name__ == '__main__':
    from a2c_config import config

    actor = Actor(config)
    actor.as_remote(config['server_ip'], config['server_port'])
