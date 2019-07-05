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

import paddle.fluid as fluid
import parl.layers as layers
import numpy as np
from parl.framework.model_base import Model
from parl.framework.agent_base import Agent

from parl.utils.scheduler import PiecewiseScheduler, LinearDecayScheduler

class RLDispatcherModel(Model):
    def __init__(self, act_dim):
        self._act_dim = act_dim
        self._fc_1 = layers.fc(size=512, act='relu')
        self._fc_2 = layers.fc(size=256, act='relu')
        self._fc_3 = layers.fc(size=128, act='tanh')

        self.value_fc = layers.fc(size=1)
        self.policy_fc = layers.fc(size=act_dim)

    def policy(self, obs):
        """
        Args:obs: A float32 tensor 
        Returns:policy_logits: B * ACT_DIM
        """
        h_1 = self._fc_1(obs)
        h_2 = self._fc_2(h_1)
        h_3 = self._fc_3(h_2)
        policy_logits = self.policy_fc(h_3)
        return policy_logits

    def value(self, obs):
        """
        Args:       obs: A float32 tensor 
        Returns:    values: B
        """
        h_1 = self._fc_1(obs)
        h_2 = self._fc_2(h_1)
        h_3 = self._fc_3(h_2)
        values = self.value_fc(h_3)
        values = layers.squeeze(values, axes=[1])
        return values

    def policy_and_value(self, obs):
        """
        Args:       obs: A float32 tensor
        Returns:    policy_logits: B * ACT_DIM
                    values: B
        """
        # print('obs.shape: ', obs.shape)
        h_1 = self._fc_1(obs)
        h_2 = self._fc_2(h_1)
        h_3 = self._fc_3(h_2)
        policy_logits = self.policy_fc(h_3)
        values = self.value_fc(h_3)
        values = layers.squeeze(values, axes=[1])

        return policy_logits, values


class ElevatorAgent(Agent):
    def __init__(self, algorithm, config):
        self.config = config

        self._action_dim = config['act_dim']
        self._obs_dim = config['obs_shape']

        self._ele_num = config['ele_num']
        self._max_floor = config['max_floor']
        # self._update_target_steps = 1000
        # self._global_step = 0
        super(ElevatorAgent, self).__init__(algorithm)

        self.lr_scheduler = LinearDecayScheduler(config['start_lr'],
                                                 config['max_sample_steps'])
        self.entropy_coeff_scheduler = PiecewiseScheduler(
            config['entropy_coeff_scheduler'])

        use_cuda = True if self.gpu_id >= 0 else False

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.use_experimental_executor = True
        exec_strategy.num_threads = 4
        build_strategy = fluid.BuildStrategy()
        build_strategy.remove_unnecessary_lock = True

        # Use ParallelExecutor to make learn program run faster
        self.learn_exe = fluid.ParallelExecutor(
            use_cuda=use_cuda,
            main_program=self.learn_program,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

    def build_program(self):
        self.sample_program = fluid.Program()
        self.predict_program = fluid.Program()
        self.value_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.sample_program):
            # obs = layers.data(name='obs', shape=[self._obs_dim], dtype='float32')
            obs = layers.data(
                name='obs', shape=[self.config['obs_shape']], dtype='float32')
            sample_actions, values = self.alg.sample(obs)
            self.sample_outputs = [sample_actions, values]

        with fluid.program_guard(self.predict_program):
            obs = layers.data(
                name='obs', shape=[self.config['obs_shape']], dtype='float32')
            self.predict_actions = self.alg.predict(obs)

        with fluid.program_guard(self.value_program):
            obs = layers.data(
                name='obs', shape=[self.config['obs_shape']], dtype='float32')
            self.values = self.alg.value(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(
                name='obs', shape=[self.config['obs_shape']], dtype='float32')
            # action = layers.data(name='act', shape=[1], dtype='int32')
            # reward = layers.data(name='reward', shape=[], dtype='float32')
            # next_obs = layers.data(
            #         name = 'next_obs',
            #         shape = [self._obs_dim],
            #         dtype = 'float32'
            #         )
            actions = layers.data(name='actions', shape=[], dtype='int32')
            advantages = layers.data(
                name='advantages', shape=[], dtype='float32')
            target_values = layers.data(
                name='target_values', shape=[], dtype='float32')
            lr = layers.data(
                name='lr', shape=[1], dtype='float32', append_batch_size=False)
            entropy_coeff = layers.data(
                name='entropy_coeff', shape=[], dtype='float32')

            total_loss, pi_loss, vf_loss, entropy = self.alg.learn(
                obs, actions, advantages, target_values, lr, entropy_coeff)
            self.learn_outputs = [
                total_loss.name, pi_loss.name, vf_loss.name, entropy.name
            ]
            # terminal = layers.data(name='terminal', shape=[], dtype='bool')
            #self._cost = self.alg.define_learn(obs, action, reward, next_obs, terminal)

    def sample(self, obs_np):
        """
        Args:       obs_np: a numpy float32 array of shape ([B] + observation_space).
        Returns:    sample_ids: a numpy int64 array of shape [B]
                    values: a numpy float32 array of shape [B]
        """
        obs_np = obs_np.astype('float32')

        sample_actions, values = self.fluid_executor.run(
            self.sample_program,
            feed={'obs': obs_np},
            fetch_list=self.sample_outputs)
        return sample_actions, values


    def predict(self, obs_np):
        """
        Args:       obs_np: a numpy float32 array of shape ([B] + observation_space).
        Returns:    sample_ids: a numpy int64 array of shape [B]
        """
        obs_np = obs_np.astype('float32')

        predict_actions = self.fluid_executor.run(
            self.predict_program,
            feed={'obs': obs_np},
            fetch_list=[self.predict_actions])[0]
        return predict_actions

    def value(self, obs_np):
        """
        Args:       obs_np: a numpy float32 array of shape ([B] + observation_space).
        Returns:    values: a numpy float32 array of shape [B]
        """
        obs_np = obs_np.astype('float32')

        values = self.fluid_executor.run(
            self.value_program, feed={'obs': obs_np},
            fetch_list=[self.values])[0]
        return values

    def learn(self, obs_np, actions_np, advantages_np, target_values_np):
        """
        Args:       obs_np: a numpy float32 array of shape ([B] + observation_space).
                    actions_np: a numpy int64 array of shape [B]
                    advantages_np: a numpy float32 array of shape [B]
                    target_values_np: a numpy float32 array of shape [B]
        """

        obs_np = obs_np.astype('float32')
        actions_np = actions_np.astype('int64')
        advantages_np = advantages_np.astype('float32')
        target_values_np = target_values_np.astype('float32')

        lr = self.lr_scheduler.step(step_num=obs_np.shape[0])
        entropy_coeff = self.entropy_coeff_scheduler.step()

        total_loss, pi_loss, vf_loss, entropy = self.learn_exe.run(
            feed={
                'obs': obs_np,
                'actions': actions_np,
                'advantages': advantages_np,
                'target_values': target_values_np,
                'lr': np.array([lr], dtype='float32'),
                'entropy_coeff': np.array([entropy_coeff], dtype='float32')
            },
            fetch_list=self.learn_outputs)
        return total_loss, pi_loss, vf_loss, entropy, lr, entropy_coeff
