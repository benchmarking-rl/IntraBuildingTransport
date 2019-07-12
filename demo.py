# -*- coding: UTF-8 -*-
##########################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
##########################################################################

"""

A running demo of elevators

Authors: wangfan04(wangfan04@baidu.com)
Date:    2019/05/22 19:30:16
"""

from intrabuildingtransport.env import IntraBuildingEnv
from intrabuildingtransport.mansion.person_generators.generator_proxy import PersonGenerator
from intrabuildingtransport.mansion.mansion_config import MansionConfig
from intrabuildingtransport.mansion.utils import ElevatorState, MansionState
from intrabuildingtransport.mansion.mansion_manager import MansionManager
import sys
import argparse
import configparser


def run_mansion_main(mansion_env, policy_handle, iteration, need_render=False):
    mansion_env.reset()
    policy_handle.link_mansion(mansion_env.attribute)
    policy_handle.load_settings()
    i = 0
    acc_reward = 0.0
    #acc_time = 0.0
    #acc_energy = 0.0
    while i < iteration:
        i += 1
        if need_render:
            mansion_env.render()
        state = mansion_env.state
        action = policy_handle.policy(state)
        _, r, _, _ = mansion_env.step(action)
        output_info = policy_handle.feedback(state, action, r)
        acc_reward += r
        #acc_time += time_consume
        #acc_energy += energy_consume
        if(isinstance(output_info, dict) and len(output_info) > 0):
            mansion_env.log_notice("%s", output_info)
        if(i % 3600 == 0):
            mansion_env.log_notice(
                "Accumulated Reward: %f, Mansion Status: %s",
                acc_reward, mansion_env.statistics)
            #acc_time = 0.0
            #acc_energy = 0.0
            acc_reward = 0.0

# run main program with args
def run_main(args):

    parser = argparse.ArgumentParser(description='Run elevator simulation')
    parser.add_argument('--configfile', type=str, default='./config.ini',
                            help='configuration file for running elevators')
    parser.add_argument('--iterations', type=int, default=100000000,
                            help='total number of iterations')
    parser.add_argument('--controlpolicy', type=str, default='rule_benchmark',
                            help='policy type: rule_benchmark or others')
    parser.add_argument('--render', type=str, default=False,
                            help='if set True, GUI will be shown')
    args = parser.parse_args(args)
    print('configfile:', args.configfile)
    print('iterations:', args.iterations)
    print('controlpolicy:', args.controlpolicy)
    print('render:', args.render)

    control_module = ("dispatchers.{}.dispatcher"
                      .format(args.controlpolicy))
    Dispatcher = __import__(control_module, fromlist=[None]).Dispatcher

    mansion_env = IntraBuildingEnv(args.configfile)
    dispatcher = Dispatcher()
    run_mansion_main(mansion_env, dispatcher, args.iterations, args.render)

    return 0


if __name__ == "__main__":
    run_main(sys.argv[1:])
