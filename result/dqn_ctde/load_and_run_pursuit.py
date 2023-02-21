"""
code that train pursuit with dqn
code can compile for very small parameter but have not been tested in a full train
"""
import argparse
import os
import pprint

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, MultiDiscreteToDiscrete
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net

import sys
import datetime

sys.path.append("../..")
sys.path.append("../../lib")
sys.path.append("..")
sys.path.append("../lib")
from lib.myPursuit_gym import my_parallel_env
from lib.mydqn import myDQNPolicy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--task", type=str, default="pursuit_v4")
    parser.add_argument("--reward-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--step-per-epoch", type=int, default=10000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128]
    )
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--prioritized-replay", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_known_args()[0]
    return args


def test_dqn(args=get_args()):
    task_parameter = {
        "shared_reward": False,
        "freeze_evaders": True,
        "surround": False,
        "max_cycles" : 50,  # 500

        "n_evaders": 8,
        #"n_pursuers": 7,
        #"x_size" : 8,  # 16
        #"y_size" : 8,  # 16
        #"obs_range" : 4,  # 7
    }

    args.render = 0.01

    # seed
    if args.seed is None:
        args.seed = int(np.random.rand()*10000)
        print(f'Seed is not given in arguments')
    print(f'Seed is {args.seed}')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = my_parallel_env
    env_ = my_parallel_env(**task_parameter)
    args.state_shape = env_.observation_space.shape or env_.observation_space.n
    # args.action_shape = env.action_space.shape or env.action_space.n
    args.state_shape = args.state_shape[1:]
    args.action_shape = 5
    if args.reward_threshold is None:
        default_reward_threshold = {"pursuit_v4": 195}
        args.reward_threshold = default_reward_threshold.get(
            args.task  # , env.spec.reward_threshold
        )

    # Q_param = V_param = {"hidden_sizes": [128]}
    # model
    net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        # dueling=(Q_param, V_param),
    ).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = myDQNPolicy(
        net,
        optim,
        args.gamma,
        args.n_step,
        target_update_freq=args.target_update_freq,
    )

    if __name__ == "__main__":
        policy.load_state_dict(torch.load(args.path, map_location=torch.device('cpu')))

        env = DummyVectorEnv(
            [
                lambda: env(render_mode="human", **task_parameter),
            ]
        )
        env.seed(args.seed)

        policy.eval()
        policy.set_eps(args.eps_test)
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


def test_pdqn(args=get_args()):
    args.prioritized_replay = True
    args.gamma = 0.95
    args.seed = 1
    test_dqn(args)


if __name__ == "__main__":
    test_dqn(get_args())
