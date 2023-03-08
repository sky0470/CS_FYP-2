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
from tianshou.utils.net.common import Net, ActorCritic, DataParallelNet
from tianshou.utils.net.discrete import Actor, Critic

import sys
import datetime

from pursuit_msg.policy.myppo import myPPOPolicy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--task", type=str, default="pursuit_v4")
    parser.add_argument("--reward-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=20000)
    # parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    # parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--step-per-epoch", type=int, default=50000)
    parser.add_argument("--step-per-collect", type=int, default=2000)
    parser.add_argument("--repeat-per-collect", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    # parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128])
    parser.add_argument("--training-num", type=int, default=20)
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    # parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--render", type=float, default=0.001)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    # ppo special
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--rew-norm", type=int, default=0)
    parser.add_argument("--norm-adv", type=int, default=0)
    parser.add_argument("--recompute-adv", type=int, default=0)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=0)

    # switch env
    parser.add_argument('--env', type=str, default=None)

    args = parser.parse_args()
    return args


def test_ppo(args=get_args()):
    task_parameter = {
        "shared_reward": False,
        "surround": False,
        "freeze_evaders": True,

        "x_size": 10,
        "y_size": 10,
        "obs_range": 3,
        "max_cycles": 40,

        "n_evaders": 2,
        "n_pursuers": 5,

        "catch_reward": 0.5,
        "urgency_reward": -0.05,
        "n_catch": 1,
        "tag_reward": 0,
    }

    # switch env
    print(f"env: {args.env}")
    if args.env is None:
        from pursuit_msg.pursuit import my_parallel_env as my_env
    elif args.env == "msg":
        from pursuit_msg.pursuit import my_parallel_env_message as my_env
    elif args.env == "grid-loc":
        from pursuit_msg.pursuit import my_parallel_env_grid_loc as my_env
    else:
        raise NotImplementedError(f"env '{args.env}' is not implemented")

    # seed
    if args.seed is None:
        args.seed = int(np.random.rand() * 10000)
        print(f"Seed is not given in arguments")
    print(f"Seed is {args.seed}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = my_env(**task_parameter)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    # args.action_shape = env.action_space.shape or env.action_space.n
    args.state_shape = args.state_shape[1:]
    args.action_shape = 5
    if args.reward_threshold is None:
        default_reward_threshold = {"pursuit_v4": 195}
        args.reward_threshold = default_reward_threshold.get(
            args.task  # , env.spec.reward_threshold
        )

    # model
    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    if torch.cuda.is_available():
        actor = DataParallelNet(
            Actor(net, args.action_shape, device=None).to(args.device)
        )
        critic = DataParallelNet(Critic(net, device=None).to(args.device))
    else:
        actor = Actor(net, args.action_shape, device=args.device).to(args.device)
        critic = Critic(net, device=args.device).to(args.device)
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
    dist = torch.distributions.Categorical
    policy = myPPOPolicy(
        num_agents=task_parameter["n_pursuers"],
        state_shape=args.state_shape,
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        reward_normalization=args.rew_norm,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        action_space=env.action_space,
        deterministic_eval=True,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
    )

    if __name__ == "__main__":
        checkpoint = torch.load(args.path, map_location=args.device)
        model = {k.replace(".net.module", ""): v for k, v in checkpoint.items()}
        policy.load_state_dict(model)

        envs = DummyVectorEnv(
            [
                lambda: my_env(render_mode="human", **task_parameter),
            ]
        )
        envs.seed(args.seed)

        policy.eval()
        collector = Collector(policy, envs)
        result = collector.collect(n_episode=10, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    test_ppo(get_args())
