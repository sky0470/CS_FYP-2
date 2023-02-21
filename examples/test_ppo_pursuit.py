"""
code that train pursuit with ppo
tested with myPursuit and myPursuit_message for small parameters

not yet test for reproducibility
parameters are set somewhat arbitrarly, should find a paper someday
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
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.policy import PPOPolicy

import sys
import datetime

sys.path.append("..")
sys.path.append("../lib")
sys.path.append("../lib/policy_lib")
# from lib.myPursuit_gym import my_parallel_env
from lib.myPursuit_gym import my_parallel_env
from lib.myppo import myPPOPolicy
from lib.myPursuit_gym_message import my_parallel_env_message


def get_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--repeat-per-collect", type=int, default=4)
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


def test_ppo(args=get_args()):
    task_parameter = {
        "shared_reward": False,
        "surround": False,
        "freeze_evaders": True,
        "max_cycles": 50,
        "n_evaders": 8,
        "n_pursuers": 8,
    }
    args.render = 0.05
    args.step_per_epoch = 1000
    if args.seed is None:
        args.seed = int(np.random.rand() * 100000)

    train_very_fast = True
    if train_very_fast:
        # Set the following parameters so that the program run very fast but train nothing
        task_parameter["max_cycles"] = 50  # 500
        task_parameter["x_size"] = 8  # 16
        task_parameter["y_size"] = 8  # 16
        task_parameter["obs_range"] = 5  # 7, should be odd
        args.training_num = 5  # 10
        args.test_num = 50  # 100
        args.hidden_sizes = [64, 64]  # [128, 128, 128, 128]
        args.epoch = 2  # 10  # 20
        args.step_per_epoch = 10  # 500  # 10000
        args.render = 0.05

    env = my_parallel_env_message

    env_ = env(**task_parameter)
    args.state_shape = env_.observation_space.shape or env_.observation_space.n
    args.action_shape = env_.action_space.shape or env_.action_space.n
    args.state_shape = args.state_shape[1:]
    args.action_shape = 5
    if args.reward_threshold is None:
        default_reward_threshold = {"pursuit_v4": 1000}
        args.reward_threshold = default_reward_threshold.get(
            args.task  # , env.spec.reward_threshold
        )
    # train_envs = gym.make(args.task)
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = DummyVectorEnv(
        [lambda: env(**task_parameter) for _ in range(args.training_num)]
    )
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv(
        [lambda: env(**task_parameter) for _ in range(args.test_num)]
    )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # Q_param = V_param = {"hidden_sizes": [128]}
    # model
    net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        # dueling=(Q_param, V_param),
    ).to(args.device)
    actor = Actor(net, args.action_shape, device=args.device)
    critic = Critic(net, device=args.device)
    optim = torch.optim.Adam(
        ActorCritic(actor, critic).parameters(), lr=args.lr
    )
    def dist(p):
        return torch.distributions.Categorical(logits=p)

    policy = myPPOPolicy(
        task_parameter["n_pursuers"],
        5,
        actor,
        critic,
        optim,
        dist,
    )

    # # buffer
    # if args.prioritized_replay:
    #     buf = PrioritizedVectorReplayBuffer(
    #         args.buffer_size,
    #         buffer_num=len(train_envs),
    #         alpha=args.alpha,
    #         beta=args.beta,
    #     )
    # else:
    buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))
    # collector
    train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num)
    train_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # log
    log_path = os.path.join(args.logdir, args.task, "ppo", train_datetime)
    print(train_datetime)
    print(str(env))
    print(str(args))
    print(str(task_parameter))
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    writer.add_text("env_para", str(task_parameter))
    writer.add_text("env_name", str(task_parameter))
    writer.add_text("date_time", train_datetime)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold

    # trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num,
        args.batch_size,
        step_per_collect=args.step_per_collect,
        update_per_step=args.update_per_step,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    )
    # assert stop_fn(result['best_reward'])

    if __name__ == "__main__":
        pprint.pprint(result)
        # Let's watch its performance!

        env = DummyVectorEnv(
            [
                lambda: env(**task_parameter)
            ]
        )

        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=None)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")

if __name__ == "__main__":
    test_ppo(get_args())
