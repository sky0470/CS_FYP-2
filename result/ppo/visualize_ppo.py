"""
code that train pursuit with ppo
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
import json
import matplotlib.pyplot as plt

from pursuit_msg.policy.myppo import myPPOPolicy
from pursuit_msg.net.msgnet import MsgNet
from pursuit_msg.net.noisy_actor import NoisyActor
from pursuit_msg.my_collector import MyCollector

def plot_graph(data, path, has_noise):
    n_episode = data["n/ep"]
    rwds = np.array(data["rews"])
    max_cycles = data["len"]
    noise_mu = np.array(data["noises_mu"])
    noise_sig = np.array(data["noises_sig"])
    rwds_detail = np.array(data["rews_detail"])
    num_agents = rwds.shape[1]


    tmp = np.mean(np.sort(rwds, 1), 0)
    mean = [tmp.mean()] * 5

    plt.plot(tmp, 'o')
    plt.plot(mean, '-')
    img_path = os.path.join(path, "rewards-summary.png")
    plt.savefig(img_path, dpi=200)
    plt.close()


    if has_noise:
        if noise_mu.ndim != 4:
            exit()
        num_noise_type = noise_mu.shape[-1]

    x = np.arange(max_cycles)

    for ep in range(n_episode):
        def plot_reward(ep):
            fig = plt.figure(figsize=(8, 6))
            fig.suptitle(f"episode {ep + 1}")
            ax = fig.add_subplot(1, 1, 1, title=f"reward", xlabel="step", xlim=(0, max_cycles))
            for i in range(num_agents):
                ax.plot(x, np.add.accumulate(rwds_detail[:, ep, i], axis=0), label=f"P{i}", marker='.')

            ax.legend(loc=1, fontsize=8)
            img_path = os.path.join(path, f"render-{ep + 1:02d}", "rewards.png")
            fig.savefig(img_path, dpi=200)
            print(f"reward graph generated to {img_path}")
            return fig
        
        def plot_noise(ep):
            fig = plt.figure(figsize=(8, 6))
            fig.suptitle(f"episode {ep + 1}")
            for t in range(num_noise_type):
                # horizontal align
                # ax_mu = fig.add_subplot(num_noise_type, 2, t + 1, title=f"mu-{t}", xlabel="step")
                # ax_sig = fig.add_subplot(num_noise_type, 2, t + 2, title=f"sig-{t}", xlabel="step")

                # vertical align
                ax_mu = fig.add_subplot(2, num_noise_type, t + 1, title=f"mu-{t}", xlabel="step", xlim=(0, max_cycles))
                ax_sig = fig.add_subplot(2, num_noise_type, t + 1 + num_noise_type, title=f"sig-{t}", xlabel="step", xlim=(0, max_cycles))
                for i in range(num_agents):
                    ax_mu.plot(x, noise_mu[:, ep, i, t], label=f"P{i}", marker='.')
                    ax_mu.plot(x, np.zeros_like(x), color="black")
                    ax_sig.plot(x, noise_sig[:, ep, i, t], label=f"P{i}", marker='.')
                    ax_sig.plot(x, np.zeros_like(x), color="black")

                ax_mu.legend(loc=1, fontsize=8)
                ax_sig.legend(loc=1, fontsize=8)
            
            fig.tight_layout()
            img_path = os.path.join(path, f"render-{ep + 1:02d}", "noise.png")
            fig.savefig(img_path, dpi=200)
            print(f"noise graph generated to {img_path}")
            return fig

        fig_rwd = plot_reward(ep)
        if has_noise:
            fig_noise = plot_noise(ep)

def get_args():
    parser = argparse.ArgumentParser()
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

    # task param
    parser.add_argument('--catch-reward-ratio', type=float, nargs="+", default=None)
    parser.add_argument('--noise-shape', type=int, nargs=2, default=(-1, 1))
    parser.add_argument('--apply-noise', type=int, default=1)

    # switch env
    parser.add_argument('--env', type=str, default=None)

    parser.add_argument('--resume-path', type=str, default=None)
    # visualize special
    parser.add_argument("--n_episode", type=int, default=10)
    args = parser.parse_args()

    # filter overrode args
    args_overrode = {
        opt.dest: getattr(args, opt.dest)
        for opt in parser._option_string_actions.values()
        if hasattr(args, opt.dest) and opt.default != getattr(args, opt.dest)
    }
    return args, args_overrode


def test_ppo(args=get_args()[0], args_overrode=dict()):
    task_parameter = dict(
        shared_reward=False,
        surround=False,
        freeze_evaders=True,

        x_size=10,
        y_size=10,
        obs_range=3,
        max_cycles=40,

        n_evaders=2,
        n_pursuers=5,

        catch_reward=0.5,
        urgency_reward=-0.05,
        n_catch=1,
        tag_reward=0,
        catch_reward_ratio=None, # redefine later

        # noise
        has_noise=False,
        noise_shape=None, # redefine later
        apply_noise=args.apply_noise,
        # note: only (2, 1), (-1, 1) are implemented, if has_noise is true
    )

    if args.seed is None:
        args.seed = int(np.random.rand() * 100000)
        args_overrode["seed"] = args.seed
        print(f"Seed is not given in arguments")
    print(f"Seed is {args.seed}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.resume_path:
        # load from existing checkpoint
        print(f"Loading agent under {args.resume_path}")
        if os.path.exists(args.resume_path):
            checkpoint = torch.load(args.resume_path, map_location=args.device)
            """
            if "args" in checkpoint:
                args = checkpoint["args"]
                args.device = "cuda" if torch.cuda.is_available() else "cpu"
                args.n_episode = 10
                for k, v in args_overrode.items():
                    setattr(args, k, v)
            if "task_parameter" in checkpoint:
                task_parameter = checkpoint["task_parameter"]
                task_parameter["apply_noise"] = args.apply_noise
            """
        else:
            print("Fail to restore policy and optim.")
            exit()

    # switch env
    print(f"env: {args.env}")
    if args.env is None:
        from pursuit_msg.pursuit import my_parallel_env as my_env
    elif args.env == "msg":
        from pursuit_msg.pursuit import my_parallel_env_message as my_env
    elif args.env == "grid-loc":
        from pursuit_msg.pursuit import my_parallel_env_grid_loc as my_env
    elif args.env == "full":
        from pursuit_msg.pursuit import my_parallel_env_full as my_env
    elif args.env == "ic3":
        from pursuit_msg.pursuit import my_parallel_env_ic3 as my_env
    elif args.env == 'noise':
        from pursuit_msg.pursuit import my_parallel_env_noise as my_env
        task_parameter["has_noise"] = True
    else:
        raise NotImplementedError(f"env '{args.env}' is not implemented")

    # redefine task_param 
    task_parameter["catch_reward_ratio"] = args.catch_reward_ratio or [num for num in range(task_parameter["n_pursuers"] + 1)]
    task_parameter["noise_shape"] = tuple(args.noise_shape) if task_parameter["has_noise"] else None
    implemented_noises = [(-1, 1), (2, 1), (2, 9), (-1, 9), None]
    if task_parameter["noise_shape"] not in implemented_noises:
        raise NotImplementedError(f"Please use noise_shape={implemented_noises}")

    env = my_env(**task_parameter)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    # args.action_shape = env.action_space.shape or env.action_space.n
    args.state_shape = args.state_shape[1:]
    args.action_shape = 5
    if args.reward_threshold is None:
        default_reward_threshold = {"pursuit_v4": 1000}
        args.reward_threshold = default_reward_threshold.get(
            args.task  # , env.spec.reward_threshold
        )

    # model
    net = MsgNet(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    # net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    if task_parameter["has_noise"]:
        actor = NoisyActor(net, args.action_shape, device=args.device, filter_noise=True, noise_shape=task_parameter["noise_shape"]).to(args.device)
    else:
        actor = Actor(net, args.action_shape, device=args.device).to(args.device)
    critic = Critic(net, device=args.device).to(args.device)
    actor_critic = ActorCritic(actor, critic)
    
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
        device=args.device,
        num_actions=args.action_shape,
        noise_shape=task_parameter["noise_shape"],
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

    pprint.pprint(dict(
        args=vars(args),
        args_overrode=args_overrode,
        task_parameter=task_parameter,
    ))

    if __name__ == "__main__":
        checkpoint = torch.load(args.resume_path, map_location=args.device)
        if isinstance(checkpoint, dict):
            policy.load_state_dict(checkpoint["model"])
        else:
            model = {k.replace(".net.module", ""): v for k, v in checkpoint.items()}
            policy.load_state_dict(model)

        render_vdo_path = args.logdir
        if render_vdo_path:
            # find first unused number
            cnt = 0
            while os.path.exists(render_vdo_path):
                cnt += 1
                render_vdo_path = f"{args.logdir}-{cnt}"
            os.makedirs(render_vdo_path)

        envs = DummyVectorEnv(
            [
                lambda: my_env(render_mode="human", 
                               render_vdo_path=render_vdo_path, 
                               **task_parameter),
            ]
        )
        envs.seed(args.seed)

        policy.eval()
        collector = MyCollector(policy, envs,
                                num_actions=args.action_shape,
                                has_noise=task_parameter["has_noise"],
                                noise_shape=task_parameter["noise_shape"],
                                visualize=True
                                )
        result = collector.collect(n_episode=args.n_episode, render=args.render)
        pprint.pprint(result)

        result_json = {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items() 
        }
        with open(os.path.join(render_vdo_path, "summary.json"), "w") as f:
            json.dump(result_json, f, indent=4)

        print(f"summary generated to {render_vdo_path}")

        # plot noise
        plot_graph(result, render_vdo_path, task_parameter["has_noise"])
            

        # rews, lens = result["rews"], result["lens"]
        # print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    args, args_overrode = get_args()
    test_ppo(args, args_overrode)
