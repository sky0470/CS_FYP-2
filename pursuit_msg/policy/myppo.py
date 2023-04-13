from typing import Any, Dict, List, Optional, Type, Union, Sequence

import numpy as np
import torch
from torch import nn

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as, to_torch
from tianshou.policy import A2CPolicy, PPOPolicy
from tianshou.utils.net.common import ActorCritic

from torch.distributions import Independent, Normal

class myPPOPolicy(PPOPolicy):
    def __init__(
        self,
        num_agents: int = None,
        state_shape = None,
        device = None,
        num_actions: int = None,
        noise_shape: Sequence[int] = None,
        *args,
        **kwargs: Any,
    ) -> None:
        assert num_agents is not None
        super().__init__(*args, **kwargs)
        self.num_actions = num_actions
        self.noise_shape = noise_shape
        self.num_norm = 0 if noise_shape is None else abs(noise_shape[0])
        self.num_noise = 0 if noise_shape is None else abs(np.prod(noise_shape))

        self.state_shape = state_shape
        self.num_agents = num_agents
        self.bases = self.num_actions ** (num_agents - 1)
        self.bases_arr = self.num_actions ** np.arange(num_agents - 1, -1, -1)
        self.device = device # "cuda" if torch.cuda.is_available() else "cpu"

    def dist_2_fn(self, *logits):
        normal = Normal(*logits)
        return normal

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        if self._recompute_adv:
            # buffer input `buffer` and `indices` to be used in `learn()`.
            self._buffer, self._indices = buffer, indices
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        if not self.num_noise:
            with torch.no_grad():
                b = self(batch)
                batch.logp_old = b.dist.log_prob(torch.unsqueeze(batch.act, 1))
        else:
            batch.act_noise = to_torch_as(batch.act_noise, batch.v_s)
            with torch.no_grad():
                b = self(batch)
                if self.noise_shape[1] == 9:
                    b_act_noise = batch.act_noise.reshape(batch.act_noise.shape[0], self.num_norm, -1).permute(2, 0, 1) #  noise_per_norm, btz, num_norm
                    d_2_log_prob = b.dist_2.log_prob(b_act_noise).sum(0)
                else:
                    d_2_log_prob = b.dist_2.log_prob(batch.act_noise)
                d_2_log_prob = d_2_log_prob.sum(-1, keepdim=True)
                batch.logp_old = b.dist.log_prob(torch.unsqueeze(batch.act, 1)) \
                                 + d_2_log_prob
        return batch

    # modified
    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        for step in range(repeat):
            if self._recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)
            for minibatch in batch.split(batch_size, merge_last=True):
                # calculate loss for actor
                b = self(minibatch)
                dist = b.dist
                dist_2 = b.dist_2
                if self._norm_adv:
                    mean, std = minibatch.adv.mean(), minibatch.adv.std()
                    minibatch.adv = (minibatch.adv - mean) / (
                        std + self._eps
                    )  # per-batch norm
                if not self.num_noise:
                    ratio = dist.log_prob(torch.unsqueeze(minibatch.act, 1)) - minibatch.logp_old  # modified
                else:
                    if self.noise_shape[1] == 9:
                        b_act_noise = minibatch.act_noise.reshape(minibatch.act_noise.shape[0], self.num_norm, -1).permute(2, 0, 1) # noise_per_norm, btz, num_norm, 
                        d_2_log_prob = b.dist_2.log_prob(b_act_noise).sum(0)
                    else:
                        d_2_log_prob = b.dist_2.log_prob(minibatch.act_noise)
                    d_2_log_prob = d_2_log_prob.sum(-1, keepdim=True)
                    ratio = dist.log_prob(torch.unsqueeze(minibatch.act, 1)) \
                            + d_2_log_prob \
                            - minibatch.logp_old  # modified
                ratio = ratio.exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                surr1 = ratio * minibatch.adv
                surr2 = (
                    ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip)
                    * minibatch.adv
                )
                if self._dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self._dual_clip * minibatch.adv)
                    clip_loss = -torch.where(minibatch.adv < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                # calculate loss for critic
                value = self.critic(minibatch.obs).flatten()
                if self._value_clip:
                    v_clip = minibatch.v_s + (value - minibatch.v_s).clamp(
                        -self._eps_clip, self._eps_clip
                    )
                    vf1 = (minibatch.returns - value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (minibatch.returns - value).pow(2).mean()
                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean() + dist_2.entropy().mean()
                loss = (
                    clip_loss + self._weight_vf * vf_loss - self._weight_ent * ent_loss
                )
                print({vf_loss: vf_loss, clip_loss: clip_loss})
                self.optim.zero_grad()
                loss.backward()
                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(), max_norm=self._grad_norm
                    )
                self.optim.step()
                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())

        return (
            losses,
            clip_losses,
            vf_losses,
            ent_losses,
        )
        # return {
        # "loss": losses,
        # "loss/clip": clip_losses,
        # "loss/vf": vf_losses,
        # "loss/ent": ent_losses,
        # }

    # forward from pg.py
    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 4 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """

        training_num = batch.obs.shape[0]
        num_agents = batch.obs.shape[1]
        num_actions = self.num_actions

        # model input: 4x4x3 output: 5
        # old method input: 5x7x4x4x3 (train_num x n_pursuer x **obs) output logit: 5x5^7 (train_num x 5 action ^ n_pursuer); act: 5 (train_num)
        # new method input: 5x7x4x4x3 (train_num x n_pursuer x **obs) output logit: 5x7x5 (train_num x n_pursuer x 5 actions)); act: 5 (train_num)
        act_ = np.zeros(training_num, dtype=int)
        bases = self.bases

        # set logit for each agent
        logits = torch.empty(
            [num_agents, training_num, num_actions + self.num_norm * 2], device=self.device
        )
        state_ret = None
        for i in range(num_agents):
            if state is None:
                logits[i], _state = self.actor(
                    batch.obs[:, i], state=state, info=batch.info
                )
            else:
                hidden = torch.unsqueeze(state["hidden"][:, i], axis=1)
                cell = torch.unsqueeze(state["cell"][:, i], axis=1)
                logits[i], _state = self.actor(
                    batch.obs[:, i],
                    state={"hidden": hidden, "cell": cell},
                    info=batch.info,
                )
            if _state is not None:
                if state_ret is None:
                    state_ret = {
                        "hidden": torch.empty(
                            [training_num, num_agents, 128], device=self.device
                        ),
                        "cell": torch.empty(
                            [training_num, num_agents, 128], device=self.device
                        ),
                    }
                state_ret["hidden"][:, (i,)] = _state["hidden"]
                state_ret["cell"][:, (i,)] = _state["cell"]
        logits = logits.transpose(1, 0)

        logits_act = logits[:, :, :num_actions]
        if isinstance(logits_act, tuple):
            dist = self.dist_fn(*logits_act)
        else:
            dist = self.dist_fn(logits_act)
        if self._deterministic_eval and not self.training:
            act = logits_act.argmax(-1)
        else:
            act = dist.sample()

        # encode action from [num_agent] to int
        act = to_numpy(act)
        for i in range(num_agents):
            act_ = act_ + bases * act[:, i]
            bases = bases // num_actions


        if self.num_noise: # the format of the NN output is: num_action + mu * num_noise+ sig * num_noise
            # pack all mu and sig tgt
            noise_shape = logits.shape[:2] + (self.num_norm,)
            logits_noise_mu = logits[:, :, num_actions:num_actions + self.num_norm].reshape(logits.shape[0], -1)
            # logits_noise_mu = logits_noise_mu.clone().mul(0) # set mu to 0
            logits_noise_sig = logits[:, :, num_actions + self.num_norm:].reshape(logits.shape[0], -1)
            logits_noise = (logits_noise_mu, logits_noise_sig)
            dist_2 = self.dist_2_fn(*logits_noise)
            # if self._deterministic_eval and not self.training:
            #     act_noise = logits_noise_mu.unsqueeze(0).repeat(self.noise_shape[1], *([1]*logits_noise_mu.ndim))
            # else:
            #     act_noise = dist_2.sample(torch.tensor([1]))

            # act_noise = logits_noise_mu.unsqueeze(0).repeat(self.noise_shape[1], *([1]*logits_noise_mu.ndim))
            act_noise = dist_2.sample(torch.tensor([self.noise_shape[1]]))
            act_noise = to_numpy(act_noise) # act_noise shape = (noise_per_norm, btz, num_agent*num_norm)
            act_noise = act_noise.transpose((1,2,0)) # before reshape : (btz, num_agent*num_norm, noise_per_norm)
            act_noise = act_noise.reshape(act_noise.shape[0], -1)

            obs = batch.obs[:, :, 0]
            # if self.noise_shape == (-1, 1):
            #     obs += act_noise.reshape(noise_shape)[:, :, None, None]
            # else:
            #     obs[:, :, :, :, :2] += act_noise.reshape(noise_shape)[:,:, None, None]
            obs = obs.reshape(obs.shape[0], -1)

            act_ = np.concatenate((act_[:, None], act_noise, obs), 1)
        else: # no noise
            act_ = np.concatenate((act_[:, None], batch.obs[:, :, 0].reshape(batch.obs.shape[0], -1) ),1)
            dist_2 = None

        # act_ shape = btz, 6 -> 6 = 1 + 5, combined act + noise for 0..4
        # logits shape = btz, agent, 7 -> 7 = 5 + 2, logit for act + (mean, sig) for noise
        return Batch(logits=logits, act=act_, state=state_ret, dist=dist, dist_2=dist_2)

    # from base.py
    def update(
        self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any
    ) -> Dict[str, Any]:
        """Update the policy network and replay buffer.

        It includes 3 function steps: process_fn, learn, and post_process_fn. In
        addition, this function will change the value of ``self.updating``: it will be
        False before this function and will be True when executing :meth:`update`.
        Please refer to :ref:`policy_state` for more detailed explanation.

        :param int sample_size: 0 means it will extract all the data from the buffer,
            otherwise it will sample a batch with given sample_size.
        :param ReplayBuffer buffer: the corresponding replay buffer.

        :return: A dict, including the data needed to be logged (e.g., loss) from
            ``policy.learn()``.
        """
        if buffer is None:
            return {}
        batch, indices = buffer.sample(sample_size)
        self.updating = True

        num_agents = buffer.obs.shape[1]

        def action(act: np.ndarray) -> np.ndarray:
            # decode int to n-base representation
            converted_act = []
            for b in self.bases_arr:
                converted_act.append(act // b)
                act = act % b
            return np.array(converted_act, dtype=int).transpose()

        buffer_act = action(buffer.act[:, 0])
        batch_act = action(batch.act[:, 0])
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        for i in range(num_agents):
            _buffer_batch = Batch(
                obs=np.expand_dims(buffer.obs[:, i], axis=1),
                rew=buffer.rew[:, i],
                info=buffer.info,
                policy=buffer.policy,
                terminated=buffer.terminated,
                truncated=buffer.truncated,
                obs_next=np.expand_dims(buffer.obs_next[:, i], axis=1),
                act = buffer_act[:, i],
                # act_noise = buffer.act[:, i+1],
                act_noise = buffer_act[:,i] if not self.num_noise else buffer.act[:, 1 + i * self.num_noise:1 + (i + 1) * self.num_noise],
                #act_noise = np.expand_dims(buffer.act[:, i+1], axis=1)
            )
            print(_buffer_batch.act_noise)
            _buffer = ReplayBuffer(
                size=buffer.maxsize,
                stack_num=buffer.options["stack_num"],
                ignore_obs_next=buffer.options["ignore_obs_next"],
                save_only_last_obs=buffer.options["save_only_last_obs"],
                sample_avail=buffer.options["sample_avail"],
            )
            print(_buffer_batch)
            for b in _buffer_batch:
                _buffer.add(b)

            _batch = Batch(
                obs=np.expand_dims(
                    batch.obs.swapaxes(1, 2)[:, i]
                    if batch.obs.ndim == 3 + len(self.state_shape)
                    else batch.obs[:, i],
                    axis=1,
                ),
                rew=batch.rew[:, i],
                info=batch.info,
                policy=batch.policy,
                terminated=batch.terminated,
                truncated=batch.truncated,
                done=batch.done,
                obs_next=np.expand_dims(
                    batch.obs_next.swapaxes(1, 2)[:, i]
                    if batch.obs_next.ndim == 3 + len(self.state_shape)
                    else batch.obs_next[:, i],
                    axis=1,
                ),
                act = batch_act[:, i],
                act_noise = None if not self.num_noise else batch.act[:, 1 + i * self.num_noise:1 + (i + 1) * self.num_noise],
            )
            _batch = self.process_fn(_batch, _buffer, indices)
            (losses_, clip_losses_, vf_losses_, ent_losses_) = self.learn(
                _batch, **kwargs
            )
            losses += losses_
            clip_losses += clip_losses_
            vf_losses += vf_losses_
            ent_losses += ent_losses_
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        self.post_process_fn(batch, buffer, indices)
        self.updating = False
        return {
            "loss": losses,
            "loss/clip": clip_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }
