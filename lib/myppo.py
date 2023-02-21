from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
from torch import nn

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as, to_torch
from tianshou.policy import A2CPolicy, PPOPolicy
from tianshou.utils.net.common import ActorCritic


class myPPOPolicy(PPOPolicy):
    def __init__(
        self,
        num_agents: int = None,
        *args,
        **kwargs: Any,
    ) -> None:
        assert num_agents is not None
        super().__init__(*args, **kwargs)
        self.num_actions = 5
        self.num_agents = num_agents
        self.bases = self.num_actions ** (num_agents - 1)
        self.bases_arr = self.num_actions ** np.arange(num_agents - 1, -1, -1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
                dist = self(minibatch).dist
                if self._norm_adv:
                    mean, std = minibatch.adv.mean(), minibatch.adv.std()
                    minibatch.adv = (minibatch.adv - mean) / (
                        std + self._eps
                    )  # per-batch norm

                ratio = (
                    (
                        dist.log_prob(torch.unsqueeze(minibatch.act, 1))
                        - minibatch.logp_old  # modified
                    )
                    .exp()
                    .float()
                )
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
                ent_loss = dist.entropy().mean()
                loss = (
                    clip_loss + self._weight_vf * vf_loss - self._weight_ent * ent_loss
                )
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
        num_actions = 5

        # model input: 4x4x3 output: 5
        # old method input: 5x7x4x4x3 (train_num x n_pursuer x **obs) output logit: 5x5^7 (train_num x 5 action ^ n_pursuer); act: 5 (train_num)
        # new method input: 5x7x4x4x3 (train_num x n_pursuer x **obs) output logit: 5x7x5 (train_num x n_pursuer x 5 actions)); act: 5 (train_num)
        act_ = np.zeros(training_num, dtype=int)
        bases = self.bases

        # set logit for each agent
        logits = torch.empty(
            [num_agents, training_num, num_actions], device=self.device
        )
        for i in range(num_agents):
            logits[i], hidden = self.actor(
                batch.obs[:, i], state=state, info=batch.info
            )
        logits = logits.transpose(1, 0)

        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            if self.action_type == "discrete":
                act = logits.argmax(-1)
            elif self.action_type == "continuous":
                act = logits[0]
        else:
            act = dist.sample()

        # encode action from [num_agent] to int]
        act = to_numpy(act)
        for i in range(num_agents):
            act_ = act_ + bases * act[:, i]
            bases = bases // num_actions
        return Batch(logits=logits, act=act_, state=None, dist=dist)

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

        num_agents = batch.obs.shape[1]
        num_actions = 5

        def action(act: np.ndarray) -> np.ndarray:
            # convert int to n-base representation
            converted_act = []
            for b in self.bases_arr:
                converted_act.append(act // b)
                act = act % b
            return np.array(converted_act, dtype=int).transpose()

        buffer_act = action(buffer.act)
        batch_act = action(batch.act)

        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        for i in range(num_agents):
            _buffer_batch = Batch(
                obs=np.expand_dims(buffer.obs[:, i], axis=1),
                rew=buffer.rew[:, i],  # np.expand_dims(buffer.rew[:, i], axis=1),
                info=buffer.info,
                policy=buffer.policy,
                terminated=buffer.terminated,
                truncated=buffer.truncated,
                obs_next=np.expand_dims(buffer.obs_next[:, i], axis=1),
                act=buffer_act[:, i],
            )
            _buffer = ReplayBuffer(
                size=buffer.maxsize,
                stack_num=buffer.options["stack_num"],
                ignore_obs_next=buffer.options["ignore_obs_next"],
                save_only_last_obs=buffer.options["save_only_last_obs"],
                sample_avail=buffer.options["sample_avail"],
            )
            for b in _buffer_batch:
                _buffer.add(b)

            _batch = Batch(
                obs=np.expand_dims(batch.obs[:, i], axis=1),
                rew=batch.rew[:, i],  # np.expand_dims(batch.rew[:, i], axis=1),
                info=batch.info,
                policy=batch.policy,
                terminated=batch.terminated,
                truncated=batch.truncated,
                done=batch.done,
                obs_next=np.expand_dims(batch.obs_next[:, i], axis=1),
                act=batch_act[:, i],
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
