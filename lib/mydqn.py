from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.policy import BasePolicy, DQNPolicy


class myDQNPolicy(DQNPolicy):
    """Implementation of Deep Q Network. arXiv:1312.5602.

    Implementation of Double Q-Learning. arXiv:1509.06461.

    Implementation of Dueling DQN. arXiv:1511.06581 (the dueling DQN is
    implemented in the network side, not here).

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network). Default to 0.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param bool is_double: use double dqn. Default to True.
    :param bool clip_loss_grad: clip the gradient of the loss in accordance
        with nature14236; this amounts to using the Huber loss instead of
        the MSE loss. Default to False.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self, *args, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def train(self, mode: bool = True) -> "DQNPolicy":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.model.train(mode)
        return self

    def sync_weight(self) -> None:
        """Synchronize the weight for the target network."""
        self.model_old.load_state_dict(self.model.state_dict())

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        result = self(batch, input="obs_next")
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            target_q = self(batch, model="model_old", input="obs_next").logits
        else:
            target_q = result.logits
        target_q = torch.squeeze(target_q)
        if self._is_double:
            return target_q[np.arange(len(result.act)), result.act]
        else:  # Nature DQN, over estimate
            return target_q.max(dim=1)[0]

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Compute the n-step return for Q-learning targets.

        More details can be found at
        :meth:`~tianshou.policy.BasePolicy.compute_nstep_return`.
        """
        batch = self.compute_nstep_return(
            batch,
            buffer,
            indices,
            self._target_q,
            self._gamma,
            self._n_step,
            self._rew_norm,
        )
        return batch

    def compute_q_value(
        self, logits: torch.Tensor, mask: Optional[np.ndarray]
    ) -> torch.Tensor:
        """Compute the q value based on the network's raw output and action mask."""
        if mask is not None:
            # the masked q value should be smaller than logits.min()
            min_value = logits.min() - logits.max() - 1.0
            logits = logits + to_torch_as(1 - mask, logits) * min_value
        return logits

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        If you need to mask the action, please add a "mask" into batch.obs, for
        example, if we have an environment that has "0/1/2" three actions:
        ::

            batch == Batch(
                obs=Batch(
                    obs="original obs, with batch_size=1 for demonstration",
                    mask=np.array([[False, True, False]]),
                    # action 1 is available
                    # action 0 and 2 are unavailable
                ),
                ...
            )

        :return: A :class:`~tianshou.data.Batch` which has 3 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = batch[input]
        obs_next = obs.obs if hasattr(obs, "obs") else obs

        num_agents = obs.shape[1]

        # model in 4x4x3 outupt 5
        # old method in 5x7x4x4x3 output logit 5x78125  act 5
        # new method in 5x7x4x4x3 output logit 5x7x5  act 5
        act_ = np.zeros(obs.shape[0], dtype=int)
        bases = 5 ** (num_agents - 1)
        logits_ = torch.empty([num_agents, obs.shape[0], 5])
        # logits = np.empty([obs.shape[0], num_agents, 5])
        for i in range(num_agents):
            logits, hidden = model(obs_next[:, i], state=state, info=batch.info)
            q = self.compute_q_value(logits, getattr(obs[:, i], "mask", None))
            logits_[i] = logits
            if not hasattr(self, "max_action_num"):
                self.max_action_num = q.shape[1]
            act = to_numpy(q.max(dim=1)[1])
            act_ = act_ + bases * act
            bases = bases // 5
        logits_ = logits_.transpose(1, 0)

        # logit 5x78125, act 5,, hidden = None
        return Batch(logits=logits_, act=act_, state=None)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        q = self(batch).logits
        q = torch.squeeze(q)
        q = q[np.arange(len(q)), batch.act]
        returns = to_torch_as(batch.returns.flatten(), q)
        td_error = returns - q

        if self._clip_loss_grad:
            y = q.reshape(-1, 1)
            t = returns.reshape(-1, 1)
            loss = torch.nn.functional.huber_loss(y, t, reduction="mean")
        else:
            loss = (td_error.pow(2) * weight).mean()

        batch.weight = td_error  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1
        return loss.item()
        # return {"loss": loss.item()}

    def exploration_noise(
        self,
        act: Union[np.ndarray, Batch],
        batch: Batch,
    ) -> Union[np.ndarray, Batch]:
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            q = np.random.rand(bsz, self.max_action_num)  # [0, 1]
            if hasattr(batch.obs, "mask"):
                q += batch.obs.mask
            rand_act = q.argmax(axis=1)
            act[rand_mask] = rand_act[rand_mask]
        return act

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

        nvec = np.array([5] * num_agents)
        bases = np.ones_like(nvec)
        for i in range(1, len(bases)):
            bases[i] = bases[i - 1] * nvec[-i]
        def action(act: np.ndarray) -> np.ndarray:
            converted_act = []
            for b in np.flip(bases):
                converted_act.append(act // b)
                act = act % b
            return np.array(converted_act, dtype=int).transpose()
        buffer_act = action(buffer.act)
        batch_act = action(batch.act)

        loss = 0
        for i in range(num_agents):
            _buffer_batch = Batch(
                obs=np.expand_dims(buffer.obs[:, i], axis=1),
                rew=np.expand_dims(buffer.rew[:, i], axis=1),
                info=buffer.info,
                policy=buffer.policy,
                terminated=buffer.terminated,
                truncated=buffer.truncated,
                obs_next=np.expand_dims(buffer.obs_next[:, i], axis=1),
                act=buffer_act[:, i]
            )
            _buffer = ReplayBuffer(size = buffer.maxsize, stack_num = buffer.options["stack_num"], ignore_obs_next=buffer.options["ignore_obs_next"], save_only_last_obs=buffer.options["save_only_last_obs"], sample_avail=buffer.options["sample_avail"])
            for b in _buffer_batch:
                _buffer.add(b)

            _batch = Batch(
                obs=np.expand_dims(batch.obs[:, i], axis=1),
                rew=np.expand_dims(batch.rew[:, i], axis=1),
                info=batch.info,
                policy=batch.policy,
                terminated=batch.terminated,
                truncated=batch.truncated,
                done=batch.done,
                obs_next=np.expand_dims(batch.obs_next[:, i], axis=1),
                act=batch_act[:, i]
            )

            _batch = self.process_fn(_batch, _buffer, indices)
            _result = self.learn(_batch, **kwargs)
            loss = loss + _result
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        self.post_process_fn(batch, buffer, indices)
        self.updating = False
        return {"loss": loss}
