from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tianshou.data import Batch, to_torch
from tianshou.utils.net.common import MLP
from tianshou.utils.net.discrete import Actor

class NoisyActor(Actor):
    """Simple actor network. (with noise)

    Will create an actor operated in discrete action space with structure of
    preprocess_net ---> action_shape.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param bool softmax_output: whether to apply a softmax layer over the last
        layer's output.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        softmax_output: bool = True,
        preprocess_net_output_dim: Optional[int] = None,
        device: Union[str, int, torch.device] = "cpu",
        filter_noise: bool = False,
        noise_shape: Sequence[int] = 0,
    ) -> None:
        super().__init__(preprocess_net, action_shape + int(abs(noise_shape[0])) * 2, hidden_sizes, softmax_output, preprocess_net_output_dim, device)
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = int(abs(np.prod(action_shape))) + int(abs(noise_shape[0])) * 2
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last_act = MLP(
            input_dim,  # type: ignore
            int(abs(np.prod(action_shape))),
            hidden_sizes,
            device=self.device
        )
        self.last_noise = MLP(
            input_dim,  # type: ignore
            int(abs(noise_shape[0])) * 2,
            hidden_sizes,
            device=self.device
        )

        self.softmax_output = softmax_output
        self.filter_noise = filter_noise
        self.action_shape = action_shape
        self.num_norm = 0 if noise_shape is None else abs(noise_shape[0])

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        logits, hidden = self.preprocess(obs, state)
        logits_act = self.last_act(logits)
        logits_noise = self.last_noise(logits)
        logits = torch.concat([logits_act, logits_noise], -1)

        if self.filter_noise:
            # split into act and noise
            logits_act = logits[:, :self.action_shape]
            logits_noise = logits[:, self.action_shape:]
            logits_noise[:, self.num_norm:] = torch.sigmoid((logits_noise[:, self.num_norm:]) / 5) * 5

            if self.softmax_output:
                # only apply to action
                logits_act = F.softmax(logits_act, dim=-1)

            logits = torch.cat([logits_act, logits_noise], dim=1)
        else:
            if self.softmax_output:
                # only apply to action
                logits = F.softmax(logits, dim=-1)
        return logits, hidden
