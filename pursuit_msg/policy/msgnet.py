from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    no_type_check,
)

import numpy as np
import torch
from torch import nn

from tianshou.data.batch import Batch
from tianshou.utils.net.common import MLP


ModuleType = Type[nn.Module]

class MsgNet(nn.Module):

    def __init__(
        self,
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]] = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = nn.ReLU,
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        concat: bool = False,
        num_atoms: int = 1,
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
    ) -> None:
        super().__init__()
        self.device = device
        self.softmax = softmax
        self.num_atoms = num_atoms
        self.state_shape = state_shape
        input_dim = int(np.prod(state_shape[1:]))
        pre_input_dim = int(np.prod(state_shape[1:]))
        pre_output_dim = 10
        input_dim += pre_output_dim * (state_shape[0]-1)
        action_dim = int(np.prod(action_shape)) * num_atoms
        output_dim = action_dim if not concat else 0

        self.pre = MLP(
            pre_input_dim, pre_output_dim, hidden_sizes, norm_layer, activation, device,
            linear_layer
        )
        self.model = MLP(
            input_dim, output_dim, hidden_sizes, norm_layer, activation, device,
            linear_layer
        )
        self.output_dim = self.model.output_dim

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits."""
        if self.device is not None:
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
            obs = obs.squeeze()
        if obs.ndim == 4: # dim of obs: (n_pursuers, obs_range, obs_range, channels)
            pre = torch.stack([self.pre(obs[i].unsqueeze(0)) for i in range(1, obs.shape[0])],1)
            obs_ = torch.cat((obs[0].flatten(), pre.flatten()))
            obs_ = obs_.unsqueeze(0)
        elif obs.ndim == 5: # dim of obs: (train_num, n_pursuers, obs_range, obs_range, channels)
            pre = torch.stack([self.pre(obs[:, i]) for i in range(1, obs.shape[1])], 1)
            obs_ = torch.cat((obs[:, 0].flatten(1), pre.flatten(1)), 1)
        else:
            pre = torch.stack([self.pre(obs[:, :, i]) for i in range(1, obs.shape[2])], 2)
            obs_ = torch.cat((obs[:, :, 0].flatten(2), pre.flatten(2)), 2)
        logits = self.model(obs_)
        bsz = logits.shape[0]
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state

