
import logging
import random
import sys
from typing import List, Tuple
import numpy as np
import pickle
import torch
from torch import nn

sys.path.append('.')
from env.Observation import Observation
from env.Actions import Action, ChaodiAction, DeclareAction, DontChaodiAction, DontDeclareAction, FollowAction, LeadAction, PlaceKittyAction
from env.utils import Stage
from env.CardSet import CardSet, MoveType

from env.utils import AbsolutePosition

class StageModule:
    def act(self, obs: Observation, epsilon=None, training=True):
        return NotImplementedError()

    def load_model(self, model: nn.Module):
        raise NotImplementedError()

    def learn_from_samples(self, samples: List[Tuple[Observation, Action, float]]):
        raise NotImplementedError()


class DeepAgent(StageModule):
    """Base class for stage-level RL agents wrapping a single neural network."""

    def __init__(self, name: str, model: nn.Module, batch_size: int, tau: float = 0.995,
                 use_oracle: bool = False) -> None:
        import pickle
        self.name = name
        self.model = model
        self.eval_model = pickle.loads(pickle.dumps(model)).to(next(model.parameters()).device)
        self.eval_model.eval()
        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.99, eps=1e-5)
        self.tau = tau
        self.batch_size = batch_size
        self.use_oracle = use_oracle
        self.loss_fn = nn.MSELoss()
        self.train_loss_history: List[float] = []

    def load_model(self, model: nn.Module):
        self.model = model
        self.eval_model = model

    def prepare_batch_inputs(self, samples):
        raise NotImplementedError()

    def learn_from_samples(self, samples: List[Tuple[Observation, Action, float]]):
        splits = int(len(samples) / self.batch_size)
        for subsamples in np.array_split(np.array(samples, dtype=object), max(1, splits), axis=0):
            *args, rewards = self.prepare_batch_inputs(subsamples)
            pred = self.model(*args)
            loss = self.loss_fn(pred, rewards)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 80)
            self.optimizer.step()
            self.train_loss_history.append(loss.detach().item())
        for param, target_param in zip(self.model.parameters(), self.eval_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class SJAgent:
    def __init__(self, name: str) -> None:
        self.name = name

        self.declare_module: StageModule = None
        self.kitty_module: StageModule = None
        self.chaodi_module: StageModule = None
        self.main_module: StageModule = None
    
    def act(self, obs: Observation, epsilon=None, training=True):
        assert self.declare_module is not None and self.kitty_module is not None and self.main_module is not None, "At least one required model is not loaded"
        if obs.stage == Stage.declare_stage:
            return self.declare_module.act(obs, epsilon, training)
        elif obs.stage == Stage.kitty_stage:
            return self.kitty_module.act(obs, epsilon, training)
        elif obs.stage == Stage.chaodi_stage:
            assert self.chaodi_module is not None, "chaodi module must be configured when chaodi mode is turned on"
            return self.chaodi_module.act(obs, epsilon, training)
        elif obs.stage == Stage.main_stage:
            return self.main_module.act(obs, epsilon, training)
        else:
            raise NotImplementedError()
    
    def learn_from_samples(self, samples: List[Tuple[Observation, Action, float]], stage: Stage):
        if stage == Stage.declare_stage:
            self.declare_module.learn_from_samples(samples)
        elif stage == Stage.kitty_stage:
            self.kitty_module.learn_from_samples(samples)
        elif stage == Stage.chaodi_stage:
            self.chaodi_module.learn_from_samples(samples)
        elif stage == Stage.main_stage:
            self.main_module.learn_from_samples(samples)
        else:
            raise NotImplementedError()

    def optimizer_states(self):
        return {}

    def load_optimizer_states(self, state):
        pass
    
    # Try to load models from disk. Return whether the models were loaded successfully.
    def load_models_from_disk(self) -> bool:
        raise NotImplementedError()

    def save_models_to_disk(self):
        raise NotImplementedError()

