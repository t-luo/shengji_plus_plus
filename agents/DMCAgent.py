from collections import deque
import copy
import logging
import pickle
import random
import sys
import time as _time
from typing import Deque, List, Tuple
import numpy as np
import os
from env.CardSet import CardSet, MoveType

from .Agent import SJAgent, StageModule

sys.path.append('.')
from env.Actions import Action, ChaodiAction, DeclareAction, DontChaodiAction, DontDeclareAction, FollowAction, LeadAction, AppendLeadAction, EndLeadAction, PlaceAllKittyAction, PlaceKittyAction
from env.utils import ORDERING_INDEX, Stage, softmax
from env.Observation import Observation
from networks.Models import *

# A generic class that describes a stage module for the DMC agent.
class DMCModule(StageModule):
    def __init__(self, batch_size: int, tau=0.1, dynamic_encoding=True) -> None:
        self.batch_size = batch_size # preferred batch size
        self.tau = tau # soft weight update parameter
        self._model: nn.Module = None # don't set directly
        self._eval_model: nn.Module = None # don't set directly
        self.loss_fn = nn.MSELoss()
        self.train_loss_history: List[float] = []
        self.optimizer: torch.optim.Optimizer = None
        self.dynamic_encoding = dynamic_encoding
        self._last_inference_ms: float = 0.0

    # Use this function to load a pretrained model
    def load_model(self, model: nn.Module):
        self._model = model
        self._model.share_memory()
        self._eval_model = copy.deepcopy(model)
        self._eval_model.eval().share_memory()
        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.99, eps=1e-5)

    # Helper function to prepare `Observation` and `Action` objects into CPU tensors.
    # Returns a tuple of CPU tensors: (*args, rewards)
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float]]):
        raise NotImplementedError

    # Training function from raw samples (used in tests / single-process fallback)
    def learn_from_samples(self, samples: List[Tuple[Observation, Action, float]]):
        if not samples:
            return
        device = next(self._model.parameters()).device
        splits = int(len(samples) / self.batch_size)
        for subsamples in np.array_split(np.array(samples, dtype=object), max(1, splits), axis=0):
            *args, rewards = self.prepare_batch_inputs(subsamples)
            args = [a.to(device) for a in args]
            rewards = rewards.to(device)
            pred = self._model(*args)
            loss = self.loss_fn(pred, rewards)
            self.optimizer.zero_grad()
            loss.backward()
            if torch.isnan(loss):
                def init_weights(m):
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight.data)
                self._model.apply(init_weights)
                print(f"Model {self} encountered nan, reset weights to random.")
            else:
                nn.utils.clip_grad_norm_(self._model.parameters(), 80)
                self.optimizer.step()
                self.train_loss_history.append(loss.detach().item())

        for param, target_param in zip(self._model.parameters(), self._eval_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # Training function from pre-built CPU tensors.
    # Actors call prepare_batch_inputs and put tensors on the queue;
    # the main process calls this to do GPU forward/backward without any Python loops.
    def learn_from_tensors(self, tensors):
        if tensors is None:
            return
        *args, rewards = tensors
        device = next(self._model.parameters()).device
        args = [a.to(device) for a in args]
        rewards = rewards.to(device)
        n = rewards.shape[0]
        indices = torch.randperm(n)
        for idx_chunk in torch.chunk(indices, max(1, n // self.batch_size)):
            batch_args = [a[idx_chunk] for a in args]
            batch_rewards = rewards[idx_chunk]
            pred = self._model(*batch_args)
            loss = self.loss_fn(pred, batch_rewards)
            self.optimizer.zero_grad()
            loss.backward()
            if torch.isnan(loss):
                def init_weights(m):
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight.data)
                self._model.apply(init_weights)
                print(f"Model {self} encountered nan, reset weights to random.")
            else:
                nn.utils.clip_grad_norm_(self._model.parameters(), 80)
                self.optimizer.step()
                self.train_loss_history.append(loss.detach().item())

        for param, target_param in zip(self._model.parameters(), self._eval_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _eval_in_chunks(self, tensors: list, chunk_size: int = 32) -> list:
        """Evaluate actions in chunks to bound activation memory per forward pass."""
        t0 = _time.monotonic()
        n = tensors[0].shape[0]
        device = next(self._eval_model.parameters()).device
        results = []
        with torch.no_grad():
            for start in range(0, n, chunk_size):
                chunk = [t[start:start + chunk_size].to(device) for t in tensors]
                results.append(self._eval_model(*chunk).cpu())
        self._last_inference_ms = (_time.monotonic() - t0) * 1000
        return torch.cat(results, dim=0).squeeze(1).tolist()

    def act(self, obs: Observation, epsilon=None, training=True):
        if epsilon and random.random() < epsilon:
            return random.choice(obs.actions)
            # return random.choices(obs.actions, softmax([reward(a).cpu().item() for a in obs.actions]))[0]
        else:
            *state_and_actions, _ = self.prepare_batch_inputs([(obs, a, 0) for a in obs.actions])
            rewards = self._eval_in_chunks(state_and_actions)

            # If in verbose mode, log actions and their probabilities in test time
            if not training and logging.getLogger().level == logging.DEBUG:
                logging.debug(f"Probability of actions ({obs.position.value}):")
                exp_total = np.sum(np.exp(rewards))
                sorted_actions = sorted(zip(obs.actions, rewards), key=lambda x: x[1], reverse=True)
                for i, (action, rw) in enumerate(sorted_actions):
                    logging.debug(f"{i:2}. {action} (reward={round(rw, 4)}, prob={np.exp(rw) / exp_total:.4f})")
                return sorted_actions[0][0]
            else:
                return max(zip(obs.actions, rewards), key=lambda x: x[1])[0]

class DeclareModule(DMCModule):
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float]]):
        x_batch = torch.zeros((len(samples), 179))
        gt_rewards = torch.zeros((len(samples), 1))
        for i, (obs, ac, rw) in enumerate(samples):
            assert isinstance(ac, DeclareAction) or isinstance(ac, DontDeclareAction), "DeclareAgent can only handle declare actions"
            state_tensor = torch.cat([
                obs.dynamic_hand_tensor if self.dynamic_encoding else obs.hand.tensor, # (108,)
                obs.dealer_position_tensor, # (4,)
                obs.trump_tensor, # (20,)
                obs.declarer_position_tensor, # (4,)
                obs.perceived_trump_cardsets, # (36,)
            ])
            x_batch[i] = torch.cat([state_tensor, ac.tensor])
            gt_rewards[i] = rw
        return x_batch, gt_rewards  # CPU tensors


class KittyModule(DMCModule):
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float]]):
        state_batch = torch.zeros((len(samples), 172))
        action_batch = torch.zeros(len(samples), dtype=torch.int)
        gt_rewards = torch.zeros((len(samples), 1))
        for i, (obs, ac, rw) in enumerate(samples):
            assert isinstance(ac, PlaceKittyAction), "KittyAgent can only handle place kitty actions"
            state_tensor = torch.cat([
                obs.dynamic_hand_tensor if self.dynamic_encoding else obs.hand.tensor, # (108,)
                obs.dealer_position_tensor, # (4,)
                obs.trump_tensor, # (20,)
                obs.declarer_position_tensor, # (4,)
                obs.perceived_trump_cardsets, # (36,)
                # TODO: add kitty to state
            ])
            state_batch[i] = state_tensor
            if self.dynamic_encoding:
                action_batch[i] = ac.get_dynamic_tensor(obs.dominant_suit, obs.dominant_rank)
            else:
                action_batch[i] = ac.tensor
            gt_rewards[i] = rw
        return state_batch, action_batch, gt_rewards  # CPU tensors


class ChaodiModule(DMCModule):
    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float]]):
        x_batch = torch.zeros((len(samples), 178))
        gt_rewards = torch.zeros((len(samples), 1))
        for i, (obs, ac, rw) in enumerate(samples):
            assert isinstance(ac, ChaodiAction) or isinstance(ac, DontChaodiAction), "ChaodiAgent can only handle chaodi decisions"
            state_tensor = torch.cat([
                obs.dynamic_hand_tensor if self.dynamic_encoding else obs.hand.tensor, # (108,)
                obs.dealer_position_tensor, # (4,)
                obs.trump_tensor, # (20,)
                obs.declarer_position_tensor, # (4,)
                obs.perceived_trump_cardsets, # (36,)
            ])
            x_batch[i] = torch.cat([state_tensor, ac.tensor])
            gt_rewards[i] = rw
        return x_batch, gt_rewards  # CPU tensors


class MainModule(DMCModule):
    def __init__(self, batch_size: int, use_oracle: bool, tau=0.1, dynamic_encoding=True, sac=False) -> None:
        super().__init__(batch_size, tau, dynamic_encoding=dynamic_encoding)

        self.sac = sac
        self.use_oracle = use_oracle
        self.log_alpha = torch.tensor(0.0, requires_grad=True)  # log(1.0) = 0.0, moved to device in load_models_from_disk
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

    def prepare_batch_inputs(self, samples: List[Tuple[Observation, Action, float, Tuple[Observation, float, float]]], training=True):
        if self.use_oracle:
            x_batch = torch.zeros((len(samples), 1089 + 108)) # additionally provide other players' hands
        else:
            x_batch = torch.zeros((len(samples), 1089 - 2 * 108))
        history_batch = torch.zeros((len(samples), 15, 436)) # Store up to last 15 rounds of history
        gt_rewards = torch.zeros((len(samples), 1))
        next_action_entropy = torch.zeros(len(samples))
        current_action_entropy = torch.zeros(len(samples))

        for i, (obs, ac, rw, aux) in enumerate(samples):
            assert isinstance(ac, LeadAction) or isinstance(ac, AppendLeadAction) or isinstance(ac, EndLeadAction) or isinstance(ac, FollowAction)
            historical_moves, current_moves = obs.historical_moves_dynamic_tensor if self.dynamic_encoding else obs.historical_moves_tensor
            cardset = ac.cardset
            if aux is not None:
                (next_obs, current_entropy, next_entropy) = aux
                if current_entropy is not None:
                    current_action_entropy[i] = current_entropy
                if next_entropy is not None:
                    next_action_entropy[i] = next_entropy
            else:
                next_obs, current_entropy, next_entropy = None, None, None

            state_tensor = torch.cat([
                obs.dynamic_hand_tensor if self.dynamic_encoding else obs.hand.tensor, # (108,),
                obs.dealer_position_tensor, # (4,)
                obs.trump_tensor, # (20,)
                obs.declarer_position_tensor, # (4,)
                obs.chaodi_times_tensor, # (4,)
                obs.points_tensor, # (80,)
                obs.unplayed_cards_dynamic_tensor if self.dynamic_encoding else obs.unplayed_cards_tensor, # (108,)
                current_moves, # (328,)
                obs.kitty_dynamic_tensor if self.dynamic_encoding else obs.kitty_tensor, # (108,)
                # obs.current_dominating_player_index, # (3,)
                obs.dominates_all_tensor(cardset), # (1,)
            ])

            if self.use_oracle and training:
                state_tensor = torch.cat([obs.oracle_cardsets, state_tensor])
            elif self.use_oracle:
                state_tensor = torch.cat([torch.zeros(108 * 3), state_tensor])

            if self.dynamic_encoding:
                x_batch[i] = torch.cat([state_tensor, ac.dynamic_tensor(obs.dominant_suit, obs.dominant_rank)])
            else:
                x_batch[i] = torch.cat([state_tensor, ac.tensor])
            history_batch[i] = historical_moves
            gt_rewards[i] = rw
        return x_batch, history_batch, current_action_entropy, next_action_entropy, gt_rewards  # CPU tensors

    def act(self, obs: Observation, epsilon=None, training=True):
        x_batch, history_batch, *_ = self.prepare_batch_inputs([(obs, a, 0, None) for a in obs.actions])
        rewards = self._eval_in_chunks([x_batch, history_batch])
        action_distribution = np.exp(rewards) / np.sum(np.exp(rewards))
        entropy = -np.mean(action_distribution * np.log2(action_distribution))
        optimal_index = np.argmax(rewards)
        if epsilon and random.random() < epsilon:
            # If in verbose mode, log actions and their probabilities in test time
            if not training and logging.getLogger().level == logging.DEBUG:
                logging.debug("Probability of actions:")
                sorted_actions = sorted(zip(obs.actions, rewards), key=lambda x: x[1], reverse=True)
                for i, (action, rw) in enumerate(sorted_actions):
                    logging.debug(f"{i:2}. {action} (reward={round(rw, 4)})")
                return sorted_actions[0], action_distribution[optimal_index], entropy
            else:
                return obs.actions[optimal_index], action_distribution[optimal_index], entropy
        else:
            return obs.actions[optimal_index], action_distribution[optimal_index], entropy

    def learn_from_samples(self, samples: List[Tuple[Observation, Action, float, Tuple[Observation, float, float]]]):
        if not samples:
            return
        device = next(self._model.parameters()).device
        splits = int(len(samples) / self.batch_size)
        for subsamples in np.array_split(np.array(samples, dtype=object), max(1, splits), axis=0):
            *args, current_action_entropy, next_action_entropy, rewards = self.prepare_batch_inputs(subsamples)
            args = [a.to(device) for a in args]
            current_action_entropy = current_action_entropy.to(device)
            next_action_entropy = next_action_entropy.to(device)
            rewards = rewards.to(device)
            pred = self._model(*args)
            loss = self.loss_fn(pred, rewards + torch.exp(self.log_alpha) * next_action_entropy.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            if torch.isnan(loss):
                def init_weights(m):
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight.data)
                self._model.apply(init_weights)
                print(f"Model {self} encountered nan, reset weights to random.")
            else:
                nn.utils.clip_grad_norm_(self._model.parameters(), 80)
                self.optimizer.step()
                self.train_loss_history.append(loss.detach().item())

            # Update alpha
            if self.sac:
                alpha_loss = self.log_alpha.exp() * torch.mean(current_action_entropy) + self.log_alpha.exp() * 10
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

        for param, target_param in zip(self._model.parameters(), self._eval_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def learn_from_tensors(self, tensors):
        """Train from pre-built CPU tensors (x, history, cur_entropy, next_entropy, rewards)."""
        if tensors is None:
            return
        x_batch, history_batch, current_action_entropy, next_action_entropy, rewards = tensors
        device = next(self._model.parameters()).device
        x_batch = x_batch.to(device)
        history_batch = history_batch.to(device)
        current_action_entropy = current_action_entropy.to(device)
        next_action_entropy = next_action_entropy.to(device)
        rewards = rewards.to(device)
        n = rewards.shape[0]
        indices = torch.randperm(n)
        for idx_chunk in torch.chunk(indices, max(1, n // self.batch_size)):
            pred = self._model(x_batch[idx_chunk], history_batch[idx_chunk])
            loss = self.loss_fn(pred, rewards[idx_chunk] + torch.exp(self.log_alpha) * next_action_entropy[idx_chunk].unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            if torch.isnan(loss):
                def init_weights(m):
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight.data)
                self._model.apply(init_weights)
                print(f"Model {self} encountered nan, reset weights to random.")
            else:
                nn.utils.clip_grad_norm_(self._model.parameters(), 80)
                self.optimizer.step()
                self.train_loss_history.append(loss.detach().item())

            if self.sac:
                alpha_loss = self.log_alpha.exp() * torch.mean(current_action_entropy[idx_chunk]) + self.log_alpha.exp() * 10
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

        for param, target_param in zip(self._model.parameters(), self._eval_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class DMCAgent(SJAgent):
    def __init__(self, name: str, use_oracle: bool, dynamic_encoding=True, sac=False) -> None:
        super().__init__(name)

        self.declare_module: DeclareModule = DeclareModule(batch_size=64, dynamic_encoding=dynamic_encoding)
        self.kitty_module: KittyModule = KittyModule(batch_size=32, dynamic_encoding=dynamic_encoding)
        self.chaodi_module: ChaodiModule = ChaodiModule(batch_size=32, dynamic_encoding=dynamic_encoding)
        self.main_module: MainModule = MainModule(batch_size=64, use_oracle=use_oracle, dynamic_encoding=dynamic_encoding, sac=sac)
        self.dynamic_encoding = dynamic_encoding
        self.sac = sac

    def optimizer_states(self):
        return {
            'declare_optim_state': self.declare_module.optimizer.state_dict(),
            'kitty_optim_state': self.kitty_module.optimizer.state_dict(),
            'chaodi_optim_state': self.chaodi_module.optimizer.state_dict(),
            'main_optim_state': self.main_module.optimizer.state_dict(),
            'alpha_optim_state': self.main_module.alpha_optimizer.state_dict() if self.sac else None,
            'alpha': self.main_module.log_alpha.exp().cpu().item()
        }

    def load_optimizer_states(self, state):
        self.main_module.optimizer.load_state_dict(state['main_optim_state'])
        self.kitty_module.optimizer.load_state_dict(state['kitty_optim_state'])
        self.declare_module.optimizer.load_state_dict(state['declare_optim_state'])
        self.chaodi_module.optimizer.load_state_dict(state['chaodi_optim_state'])
        if self.sac:
            self.main_module.log_alpha.data = torch.tensor(state['alpha']).log()
            self.main_module.alpha_optimizer.load_state_dict(state['alpha_optim_state'])

    def load_models_from_disk(self, train_models):
        # Load models for DMC
        loaded_models = True
        declare_model: nn.Module = train_models.DeclarationModel().cuda()
        if os.path.exists(f'{self.name}/declare.pt'):
            declare_model.load_state_dict(torch.load(f'{self.name}/declare.pt', map_location='cuda', weights_only=True), strict=False)
            print("Using loaded model for declaration")
        else:
            loaded_models = False
        self.declare_module.load_model(declare_model)

        kitty_model: nn.Module = train_models.KittyModel().cuda()
        if os.path.exists(f'{self.name}/kitty.pt'):
            kitty_model.load_state_dict(torch.load(f'{self.name}/kitty.pt', map_location='cuda', weights_only=True), strict=False)
            print("Using loaded model for kitty")
        else:
            loaded_models = False
        self.kitty_module.load_model(kitty_model)

        chaodi_model: nn.Module = train_models.ChaodiModel().cuda()
        if os.path.exists(f'{self.name}/chaodi.pt'):
            chaodi_model.load_state_dict(torch.load(f'{self.name}/chaodi.pt', map_location='cuda', weights_only=True), strict=False)
            print("Using loaded model for chaodi")
        else:
            loaded_models = False
        self.chaodi_module.load_model(chaodi_model)

        try:
            with open(f'{self.name}/state.pkl', mode='rb') as f:
                state = pickle.load(f)
                self.main_module.use_oracle = state['oracle_duration'] > 0
            with open(f'{self.name}/stats.pkl', mode='rb') as f:
                stats = pickle.load(f)
                iterations = stats[-1]['iterations']
            # If resuming from checkpoint, subtract iterations from oracle duration
            oracle_duration = max(0, state['oracle_duration'] - iterations)
            print(f"Resuming with remaining oracle duration {oracle_duration}")
        except Exception as e:
            loaded_models = False
        main_model: nn.Module = train_models.MainModel(use_oracle=self.main_module.use_oracle).cuda()
        if os.path.exists(f'{self.name}/main.pt'):
            main_model.load_state_dict(torch.load(f'{self.name}/main.pt', map_location='cuda', weights_only=True), strict=False)
            print("Using loaded model for main game")
        else:
            loaded_models = False
        self.main_module.load_model(main_model)

        device = next(main_model.parameters()).device
        self.main_module.log_alpha = self.main_module.log_alpha.to(device).detach().requires_grad_(True)
        self.main_module.alpha_optimizer = torch.optim.Adam([self.main_module.log_alpha], lr=3e-4)

        return loaded_models, stats[-1]['iterations'] if loaded_models else 0

    def save_models_to_disk(self):
        torch.save(self.declare_module._model.state_dict(), self.name + '/declare.pt')
        torch.save(self.kitty_module._model.state_dict(), self.name + '/kitty.pt')
        if self.chaodi_module._model is not None:
            torch.save(self.chaodi_module._model.state_dict(), self.name + '/chaodi.pt')
        torch.save(self.main_module._model.state_dict(), self.name + '/main.pt')

    def clear_loss_histories(self):
        self.declare_module.train_loss_history.clear()
        self.kitty_module.train_loss_history.clear()
        self.chaodi_module.train_loss_history.clear()
        self.main_module.train_loss_history.clear()
