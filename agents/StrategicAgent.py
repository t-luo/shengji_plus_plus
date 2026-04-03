from .Agent import SJAgent, StageModule
from typing import List, Tuple
import sys
import random
from env.CardSet import CardSet, MoveType, TrumpSuit

sys.path.append('.')
from env.Actions import *
from env.utils import ORDERING_INDEX, Stage, softmax, CardSuit
from env.Observation import Observation
from networks.Models import *

class DeclarationModule(StageModule):
    def act(self, obs: Observation, epsilon=None, training=True):
        assert obs.stage == Stage.declare_stage

        best_declare_option = None
        best_relative_count = 0
        for declare_action in obs.actions:
            if isinstance(declare_action, DeclareAction) and declare_action.declaration.suit not in ['XJ', 'DJ']:
                trump_count = obs.hand.count_suit(CardSuit.TRUMP, declare_action.declaration.suit, obs.dominant_rank)
                if trump_count > best_relative_count:
                    best_relative_count = trump_count
                    best_declare_option = declare_action
        if best_relative_count > 5 and best_relative_count / obs.hand.size >= 0.4:
            return best_declare_option
        else:
            return DontDeclareAction()

class KittyModule(StageModule):
    def act(self, obs: Observation, epsilon=None, training=True):
        assert obs.stage == Stage.kitty_stage

        best_actions: List[Tuple[int, Action]] = []
        second_best_actions: List[Tuple[int, Action]] = []
        worse_actions: List[Tuple[int, Action]] = []
        current_suit = obs.declaration.suit if obs.declaration else TrumpSuit.XJ
        for action in obs.actions:
            assert isinstance(action, PlaceKittyAction)
            action: PlaceKittyAction
            
            rank = get_rank(action.card, current_suit, obs.dominant_rank)
            suit = get_suit(action.card, current_suit, obs.dominant_rank)
            
            # Best actions
            if suit != CardSuit.TRUMP and rank <= 11 and action.count == 1:
                best_actions.append((rank, action))
            
            # Second best actions
            elif suit != CardSuit.TRUMP and (rank <= 8 and action.count == 2) or (rank <= 13 and action.count == 1):
                second_best_actions.append((rank * 2, action))
            elif suit == CardSuit.TRUMP and rank <= 7:
                worse_actions.append((rank, action))
        
        if best_actions:
            return min(best_actions, key=lambda x: x[0])[1]
        elif second_best_actions:
            return min(second_best_actions, key=lambda x: x[0])[1]
        else:
            return min(worse_actions, key=lambda x: x[0])[1]
    
class ChaodiModule(StageModule):
    def act(self, obs: Observation, epsilon=None, training=True):
        assert obs.stage == Stage.chaodi_stage
        dominant_suit = obs.declaration.suit if obs.declaration else TrumpSuit.XJ
        best_suit_action: Action = None
        nt_action: Action = None
        current_trump_count = obs.hand.count_suit(CardSuit.TRUMP, dominant_suit, obs.dominant_rank)
        best_chaodi_trump_count = 0
        for option in obs.actions:
            if isinstance(option, ChaodiAction):
                if option.declaration.suit in ('XJ', 'DJ'):
                    nt_action = option
                else:
                    new_trump_count = obs.hand.count_suit(CardSuit.TRUMP, option.declaration.suit, obs.dominant_rank)
                    best_chaodi_trump_count = max(best_chaodi_trump_count, new_trump_count)
        
        if best_suit_action and best_chaodi_trump_count >= 10 and best_chaodi_trump_count + 2 >= current_trump_count:
            return best_suit_action
        elif nt_action and current_trump_count < 12:
            return nt_action
        else:
            return DontChaodiAction()

class MainModule(StageModule):
    def act(self, obs: Observation, epsilon=None, training=True):
        if obs.leads_current_round:
            optimal_actions = []
            second_best_actions = []
            other_actions = []
            for action in obs.actions:
                action: LeadAction
                d = action.move.cardset.decompose(obs.dominant_suit, obs.dominant_rank)
                
                # For simplicity, this agent doesn't play combos
                if len(d) > 1:
                    # If the combo is unbeatable (unless using trump cards)
                    if CardSet.is_bigger_than(obs.unplayed_cards, action.move, obs.dominant_suit, obs.dominant_rank) is None:
                        # If the combo contains a pair of a tractor, then it's probably a good move to play
                        if not all([isinstance(component, MoveType.Single) for component in d]):
                            optimal_actions.append(action)
                        else:
                            second_best_actions.append(action)
                    print(action.move)
                elif action.move.cardset.count_suit(CardSuit.TRUMP, obs.dominant_suit, obs.dominant_rank) == 0:
                    min_rank = d[0].cardset.min_rank(obs.dominant_suit, obs.dominant_rank)
                    if isinstance(d[0], MoveType.Tractor) or isinstance(d[0], MoveType.Pair) and min_rank >= 8 and isinstance(d[0], MoveType.Single) and (min_rank == 14 or obs.dominant_rank == 14 and min_rank == 13):
                        optimal_actions.append(action)
                    elif isinstance(d[0], MoveType.Pair) or d[0].cardset.min_rank(obs.dominant_suit, obs.dominant_rank) > 10:
                        second_best_actions.append(action)
                    else:
                        other_actions.append(action)
                else:
                    if action.move.cardset.total_points() == 0:
                        optimal_actions.append(action)
                    else:
                        other_actions.append(action)

            if optimal_actions:
                return random.choice(optimal_actions), None, None
            elif second_best_actions:
                return random.choice(second_best_actions), None, None
            elif other_actions:
                return random.choice(other_actions), None, None
            else:
                return random.choice(obs.actions), None, None
                        
        else:
            current_round_points = 0
            dominant_suit = obs.declaration.suit if obs.declaration else TrumpSuit.XJ
            for h in obs.round_history[-1][1]:
                current_round_points += h.total_points()
            
            good_action_choices = []
            other_action_choices = []
            lead_index = CardSet.round_winner(obs.round_history[-1][1], dominant_suit, obs.dominant_rank)
            teammate_is_biggest = len(obs.round_history[-1][1]) - lead_index == 2
            
            for action in obs.actions:
                if not isinstance(action, FollowAction): continue

                is_dominating = CardSet.round_winner(obs.round_history[-1][1] + [action.cardset], dominant_suit, obs.dominant_rank) == len(obs.round_history[-1][1])
                if (current_round_points > 0 or len(obs.round_history[-1][1]) == 2) and not teammate_is_biggest:
                    if is_dominating:
                        good_action_choices.append(action)
                    else:
                        other_action_choices.append(action)
                else:
                    good_action_choices.append(action)
            
            if good_action_choices:
                return random.choice(good_action_choices), None, None
            else:
                return random.choice(other_action_choices), None, None

class StrategicAgent(SJAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)

        self.declare_module = DeclarationModule()
        self.kitty_module = KittyModule()
        self.chaodi_module = ChaodiModule()
        self.main_module = MainModule()

