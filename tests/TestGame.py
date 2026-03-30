import logging
import random
import unittest
import sys
import torch
import tqdm
import numpy as np

sys.path.append('.')
from agents.RLAgents import ChaodiAgent, DeclareAgent, KittyAgent, MainAgent
from networks.Models import ChaodiModel, DeclarationModel, KittyModel, MainModel
from Simulation import Simulation
from agents.Agent import SJAgent
from agents.RandomAgent import RandomAgent
from agents.StrategicAgent import StrategicAgent
from env.Actions import DeclareAction, DontDeclareAction, PlaceKittyAction
from env.utils import AbsolutePosition, Declaration, TrumpSuit
from env.CardSet import CardSet
from env.Game import Game

class TestGame(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(101)
        torch.manual_seed(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using", self.device)
        return super().setUp()

    def test_gameplay(self):
        random.seed(1235)

        game = Game()
        current_player = game.start_game()
        for _ in range(99):
            game.run_action(DontDeclareAction(), current_player)
            current_player = current_player.next_position

        # Test action list
        north_actions = game.get_observation(AbsolutePosition.NORTH).actions
        west_actions = game.get_observation(AbsolutePosition.WEST).actions
        # print('West actions:', west_actions) 
        # print('North actions:', north_actions) # NORTH has double red jokers in this game at action 1.
        self.assertEqual(len(west_actions), 2)
        self.assertEqual(len(north_actions), 3)
        game.run_action(west_actions[0], AbsolutePosition.WEST) # West declares something
        game.run_action(DontDeclareAction(), AbsolutePosition.SOUTH)
        game.run_action(DontDeclareAction(), AbsolutePosition.EAST)
        game.run_action(north_actions[0], AbsolutePosition.NORTH) # North overrides the declaration with double black joker
        game.run_action(DontDeclareAction(), AbsolutePosition.WEST)
        game.run_action(DontDeclareAction(), AbsolutePosition.SOUTH)
        game.run_action(DontDeclareAction(), AbsolutePosition.EAST)

        self.assertEqual(sum([hand.size for hand in game.hands.values()]), 108) # all cards should be distributed
        self.assertEqual(game.hands[game.dealer_position].size, 33) # Dealer has 25 + 8 = 33 cards
        self.assertTrue(all([hand.size == 25 for player, hand in game.hands.items() if player != game.dealer_position]))
        self.assertTrue(game.dealer_position == game.declarations[-1].absolute_position == AbsolutePosition.NORTH) # WEST is now the new dealer

        north_actions = game.get_observation(AbsolutePosition.NORTH).actions
        self.assertEqual(sum([a.count for a in north_actions if isinstance(a, PlaceKittyAction)]), 33) # At this point, NORTH can choose from all 33 cards

        for i in range(8):
            game.run_action(north_actions[i], AbsolutePosition.NORTH)
        
        # Skip chaodi
        north_actions = game.get_observation(AbsolutePosition.NORTH).actions
        self.assertEqual(len(north_actions), 26)

    def test_random_game_simulation(self):
        random.seed(101)
        sim = Simulation(player1=RandomAgent('random'), enable_chaodi=True)
        logging.getLogger().setLevel(logging.DEBUG)

        while sim.step()[0]: pass
        print("Game summary:")
        print(sim.game_engine.print_status())

    def test_rl_agents(self):
        from agents.DMCAgent import DMCAgent
        declare_model = DeclarationModel().to(self.device)
        kitty_model = KittyModel().to(self.device)
        chaodi_model = ChaodiModel().to(self.device)
        main_model = MainModel().to(self.device)
        agent = DMCAgent('test', use_oracle=False)
        agent.declare_module.load_model(declare_model)
        agent.kitty_module.load_model(kitty_model)
        agent.chaodi_module.load_model(chaodi_model)
        agent.main_module.load_model(main_model)
        sim = Simulation(
            player1=agent,
            enable_chaodi=True,
        )

        logging.getLogger().setLevel(logging.DEBUG)
        
        while sim.step()[0]: pass
        print(sim.game_engine.print_status())

        sim.reset()

        while sim.step()[0]: pass
    
    def strategic_vs_random(self):
        "Tests how much better the StrategicAgent is relative to the RandomAgent."

        random.seed(101)

        random_agent = RandomAgent("random")
        strategic_agent = StrategicAgent("strategic")
        sim = Simulation(
            strategic_agent, strategic_agent, strategic_agent, strategic_agent,
            eval=True,
            eval_main=random_agent,
            eval_declare=random_agent,
            eval_kitty=random_agent,
            eval_chaodi=random_agent,
            enable_combos=False
        )
        
        # stats
        wins = [0, 0]
        level_counts = [0, 0]
        points = [[], []]

        for _ in tqdm.tqdm(range(3000)):
            while sim.step()[0]: pass # Play a game
            opponent_index = int(sim.game_engine.dealer_position in ['N', 'S'])
            opponents_won = sim.game_engine.opponent_points >= 80
            win_index = int(opponents_won) if opponent_index == 1 else (1 - opponents_won)
            wins[win_index] += 1
            level_counts[win_index] += abs(sim.game_engine.final_defender_reward)
            points[opponent_index].append(sim.game_engine.opponent_points)

            sim.reset()
        
        print('Wins:', wins)
        print('Level count:', level_counts)
        print(np.mean(points[0]), np.mean(points[1]))



if __name__ == '__main__':
    unittest.main()