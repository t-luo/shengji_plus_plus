import importlib
import logging
import pickle
import random
import shutil
import time
from torch.multiprocessing import Process, Lock, Queue
import torch, os, sys
from Simulation import Simulation
from agents.Agent import SJAgent
from agents.RandomAgent import RandomAgent
from agents.StrategicAgent import StrategicAgent
from agents.InteractiveAgent import InteractiveAgent
from agents.DMCAgent import DMCAgent
from agents.DQNAgent import DQNAgent
from env.utils import Stage
import argparse
import tqdm
import numpy as np
from torch import nn

ctx = torch.multiprocessing.get_context('spawn')
global_main_queue = ctx.Queue(maxsize=25)
global_chaodi_queue = ctx.Queue(maxsize=25)
global_declare_queue = ctx.Queue(maxsize=25)
global_kitty_queue = ctx.Queue(maxsize=25)
actor_processes = []

# Parallelized data sampling
def sampler(idx: int, player: SJAgent, discount, decay_factor, global_main_queue, global_chaodi_queue, global_declare_queue, global_kitty_queue, enable_chaodi: bool, enable_combos: bool, epsilon=0.02, reuse_times=0, oracle_duration=0, game_count=0, log_file='', combo_penalty=0.1, combo_alternation=False, epsilon_start=0.5):
    logging.getLogger().setLevel(logging.ERROR)
    # logging.basicConfig(format="%(process)d %(message)s", filename=log_file, encoding='utf-8', level=logging.DEBUG)
    train_sim = Simulation(
        player1=player,
        player2=None,
        enable_chaodi=enable_chaodi,
        enable_combos=enable_combos,
        discount=discount,
        epsilon=epsilon_start,
        oracle_duration=oracle_duration,
        game_count=game_count,
        combo_penalty=combo_penalty
    )
    while True:
        local_main, local_declare, local_kitty, local_chaodi = [], [], [], []
        same_deck_count = 0
        t_sim_total = 0.0
        t_infer_total = 0.0
        for i in range(10):
            t0 = time.monotonic()
            with torch.no_grad():
                while train_sim.step()[0]: pass
            t_sim_total += time.monotonic() - t0
            t_infer_total += train_sim.total_inference_ms / 1000
            local_main.extend(train_sim.main_history)
            local_declare.extend(train_sim.declaration_history)
            local_chaodi.extend(train_sim.chaodi_history)
            local_kitty.extend(train_sim.kitty_history)

            # Get new deck every `reuse_times` times
            if same_deck_count < reuse_times:
                same_deck_count += 1
                train_sim.reset(reuse_old_deck=True)
            else:
                same_deck_count = 0
                train_sim.reset(reuse_old_deck=False)
        train_sim.epsilon = max(epsilon, train_sim.epsilon / decay_factor)

        # Pre-tensorize in the actor process so the main process only does GPU work
        main_tensors = player.main_module.prepare_batch_inputs(local_main) if local_main else None
        declare_tensors = player.declare_module.prepare_batch_inputs(local_declare) if local_declare else None
        kitty_tensors = player.kitty_module.prepare_batch_inputs(local_kitty) if local_kitty else None
        chaodi_tensors = player.chaodi_module.prepare_batch_inputs(local_chaodi) if local_chaodi else None

        global_main_queue.put((main_tensors, t_sim_total / 10, t_infer_total / 10))
        global_declare_queue.put(declare_tensors)
        global_chaodi_queue.put(chaodi_tensors)
        global_kitty_queue.put(kitty_tensors)

def evaluator(idx: int, player1: SJAgent, player2: SJAgent, enable_chaodi: bool, enable_combos: bool, eval_size: int, eval_results_queue: Queue, verbose=False, learn_from_eval=False, log_file=''):
    logging.getLogger().setLevel(logging.ERROR)
    # if not verbose:
    #     logging.basicConfig(format="%(process)d %(message)s", filename=log_file, encoding='utf-8', level=logging.DEBUG)
    random.seed(idx)
    eval_sim = Simulation(
        player1=player1,
        player2=player2,
        enable_chaodi=enable_chaodi,
        enable_combos=enable_combos,
        eval=True,
        learn_from_eval=learn_from_eval
    )
    iterations = 0
    while True:
        with torch.no_grad():
            while eval_sim.step()[0]: pass
        opponent_index = int(eval_sim.game_engine.dealer_position in ['N', 'S'])
        opponents_won = eval_sim.game_engine.opponent_points >= 80
        win_index = int(opponents_won) if opponent_index == 1 else (1 - opponents_won)
        eval_results_queue.put((
            win_index,
            opponent_index,
            eval_sim.game_engine.opponent_points,
            abs(eval_sim.game_engine.final_defender_reward)
        ))
        eval_sim.reset()
        # with open(log_file, 'w') as f:
        #     f.write('')
        iterations += 1
        
        # if iterations > eval_size:
        #     exit(0)
    


def train(agent_type: str, games: int, model_folder: str, eval_only: bool, eval_size: int, compare: str = None, discount=0.99, decay_factor=1.2, chaodi=True, combos=False, verbose=False, random_seed=1, single_process=False, epsilon=0.01, tau=0.995, kitty_agent='fc', eval_agent_type='random', learn_from_eval=False, reuse_times=0, oracle_duration=0, max_games=500000, combo_penalty=0.1, dynamic_encoding=True, combo_alternation=False, actor_process_count=6, eval_process_count=7, model_architecture='mlp', eval_agents=None, epsilon_start=0.5):
    os.makedirs(model_folder, exist_ok=True)
    torch.manual_seed(0)
    random.seed(random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    oracle_duration_input = oracle_duration

    # Copy the active Models.py to networks/ActiveModels.py so spawn child
    # processes can import it by a stable, fully-qualified module name.
    _models_path = f"{model_folder}/Models.py"
    if os.path.exists(_models_path):
        shutil.copyfile(_models_path, "networks/ActiveModels.py")
    else:
        shutil.copyfile("networks/Models.py", "networks/ActiveModels.py")
    import importlib
    if "networks.ActiveModels" in sys.modules:
        del sys.modules["networks.ActiveModels"]
    train_models = importlib.import_module("networks.ActiveModels")

    agent: SJAgent
    iterations = 0
    stats = []

    if agent_type.startswith('dmc'):
        agent = DMCAgent(model_folder, use_oracle=oracle_duration_input > 0, dynamic_encoding=dynamic_encoding, sac=agent_type.endswith('sac'))
        print(f"Using DMC model {'with' if dynamic_encoding else 'without'} dynamic encoding and " + ("with sac" if agent_type.endswith('sac') else "without sac"))
    elif agent_type.startswith('dqn'):
        agent = DQNAgent(model_folder, discount=discount, sac=agent_type.endswith('sac'))
        print("Using DQN model" + (" with sac" if agent_type.endswith('sac') else ""))
    loaded_from_disk, iterations = agent.load_models_from_disk(train_models)
    if loaded_from_disk:
        print(f"Using checkpoint at iteration {iterations}")
        with open(model_folder + '/stats.pkl', 'rb') as f:
            stats = pickle.load(f)

    def _build_eval_agent(spec: str) -> tuple:
        """Parse an eval agent spec and return (label, agent).
        Spec format: 'random' | 'strategic' | 'interactive' | 'model:<folder>'
        """
        if spec == 'random':
            return 'random', RandomAgent('random')
        elif spec == 'strategic':
            return 'strategic', StrategicAgent('strategic')
        elif spec == 'interactive':
            return 'interactive', InteractiveAgent('interactive')
        elif spec.startswith('model:'):
            folder = spec[len('model:'):]
            try:
                eval_state = torch.load(f'{folder}/state.pkl', map_location='cpu', weights_only=False)
            except Exception:
                eval_state = {}
            _eval_src = f'{folder}/Models.py' if os.path.exists(f'{folder}/Models.py') else 'networks/Models.py'
            shutil.copyfile(_eval_src, 'networks/EvalModels.py')
            if 'networks.EvalModels' in sys.modules:
                del sys.modules['networks.EvalModels']
            eval_models = importlib.import_module('networks.EvalModels')
            ea = DMCAgent(folder, use_oracle=eval_state.get('oracle_duration', 0) > 0,
                          dynamic_encoding=eval_state.get('dynamic_encoding', True),
                          sac=eval_state.get('agent_type', 'dmc').endswith('sac'))
            ea.load_models_from_disk(eval_models)
            return os.path.basename(folder), ea
        else:
            raise ValueError(f"Unknown eval agent spec: {spec!r}. Use 'random', 'strategic', or 'model:<folder>'")

    # Build list of (label, agent) pairs to evaluate against
    if eval_agents:
        eval_agent_list = [_build_eval_agent(s) for s in eval_agents]
    elif compare:
        eval_agent_list = [_build_eval_agent(f'model:{compare}')]
    else:
        eval_agent_list = [_build_eval_agent(eval_agent_type)]
    print(f"Evaluating against: {[label for label, _ in eval_agent_list]}")

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.ERROR)
        # logging.basicConfig(format="%(process)d %(message)s", filename=f'{model_folder}/debug.log', encoding='utf-8', level=logging.DEBUG)
    
    # Record the command used to run the script
    if not eval_only:
        with open(f'{model_folder}/command.txt', mode='w') as f:
            f.write(' '.join(sys.argv))
    
    # Load saved optimizer states
    _state_path = f'{model_folder}/state.pkl'
    try:
        state = torch.load(_state_path, map_location='cpu', weights_only=False)
        agent.load_optimizer_states(state)
        print("Loaded optimizer states from checkpoint")
    except FileNotFoundError:
        print("Starting new training session")
        if model_architecture == 'transformer':
            shutil.copyfile('networks/TransformerModels.py', f'{model_folder}/Models.py')
        else:
            shutil.copyfile('networks/Models.py', f'{model_folder}/Models.py')
    except Exception as e:
        print(f"WARNING: Failed to load optimizer states: {e}")
        print("Model weights are loaded but optimizer state is reset — this may cause instability")
    
    if not eval_only:
        for i in range(1 if single_process else actor_process_count):
            actor = ctx.Process(target=sampler, args=(i, agent, discount, decay_factor ** (1 / games), global_main_queue, global_chaodi_queue, global_declare_queue, global_kitty_queue, chaodi, combos, epsilon, reuse_times, oracle_duration // actor_process_count, iterations // actor_process_count, f"{model_folder}/debug{i}.log", combo_penalty, combo_alternation, epsilon_start))
            actor.start()
            actor_processes.append(actor)
            
            print(f"Spawned process {i}")    
  
    while iterations < max_games or eval_only:
        if not eval_only:
            print(f"Training iterations {iterations}-{iterations + games}...")
            t0 = time.monotonic()
            t_queue = 0.0
            t_learn = 0.0
            avg_sim_times = []
            avg_infer_times = []
            for _ in tqdm.tqdm(range(0, games, 10)):
                tq = time.monotonic()
                declare_batch = global_declare_queue.get()
                kitty_batch = global_kitty_queue.get()
                main_batch, avg_sim, avg_infer = global_main_queue.get()
                chaodi_batch = global_chaodi_queue.get()
                t_queue += time.monotonic() - tq
                avg_sim_times.append(avg_sim)
                avg_infer_times.append(avg_infer)
                tl = time.monotonic()
                agent.declare_module.learn_from_tensors(declare_batch)
                agent.kitty_module.learn_from_tensors(kitty_batch)
                agent.chaodi_module.learn_from_tensors(chaodi_batch)
                agent.main_module.learn_from_tensors(main_batch)
                t_learn += time.monotonic() - tl
            training_time = time.monotonic() - t0
            agent.save_models_to_disk()
            print(f'Training time: {training_time:.1f}s ({games / training_time:.1f} games/s)')
            print(f'  Queue wait: {t_queue:.1f}s ({100*t_queue/training_time:.0f}%)  Learn: {t_learn:.1f}s ({100*t_learn/training_time:.0f}%)')
            print(f'  Avg sim/game: {np.mean(avg_sim_times):.2f}s  Avg inference/game: {np.mean(avg_infer_times):.2f}s ({100*np.mean(avg_infer_times)/np.mean(avg_sim_times):.0f}% of sim)')
            print('main loss:', np.mean(agent.main_module.train_loss_history), 'declare loss:', np.mean(agent.declare_module.train_loss_history), 'kitty loss:', np.mean(agent.kitty_module.train_loss_history), 'chaodi loss:', np.mean(agent.chaodi_module.train_loss_history))
            if agent.sac:
                print("Current alpha:", agent.main_module.log_alpha.exp().cpu().item())
                if isinstance(agent, DQNAgent):
                    print("value loss:", np.mean(agent.main_module.value_loss_history))
            agent.clear_loss_histories()
        
        per_opponent_stats = {}
        games_per_eval = max(1, eval_size // len(eval_agent_list))

        for opp_label, eval_agent in eval_agent_list:
            if single_process:
                eval_sim = Simulation(
                    player1=agent,
                    player2=eval_agent,
                    enable_chaodi=chaodi,
                    enable_combos=combos,
                    eval=True
                )
                for _ in tqdm.tqdm(range(games_per_eval), desc=opp_label):
                    with torch.no_grad():
                        while eval_sim.step()[0]: pass
                    eval_sim.reset()
                win_counts = eval_sim.win_counts
                level_counts = eval_sim.level_counts
                opposition_points = eval_sim.opposition_points
            else:
                win_counts = [0, 0]
                level_counts = [0, 0]
                opposition_points = [[], []]
                torch.cuda.empty_cache()
                eval_queue = ctx.Queue()
                eval_actors = []
                procs = min(games_per_eval, eval_process_count)
                for i in range(procs):
                    actor = ctx.Process(target=evaluator, args=(i, agent, eval_agent, chaodi, combos, max(1, games_per_eval // procs), eval_queue, verbose, learn_from_eval, f"{model_folder}/eval{i}.log"))
                    actor.start()
                    eval_actors.append(actor)
                with tqdm.tqdm(total=games_per_eval, desc=opp_label) as progress_bar:
                    for i in range(games_per_eval):
                        win_index, opponent_index, points, levels = eval_queue.get()
                        win_counts[win_index] += 1
                        level_counts[win_index] += levels
                        opposition_points[opponent_index].append(points)
                        progress_bar.update(1)
                for a in eval_actors:
                    a.kill()

            per_opponent_stats[opp_label] = {
                'win_counts': win_counts,
                'level_counts': level_counts,
                'avg_points': [float(np.mean(opposition_points[0])) if opposition_points[0] else 0.0,
                               float(np.mean(opposition_points[1])) if opposition_points[1] else 0.0],
            }
            win_rate = win_counts[0] / sum(win_counts) if sum(win_counts) > 0 else 0
            lvl_rate = level_counts[0] / sum(level_counts) if sum(level_counts) > 0 else 0
            print(f'  vs {opp_label}: win={win_rate:.1%}  level={lvl_rate:.1%}  opp_pts={per_opponent_stats[opp_label]["avg_points"]}')
        
        iterations += games

        if not eval_only and iterations % 100000 < games:
            agent.save_snapshot(iterations)

        if not eval_only:
            # Aggregate across all opponents for backwards-compatible summary fields
            all_win = [0, 0]
            all_lvl = [0, 0]
            all_pts = [[], []]
            for s in per_opponent_stats.values():
                all_win[0] += s['win_counts'][0]; all_win[1] += s['win_counts'][1]
                all_lvl[0] += s['level_counts'][0]; all_lvl[1] += s['level_counts'][1]
                all_pts[0].append(s['avg_points'][0]); all_pts[1].append(s['avg_points'][1])
            stats.append({
                "iterations": iterations,
                "win_counts": all_win[0] / sum(all_win) if sum(all_win) > 0 else 0,
                "level_counts": all_lvl[0] / sum(all_lvl) if sum(all_lvl) > 0 else 0,
                "avg_points": [float(np.mean(all_pts[0])), float(np.mean(all_pts[1]))],
                "per_opponent": per_opponent_stats,
                "training_time": training_time,
                "games_per_sec": games / training_time,
                "queue_wait": t_queue,
                "learn_time": t_learn,
                "avg_sim_per_game": float(np.mean(avg_sim_times)),
                "avg_infer_per_game": float(np.mean(avg_infer_times)),
            })
            with open(f'{model_folder}/stats.pkl', mode='w+b') as f:
                pickle.dump(stats, f)
            torch.save({
                    'agent_type': agent_type,
                    'iterations': iterations,
                    **agent.optimizer_states(),
                    'oracle_duration': oracle_duration_input,
                    'dynamic_encoding': dynamic_encoding
                }, f'{model_folder}/state.pkl')
        else:
            break
    
    for c in ctx.active_children():
        c.kill()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train loop')
    parser.add_argument('--agent-type', type=str, default='dmc', choices=['dmc', 'dqn', 'dqnsac', 'dmcsac'])
    parser.add_argument('--games', type=int, default=500)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--eval-size', type=int, default=300)
    parser.add_argument('--model-folder', type=str, default='pretrained')
    parser.add_argument('--compare', type=str, default='')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--decay-factor', type=float, default=1.2)
    parser.add_argument('--random-seed', type=int, default=1)
    parser.add_argument('--disable-chaodi', action='store_true')
    parser.add_argument('--enable-combos', action='store_true')
    parser.add_argument('--single-process', action='store_true')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Epsilon floor (minimum exploration rate)')
    parser.add_argument('--epsilon-start', type=float, default=0.5, help='Initial epsilon before decay')
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--kitty-agent', type=str, default='fc', choices=['fc', 'argmax', 'rnn', 'lstm'])
    parser.add_argument('--eval-agent', type=str, default='random', choices=['random', 'interactive', 'strategic'])
    parser.add_argument('--eval-agents', type=str, nargs='*', default=None, help="List of eval agent specs: 'random', 'strategic', 'model:<folder>'")
    parser.add_argument('--learn-from-eval', action='store_true')
    parser.add_argument('--reuse-times', type=int, default=0)
    parser.add_argument('--oracle-duration', type=int, default=0)
    parser.add_argument('--max-games', type=int, default=500000)
    parser.add_argument('--combo-penalty', type=float, default=0.1)
    parser.add_argument('--static-encoding', action='store_true')
    parser.add_argument('--combo-alternation', action='store_true')
    parser.add_argument('--actor-processes', type=int, default=6)
    parser.add_argument('--eval-processes', type=int, default=7)
    parser.add_argument('--model-architecture', type=str, default='mlp', choices=['mlp', 'transformer'])
    args = parser.parse_args()
    train(args.agent_type, args.games, args.model_folder, args.eval_only, args.eval_size, args.compare, args.discount, args.decay_factor, not args.disable_chaodi, args.enable_combos, args.verbose, args.random_seed, args.single_process, args.epsilon, args.tau, args.kitty_agent, args.eval_agent, args.learn_from_eval, args.reuse_times, args.oracle_duration, args.max_games, args.combo_penalty, not args.static_encoding, args.combo_alternation, args.actor_processes, args.eval_processes, args.model_architecture, args.eval_agents, args.epsilon_start)