"""
Plot training stats from stats.pkl.
Usage: python plot_stats.py --model-folder exps/transformer_v2
"""
import argparse
import pickle
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model-folder', type=str, required=True)
args = parser.parse_args()

with open(f'{args.model_folder}/stats.pkl', 'rb') as f:
    stats = pickle.load(f)

print(f"Loaded {len(stats)} checkpoints")

iterations = [s['iterations'] for s in stats]
win_counts = [s['win_counts'] for s in stats]
level_counts = [s['level_counts'] for s in stats]
avg_points_ns = [s['avg_points'][0] for s in stats]
avg_points_we = [s['avg_points'][1] for s in stats]
games_per_sec = [s.get('games_per_sec', None) for s in stats]

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(f'Training Progress — {args.model_folder}', fontsize=14)

    # Win rate
    axes[0].plot(iterations, win_counts, 'b-o', markersize=4, label='Win rate (N/S)')
    axes[0].axhline(0.5, color='gray', linestyle='--', linewidth=1, label='50% baseline')
    axes[0].set_ylabel('Win rate')
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Win Rate Over Time')

    # Level counts (margin of victory)
    axes[1].plot(iterations, level_counts, 'g-o', markersize=4, label='Level win rate')
    axes[1].axhline(0.5, color='gray', linestyle='--', linewidth=1)
    axes[1].set_ylabel('Level win rate')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Level Win Rate (margin of victory)')

    # Opposition points
    axes[2].plot(iterations, avg_points_ns, 'r-o', markersize=4, label='Opp points when N/S is dealer')
    axes[2].plot(iterations, avg_points_we, 'm-o', markersize=4, label='Opp points when W/E is dealer')
    axes[2].axhline(80, color='gray', linestyle='--', linewidth=1, label='80pt threshold')
    axes[2].set_ylabel('Avg opposition points')
    axes[2].set_xlabel('Training games')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Average Opposition Points (lower = better defense, higher = better offense)')

    plt.tight_layout()
    out_path = f'{args.model_folder}/training_curve.png'
    plt.savefig(out_path, dpi=150)
    print(f'Saved to {out_path}')
    plt.show()

except ImportError:
    # matplotlib not available — print a text summary instead
    print(f"\n{'Iter':>8}  {'Win%':>6}  {'Lvl%':>6}  {'Pts(NS)':>8}  {'Pts(WE)':>8}  {'G/s':>6}")
    print('-' * 55)
    for s in stats:
        gps = s.get('games_per_sec', 0) or 0
        print(f"{s['iterations']:>8}  {s['win_counts']:>6.3f}  {s['level_counts']:>6.3f}  "
              f"{s['avg_points'][0]:>8.1f}  {s['avg_points'][1]:>8.1f}  {gps:>6.1f}")
