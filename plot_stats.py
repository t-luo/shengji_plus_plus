"""
Plot training stats from stats.pkl.

Usage:
  # Single model
  python plot_stats.py --model-folder exps/transformer_v2

  # Compare multiple models
  python plot_stats.py --model-folder exps/lstm_baseline --compare exps/transformer_v2 exps/transformer_v3
"""
import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model-folder', type=str, required=True, help='Primary model folder')
parser.add_argument('--compare', type=str, nargs='*', default=[], help='Additional model folders to overlay')
args = parser.parse_args()

def load_stats(folder):
    with open(f'{folder}/stats.pkl', 'rb') as f:
        stats = pickle.load(f)
    return {
        'label':         folder.split('/')[-1].split('\\')[-1],
        'iterations':    [s['iterations'] for s in stats],
        'level_counts':  [s['level_counts'] for s in stats],
        'win_counts':    [s['win_counts'] for s in stats],
        'avg_points_ns': [s['avg_points'][0] for s in stats],
        'avg_points_we': [s['avg_points'][1] for s in stats],
    }

all_models = [load_stats(args.model_folder)] + [load_stats(f) for f in args.compare]

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle('Training Progress', fontsize=14)

    for m in all_models:
        axes[0].plot(m['iterations'], m['level_counts'], '-', linewidth=1.5, label=m['label'])

    axes[0].axhline(0.5, color='gray', linestyle='--', linewidth=1)
    axes[0].set_ylabel('Leveling Rate')
    axes[0].set_ylim(0.75, 1.0)
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Leveling Rate Over Training Games')

    for m in all_models:
        axes[1].plot(m['iterations'], m['avg_points_ns'], '-',  linewidth=1.2, label=f"{m['label']} (N/S dealer)")
        axes[1].plot(m['iterations'], m['avg_points_we'], '--', linewidth=1.2, label=f"{m['label']} (W/E dealer)")

    axes[1].axhline(80, color='gray', linestyle='--', linewidth=1, label='80pt win threshold')
    axes[1].set_ylabel('Avg Opposition Points')
    axes[1].set_xlabel('Training Games')
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Average Opposition Points (higher offense / lower defense = better)')

    plt.tight_layout()
    out_path = f'{args.model_folder}/training_curve.png'
    plt.savefig(out_path, dpi=150)
    print(f'Saved to {out_path}')
    plt.show()

except ImportError:
    print(f"\n{'Model':<20} {'Iter':>8}  {'LvlRate':>8}  {'Win%':>6}  {'Pts(NS)':>8}  {'Pts(WE)':>8}")
    print('-' * 65)
    for m in all_models:
        for i in range(len(m['iterations'])):
            print(f"{m['label']:<20} {m['iterations'][i]:>8}  {m['level_counts'][i]:>8.3f}  "
                  f"{m['win_counts'][i]:>6.3f}  {m['avg_points_ns'][i]:>8.1f}  {m['avg_points_we'][i]:>8.1f}")
