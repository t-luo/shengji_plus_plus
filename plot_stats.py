"""
Plot training stats from stats.pkl.

Usage:
  # Single model
  python plot_stats.py --model-folder exps/transformer_v2

  # Compare two models
  python plot_stats.py --model-folder exps/lstm_baseline --compare exps/transformer_v2
"""
import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model-folder', type=str, required=True, help='Primary model folder')
parser.add_argument('--compare', type=str, default=None, help='Optional second model folder to overlay')
args = parser.parse_args()

def load_stats(folder):
    with open(f'{folder}/stats.pkl', 'rb') as f:
        stats = pickle.load(f)
    return {
        'iterations':   [s['iterations'] for s in stats],
        'level_counts': [s['level_counts'] for s in stats],
        'win_counts':   [s['win_counts'] for s in stats],
        'avg_points_ns': [s['avg_points'][0] for s in stats],
        'avg_points_we': [s['avg_points'][1] for s in stats],
    }

primary = load_stats(args.model_folder)
compare = load_stats(args.compare) if args.compare else None

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Training Progress', fontsize=14)

    primary_label = args.model_folder.split('/')[-1]
    compare_label = args.compare.split('/')[-1] if args.compare else None

    # --- Leveling rate ---
    axes[0].plot(primary['iterations'], primary['level_counts'], '-', linewidth=1.5, label=primary_label)
    if compare:
        axes[0].plot(compare['iterations'], compare['level_counts'], '-', linewidth=1.5, label=compare_label)
    axes[0].axhline(0.5, color='gray', linestyle='--', linewidth=1, label='50% baseline')
    axes[0].set_ylabel('Leveling Rate')
    axes[0].set_ylim(0.75, 1.0)
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Leveling Rate Over Training Games')

    # --- Average opposition points ---
    axes[1].plot(primary['iterations'], primary['avg_points_ns'], '-', linewidth=1.5,
                 label=f'{primary_label} (N/S dealer)')
    axes[1].plot(primary['iterations'], primary['avg_points_we'], '--', linewidth=1.5,
                 label=f'{primary_label} (W/E dealer)')
    if compare:
        axes[1].plot(compare['iterations'], compare['avg_points_ns'], '-', linewidth=1.5,
                     label=f'{compare_label} (N/S dealer)')
        axes[1].plot(compare['iterations'], compare['avg_points_we'], '--', linewidth=1.5,
                     label=f'{compare_label} (W/E dealer)')
    axes[1].axhline(80, color='gray', linestyle='--', linewidth=1, label='80pt win threshold')
    axes[1].set_ylabel('Avg Opposition Points')
    axes[1].set_xlabel('Training Games')
    axes[1].legend(fontsize=8)
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
    for label, data in [(primary_label, primary)] + ([(compare_label, compare)] if compare else []):
        for i in range(len(data['iterations'])):
            print(f"{label:<20} {data['iterations'][i]:>8}  {data['level_counts'][i]:>8.3f}  "
                  f"{data['win_counts'][i]:>6.3f}  {data['avg_points_ns'][i]:>8.1f}  {data['avg_points_we'][i]:>8.1f}")
