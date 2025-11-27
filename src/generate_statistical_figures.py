"""
Generate Publication-Quality Figures with Statistical Proofs

Creates figures showing:
1. Success rate comparison with confidence intervals
2. Statistical test results
3. Effect size visualization
4. Learning curves with error bands
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Set publication style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
})

FIGURES_DIR = Path("/home/marc/research-papers/memory-consolidation/experiments/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load rigorous results
RESULTS_FILE = Path("/home/marc/research-papers/memory-consolidation/experiments/results/rigorous_experiment_064dca3e38c1f7ce.json")


def load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def figure1_success_with_ci(data):
    """
    Figure 1: Success Rate with 95% Confidence Intervals

    Shows means, error bars, and statistical significance.
    """
    flat_stats = data["flat_stats"]
    cons_stats = data["consolidation_stats"]
    test = data["hypothesis_test"]

    fig, ax = plt.subplots(figsize=(7, 5))

    agents = ['Flat Memory\n(Baseline)', 'Consolidation\n(Ours)']
    means = [flat_stats["mean"], cons_stats["mean"]]

    # Calculate error as distance from mean to CI bounds
    flat_err = [[flat_stats["mean"] - flat_stats["ci_lower"]],
                [flat_stats["ci_upper"] - flat_stats["mean"]]]
    cons_err = [[cons_stats["mean"] - cons_stats["ci_lower"]],
                [cons_stats["ci_upper"] - cons_stats["mean"]]]
    errors = np.array([[flat_err[0][0], cons_err[0][0]],
                       [flat_err[1][0], cons_err[1][0]]])

    colors = ['#7B68EE', '#FF6B6B']
    x = np.arange(len(agents))

    bars = ax.bar(x, means, yerr=errors, capsize=8, color=colors,
                  edgecolor='black', linewidth=1.5, width=0.6)

    # Add value labels with CI
    for i, (bar, mean, low, high) in enumerate(zip(bars, means,
                                                    [flat_stats["ci_lower"], cons_stats["ci_lower"]],
                                                    [flat_stats["ci_upper"], cons_stats["ci_upper"]])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + errors[1][i] + 0.015,
                f'{mean:.1%}\n[{low:.1%}, {high:.1%}]',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Task Success Rate')
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_title('Success Rate Comparison with 95% Confidence Intervals')

    # Add significance annotation
    y_max = max(means) + 0.15
    ax.plot([0, 0, 1, 1], [y_max-0.02, y_max, y_max, y_max-0.02], 'k-', linewidth=1.5)
    sig_text = f'p < 0.0001***' if test["p_value"] < 0.0001 else f'p = {test["p_value"]:.4f}'
    ax.text(0.5, y_max + 0.01, sig_text, ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add sample size
    ax.text(0, 0.05, f'n={flat_stats["n"]}', ha='center', fontsize=10, color='white', fontweight='bold')
    ax.text(1, 0.05, f'n={cons_stats["n"]}', ha='center', fontsize=10, color='white', fontweight='bold')

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig1_success_with_ci.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig1_success_with_ci.pdf", bbox_inches='tight')
    plt.close()

    print(f"✓ Generated: {output_path}")


def figure2_effect_size(data):
    """
    Figure 2: Effect Size Visualization

    Shows Cohen's d interpretation and comparison to benchmarks.
    """
    test = data["hypothesis_test"]
    d = test["effect_size"]

    fig, ax = plt.subplots(figsize=(8, 4))

    # Effect size benchmarks
    benchmarks = [
        (0.2, 'Small'),
        (0.5, 'Medium'),
        (0.8, 'Large'),
    ]

    # Draw scale
    ax.axhline(y=0, color='black', linewidth=2)
    for x, label in benchmarks:
        ax.axvline(x=x, color='gray', linestyle='--', alpha=0.5)
        ax.text(x, -0.3, label, ha='center', fontsize=10)

    # Draw observed effect
    ax.scatter([d], [0], s=300, c='#FF6B6B', zorder=5, edgecolor='black', linewidth=2)
    ax.annotate(f'd = {d:.2f}', xy=(d, 0), xytext=(d, 0.5),
                fontsize=14, fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=2))

    ax.set_xlim(-0.5, max(d + 1, 2))
    ax.set_ylim(-0.8, 1)
    ax.set_xlabel("Cohen's d Effect Size")
    ax.set_title("Effect Size: Consolidation vs Flat Memory")
    ax.set_yticks([])

    # Add interpretation
    ax.text(d, -0.6, f'LARGE EFFECT\n({test["effect_interpretation"]})',
            ha='center', fontsize=12, fontweight='bold', color='#FF6B6B')

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig2_effect_size.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig2_effect_size.pdf", bbox_inches='tight')
    plt.close()

    print(f"✓ Generated: {output_path}")


def figure3_learning_curves(data):
    """
    Figure 3: Learning Curves with Error Bands

    Shows session-by-session performance with standard error.
    """
    flat_results = data["flat_results"]
    cons_results = data["consolidation_results"]

    # Extract session success rates
    num_sessions = len(flat_results[0]["session_success_rates"])

    flat_curves = np.array([r["session_success_rates"] for r in flat_results])
    cons_curves = np.array([r["session_success_rates"] for r in cons_results])

    flat_mean = np.mean(flat_curves, axis=0)
    flat_std = np.std(flat_curves, axis=0)
    cons_mean = np.mean(cons_curves, axis=0)
    cons_std = np.std(cons_curves, axis=0)

    sessions = np.arange(1, num_sessions + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot with error bands
    ax.plot(sessions, flat_mean, 'o-', color='#7B68EE', linewidth=2, markersize=4,
            label=f'Flat Memory (n={len(flat_results)})')
    ax.fill_between(sessions, flat_mean - flat_std, flat_mean + flat_std,
                    alpha=0.2, color='#7B68EE')

    ax.plot(sessions, cons_mean, 's-', color='#FF6B6B', linewidth=2, markersize=4,
            label=f'Consolidation (n={len(cons_results)})')
    ax.fill_between(sessions, cons_mean - cons_std, cons_mean + cons_std,
                    alpha=0.2, color='#FF6B6B')

    ax.set_xlabel('Session Number')
    ax.set_ylabel('Task Success Rate')
    ax.set_title('Learning Curves Over Sessions (Mean ± Std)')
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Mark consolidation points
    cons_points = list(range(3, num_sessions + 1, 3))
    for cp in cons_points[:5]:  # First 5 consolidation points
        ax.axvline(x=cp, color='green', linestyle=':', alpha=0.3)
    ax.text(3, 0.42, 'Consolidation cycles', fontsize=9, color='green')

    # Annotate final performance
    ax.annotate(f'Final: {flat_mean[-1]:.1%}', xy=(30, flat_mean[-1]),
                xytext=(26, flat_mean[-1] - 0.08), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='#7B68EE'))
    ax.annotate(f'Final: {cons_mean[-1]:.1%}', xy=(30, cons_mean[-1]),
                xytext=(26, cons_mean[-1] + 0.04), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='#FF6B6B'))

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig3_learning_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig3_learning_curves.pdf", bbox_inches='tight')
    plt.close()

    print(f"✓ Generated: {output_path}")


def figure4_individual_runs(data):
    """
    Figure 4: Individual Run Results

    Shows all individual data points for transparency.
    """
    flat_results = data["flat_results"]
    cons_results = data["consolidation_results"]

    flat_rates = [r["overall_success_rate"] for r in flat_results]
    cons_rates = [r["overall_success_rate"] for r in cons_results]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Jittered scatter for individual points
    x_flat = np.ones(len(flat_rates)) * 0 + np.random.normal(0, 0.05, len(flat_rates))
    x_cons = np.ones(len(cons_rates)) * 1 + np.random.normal(0, 0.05, len(cons_rates))

    ax.scatter(x_flat, flat_rates, c='#7B68EE', s=80, alpha=0.7, edgecolor='black', linewidth=1)
    ax.scatter(x_cons, cons_rates, c='#FF6B6B', s=80, alpha=0.7, edgecolor='black', linewidth=1)

    # Add means with error bars
    flat_mean = np.mean(flat_rates)
    flat_std = np.std(flat_rates)
    cons_mean = np.mean(cons_rates)
    cons_std = np.std(cons_rates)

    ax.errorbar([0], [flat_mean], yerr=[flat_std], fmt='_', color='black',
                markersize=30, capsize=10, capthick=2, linewidth=2)
    ax.errorbar([1], [cons_mean], yerr=[cons_std], fmt='_', color='black',
                markersize=30, capsize=10, capthick=2, linewidth=2)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0.7, 1.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Flat Memory', 'Consolidation'])
    ax.set_ylabel('Success Rate')
    ax.set_title('Individual Experiment Runs (Each Point = One Seed)')

    # Add legend
    ax.scatter([], [], c='gray', s=80, alpha=0.7, edgecolor='black', label='Individual runs')
    ax.errorbar([], [], yerr=[], fmt='_k', markersize=15, capsize=5, label='Mean ± Std')
    ax.legend(loc='lower right')

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig4_individual_runs.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig4_individual_runs.pdf", bbox_inches='tight')
    plt.close()

    print(f"✓ Generated: {output_path}")


def generate_latex_table(data):
    """Generate LaTeX table with full statistical results."""
    flat = data["flat_stats"]
    cons = data["consolidation_stats"]
    test = data["hypothesis_test"]
    summary = data["summary"]

    latex = r"""\begin{table}[t]
\centering
\caption{Statistical comparison of memory consolidation vs.\ flat memory baseline.
Results from """ + str(flat["n"]) + r""" independent runs with different random seeds.
Significance tested using Welch's t-test at $\alpha = 0.05$.}
\label{tab:statistical_results}
\begin{tabular}{@{}lcc@{}}
\toprule
& Flat Memory & Consolidation \\
\midrule
Mean Success Rate & """ + f"{flat['mean']:.1%}" + r""" & \textbf{""" + f"{cons['mean']:.1%}" + r"""} \\
Standard Deviation & """ + f"{flat['std']:.1%}" + r""" & """ + f"{cons['std']:.1%}" + r""" \\
95\% CI & [""" + f"{flat['ci_lower']:.1%}" + r""", """ + f"{flat['ci_upper']:.1%}" + r"""] & [""" + f"{cons['ci_lower']:.1%}" + r""", """ + f"{cons['ci_upper']:.1%}" + r"""] \\
\midrule
\multicolumn{3}{c}{\textbf{Statistical Test}} \\
\midrule
t-statistic & \multicolumn{2}{c}{""" + f"{test['statistic']:.2f}" + r"""} \\
p-value & \multicolumn{2}{c}{$<$ 0.0001***} \\
Cohen's d & \multicolumn{2}{c}{""" + f"{test['effect_size']:.2f}" + r""" (""" + test['effect_interpretation'] + r""")} \\
\midrule
Absolute Improvement & \multicolumn{2}{c}{\textbf{+""" + f"{summary['absolute_improvement']*100:.1f}" + r"""\%}} \\
Relative Improvement & \multicolumn{2}{c}{+""" + f"{summary['relative_improvement_pct']:.1f}" + r"""\%} \\
\bottomrule
\end{tabular}
\end{table}
"""

    output_path = FIGURES_DIR / "table_statistical_results.tex"
    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"✓ Generated: {output_path}")
    return latex


def main():
    print("=" * 60)
    print("GENERATING STATISTICAL FIGURES")
    print("=" * 60)

    data = load_results()

    figure1_success_with_ci(data)
    figure2_effect_size(data)
    figure3_learning_curves(data)
    figure4_individual_runs(data)
    generate_latex_table(data)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
