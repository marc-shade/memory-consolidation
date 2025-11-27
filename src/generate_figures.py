"""
Generate Figures for Memory Consolidation Paper

Creates publication-quality figures from experiment results.
Uses matplotlib with no display (saves to files).
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

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

# Output directory
FIGURES_DIR = Path("/home/marc/research-papers/memory-consolidation/experiments/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load results
RESULTS_FILE = Path("/home/marc/research-papers/memory-consolidation/experiments/results/realistic_results.json")


def load_results():
    """Load experiment results."""
    with open(RESULTS_FILE) as f:
        return json.load(f)


def figure1_success_comparison(data):
    """
    Figure 1: Overall Success Rate Comparison

    Bar chart comparing consolidation vs flat memory.
    """
    analysis = data["analysis"]

    fig, ax = plt.subplots(figsize=(6, 4))

    agents = ['Flat Memory\n(Baseline)', 'Consolidation\n(Ours)']
    means = [
        analysis["flat_memory"]["mean_success_rate"],
        analysis["consolidation"]["mean_success_rate"]
    ]
    stds = [
        analysis["flat_memory"]["std_success_rate"],
        analysis["consolidation"]["std_success_rate"]
    ]

    colors = ['#7B68EE', '#FF6B6B']  # Purple and coral

    bars = ax.bar(agents, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.1%}', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Task Success Rate')
    ax.set_ylim(0, 1.05)
    ax.set_title('Overall Task Success Rate')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='Random baseline')

    # Add improvement annotation
    improvement = analysis["comparison"]["success_improvement_percent"]
    ax.annotate(f'+{improvement:.1f}%', xy=(1, means[1]), xytext=(1.3, means[0]),
                fontsize=12, fontweight='bold', color='green',
                arrowprops=dict(arrowstyle='->', color='green'))

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig1_success_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig1_success_comparison.pdf", bbox_inches='tight')
    plt.close()

    print(f"✓ Generated: {output_path}")
    return output_path


def figure2_learning_curves(data):
    """
    Figure 2: Learning Curves Over Sessions

    Shows how each agent type improves (or doesn't) over time.
    """
    results = data["results"]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Calculate average learning curve for each agent type
    for agent_type, exp_results in results.items():
        all_sessions = [r["sessions"] for r in exp_results]
        num_sessions = len(all_sessions[0])

        # Average across seeds
        avg_by_session = []
        std_by_session = []
        for i in range(num_sessions):
            rates = [s[i]["success_rate"] for s in all_sessions if i < len(s)]
            avg_by_session.append(np.mean(rates))
            std_by_session.append(np.std(rates))

        avg_by_session = np.array(avg_by_session)
        std_by_session = np.array(std_by_session)
        sessions = np.arange(1, num_sessions + 1)

        # Plot
        label = "Consolidation (Ours)" if agent_type == "consolidation" else "Flat Memory (Baseline)"
        color = '#FF6B6B' if agent_type == "consolidation" else '#7B68EE'
        marker = 's' if agent_type == "consolidation" else 'o'

        ax.plot(sessions, avg_by_session, marker=marker, markersize=4,
                label=label, color=color, linewidth=2)
        ax.fill_between(sessions, avg_by_session - std_by_session,
                       avg_by_session + std_by_session, alpha=0.2, color=color)

    ax.set_xlabel('Session Number')
    ax.set_ylabel('Task Success Rate')
    ax.set_title('Learning Curves: Success Rate Over Sessions')
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Add consolidation markers
    ax.axvline(x=3, color='green', linestyle=':', alpha=0.3, linewidth=1)
    ax.axvline(x=6, color='green', linestyle=':', alpha=0.3, linewidth=1)
    ax.axvline(x=9, color='green', linestyle=':', alpha=0.3, linewidth=1)
    ax.text(3, 0.42, 'C', fontsize=8, color='green', ha='center')
    ax.text(6, 0.42, 'C', fontsize=8, color='green', ha='center')
    ax.text(9, 0.42, 'C', fontsize=8, color='green', ha='center')
    ax.text(12, 0.42, '(C = Consolidation)', fontsize=8, color='green', ha='left')

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig2_learning_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig2_learning_curves.pdf", bbox_inches='tight')
    plt.close()

    print(f"✓ Generated: {output_path}")
    return output_path


def figure3_memory_growth(data):
    """
    Figure 3: Memory Growth Over Sessions

    Shows how memory accumulates with forgetting vs without.
    """
    results = data["results"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for agent_type, exp_results in results.items():
        all_sessions = [r["sessions"] for r in exp_results]
        num_sessions = len(all_sessions[0])

        # Average memory count across seeds
        avg_memory = []
        for i in range(num_sessions):
            counts = [s[i]["memory_count"] for s in all_sessions if i < len(s)]
            avg_memory.append(np.mean(counts))

        sessions = np.arange(1, num_sessions + 1)

        label = "Consolidation (with forgetting)" if agent_type == "consolidation" else "Flat Memory (no forgetting)"
        color = '#FF6B6B' if agent_type == "consolidation" else '#7B68EE'
        linestyle = '-' if agent_type == "consolidation" else '--'

        ax.plot(sessions, avg_memory, label=label, color=color, linewidth=2, linestyle=linestyle)

    ax.set_xlabel('Session Number')
    ax.set_ylabel('Total Memory Count')
    ax.set_title('Memory Growth Over Sessions')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Highlight that consolidation creates semantic + procedural memories
    ax.annotate('Semantic concepts\n+ procedures', xy=(25, 500), fontsize=9,
               color='#FF6B6B', style='italic')
    ax.annotate('Raw episodic\nonly', xy=(25, 450), fontsize=9,
               color='#7B68EE', style='italic')

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig3_memory_growth.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig3_memory_growth.pdf", bbox_inches='tight')
    plt.close()

    print(f"✓ Generated: {output_path}")
    return output_path


def figure4_consolidation_effects(data):
    """
    Figure 4: Consolidation Effects Analysis

    Shows patterns extracted, procedures learned, memories forgotten.
    """
    results = data["results"]

    # Get consolidation stats
    cons_results = results.get("consolidation", [])

    if not cons_results:
        print("✗ No consolidation results found")
        return None

    all_consolidations = []
    for r in cons_results:
        all_consolidations.extend(r.get("consolidations", []))

    if not all_consolidations:
        print("✗ No consolidation data found")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel A: Memories forgotten per cycle
    forgotten = [c.get("memories_forgotten", 0) for c in all_consolidations]
    ax1 = axes[0]
    ax1.bar(range(1, len(forgotten) + 1), forgotten, color='#7B68EE', edgecolor='black')
    ax1.set_xlabel('Consolidation Cycle')
    ax1.set_ylabel('Memories Forgotten')
    ax1.set_title('A. Memory Forgetting')

    # Panel B: Concepts extracted cumulative
    concepts = []
    cumulative = 0
    for c in all_consolidations:
        cumulative += c.get("concepts_extracted", 0)
        concepts.append(cumulative)
    ax2 = axes[1]
    ax2.plot(range(1, len(concepts) + 1), concepts, 'o-', color='#FF6B6B', linewidth=2, markersize=6)
    ax2.set_xlabel('Consolidation Cycle')
    ax2.set_ylabel('Cumulative Semantic Concepts')
    ax2.set_title('B. Pattern Extraction')
    ax2.fill_between(range(1, len(concepts) + 1), concepts, alpha=0.2, color='#FF6B6B')

    # Panel C: Procedures learned cumulative
    procedures = []
    cumulative = 0
    for c in all_consolidations:
        cumulative += c.get("procedures_learned", 0)
        procedures.append(cumulative)
    ax3 = axes[2]
    ax3.plot(range(1, len(procedures) + 1), procedures, 's-', color='#2ECC71', linewidth=2, markersize=6)
    ax3.set_xlabel('Consolidation Cycle')
    ax3.set_ylabel('Cumulative Procedures')
    ax3.set_title('C. Procedure Learning')
    ax3.fill_between(range(1, len(procedures) + 1), procedures, alpha=0.2, color='#2ECC71')

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig4_consolidation_effects.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig4_consolidation_effects.pdf", bbox_inches='tight')
    plt.close()

    print(f"✓ Generated: {output_path}")
    return output_path


def generate_table1(data):
    """
    Table 1: Main Results Comparison (LaTeX format)
    """
    analysis = data["analysis"]

    flat = analysis["flat_memory"]
    cons = analysis["consolidation"]
    comp = analysis["comparison"]

    latex = r"""
\begin{table}[t]
\centering
\caption{Main experimental results comparing memory consolidation against flat memory baseline.
Results averaged over 5 random seeds. \textbf{Bold} indicates best performance.}
\label{tab:main_results}
\begin{tabular}{@{}lcc@{}}
\toprule
Metric & Flat Memory & Consolidation (Ours) \\
\midrule
Task Success Rate & """ + f"{flat['mean_success_rate']:.1%}" + r""" $\pm$ """ + f"{flat['std_success_rate']:.1%}" + r""" & \textbf{""" + f"{cons['mean_success_rate']:.1%}" + r"""} $\pm$ """ + f"{cons['std_success_rate']:.1%}" + r""" \\
Avg Retrieval Score & """ + f"{flat['mean_retrieval_score']:.3f}" + r""" & \textbf{""" + f"{cons['mean_retrieval_score']:.3f}" + r"""} \\
Final Memory Count & \textbf{""" + f"{flat['mean_memory_count']:.0f}" + r"""} & """ + f"{cons['mean_memory_count']:.0f}" + r""" \\
\midrule
\textbf{Improvement} & \multicolumn{2}{c}{\textbf{+""" + f"{comp['success_improvement_percent']:.1f}" + r"""\%}} \\
\bottomrule
\end{tabular}
\end{table}
"""

    output_path = FIGURES_DIR / "table1_main_results.tex"
    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"✓ Generated: {output_path}")
    return latex


def main():
    """Generate all figures and tables."""
    print("=" * 60)
    print("GENERATING PAPER FIGURES")
    print("=" * 60)

    # Load results
    data = load_results()

    # Generate figures
    figure1_success_comparison(data)
    figure2_learning_curves(data)
    figure3_memory_growth(data)
    figure4_consolidation_effects(data)

    # Generate table
    generate_table1(data)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
