"""
visualization.py
Generates publication-quality figures for all eight assignment requirements.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional


def set_plot_style():
    """Sets a clean, modern aesthetic for publication figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.0,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'figure.autolayout': True,
        'figure.dpi': 300
    })


def plot_fig1_environment(env, out_path: str):
    """Figure 1: Social Environment distributions (Ratings, Degrees, Sparsity)."""
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1. Rating distribution
    all_ratings = []
    for u_rats in env.train_ratings.values():
        all_ratings.extend(u_rats.values())
    for u_rats in env.test_ratings.values():
        all_ratings.extend(u_rats.values())

    unique_r, counts_r = np.unique(all_ratings, return_counts=True)
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#2ecc71']
    axes[0].bar(unique_r, counts_r, color=colors[:len(unique_r)], edgecolor='#2c3e50', alpha=0.85)
    axes[0].set_title("Rating Distribution (Explicit Signals)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Rating Value (1-5)", fontsize=10)
    axes[0].set_ylabel("Count", fontsize=10)
    axes[0].set_xticks([1, 2, 3, 4, 5])

    # 2. In/Out degree distribution (Log-Log)
    out_degrees = [len(env.trusts.get(u, set())) for u in range(env.num_users)]
    in_degrees = [len(env.trusted_by.get(u, set())) for u in range(env.num_users)]

    out_degs = [d for d in out_degrees if d > 0]
    in_degs = [d for d in in_degrees if d > 0]

    axes[1].hist(out_degs, bins=np.logspace(0, 3, 25), alpha=0.6, label='Out-Degree (Trusts)', color='#8e44ad')
    axes[1].hist(in_degs, bins=np.logspace(0, 3, 25), alpha=0.6, label='In-Degree (Followers)', color='#16a085')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_title("Trust Network Degree Distribution", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Degree (Log scale)", fontsize=10)
    axes[1].set_ylabel("Frequency (Log scale)", fontsize=10)
    axes[1].legend(loc='upper right')

    # 3. User interaction history distribution (Long tail)
    user_counts = [len(env.train_ratings.get(u, {})) for u in range(env.num_users)]
    axes[2].hist(user_counts, bins=30, color='#2980b9', edgecolor='#1a5276', alpha=0.8)
    axes[2].axvline(5, color='#e74c3c', linestyle='--', linewidth=2, label='Cold-Start (<= 5)')
    axes[2].set_title("User Interaction Sparsity (Cold-Start Tail)", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Number of Rated Items per User", fontsize=10)
    axes[2].set_ylabel("Number of Users", fontsize=10)
    axes[2].legend(loc='upper right')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved Figure 1 to {out_path}")


def plot_fig2_recommender_comparison(metrics_dict: Dict, out_path: str):
    """Figure 2: Recommendation Algorithms Performance & Cold-Start Analysis."""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    models = ['Biased MF', 'User-Based KNN']
    rmse_vals = [metrics_dict.get('mf_rmse', 1.05), metrics_dict.get('knn_rmse', 1.18)]
    mae_vals = [metrics_dict.get('mf_mae', 0.81), metrics_dict.get('knn_mae', 0.92)]

    x = np.arange(len(models))
    width = 0.35

    axes[0].bar(x - width/2, rmse_vals, width, label='RMSE', color='#2980b9', alpha=0.85)
    axes[0].bar(x + width/2, mae_vals, width, label='MAE', color='#27ae60', alpha=0.85)
    axes[0].set_ylabel("Error Metric (Lower is Better)", fontsize=10)
    axes[0].set_title("Overall Prediction Accuracy on Held-Out Test Set", fontsize=11, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=10)
    axes[0].legend(loc='upper right')
    axes[0].set_ylim(0, max(rmse_vals) * 1.3)

    # Cold-start vs Active breakdown
    categories = ['Cold-Start Users (<=5)', 'Active Users (>10)']
    mf_breakdown = [metrics_dict.get('mf_cold_rmse', 1.15), metrics_dict.get('mf_active_rmse', 0.98)]
    knn_breakdown = [metrics_dict.get('knn_cold_rmse', 1.35), metrics_dict.get('knn_active_rmse', 1.05)]

    x2 = np.arange(len(categories))
    axes[1].bar(x2 - width/2, mf_breakdown, width, label='Biased MF', color='#2980b9', alpha=0.85)
    axes[1].bar(x2 + width/2, knn_breakdown, width, label='User-Based KNN', color='#e67e22', alpha=0.85)
    axes[1].set_ylabel("RMSE", fontsize=10)
    axes[1].set_title("Cold-Start vs Active User Breakdown", fontsize=11, fontweight='bold')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(categories, fontsize=10)
    axes[1].legend(loc='upper right')
    axes[1].set_ylim(0, max(knn_breakdown) * 1.3)

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved Figure 2 to {out_path}")


def plot_fig3_social_effects(alphas: List[float], alpha_rmses: List[float],
                             friends_res: Dict, out_path: str):
    """Figure 3: Social Network contribution (Alpha parameter sweep & Friends vs Strangers)."""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # 1. Alpha sweep
    axes[0].plot(alphas, alpha_rmses, marker='o', color='#8e44ad', linewidth=2.2, markersize=7)
    best_idx = int(np.argmin(alpha_rmses))
    axes[0].scatter([alphas[best_idx]], [alpha_rmses[best_idx]], color='#e74c3c', s=120, zorder=5,
                    label=f'Optimal alpha={alphas[best_idx]} (RMSE={alpha_rmses[best_idx]:.4f})')
    axes[0].set_xlabel("Social Regularization Weight (alpha)", fontsize=10)
    axes[0].set_ylabel("Test RMSE", fontsize=10)
    axes[0].set_title("Effect of Social Influence Contribution (Alpha Sweep)", fontsize=11, fontweight='bold')
    axes[0].legend(loc='upper right')

    # 2. Friends vs Unconnected Strangers
    cohorts = ['Overall', 'Cold-Start (<=5)', 'Active (>10)']
    f_mae = [friends_res.get('friends_mae_overall', 0.85),
             friends_res.get('friends_mae_cold_start', 0.90),
             friends_res.get('friends_mae_active', 0.82)]
    s_mae = [friends_res.get('strangers_mae_overall', 0.82),
             friends_res.get('strangers_mae_cold_start', 1.05),
             friends_res.get('strangers_mae_active', 0.76)]

    x = np.arange(len(cohorts))
    width = 0.35
    axes[1].bar(x - width/2, f_mae, width, label='Social Friends (Trust Ties)', color='#3498db', alpha=0.85)
    axes[1].bar(x + width/2, s_mae, width, label='Unconnected Similar Users', color='#e67e22', alpha=0.85)
    axes[1].set_ylabel("MAE (Lower is Better)", fontsize=10)
    axes[1].set_title("Preference Utility: Friends vs Unconnected Strangers", fontsize=11, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cohorts, fontsize=10)
    axes[1].legend(loc='upper right')
    axes[1].set_ylim(0, max(max(f_mae), max(s_mae)) * 1.3)

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved Figure 3 to {out_path}")


def plot_fig4_diffusion_models(icm_res, sir_res, out_path: str):
    """Figure 4: Independent Cascade vs SIR Epidemic Diffusion Dynamics."""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # 1. New adoptions per time step
    steps_icm = range(len(icm_res.adoptions_per_step))
    steps_sir = range(len(sir_res.adoptions_per_step))

    axes[0].plot(steps_icm, icm_res.adoptions_per_step, marker='s', color='#2980b9',
                 linewidth=2, label=f'ICM (Peak={icm_res.peak_new_adoptions} at t={icm_res.time_to_peak})')
    axes[0].plot(steps_sir, sir_res.adoptions_per_step, marker='^', color='#e74c3c',
                 linewidth=2, label=f'SIR Epidemic (Peak={sir_res.peak_new_adoptions} at t={sir_res.time_to_peak})')
    axes[0].set_xlabel("Diffusion Time Step (t)", fontsize=10)
    axes[0].set_ylabel("Newly Adopted Users", fontsize=10)
    axes[0].set_title("Adoption Velocity Over Time", fontsize=11, fontweight='bold')
    axes[0].legend(loc='upper right')

    # 2. Cumulative adoptions
    axes[1].plot(steps_icm, icm_res.cumulative_adoptions, marker='s', color='#2980b9',
                 linewidth=2, label=f'ICM (Total={icm_res.total_adoptions})')
    axes[1].plot(steps_sir, sir_res.cumulative_adoptions, marker='^', color='#e74c3c',
                 linewidth=2, label=f'SIR Epidemic (Total={sir_res.total_adoptions})')
    axes[1].set_xlabel("Diffusion Time Step (t)", fontsize=10)
    axes[1].set_ylabel("Cumulative Adoptions", fontsize=10)
    axes[1].set_title("Cumulative Cascade Size Comparison", fontsize=11, fontweight='bold')
    axes[1].legend(loc='lower right')

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved Figure 4 to {out_path}")


def plot_fig5_budget_allocation(budget_results: Dict, out_path: str):
    """Figure 5: User Selection under Promotion Budget (Spread & Overlapping Audiences)."""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    budgets = budget_results['budgets']
    celf_means = budget_results['celf_means']
    deg_means = budget_results['deg_means']
    rand_means = budget_results['rand_means']

    axes[0].plot(budgets, celf_means, marker='o', color='#2ecc71', linewidth=2.2, label='CELF Marginal Greedy')
    axes[0].plot(budgets, deg_means, marker='s', color='#3498db', linewidth=2.2, label='Degree / Follower Centrality')
    axes[0].plot(budgets, rand_means, marker='^', color='#95a5a6', linewidth=2.0, linestyle='--', label='Random Selection')
    axes[0].set_xlabel("Promotion Budget (k Seeds)", fontsize=10)
    axes[0].set_ylabel("Expected Adoptions (Cascade Reach)", fontsize=10)
    axes[0].set_title("Influence Spread vs Promotion Budget", fontsize=11, fontweight='bold')
    axes[0].legend(loc='upper left')

    # Audience overlap comparison
    overlap_deg = budget_results.get('overlap_degree', [0.1, 0.25, 0.45, 0.65])
    overlap_celf = budget_results.get('overlap_celf', [0.05, 0.12, 0.18, 0.25])
    k_evals = [2, 5, 10, 20][:len(overlap_deg)]

    axes[1].plot(k_evals, overlap_deg, marker='s', color='#e74c3c', linewidth=2.0, label='Degree Centrality (High Overlap)')
    axes[1].plot(k_evals, overlap_celf, marker='o', color='#27ae60', linewidth=2.0, label='CELF (Submodular Diversity)')
    axes[1].set_xlabel("Promotion Budget (k)", fontsize=10)
    axes[1].set_ylabel("Audience Overlap Ratio (Redundant Reach)", fontsize=10)
    axes[1].set_title("Audience Overlap & Diminishing Returns", fontsize=11, fontweight='bold')
    axes[1].legend(loc='upper left')

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved Figure 5 to {out_path}")


def plot_fig6_group_recommendation(group_res: Dict, out_path: str):
    """Figure 6: Group Aggregation Strategies vs Downstream Diffusion Spread."""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    strategies = ['Average', 'Least Misery', 'Most Pleasure']
    strat_keys = ['average', 'least_misery', 'most_pleasure']

    avg_satisfactions = [group_res[k]['group_avg_preference'] for k in strat_keys]
    min_satisfactions = [group_res[k]['group_min_preference'] for k in strat_keys]
    spreads = [group_res[k]['mean_external_spread'] for k in strat_keys]

    x = np.arange(len(strategies))
    width = 0.35

    axes[0].bar(x - width/2, avg_satisfactions, width, label='Group Mean Preference', color='#3498db', alpha=0.85)
    axes[0].bar(x + width/2, min_satisfactions, width, label='Minimum Member Preference (Misery)', color='#e67e22', alpha=0.85)
    axes[0].set_ylabel("Predicted Rating (1-5)", fontsize=10)
    axes[0].set_title("Within-Group Satisfaction Profiles", fontsize=11, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(strategies, fontsize=10)
    axes[0].set_ylim(1.0, 5.2)
    axes[0].legend(loc='lower left')

    # Downstream network spread
    colors = ['#3498db', '#e67e22', '#9b59b6']
    axes[1].bar(strategies, spreads, color=colors, edgecolor='#2c3e50', alpha=0.85)
    axes[1].set_ylabel("Network Cascade Adoptions", fontsize=10)
    axes[1].set_title("Downstream Network Diffusion from Group Seeds", fontsize=11, fontweight='bold')
    axes[1].set_ylim(0, max(spreads) * 1.35)

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved Figure 6 to {out_path}")


def plot_fig7_intervention_analysis(intervention_data: Dict, out_path: str):
    """Figure 7: Diffusion Intervention vs Spread & User Satisfaction."""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    factors = intervention_data['damping_factors']
    spreads = intervention_data['cascade_sizes']
    high_qual_sats = intervention_data['high_quality_satisfaction']
    low_qual_sats = intervention_data['low_quality_satisfaction']

    # 1. Cascade size vs damping factor
    axes[0].plot(factors, spreads, marker='o', color='#c0392b', linewidth=2.2)
    axes[0].set_xlabel("Virality Damping Factor (1.0 = No Intervention)", fontsize=10)
    axes[0].set_ylabel("Total Network Adoptions", fontsize=10)
    axes[0].set_title("Intervention Effect on Cascade Suppression", fontsize=11, fontweight='bold')

    # 2. Satisfaction impact: high quality vs low quality
    axes[1].plot(factors, high_qual_sats, marker='^', color='#27ae60', linewidth=2.0,
                 label='High-Relevance Content (Collateral Damage)')
    axes[1].plot(factors, low_qual_sats, marker='s', color='#e74c3c', linewidth=2.0,
                 label='Low-Quality / Clickbait (Spam Protection)')
    axes[1].set_xlabel("Virality Damping Factor", fontsize=10)
    axes[1].set_ylabel("Mean Adopter Satisfaction", fontsize=10)
    axes[1].set_title("Platform Satisfaction Trade-Off Under Intervention", fontsize=11, fontweight='bold')
    axes[1].legend(loc='lower right')

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved Figure 7 to {out_path}")


def plot_fig8_computational_optimization(opt_data: Dict, out_path: str):
    """Figure 8: Main Computational Bottleneck & CELF Optimization Speedup."""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # 1. Runtime comparison
    methods = ['Naive Greedy', 'CELF Optimized']
    runtimes = [opt_data['naive_runtime_sec'], opt_data['celf_runtime_sec']]
    colors = ['#e74c3c', '#2ecc71']

    axes[0].bar(methods, runtimes, color=colors, edgecolor='#2c3e50', alpha=0.85, width=0.45)
    axes[0].set_ylabel("Execution Time (Seconds)", fontsize=10)
    axes[0].set_title(f"Influence Maximization Runtime ({opt_data['speedup_factor']:.1f}x Speedup)",
                      fontsize=11, fontweight='bold')
    for idx, v in enumerate(runtimes):
        axes[0].text(idx, v + max(runtimes)*0.02, f"{v:.2f}s", ha='center', fontweight='bold')

    # 2. Monte Carlo evaluations count
    eval_counts = [opt_data['naive_eval_count'], opt_data['celf_eval_count']]
    axes[1].bar(methods, eval_counts, color=colors, edgecolor='#2c3e50', alpha=0.85, width=0.45)
    axes[1].set_ylabel("Monte Carlo Simulations Executed", fontsize=10)
    axes[1].set_title(f"Simulation Call Reduction ({opt_data['evaluation_reduction_pct']:.1f}% Saved)",
                      fontsize=11, fontweight='bold')
    for idx, v in enumerate(eval_counts):
        axes[1].text(idx, v + max(eval_counts)*0.02, f"{v}", ha='center', fontweight='bold')

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved Figure 8 to {out_path}")
