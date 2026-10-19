"""
run_full_experiments.py
Full experiment execution runner supporting both sample and full datasets.
Usage:
  python run_full_experiments.py --mode sample
  python run_full_experiments.py --mode full --sample_users 5000
"""

import os
import sys
import json
import time
import argparse
import numpy as np

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_dir)

from environment import SocialEnvironment
from recommenders import BiasedMatrixFactorization, UserBasedKNNRecommender
from social_recommender import SocialRegularizedMF, evaluate_friends_vs_strangers
from diffusion_models import (IndependentCascadeSimulator, SIREpidemicSimulator,
                              run_monte_carlo_diffusion)
from budget_allocation import SeedSelector, evaluate_audience_overlap
from group_recommendation import GroupRecommender, find_conflicting_and_cohesive_groups
from evaluation import (compute_rmse_mae, compute_ranking_metrics_at_k,
                        compute_classification_metrics, compute_intra_list_diversity)
from intervention_and_scaling import (InterventionSimulator, benchmark_computational_scaling,
                                     benchmark_naive_vs_celf_optimization)
from visualization import (plot_fig1_environment, plot_fig2_recommender_comparison,
                           plot_fig3_social_effects, plot_fig4_diffusion_models,
                           plot_fig5_budget_allocation, plot_fig6_group_recommendation,
                           plot_fig7_intervention_analysis, plot_fig8_computational_optimization)


def main():
    parser = argparse.ArgumentParser(description="Run Full SNTP Experiments")
    parser.add_argument("--mode", type=str, default="sample", choices=["sample", "full"],
                        help="Execution mode: sample (fast) or full (larger dataset)")
    parser.add_argument("--sample_users", type=int, default=2500,
                        help="Number of users to sample if mode=full")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 80)
    print(f"SNTP ASSIGNMENT #02 - FULL EXPERIMENT SUITE (Mode: {args.mode.upper()})")
    print("Author: Muhammad Anas | Roll Number: 23F-3094 | FAST-NUCES")
    print("=" * 80)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    fig_dir = os.path.join(results_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    env = SocialEnvironment(random_seed=42)

    if args.mode == "sample":
        ratings_csv = os.path.join(data_dir, 'sample_ratings.csv')
        trust_csv = os.path.join(data_dir, 'sample_trust.csv')
        env.load_from_csvs(ratings_csv, trust_csv, train_ratio=0.8)
    else:
        parent_dir = os.path.abspath(os.path.join(base_dir, '..'))
        r_zip = os.path.join(parent_dir, "ratings_data.zip")
        t_zip = os.path.join(parent_dir, "trust_data.zip")
        env.load_from_zips(r_zip, t_zip, sample_users=args.sample_users, train_ratio=0.8)

    print(f"Loaded Environment: {env.num_users} users, {env.num_items} items, {len(env.trust_df)} trust edges.")

    # Generate Figures & Execute Full Pipeline
    plot_fig1_environment(env, os.path.join(fig_dir, "fig1_rating_and_trust_distributions.png"))

    # Train Recommenders
    mf = BiasedMatrixFactorization(latent_dim=20, learning_rate=0.015, reg_lambda=0.05, epochs=20, random_seed=42)
    mf.fit(env)
    mf_rmse, mf_mae = compute_rmse_mae(mf, env)

    knn = UserBasedKNNRecommender(k_neighbors=20, min_overlap=2)
    knn.fit(env)
    knn_rmse, knn_mae = compute_rmse_mae(knn, env)

    plot_fig2_recommender_comparison({'mf_rmse': mf_rmse, 'mf_mae': mf_mae,
                                      'knn_rmse': knn_rmse, 'knn_mae': knn_mae},
                                     os.path.join(fig_dir, "fig2_recommender_comparison_cold_start.png"))

    # Social MF
    alphas = [0.0, 0.05, 0.1, 0.25]
    alpha_rmses = []
    social_models = {}
    for a in alphas:
        smf = SocialRegularizedMF(latent_dim=20, learning_rate=0.015, reg_lambda=0.05, alpha=a, epochs=15, random_seed=42)
        smf.fit(env)
        s_rmse, _ = compute_rmse_mae(smf, env)
        alpha_rmses.append(s_rmse)
        social_models[a] = smf

    best_smf = social_models[alphas[int(np.argmin(alpha_rmses))]]
    friends_res = evaluate_friends_vs_strangers(env, top_k_neighbors=15, sample_eval_users=150)
    plot_fig3_social_effects(alphas, alpha_rmses, friends_res,
                             os.path.join(fig_dir, "fig3_social_alpha_impact_and_friends_vs_strangers.png"))

    # Diffusion
    icm_sim = IndependentCascadeSimulator(env, best_smf, base_propagation_prob=0.15, max_steps=25)
    sir_sim = SIREpidemicSimulator(env, best_smf, transmission_beta=0.25, recovery_gamma=0.12, max_steps=25)
    seed_nodes = set(list(np.argsort([len(env.trusted_by.get(u, set())) for u in range(env.num_users)]))[-5:])
    test_item = 100 if 100 < env.num_items else 0

    rng = np.random.RandomState(42)
    icm_res = icm_sim.simulate(seed_nodes, test_item, rng)
    sir_res = sir_sim.simulate(seed_nodes, test_item, rng)
    plot_fig4_diffusion_models(icm_res, sir_res, os.path.join(fig_dir, "fig4_icm_vs_sir_diffusion_dynamics.png"))

    # Budget
    seed_selector = SeedSelector(env, best_smf, icm_sim, random_seed=42)
    budgets = [1, 2, 5, 10, 20]
    max_k = max(budgets)
    celf_all, _ = seed_selector.select_greedy_celf(test_item, max_k, mc_trials=5, candidate_pool_size=40)
    deg_all = seed_selector.select_high_degree(max_k)

    celf_means, deg_means, rand_means = [], [], []
    overlap_deg, overlap_celf = [], []
    for k in budgets:
        r_seeds = seed_selector.select_random(k)
        rand_means.append(run_monte_carlo_diffusion(icm_sim, set(r_seeds), test_item, num_trials=5)['mean_adoptions'])
        deg_k = deg_all[:k]
        deg_means.append(run_monte_carlo_diffusion(icm_sim, set(deg_k), test_item, num_trials=5)['mean_adoptions'])
        overlap_deg.append(evaluate_audience_overlap(env, deg_k)['overlap_ratio'])
        celf_k = celf_all[:k]
        celf_means.append(run_monte_carlo_diffusion(icm_sim, set(celf_k), test_item, num_trials=5)['mean_adoptions'])
        overlap_celf.append(evaluate_audience_overlap(env, celf_k)['overlap_ratio'])

    plot_fig5_budget_allocation({
        'budgets': budgets, 'celf_means': celf_means, 'deg_means': deg_means, 'rand_means': rand_means,
        'overlap_degree': overlap_deg, 'overlap_celf': overlap_celf
    }, os.path.join(fig_dir, "fig5_budget_allocation_and_marginal_gains.png"))

    # Group
    group_rec = GroupRecommender(env, best_smf, icm_sim)
    _, conflicting_g = find_conflicting_and_cohesive_groups(env, group_size=5, seed=42)
    cand_items = list(range(min(150, env.num_items)))
    conflicting_eval = group_rec.evaluate_group_diffusion_tradeoff(conflicting_g, cand_items, mc_trials=5)
    plot_fig6_group_recommendation(conflicting_eval, os.path.join(fig_dir, "fig6_group_recommendation_vs_diffusion_spread.png"))

    # Intervention & Optimization
    interv_sim = InterventionSimulator(env, best_smf)
    dfs = [0.2, 0.4, 0.6, 0.8, 1.0]
    c_sizes, h_sats, l_sats = [], [], []
    item_means_idx = np.argsort(env.item_means)
    low_item = int(item_means_idx[0])
    high_item = int(item_means_idx[-1])
    for df in dfs:
        res_h = interv_sim.simulate_with_intervention(seed_nodes, high_item, damping_factor=df, fatigue_limit=2)
        res_l = interv_sim.simulate_with_intervention(seed_nodes, low_item, damping_factor=df, fatigue_limit=2)
        c_sizes.append(res_h['cascade_size'])
        h_sats.append(res_h['mean_adopter_satisfaction'])
        l_sats.append(res_l['mean_adopter_satisfaction'])

    plot_fig7_intervention_analysis({
        'damping_factors': dfs, 'cascade_sizes': c_sizes,
        'high_quality_satisfaction': h_sats, 'low_quality_satisfaction': l_sats
    }, os.path.join(fig_dir, "fig7_intervention_spread_vs_satisfaction_tradeoff.png"))

    opt_bench = benchmark_naive_vs_celf_optimization(env, best_smf, test_item, k=4, candidate_pool_size=30, mc_trials=4)
    plot_fig8_computational_optimization(opt_bench, os.path.join(fig_dir, "fig8_computational_scaling_and_celf_speedup.png"))

    print(f"\nFull experiments completed in {time.time() - t_start:.2f}s.")


if __name__ == "__main__":
    main()
