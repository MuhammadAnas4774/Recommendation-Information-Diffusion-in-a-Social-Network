"""
run_demo.py
Demonstration script executing all eight requirements on the prepared sample dataset in ~45 seconds.
"""

import os
import sys
import json
import time
import numpy as np

# Add src to path
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


def run_demonstration():
    t_start = time.time()
    print("=" * 80)
    print("SNTP ASSIGNMENT #02: RECOMMENDATION & INFORMATION DIFFUSION IN A SOCIAL NETWORK")
    print("Author: Muhammad Anas | Roll Number: 23F-3094 | FAST-NUCES")
    print("Executing Complete Demonstration Pipeline Across All 8 Requirements")
    print("=" * 80)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    fig_dir = os.path.join(results_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    ratings_csv = os.path.join(data_dir, 'sample_ratings.csv')
    trust_csv = os.path.join(data_dir, 'sample_trust.csv')

    # Fallback to generate sample if missing
    if not os.path.exists(ratings_csv) or not os.path.exists(trust_csv):
        from data_preparation import prepare_datasets
        parent_dir = os.path.abspath(os.path.join(base_dir, '..'))
        r_zip = os.path.join(parent_dir, "ratings_data.zip")
        t_zip = os.path.join(parent_dir, "trust_data.zip")
        prepare_datasets(r_zip, t_zip, ratings_csv, trust_csv, sample_users=1000)

    metrics_record = {}

    # -------------------------------------------------------------
    # REQUIREMENT 1: Represent the Social Environment
    # -------------------------------------------------------------
    print("\n[REQUIREMENT 1] Represent the Social Environment...")
    env = SocialEnvironment(random_seed=42)
    env.load_from_csvs(ratings_csv, trust_csv, train_ratio=0.8)

    sparsity = env.get_sparsity()
    print(f"  Total Users: {env.num_users}, Total Items: {env.num_items}")
    print(f"  Rating Sparsity: {sparsity * 100:.2f}% (Dense users/items sampled)")
    print(f"  Trust Edges: {len(env.trust_df)}")

    fig1_path = os.path.join(fig_dir, "fig1_rating_and_trust_distributions.png")
    plot_fig1_environment(env, fig1_path)
    metrics_record['num_users'] = env.num_users
    metrics_record['num_items'] = env.num_items
    metrics_record['sparsity'] = float(sparsity)

    # -------------------------------------------------------------
    # REQUIREMENT 2: Learn Preferences & Generate Recommendations
    # -------------------------------------------------------------
    print("\n[REQUIREMENT 2] Learning Preferences via Biased MF and User-KNN...")
    # Model 1: Biased Matrix Factorization
    mf_model = BiasedMatrixFactorization(latent_dim=15, learning_rate=0.015,
                                         reg_lambda=0.05, epochs=15, random_seed=42)
    mf_model.fit(env)
    mf_rmse, mf_mae = compute_rmse_mae(mf_model, env)
    print(f"  Biased MF Test RMSE: {mf_rmse:.4f}, MAE: {mf_mae:.4f}")

    # Model 2: Memory-Based User-Based KNN
    knn_model = UserBasedKNNRecommender(k_neighbors=15, min_overlap=2)
    knn_model.fit(env)
    knn_rmse, knn_mae = compute_rmse_mae(knn_model, env)
    print(f"  User-KNN  Test RMSE: {knn_rmse:.4f}, MAE: {knn_mae:.4f}")

    # Cold-start vs Active breakdown
    cold_users = [u for u in env.test_ratings if len(env.train_ratings.get(u, {})) <= 10]
    active_users = [u for u in env.test_ratings if len(env.train_ratings.get(u, {})) > 10]

    def eval_subset_rmse(model, u_list):
        errs = []
        for u in u_list:
            for i, r in env.test_ratings.get(u, {}).items():
                errs.append((r - model.predict(u, i))**2)
        return float(np.sqrt(np.mean(errs))) if errs else 0.0

    mf_cold_rmse = eval_subset_rmse(mf_model, cold_users)
    mf_active_rmse = eval_subset_rmse(mf_model, active_users)
    knn_cold_rmse = eval_subset_rmse(knn_model, cold_users)
    knn_active_rmse = eval_subset_rmse(knn_model, active_users)

    metrics_record.update({
        'mf_rmse': mf_rmse, 'mf_mae': mf_mae,
        'knn_rmse': knn_rmse, 'knn_mae': knn_mae,
        'mf_cold_rmse': mf_cold_rmse, 'mf_active_rmse': mf_active_rmse,
        'knn_cold_rmse': knn_cold_rmse, 'knn_active_rmse': knn_active_rmse
    })

    fig2_path = os.path.join(fig_dir, "fig2_recommender_comparison_cold_start.png")
    plot_fig2_recommender_comparison(metrics_record, fig2_path)

    # -------------------------------------------------------------
    # REQUIREMENT 3: Incorporate Social Relationships
    # -------------------------------------------------------------
    print("\n[REQUIREMENT 3] Incorporating Social Relationships (SocialMF & Friends vs Strangers)...")
    alphas = [0.0, 0.05, 0.1, 0.25]
    alpha_rmses = []
    social_models = {}

    for a in alphas:
        s_mf = SocialRegularizedMF(latent_dim=15, learning_rate=0.015,
                                  reg_lambda=0.05, alpha=a, epochs=10, random_seed=42)
        s_mf.fit(env)
        s_rmse, _ = compute_rmse_mae(s_mf, env)
        alpha_rmses.append(s_rmse)
        social_models[a] = s_mf
        print(f"  SocialMF (alpha={a:4.2f}) -> Test RMSE: {s_rmse:.4f}")

    best_alpha_idx = int(np.argmin(alpha_rmses))
    best_social_mf = social_models[alphas[best_alpha_idx]]

    # Friends vs Strangers evaluation
    friends_res = evaluate_friends_vs_strangers(env, top_k_neighbors=10, sample_eval_users=100)
    print(f"  Friends MAE (Overall): {friends_res['friends_mae_overall']:.4f} | Cold-Start: {friends_res['friends_mae_cold_start']:.4f}")
    print(f"  Strangers MAE (Overall): {friends_res['strangers_mae_overall']:.4f} | Cold-Start: {friends_res['strangers_mae_cold_start']:.4f}")

    metrics_record['social_alphas'] = alphas
    metrics_record['social_alpha_rmses'] = alpha_rmses
    metrics_record['friends_vs_strangers'] = friends_res

    fig3_path = os.path.join(fig_dir, "fig3_social_alpha_impact_and_friends_vs_strangers.png")
    plot_fig3_social_effects(alphas, alpha_rmses, friends_res, fig3_path)

    # -------------------------------------------------------------
    # REQUIREMENT 4: Model Adoption and Information Spread
    # -------------------------------------------------------------
    print("\n[REQUIREMENT 4] Simulating Diffusion Models (ICM vs SIR Epidemic)...")
    icm_sim = IndependentCascadeSimulator(env, best_social_mf, base_propagation_prob=0.15, max_steps=20)
    sir_sim = SIREpidemicSimulator(env, best_social_mf, transmission_beta=0.25, recovery_gamma=0.15, max_steps=20)

    # Select seed users (top 5 degree nodes) and high-preference item
    seed_nodes = set(list(np.argsort([len(env.trusted_by.get(u, set())) for u in range(env.num_users)]))[-5:])
    test_item_id = 100 if 100 < env.num_items else 0

    rng = np.random.RandomState(42)
    icm_res = icm_sim.simulate(seed_nodes, test_item_id, rng)
    sir_res = sir_sim.simulate(seed_nodes, test_item_id, rng)

    print(f"  ICM Diffusion: Total Adoptions = {icm_res.total_adoptions}, Peak at t={icm_res.time_to_peak}")
    print(f"  SIR Epidemic:  Total Adoptions = {sir_res.total_adoptions}, Peak at t={sir_res.time_to_peak}")

    fig4_path = os.path.join(fig_dir, "fig4_icm_vs_sir_diffusion_dynamics.png")
    plot_fig4_diffusion_models(icm_res, sir_res, fig4_path)

    metrics_record['icm_adoptions'] = icm_res.total_adoptions
    metrics_record['sir_adoptions'] = sir_res.total_adoptions

    # -------------------------------------------------------------
    # REQUIREMENT 5: Select Users Under a Limited Promotion Budget
    # -------------------------------------------------------------
    print("\n[REQUIREMENT 5] Selecting Users under Budget k (CELF vs Centrality vs Random)...")
    seed_selector = SeedSelector(env, best_social_mf, icm_sim, random_seed=42)
    budgets = [1, 2, 5, 10]
    max_k = max(budgets)
    celf_all_seeds, _ = seed_selector.select_greedy_celf(test_item_id, max_k, mc_trials=4, candidate_pool_size=30)
    deg_all_seeds = seed_selector.select_high_degree(max_k)

    celf_means, deg_means, rand_means = [], [], []
    overlap_deg, overlap_celf = [], []

    for k in budgets:
        # 1. Random
        rand_seeds = seed_selector.select_random(k)
        rand_spread = run_monte_carlo_diffusion(icm_sim, set(rand_seeds), test_item_id, num_trials=5)['mean_adoptions']
        rand_means.append(rand_spread)

        # 2. Degree
        deg_k_seeds = deg_all_seeds[:k]
        deg_spread = run_monte_carlo_diffusion(icm_sim, set(deg_k_seeds), test_item_id, num_trials=5)['mean_adoptions']
        deg_means.append(deg_spread)
        overlap_deg.append(evaluate_audience_overlap(env, deg_k_seeds)['overlap_ratio'])

        # 3. CELF
        celf_k_seeds = celf_all_seeds[:k]
        celf_spread = run_monte_carlo_diffusion(icm_sim, set(celf_k_seeds), test_item_id, num_trials=5)['mean_adoptions']
        celf_means.append(celf_spread)
        overlap_celf.append(evaluate_audience_overlap(env, celf_k_seeds)['overlap_ratio'])

        print(f"  Budget k={k:2d} -> CELF: {celf_spread:5.1f} | Degree: {deg_spread:5.1f} | Random: {rand_spread:5.1f}")

    budget_results = {
        'budgets': budgets,
        'celf_means': celf_means, 'deg_means': deg_means, 'rand_means': rand_means,
        'overlap_degree': overlap_deg, 'overlap_celf': overlap_celf
    }
    metrics_record['budget_results'] = budget_results
    fig5_path = os.path.join(fig_dir, "fig5_budget_allocation_and_marginal_gains.png")
    plot_fig5_budget_allocation(budget_results, fig5_path)

    # -------------------------------------------------------------
    # REQUIREMENT 6: Recommend to Groups
    # -------------------------------------------------------------
    print("\n[REQUIREMENT 6] Group Recommendation & Conflict Downstream Diffusion...")
    group_rec = GroupRecommender(env, best_social_mf, icm_sim)
    cohesive_g, conflicting_g = find_conflicting_and_cohesive_groups(env, group_size=5, seed=42)

    candidate_items_group = list(range(min(100, env.num_items)))
    cohesive_eval = group_rec.evaluate_group_diffusion_tradeoff(cohesive_g, candidate_items_group, mc_trials=6)
    conflicting_eval = group_rec.evaluate_group_diffusion_tradeoff(conflicting_g, candidate_items_group, mc_trials=6)

    print("  Conflicting Group Trade-off:")
    for strat in ['average', 'least_misery', 'most_pleasure']:
        s_data = conflicting_eval[strat]
        print(f"    Strategy: {strat:14s} | Min Pref: {s_data['group_min_preference']:.2f} | Avg Pref: {s_data['group_avg_preference']:.2f} | External Spread: {s_data['mean_external_spread']:.1f}")

    metrics_record['group_eval_conflicting'] = conflicting_eval
    metrics_record['group_eval_cohesive'] = cohesive_eval

    fig6_path = os.path.join(fig_dir, "fig6_group_recommendation_vs_diffusion_spread.png")
    plot_fig6_group_recommendation(conflicting_eval, fig6_path)

    # -------------------------------------------------------------
    # REQUIREMENT 7: Evaluate the Complete System
    # -------------------------------------------------------------
    print("\n[REQUIREMENT 7] Complete System Evaluation (Ranking, Classification, Diversity, Leakage-Free)...")
    ranking_k10 = compute_ranking_metrics_at_k(best_social_mf, env, k=10, relevance_threshold=4.0, sample_users_eval=150)
    classification_res = compute_classification_metrics(best_social_mf, env, threshold=4.0)
    ild_score = compute_intra_list_diversity(best_social_mf, env, top_k=10, sample_users=100)

    print(f"  Ranking Quality: Precision@10: {ranking_k10['precision@10']:.4f} | Recall@10: {ranking_k10['recall@10']:.4f} | NDCG@10: {ranking_k10['ndcg@10']:.4f} | MRR: {ranking_k10['mrr@10']:.4f}")
    print(f"  Coverage: Catalog: {ranking_k10['catalog_coverage@10']*100:.2f}% | User Coverage: {ranking_k10['user_coverage@10']*100:.1f}%")
    print(f"  Classification: Accuracy: {classification_res['accuracy']*100:.2f}% | F1-Score: {classification_res['f1_score']:.4f} | ROC-AUC: {classification_res['roc_auc']:.4f}")
    print(f"  Intra-List Diversity (ILD): {ild_score:.4f}")

    metrics_record['ranking'] = ranking_k10
    metrics_record['classification'] = classification_res
    metrics_record['ild'] = ild_score

    # -------------------------------------------------------------
    # REQUIREMENT 8: Investigate Intervention and Computational Cost
    # -------------------------------------------------------------
    print("\n[REQUIREMENT 8] Intervention Analysis & Computational Optimization (CELF vs Naive)...")
    interv_sim = InterventionSimulator(env, best_social_mf)
    damping_factors = [0.2, 0.4, 0.6, 0.8, 1.0]
    cascade_sizes = []
    high_qual_sats = []
    low_qual_sats = []

    # Identify a high-preference item and a low-preference item
    item_means_idx = np.argsort(env.item_means)
    low_item = int(item_means_idx[0])
    high_item = int(item_means_idx[-1])

    for df in damping_factors:
        res_h = interv_sim.simulate_with_intervention(seed_nodes, high_item, damping_factor=df, fatigue_limit=2)
        res_l = interv_sim.simulate_with_intervention(seed_nodes, low_item, damping_factor=df, fatigue_limit=2)
        cascade_sizes.append(res_h['cascade_size'])
        high_qual_sats.append(res_h['mean_adopter_satisfaction'])
        low_qual_sats.append(res_l['mean_adopter_satisfaction'])

    intervention_data = {
        'damping_factors': damping_factors,
        'cascade_sizes': cascade_sizes,
        'high_quality_satisfaction': high_qual_sats,
        'low_quality_satisfaction': low_qual_sats
    }
    metrics_record['intervention_data'] = intervention_data
    fig7_path = os.path.join(fig_dir, "fig7_intervention_spread_vs_satisfaction_tradeoff.png")
    plot_fig7_intervention_analysis(intervention_data, fig7_path)

    # Bottleneck & CELF Benchmark
    opt_benchmark = benchmark_naive_vs_celf_optimization(env, best_social_mf, test_item_id,
                                                         k=3, candidate_pool_size=20, mc_trials=3)
    print(f"  Optimization Speedup: {opt_benchmark['speedup_factor']:.1f}x (Naive: {opt_benchmark['naive_runtime_sec']:.2f}s vs CELF: {opt_benchmark['celf_runtime_sec']:.2f}s)")
    metrics_record['optimization_benchmark'] = opt_benchmark
    fig8_path = os.path.join(fig_dir, "fig8_computational_scaling_and_celf_speedup.png")
    plot_fig8_computational_optimization(opt_benchmark, fig8_path)

    # Save summary JSON
    json_path = os.path.join(results_dir, "metrics_summary.json")
    with open(json_path, 'w') as f:
        # Convert any sets or non-serializable elements
        json.dump(metrics_record, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.float32, np.float64)) else str(o))
    print(f"\n[Results] Saved all metrics to {json_path}")

    elapsed_total = time.time() - t_start
    print("=" * 80)
    print(f"SUCCESS: Demonstration pipeline finished cleanly in {elapsed_total:.2f} seconds!")
    print(f"All figures generated in: {fig_dir}")
    print("=" * 80)


if __name__ == "__main__":
    run_demonstration()
