"""
group_recommendation.py
Requirement 6: Recommend to groups

Implements:
1. Group preference aggregation strategies:
   - Average Satisfaction (Utilitarian)
   - Least Misery (Egalitarian)
   - Most Pleasure (Optimistic)
2. Handling conflicting vs cohesive groups
3. Downstream diffusion analysis:
   - Determines whether the group-preferred item under each strategy
     maximizes external network-wide information spread.
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from environment import SocialEnvironment
from recommenders import BaseRecommender
from diffusion_models import IndependentCascadeSimulator, run_monte_carlo_diffusion


class GroupRecommender:
    """Manages group recommendation and downstream diffusion analysis."""
    def __init__(self, env: SocialEnvironment, recommender: BaseRecommender,
                 simulator: IndependentCascadeSimulator):
        self.env = env
        self.recommender = recommender
        self.simulator = simulator

    def score_item_for_group(self, group: List[int], item_id: int) -> Dict[str, float]:
        """
        Computes group scores under Average, Least Misery, and Most Pleasure strategies,
        as well as group rating variance (conflict measure).
        """
        preds = [self.recommender.predict(u, item_id) for u in group]
        return {
            "average": float(np.mean(preds)),
            "least_misery": float(np.min(preds)),
            "most_pleasure": float(np.max(preds)),
            "variance_conflict": float(np.var(preds)),
            "member_preds": preds
        }

    def recommend_for_group(self, group: List[int], candidate_items: List[int],
                            strategy: str = "average", top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Ranks items for group according to chosen strategy:
        - 'average': maximizes mean satisfaction
        - 'least_misery': maximizes minimum member satisfaction
        - 'most_pleasure': maximizes maximum member satisfaction
        """
        strategy = strategy.lower()
        scored_items = []
        for i in candidate_items:
            scores = self.score_item_for_group(group, i)
            val = scores[strategy]
            scored_items.append((i, val))

        scored_items.sort(key=lambda x: x[1], reverse=True)
        return scored_items[:top_k]

    def evaluate_group_diffusion_tradeoff(self, group: List[int], candidate_items: List[int],
                                          mc_trials: int = 15, base_seed: int = 42) -> Dict[str, Dict]:
        """
        Compares the top recommended item under each of the 3 strategies:
        1. Identifies top-1 item for Average, Least Misery, Most Pleasure
        2. Simulates internal group adoption
        3. Seeds downstream diffusion into the network
        4. Reports within-group satisfaction vs external network cascade size
        """
        strategies = ["average", "least_misery", "most_pleasure"]
        strategy_results = {}

        for strat in strategies:
            ranked = self.recommend_for_group(group, candidate_items, strategy=strat, top_k=1)
            if not ranked:
                continue
            best_item, score = ranked[0]
            item_stats = self.score_item_for_group(group, best_item)

            # Downstream diffusion seeded through group
            diffusion_stats = run_monte_carlo_diffusion(self.simulator, set(group), best_item,
                                                        num_trials=mc_trials, base_seed=base_seed)

            strategy_results[strat] = {
                "top_item": int(best_item),
                "strategy_score": float(score),
                "group_avg_preference": item_stats["average"],
                "group_min_preference": item_stats["least_misery"],
                "group_max_preference": item_stats["most_pleasure"],
                "group_conflict_var": item_stats["variance_conflict"],
                "mean_external_spread": diffusion_stats["mean_adoptions"],
                "std_external_spread": diffusion_stats["std_adoptions"]
            }

        return strategy_results


def find_conflicting_and_cohesive_groups(env: SocialEnvironment, group_size: int = 5,
                                         num_candidates: int = 50, seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    Constructs:
    1. Cohesive Group: members connected by direct trust links
    2. Conflicting Group: members with divergent rating tastes
    """
    rng = np.random.RandomState(seed)

    # 1. Cohesive group (pick an active user and its trusted friends)
    active_users = [u for u in range(env.num_users) if len(env.trusts.get(u, set())) >= group_size - 1]
    if active_users:
        hub = active_users[0]
        cohesive_group = [hub] + list(env.trusts[hub])[:group_size - 1]
    else:
        cohesive_group = list(range(group_size))

    # 2. Conflicting group (pick users with distant rating profiles or contrasting mean ratings)
    sorted_by_means = np.argsort(env.user_means)
    # pick half from lowest raters, half from highest raters
    half = group_size // 2
    harsh_raters = list(sorted_by_means[:half])
    generous_raters = list(sorted_by_means[-(group_size - half):])
    conflicting_group = [int(u) for u in harsh_raters + generous_raters]

    return cohesive_group, conflicting_group
