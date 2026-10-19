"""
budget_allocation.py
Requirement 5: Select users under a limited promotion budget

Implements:
1. Random Selection Baseline
2. Degree / Follower Centrality Baseline (Individual Popularity)
3. Greedy Marginal Spread with CELF (Cost-Effective Lazy Forward)
   - Considers diminishing marginal returns and overlapping audiences
   - Accounts for uncertainty via Monte Carlo estimations
4. Comparative Budget Scaling experiments across budgets k in {1, 2, 5, 10, 20}
"""

import heapq
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from environment import SocialEnvironment
from recommenders import BaseRecommender
from diffusion_models import IndependentCascadeSimulator, run_monte_carlo_diffusion


class SeedSelector:
    """Provides seed selection algorithms under promotion budget k."""
    def __init__(self, env: SocialEnvironment, recommender: BaseRecommender,
                 simulator: IndependentCascadeSimulator, random_seed: int = 42):
        self.env = env
        self.recommender = recommender
        self.simulator = simulator
        self.random_seed = random_seed

    def select_random(self, k: int, rng: Optional[np.random.RandomState] = None) -> List[int]:
        """Baseline 1: Uniform random seed selection."""
        if rng is None:
            rng = np.random.RandomState(self.random_seed)
        eligible_users = list(range(self.env.num_users))
        selected = rng.choice(eligible_users, size=min(k, len(eligible_users)), replace=False)
        return [int(u) for u in selected]

    def select_high_degree(self, k: int) -> List[int]:
        """
        Baseline 2: Degree Centrality / High Follower Count.
        Picks candidates with the largest individual follower base (|trusted_by[u]|).
        Susceptible to heavy audience overlap between top influencers.
        """
        # Sort users by number of followers (in-degree in trust network)
        user_followers = [(u, len(self.env.trusted_by.get(u, set()))) for u in range(self.env.num_users)]
        user_followers.sort(key=lambda x: x[1], reverse=True)
        return [u for u, _ in user_followers[:k]]

    def select_greedy_celf(self, item_id: int, k: int, mc_trials: int = 15,
                           candidate_pool_size: int = 150) -> Tuple[List[int], List[float]]:
        """
        Our Strategy: Cost-Effective Lazy Forward (CELF) Greedy Selection.
        - Maximizes marginal gain: Delta(u | S) = sigma(S union {u}) - sigma(S)
        - Uses submodularity: marginal gains are monotonically non-increasing as seed set grows
        - Lazy evaluation with priority queue avoids evaluating all candidates at every step
        - Evaluates candidates over Monte Carlo trials to account for uncertainty
        """
        rng = np.random.RandomState(self.random_seed)

        # Pre-filter top candidates by follower count and predicted interest to speed up initial queue
        degree_ranked = sorted(range(self.env.num_users),
                               key=lambda u: len(self.env.trusted_by.get(u, set())),
                               reverse=True)
        candidates = degree_ranked[:min(candidate_pool_size, self.env.num_users)]

        # Step 1: Initial evaluation of each candidate individually
        print(f"[CELF Seed Selection] Evaluating {len(candidates)} initial candidates for budget k={k}...")
        heap = []  # store (-marginal_gain, user_id, iteration_evaluated)
        for u in candidates:
            spread_stats = run_monte_carlo_diffusion(self.simulator, {u}, item_id,
                                                     num_trials=mc_trials, base_seed=self.random_seed)
            gain = spread_stats['mean_adoptions']
            heapq.heappush(heap, (-gain, u, 0))

        # Step 2: CELF Lazy Forward selection
        selected_seeds: List[int] = []
        marginal_gains: List[float] = []
        current_spread = 0.0

        for it in range(k):
            while heap:
                neg_gain, u, last_evaluated_it = heapq.heappop(heap)
                # If this candidate was evaluated at current iteration it, it is guaranteed optimal!
                if last_evaluated_it == it:
                    selected_seeds.append(u)
                    marginal_gains.append(-neg_gain)
                    current_spread += (-neg_gain)
                    print(f"  Selected seed {it+1}/{k}: User {u} with marginal spread {(-neg_gain):.2f}")
                    break
                else:
                    # Re-evaluate marginal gain with current seed set S
                    current_set = set(selected_seeds)
                    current_set.add(u)
                    new_spread_stats = run_monte_carlo_diffusion(self.simulator, current_set, item_id,
                                                                 num_trials=mc_trials, base_seed=self.random_seed)
                    new_spread = new_spread_stats['mean_adoptions']
                    marginal = max(0.0, new_spread - current_spread)
                    # Push back into heap with current iteration stamp
                    heapq.heappush(heap, (-marginal, u, it))

        return selected_seeds, marginal_gains


def evaluate_audience_overlap(env: SocialEnvironment, seed_set: List[int]) -> Dict[str, float]:
    """
    Quantifies audience overlap among selected seeds:
    - Total combined unique audience
    - Sum of individual audience sizes
    - Overlap Ratio = 1 - (Unique / Sum)
    """
    if not seed_set:
        return {"unique_audience": 0, "sum_audience": 0, "overlap_ratio": 0.0}

    audiences = [env.trusted_by.get(u, set()) for u in seed_set]
    unique_audience = set().union(*audiences)
    sum_audience = sum(len(a) for a in audiences)

    overlap_ratio = 1.0 - (len(unique_audience) / sum_audience if sum_audience > 0 else 1.0)
    return {
        "unique_audience": len(unique_audience),
        "sum_audience": sum_audience,
        "overlap_ratio": float(overlap_ratio)
    }
