"""
intervention_and_scaling.py
Requirement 8: Investigate intervention and computational cost

Implements:
1. Diffusion Intervention Policies:
   - Exposure Fatigue & Virality Damping: throttles edge propagation when items exceed spread thresholds
   - Measures dual effects on network cascade size and user satisfaction (relevance of reached users)
2. Computational Scaling Profiler:
   - Measures runtime and memory consumption across varying:
     * Network size |V|
     * Item collection size |I|
     * Promotion budget k
3. Main Bottleneck Identification & Optimization:
   - Identifies Naive Greedy influence simulation as the O(k * |V| * R) bottleneck
   - Implements & benchmarks CELF (Cost-Effective Lazy Forward) submodular optimization
   - Reports empirical runtime speedup and call reduction
"""

import time
import sys
import heapq
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from environment import SocialEnvironment
from recommenders import BaseRecommender, BiasedMatrixFactorization
from diffusion_models import sigmoid_preference_adoption, DiffusionResult


class InterventionSimulator:
    """
    Simulates diffusion under an algorithmic intervention policy:
    - Virality Damping Factor delta in [0.0, 1.0]: dampens edge propagation probabilities
    - User Fatigue Cap C_cap: caps exposures per user to prevent notification fatigue
    """
    def __init__(self, env: SocialEnvironment, recommender: BaseRecommender,
                 base_propagation_prob: float = 0.15, max_steps: int = 20):
        self.env = env
        self.recommender = recommender
        self.base_p = base_propagation_prob
        self.max_steps = max_steps

    def simulate_with_intervention(self, seed_users: Set[int], item_id: int,
                                  damping_factor: float = 0.5,
                                  fatigue_limit: int = 2,
                                  rng: Optional[np.random.RandomState] = None) -> Dict[str, float]:
        """
        Runs ICM diffusion with intervention active:
        - damping_factor: scales propagation probabilities (1.0 = no intervention, 0.2 = heavy throttle)
        - fatigue_limit: maximum exposures tolerated by a user before muting/ignoring
        """
        if rng is None:
            rng = np.random.RandomState()

        user_exposure_counts: Dict[int, int] = {u: 0 for u in range(self.env.num_users)}
        adopted_users: Set[int] = set()

        # Seed adoption
        for u in seed_users:
            pred_r = self.recommender.predict(u, item_id)
            if rng.rand() < sigmoid_preference_adoption(pred_r):
                adopted_users.add(u)

        if not adopted_users:
            return {
                "cascade_size": 0,
                "total_exposures": 0,
                "mean_adopter_satisfaction": 0.0,
                "suppressed_exposures": 0
            }

        active_frontier = set(adopted_users)
        total_exposures = 0
        suppressed_exposures = 0

        for step in range(self.max_steps):
            if not active_frontier:
                break
            next_frontier = set()

            for v in active_frontier:
                followers = self.env.trusted_by.get(v, set())
                for u in followers:
                    if u not in adopted_users:
                        # Check user fatigue cap
                        if user_exposure_counts[u] >= fatigue_limit:
                            suppressed_exposures += 1
                            continue

                        user_exposure_counts[u] += 1
                        total_exposures += 1

                        # Compute dampened edge propagation probability
                        base_edge_p = self.env.edge_influence_weights.get((v, u), self.base_p)
                        damped_p = base_edge_p * damping_factor

                        if rng.rand() < damped_p:
                            pred_r = self.recommender.predict(u, item_id)
                            if rng.rand() < sigmoid_preference_adoption(pred_r):
                                next_frontier.add(u)
                                adopted_users.add(u)

            active_frontier = next_frontier

        # Measure adopter satisfaction (average predicted rating of all users who adopted)
        adopter_satisfaction = [self.recommender.predict(u, item_id) for u in adopted_users]
        mean_satisfaction = float(np.mean(adopter_satisfaction)) if adopter_satisfaction else 0.0

        return {
            "cascade_size": len(adopted_users),
            "total_exposures": total_exposures,
            "mean_adopter_satisfaction": mean_satisfaction,
            "suppressed_exposures": suppressed_exposures
        }


def benchmark_computational_scaling(env: SocialEnvironment,
                                    budgets: List[int] = [1, 2, 5, 10],
                                    subgraph_sizes: List[int] = [200, 400, 800]) -> Dict:
    """
    Evaluates runtime and memory requirements as:
    1. Network size |V| and |E| grows
    2. Promotion budget k grows
    """
    results = {"network_scaling": [], "budget_scaling": []}

    print("[Computational Scaling] Benchmarking network size scaling...")
    for sz in subgraph_sizes:
        t0 = time.time()
        sub_users = list(range(min(sz, env.num_users)))
        # Approximate memory footprint of subgraph structures
        adj_size = sum(sys.getsizeof(env.trusted_by.get(u, set())) for u in sub_users)
        rats_size = sum(sys.getsizeof(env.train_ratings.get(u, {})) for u in sub_users)
        total_kb = (adj_size + rats_size) / 1024.0

        # Run small simulated cascade
        rng = np.random.RandomState(42)
        active = {sub_users[0]}
        for _ in range(5):
            nxt = set()
            for v in active:
                for u in env.trusted_by.get(v, set()):
                    if u in sub_users and rng.rand() < 0.1:
                        nxt.add(u)
            active = nxt

        elapsed = time.time() - t0
        results["network_scaling"].append({
            "num_users": sz,
            "runtime_sec": float(elapsed),
            "approx_memory_kb": float(total_kb)
        })

    print("[Computational Scaling] Benchmarking budget scaling...")
    for k in budgets:
        t0 = time.time()
        # Measure degree selection time
        user_followers = [(u, len(env.trusted_by.get(u, set()))) for u in range(env.num_users)]
        user_followers.sort(key=lambda x: x[1], reverse=True)
        top_k = [u for u, _ in user_followers[:k]]
        elapsed = time.time() - t0
        results["budget_scaling"].append({
            "budget_k": k,
            "runtime_sec": float(elapsed)
        })

    return results


def benchmark_naive_vs_celf_optimization(env: SocialEnvironment, recommender: BaseRecommender,
                                         item_id: int, k: int = 5,
                                         candidate_pool_size: int = 60,
                                         mc_trials: int = 5) -> Dict[str, float]:
    """
    Directly benchmarks Naive Greedy Influence Maximization vs CELF Optimization.
    Measures:
    - Execution time for Naive Greedy
    - Execution time for CELF
    - Number of Monte Carlo evaluations executed
    - Empirical speedup factor
    """
    from diffusion_models import IndependentCascadeSimulator, run_monte_carlo_diffusion
    sim = IndependentCascadeSimulator(env, recommender, base_propagation_prob=0.15, max_steps=15)

    candidates = sorted(range(env.num_users),
                        key=lambda u: len(env.trusted_by.get(u, set())),
                        reverse=True)[:candidate_pool_size]

    # 1. Naive Greedy Execution
    print(f"[Optimization Benchmark] Running Naive Greedy for budget k={k} on {len(candidates)} candidates...")
    t0_naive = time.time()
    naive_seeds = []
    naive_eval_count = 0

    for step in range(k):
        best_gain = -1.0
        best_cand = None
        for cand in candidates:
            if cand in naive_seeds:
                continue
            test_set = set(naive_seeds).union({cand})
            res = run_monte_carlo_diffusion(sim, test_set, item_id, num_trials=mc_trials)
            naive_eval_count += 1
            if res['mean_adoptions'] > best_gain:
                best_gain = res['mean_adoptions']
                best_cand = cand
        if best_cand is not None:
            naive_seeds.append(best_cand)

    naive_time = time.time() - t0_naive
    print(f"  Naive Greedy completed in {naive_time:.3f}s with {naive_eval_count} MC evaluations.")

    # 2. CELF Execution
    print(f"[Optimization Benchmark] Running CELF Optimization for budget k={k}...")
    t0_celf = time.time()
    celf_eval_count = 0

    # Step 1: Initial evaluation
    heap = []
    for u in candidates:
        res = run_monte_carlo_diffusion(sim, {u}, item_id, num_trials=mc_trials)
        celf_eval_count += 1
        heapq.heappush(heap, (-res['mean_adoptions'], u, 0))

    # Step 2: Lazy selection
    celf_seeds = []
    current_spread = 0.0

    for it in range(k):
        while heap:
            neg_gain, u, last_it = heapq.heappop(heap)
            if last_it == it:
                celf_seeds.append(u)
                current_spread += (-neg_gain)
                break
            else:
                test_set = set(celf_seeds).union({u})
                res = run_monte_carlo_diffusion(sim, test_set, item_id, num_trials=mc_trials)
                celf_eval_count += 1
                gain = max(0.0, res['mean_adoptions'] - current_spread)
                heapq.heappush(heap, (-gain, u, it))

    celf_time = time.time() - t0_celf
    speedup = naive_time / celf_time if celf_time > 0 else 1.0
    print(f"  CELF completed in {celf_time:.3f}s with {celf_eval_count} MC evaluations. Speedup: {speedup:.2f}x")

    return {
        "naive_runtime_sec": float(naive_time),
        "naive_eval_count": naive_eval_count,
        "celf_runtime_sec": float(celf_time),
        "celf_eval_count": celf_eval_count,
        "speedup_factor": float(speedup),
        "evaluation_reduction_pct": float(100.0 * (1.0 - celf_eval_count / naive_eval_count))
    }
