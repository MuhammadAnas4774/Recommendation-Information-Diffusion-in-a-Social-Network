"""
diffusion_models.py
Requirement 4: Model adoption and information spread

Implements:
1. Multi-stage user adoption:
   - Recommendation Received (Platform Seed)
   - Exposed through Neighbor (Social Word-of-Mouth)
   - Actual Adoption (Conditioned on predicted personal preference)
2. Independent Cascade Model (ICM)
3. SIR (Susceptible-Infectious-Recovered) Epidemic Model
4. Repeated Monte Carlo simulation engine reporting mean and variability
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from environment import SocialEnvironment
from recommenders import BaseRecommender


def sigmoid_preference_adoption(pred_rating: float, threshold: float = 3.5, slope: float = 2.0) -> float:
    """
    Computes adoption probability conditioned on predicted personalized preference.
    High preference (rating > 4) -> High adoption (>80%).
    Low preference (rating < 2.5) -> Low adoption (<10%).
    """
    z = slope * (pred_rating - threshold)
    return float(1.0 / (1.0 + np.exp(-z)))


class DiffusionResult:
    """Stores the metrics and trajectory of a single or aggregated diffusion simulation."""
    def __init__(self):
        self.total_adoptions: int = 0
        self.total_exposures: int = 0
        self.recommendation_seeds_count: int = 0
        self.neighbor_exposures_count: int = 0
        self.steps_run: int = 0
        self.time_to_peak: int = 0
        self.peak_new_adoptions: int = 0
        self.adoptions_per_step: List[int] = []
        self.cumulative_adoptions: List[int] = []
        self.adopted_users: Set[int] = set()


class IndependentCascadeSimulator:
    """
    Independent Cascade Model (ICM).
    Discrete-time progressive diffusion process where newly active nodes
    get a single chance to activate their followers with edge probability p(v -> u).
    """
    def __init__(self, env: SocialEnvironment, recommender: BaseRecommender,
                 base_propagation_prob: float = 0.15, max_steps: int = 20):
        self.env = env
        self.recommender = recommender
        self.base_p = base_propagation_prob
        self.max_steps = max_steps

    def simulate(self, seed_users: Set[int], item_id: int, rng: Optional[np.random.RandomState] = None) -> DiffusionResult:
        if rng is None:
            rng = np.random.RandomState()

        result = DiffusionResult()
        result.recommendation_seeds_count = len(seed_users)

        # Level 1 & Level 3: Seeds receive recommendation and adopt if they like it
        adopted = set()
        for u in seed_users:
            pred_r = self.recommender.predict(u, item_id)
            p_adopt = sigmoid_preference_adoption(pred_r)
            if rng.rand() < p_adopt:
                adopted.add(u)

        if not adopted:
            # If no seed user adopted, cascade halts immediately
            result.total_adoptions = 0
            result.cumulative_adoptions = [0]
            return result

        result.adopted_users = set(adopted)
        active_frontier = set(adopted)
        result.adoptions_per_step.append(len(adopted))
        result.cumulative_adoptions.append(len(adopted))

        peak_new = len(adopted)
        peak_step = 0
        step = 0

        while active_frontier and step < self.max_steps:
            step += 1
            next_frontier = set()

            for v in active_frontier:
                # v spreads to its followers u (users who trust v)
                followers = self.env.trusted_by.get(v, set())
                for u in followers:
                    if u not in result.adopted_users:
                        result.total_exposures += 1
                        result.neighbor_exposures_count += 1

                        # Influence propagation probability
                        edge_p = self.env.edge_influence_weights.get((v, u), self.base_p)
                        if rng.rand() < edge_p:
                            # User is exposed! Now personal preference determines adoption
                            pred_r = self.recommender.predict(u, item_id)
                            p_adopt = sigmoid_preference_adoption(pred_r)
                            if rng.rand() < p_adopt:
                                next_frontier.add(u)
                                result.adopted_users.add(u)

            new_count = len(next_frontier)
            result.adoptions_per_step.append(new_count)
            result.cumulative_adoptions.append(len(result.adopted_users))

            if new_count > peak_new:
                peak_new = new_count
                peak_step = step

            active_frontier = next_frontier

        result.steps_run = step
        result.total_adoptions = len(result.adopted_users)
        result.time_to_peak = peak_step
        result.peak_new_adoptions = peak_new
        return result


class SIREpidemicSimulator:
    """
    SIR (Susceptible - Infectious - Recovered) Compartmental Epidemic Model.
    - S (Susceptible): Unreached users.
    - I (Infectious): Active spreaders who advocate the item to followers each step.
    - R (Recovered): Users who adopted but are no longer actively spreading it (fatigued).
    """
    def __init__(self, env: SocialEnvironment, recommender: BaseRecommender,
                 transmission_beta: float = 0.25, recovery_gamma: float = 0.15,
                 max_steps: int = 25):
        self.env = env
        self.recommender = recommender
        self.beta = transmission_beta
        self.gamma = recovery_gamma
        self.max_steps = max_steps

    def simulate(self, seed_users: Set[int], item_id: int, rng: Optional[np.random.RandomState] = None) -> DiffusionResult:
        if rng is None:
            rng = np.random.RandomState()

        result = DiffusionResult()
        result.recommendation_seeds_count = len(seed_users)

        # Initial state setup
        # S = all users not yet adopted
        # I = seed adopters
        # R = empty
        infectious = set()
        recovered = set()
        all_adopted = set()

        for u in seed_users:
            pred_r = self.recommender.predict(u, item_id)
            p_adopt = sigmoid_preference_adoption(pred_r)
            if rng.rand() < p_adopt:
                infectious.add(u)
                all_adopted.add(u)

        if not infectious:
            result.cumulative_adoptions = [0]
            return result

        result.adoptions_per_step.append(len(infectious))
        result.cumulative_adoptions.append(len(all_adopted))

        peak_new = len(infectious)
        peak_step = 0
        step = 0

        while infectious and step < self.max_steps:
            step += 1
            newly_infected = set()

            # Transmission step: each infectious user attempts to transmit to susceptible followers
            for v in infectious:
                followers = self.env.trusted_by.get(v, set())
                for u in followers:
                    if u not in all_adopted and u not in newly_infected:
                        result.total_exposures += 1
                        result.neighbor_exposures_count += 1

                        edge_weight = self.env.edge_influence_weights.get((v, u), 0.2)
                        p_trans = self.beta * edge_weight
                        if rng.rand() < p_trans:
                            # Exposed: checks personal preference
                            pred_r = self.recommender.predict(u, item_id)
                            p_adopt = sigmoid_preference_adoption(pred_r)
                            if rng.rand() < p_adopt:
                                newly_infected.add(u)

            # Recovery step: active spreaders recover with rate gamma
            newly_recovered = set()
            for v in infectious:
                if rng.rand() < self.gamma:
                    newly_recovered.add(v)

            infectious = (infectious - newly_recovered).union(newly_infected)
            recovered = recovered.union(newly_recovered)
            all_adopted = all_adopted.union(newly_infected)

            new_count = len(newly_infected)
            result.adoptions_per_step.append(new_count)
            result.cumulative_adoptions.append(len(all_adopted))

            if new_count > peak_new:
                peak_new = new_count
                peak_step = step

        result.steps_run = step
        result.total_adoptions = len(all_adopted)
        result.adopted_users = all_adopted
        result.time_to_peak = peak_step
        result.peak_new_adoptions = peak_new
        return result


def run_monte_carlo_diffusion(simulator, seed_users: Set[int], item_id: int,
                              num_trials: int = 15, base_seed: int = 42) -> Dict[str, float]:
    """
    Runs repeated Monte Carlo trials to measure diffusion variability and confidence bounds.
    """
    adoptions = []
    exposures = []
    times_to_peak = []
    durations = []

    for t in range(num_trials):
        rng = np.random.RandomState(base_seed + t * 997)
        res = simulator.simulate(seed_users, item_id, rng)
        adoptions.append(res.total_adoptions)
        exposures.append(res.total_exposures)
        times_to_peak.append(res.time_to_peak)
        durations.append(res.steps_run)

    return {
        "mean_adoptions": float(np.mean(adoptions)),
        "std_adoptions": float(np.std(adoptions)),
        "mean_exposures": float(np.mean(exposures)),
        "std_exposures": float(np.std(exposures)),
        "mean_time_to_peak": float(np.mean(times_to_peak)),
        "std_time_to_peak": float(np.std(times_to_peak)),
        "mean_duration": float(np.mean(durations)),
        "std_duration": float(np.std(durations)),
        "num_trials": num_trials
    }
