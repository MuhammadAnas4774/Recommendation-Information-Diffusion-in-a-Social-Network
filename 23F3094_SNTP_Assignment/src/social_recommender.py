"""
social_recommender.py
Requirement 3: Incorporate social relationships

Extends Matrix Factorization with Social Regularization (SocialMF):
L = 1/2 * sum((r_ui - r_hat_ui)^2) + lambda/2 * (||P||^2 + ||Q||^2 + ||b||^2)
    + alpha/2 * sum_{u} sum_{v in T(u)} S_uv * ||P_u - P_v||^2

Features:
- Controllable alpha parameter to isolate social network contribution
- Friends vs Socially Unconnected Similar Users comparative experiment
- Detailed breakdown of when social information helps vs when it hurts
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from environment import SocialEnvironment
from recommenders import BaseRecommender


class SocialRegularizedMF(BaseRecommender):
    """
    Social Regularized Matrix Factorization.
    Ties user latent preference vectors towards their trusted social neighbors.
    Alpha controls the strength of social regularization:
    - alpha = 0.0: Standard Biased Matrix Factorization (no social info)
    - alpha > 0.0: Latent preferences are constrained towards trusted peers
    """
    def __init__(self, latent_dim: int = 20, learning_rate: float = 0.01,
                 reg_lambda: float = 0.05, alpha: float = 0.1,
                 epochs: int = 25, random_seed: int = 42):
        super().__init__(f"Social_MF_alpha_{alpha:.2f}")
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.reg = reg_lambda
        self.alpha = alpha  # Social regularization weight
        self.epochs = epochs
        self.random_seed = random_seed

        self.mu: float = 3.99
        self.b_u: np.ndarray = np.array([])
        self.b_i: np.ndarray = np.array([])
        self.P: np.ndarray = np.array([[]])
        self.Q: np.ndarray = np.array([[]])
        self.loss_history: List[float] = []

    def fit(self, env: SocialEnvironment) -> None:
        np.random.seed(self.random_seed)
        self.mu = env.global_mean
        n_users = env.num_users
        n_items = env.num_items

        self.b_u = np.zeros(n_users, dtype=np.float32)
        self.b_i = np.zeros(n_items, dtype=np.float32)
        self.P = np.random.normal(0.0, 0.1, (n_users, self.latent_dim)).astype(np.float32)
        self.Q = np.random.normal(0.0, 0.1, (n_items, self.latent_dim)).astype(np.float32)

        # Build sparse trust adjacency lookup: user -> (friend_indices, weights)
        social_lookup = {}
        for u in range(n_users):
            if u in env.social_trust_weights and len(env.social_trust_weights[u]) > 0:
                friends = np.array(list(env.social_trust_weights[u].keys()), dtype=np.int32)
                weights = np.array(list(env.social_trust_weights[u].values()), dtype=np.float32)
                social_lookup[u] = (friends, weights)

        # Collect train samples
        train_samples = []
        for u, items in env.train_ratings.items():
            for i, r in items.items():
                train_samples.append((u, i, r))
        train_samples = np.array(train_samples, dtype=np.float32)
        n_samples = len(train_samples)

        print(f"[{self.name}] Training SocialMF (alpha={self.alpha}) on {n_samples} interactions...")

        for epoch in range(self.epochs):
            # Precompute friend centroid vectors for all users in O(|E|) time
            P_friends_avg = np.zeros_like(self.P)
            if self.alpha > 0.0:
                for u, (friends, weights) in social_lookup.items():
                    P_friends_avg[u] = np.dot(weights, self.P[friends])

            indices = np.random.permutation(n_samples)
            sq_err_sum = 0.0

            for idx in indices:
                u = int(train_samples[idx, 0])
                i = int(train_samples[idx, 1])
                r = train_samples[idx, 2]

                pred = self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
                err = r - pred
                sq_err_sum += err * err

                # Rating gradients
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])

                p_old = self.P[u].copy()

                # Base rating gradient for P[u] and Q[i]
                grad_p = err * self.Q[i] - self.reg * self.P[u]
                self.Q[i] += self.lr * (err * p_old - self.reg * self.Q[i])

                # Social Regularization gradient contribution: alpha * (P_friends_avg[u] - P[u])
                if self.alpha > 0.0 and u in social_lookup:
                    grad_p += self.alpha * (P_friends_avg[u] - self.P[u])

                self.P[u] += self.lr * grad_p

            epoch_rmse = np.sqrt(sq_err_sum / n_samples)
            self.loss_history.append(float(epoch_rmse))

    def predict(self, u: int, i: int) -> float:
        pred = self.mu
        if 0 <= u < len(self.b_u):
            pred += self.b_u[u]
        if 0 <= i < len(self.b_i):
            pred += self.b_i[i]
        if 0 <= u < len(self.P) and 0 <= i < len(self.Q):
            pred += np.dot(self.P[u], self.Q[i])
        return float(np.clip(pred, 1.0, 5.0))


def evaluate_friends_vs_strangers(env: SocialEnvironment, top_k_neighbors: int = 15,
                                  sample_eval_users: int = 150) -> Dict[str, float]:
    """
    Investigates whether friends provide more useful preference information
    than users who are similar in rating behavior but socially unconnected.
    
    Returns comparative evaluation metrics:
    - Friend-based RMSE & MAE
    - Unconnected Stranger-based RMSE & MAE
    - Cold-start user vs Active user performance splits
    """
    print(f"[Social Investigation] Comparing Friends vs Socially Unconnected Similar Users on {sample_eval_users} users...")

    friend_errors = []
    stranger_errors = []

    friend_errors_cold = []
    stranger_errors_cold = []

    friend_errors_active = []
    stranger_errors_active = []

    eval_users = [u for u in range(env.num_users) if len(env.test_ratings.get(u, {})) > 0][:sample_eval_users]

    for u in eval_users:
        test_items = env.test_ratings.get(u, {})
        if not test_items:
            continue

        friends = list(env.trusts.get(u, set()))
        u_active = len(env.train_ratings.get(u, {})) >= 10

        # Compute rating similarity with all other users to find unconnected similar users
        u_rats = env.train_ratings.get(u, {})
        u_mean = env.user_means[u]

        # Candidates who are NOT in trust network (neither u trusts v nor v trusts u)
        social_network_peers = env.trusts.get(u, set()).union(env.trusted_by.get(u, set()))
        stranger_candidates = []

        if len(u_rats) > 0:
            for v in range(env.num_users):
                if v == u or v in social_network_peers:
                    continue
                v_rats = env.train_ratings.get(v, {})
                overlap = set(u_rats.keys()).intersection(set(v_rats.keys()))
                if len(overlap) >= 2:
                    # Simple Pearson / Centered correlation
                    diff_u = [u_rats[it] - u_mean for it in overlap]
                    diff_v = [v_rats[it] - env.user_means[v] for it in overlap]
                    denom = np.sqrt(sum(d*d for d in diff_u)) * np.sqrt(sum(d*d for d in diff_v))
                    if denom > 1e-6:
                        sim = sum(du * dv for du, dv in zip(diff_u, diff_v)) / denom
                        if sim > 0:
                            stranger_candidates.append((v, sim))

        stranger_candidates.sort(key=lambda x: x[1], reverse=True)
        top_strangers = [v for v, _ in stranger_candidates[:top_k_neighbors]]

        # Evaluate predictions on test items using Friend ratings vs Stranger ratings
        for item, actual_r in test_items.items():
            # 1. Friends prediction
            f_ratings = [env.train_ratings[f][item] for f in friends if item in env.train_ratings.get(f, {})]
            if f_ratings:
                pred_friend = float(np.mean(f_ratings))
                err_f = abs(actual_r - pred_friend)
                friend_errors.append(err_f)
                if u_active:
                    friend_errors_active.append(err_f)
                else:
                    friend_errors_cold.append(err_f)

            # 2. Unconnected Strangers prediction
            s_ratings = [env.train_ratings[s][item] for s in top_strangers if item in env.train_ratings.get(s, {})]
            if s_ratings:
                pred_stranger = float(np.mean(s_ratings))
                err_s = abs(actual_r - pred_stranger)
                stranger_errors.append(err_s)
                if u_active:
                    stranger_errors_active.append(err_s)
                else:
                    stranger_errors_cold.append(err_s)

    results = {
        "friends_mae_overall": float(np.mean(friend_errors)) if friend_errors else 0.0,
        "strangers_mae_overall": float(np.mean(stranger_errors)) if stranger_errors else 0.0,
        "friends_mae_cold_start": float(np.mean(friend_errors_cold)) if friend_errors_cold else 0.0,
        "strangers_mae_cold_start": float(np.mean(stranger_errors_cold)) if stranger_errors_cold else 0.0,
        "friends_mae_active": float(np.mean(friend_errors_active)) if friend_errors_active else 0.0,
        "strangers_mae_active": float(np.mean(stranger_errors_active)) if stranger_errors_active else 0.0,
        "evaluated_friend_pairs": len(friend_errors),
        "evaluated_stranger_pairs": len(stranger_errors)
    }
    return results
