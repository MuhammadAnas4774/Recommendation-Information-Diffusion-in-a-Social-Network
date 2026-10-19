"""
recommenders.py
Requirement 2: Learn preferences and generate recommendations

Implements two substantially different recommendation approaches from scratch:
1. Model-Based: Biased Matrix Factorization (FunkSVD with User & Item Biases + L2 Regularization)
2. Memory-Based: User-Based Collaborative Filtering with Mean-Centering and Significance Weighting

Includes:
- Prediction for unseen items
- Top-K ranked recommendation generation
- Cold-start analysis (sparse users & sparse items)
- Robust handling of user rating scale variations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from environment import SocialEnvironment


class BaseRecommender:
    """Abstract base class for recommendation algorithms."""
    def __init__(self, name: str):
        self.name = name

    def fit(self, env: SocialEnvironment) -> None:
        raise NotImplementedError

    def predict(self, u: int, i: int) -> float:
        raise NotImplementedError

    def recommend(self, u: int, env: SocialEnvironment, top_k: int = 10,
                  filter_consumed: bool = True) -> List[Tuple[int, float]]:
        """
        Generates top-K ranked recommendations for user u for unconsumed items.
        """
        consumed = set(env.train_ratings.get(u, {}).keys()) if filter_consumed else set()
        scores = []
        for i in range(env.num_items):
            if i not in consumed:
                pred = self.predict(u, i)
                scores.append((i, pred))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class BiasedMatrixFactorization(BaseRecommender):
    """
    Approach 1: Latent Factor Model (FunkSVD with User & Item Biases)
    Model: r_hat(u, i) = mu + b_u + b_i + P_u · Q_i^T
    Optimized via Stochastic Gradient Descent (SGD) with L2 regularization.
    """
    def __init__(self, latent_dim: int = 20, learning_rate: float = 0.01,
                 reg_lambda: float = 0.05, epochs: int = 25, random_seed: int = 42):
        super().__init__("Biased_Matrix_Factorization")
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.reg = reg_lambda
        self.epochs = epochs
        self.random_seed = random_seed

        # Learned parameters
        self.mu: float = 3.99
        self.b_u: np.ndarray = np.array([])
        self.b_i: np.ndarray = np.array([])
        self.P: np.ndarray = np.array([[]])  # Users: (num_users, latent_dim)
        self.Q: np.ndarray = np.array([[]])  # Items: (num_items, latent_dim)
        self.loss_history: List[float] = []

    def fit(self, env: SocialEnvironment) -> None:
        np.random.seed(self.random_seed)
        self.mu = env.global_mean
        n_users = env.num_users
        n_items = env.num_items

        # Initialize biases to 0 and latent factors with small random Gaussian values
        self.b_u = np.zeros(n_users, dtype=np.float32)
        self.b_i = np.zeros(n_items, dtype=np.float32)
        self.P = np.random.normal(0.0, 0.1, (n_users, self.latent_dim)).astype(np.float32)
        self.Q = np.random.normal(0.0, 0.1, (n_items, self.latent_dim)).astype(np.float32)

        # Collect train samples for SGD
        train_samples = []
        for u, items in env.train_ratings.items():
            for i, r in items.items():
                train_samples.append((u, i, r))
        train_samples = np.array(train_samples, dtype=np.float32)
        n_samples = len(train_samples)

        print(f"[{self.name}] Training SGD on {n_samples} interactions for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            sq_err_sum = 0.0

            for idx in indices:
                u = int(train_samples[idx, 0])
                i = int(train_samples[idx, 1])
                r = train_samples[idx, 2]

                # Current prediction
                pred = self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
                err = r - pred
                sq_err_sum += err * err

                # Gradient updates with L2 regularization
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])

                p_old = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * p_old - self.reg * self.Q[i])

            epoch_rmse = np.sqrt(sq_err_sum / n_samples)
            self.loss_history.append(float(epoch_rmse))
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1}/{self.epochs} - Train RMSE: {epoch_rmse:.4f}")

    def predict(self, u: int, i: int) -> float:
        pred = self.mu
        if 0 <= u < len(self.b_u):
            pred += self.b_u[u]
        if 0 <= i < len(self.b_i):
            pred += self.b_i[i]
        if 0 <= u < len(self.P) and 0 <= i < len(self.Q):
            pred += np.dot(self.P[u], self.Q[i])
        return float(np.clip(pred, 1.0, 5.0))


class UserBasedKNNRecommender(BaseRecommender):
    """
    Approach 2: Memory-Based Collaborative Filtering (User-Based k-NN with Mean Centering)
    Accounts for user rating scale differences (optimistic vs harsh raters) by subtracting
    user means, computing Pearson/Centered Cosine similarity, and adding significance weighting.
    """
    def __init__(self, k_neighbors: int = 20, min_overlap: int = 2,
                 significance_threshold: int = 5):
        super().__init__("User_Based_KNN")
        self.k = k_neighbors
        self.min_overlap = min_overlap
        self.sig_thresh = significance_threshold
        self.env: Optional[SocialEnvironment] = None
        self.user_centered_ratings: Dict[int, Dict[int, float]] = {}
        self.user_norms: Dict[int, float] = {}

    def fit(self, env: SocialEnvironment) -> None:
        self.env = env
        print(f"[{self.name}] Precomputing mean-centered user rating profiles...")
        for u in range(env.num_users):
            u_rats = env.train_ratings.get(u, {})
            u_mean = env.user_means[u]
            centered = {i: (r - u_mean) for i, r in u_rats.items()}
            self.user_centered_ratings[u] = centered
            norm = np.sqrt(sum(v * v for v in centered.values()))
            self.user_norms[u] = norm if norm > 1e-8 else 1e-8

    def _compute_similarity(self, u: int, v: int) -> float:
        """
        Computes centered cosine similarity (Pearson Correlation) between u and v
        with significance weighting.
        """
        u_rats = self.user_centered_ratings.get(u, {})
        v_rats = self.user_centered_ratings.get(v, {})
        common_items = set(u_rats.keys()).intersection(set(v_rats.keys()))

        if len(common_items) < self.min_overlap:
            return 0.0

        dot = sum(u_rats[i] * v_rats[i] for i in common_items)
        denom = self.user_norms[u] * self.user_norms[v]
        sim = dot / denom if denom > 0 else 0.0

        # Significance weighting penalty for small overlap
        if len(common_items) < self.sig_thresh:
            sim *= (len(common_items) / self.sig_thresh)

        return float(np.clip(sim, -1.0, 1.0))

    def predict(self, u: int, i: int) -> float:
        if self.env is None:
            return 3.99

        u_mean = self.env.user_means[u] if u < len(self.env.user_means) else self.env.global_mean
        i_raters = self.env.train_item_ratings.get(i, {})

        if not i_raters:
            # Cold-start item fallback
            return float(np.clip(u_mean, 1.0, 5.0))

        # Find similarities with candidates who rated item i
        similarities = []
        for v, r_vi in i_raters.items():
            if v == u:
                continue
            sim = self._compute_similarity(u, v)
            if sim > 0:  # Only positive peer correlation
                similarities.append((v, sim, r_vi - self.env.user_means[v]))

        if not similarities:
            return float(np.clip(u_mean, 1.0, 5.0))

        # Select top-K neighbors
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_neighbors = similarities[:self.k]

        numerator = sum(sim * diff for _, sim, diff in top_neighbors)
        denominator = sum(sim for _, sim, _ in top_neighbors)

        if denominator < 1e-6:
            return float(np.clip(u_mean, 1.0, 5.0))

        pred = u_mean + (numerator / denominator)
        return float(np.clip(pred, 1.0, 5.0))
