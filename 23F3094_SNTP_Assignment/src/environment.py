"""
environment.py
Requirement 1: Represent the Social Environment

Defines data structures and environment representation for:
- Users, Items, and Explicit Ratings
- Directed Trust Network (Social Graph)
- Incomplete Interaction Histories (Sparsity, Cold-start analysis)
- Varying Strengths of Social Influence (Degree-normalized & Homophily weights)
- Distinguishing Observed vs Inferred Information
- Train/Test Split (leakage-free)
"""

import os
import zipfile
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional


class SocialEnvironment:
    """
    Encapsulates the social recommendation and information diffusion environment.
    """
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # Mapping structures
        self.user2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.item2idx: Dict[int, int] = {}
        self.idx2item: Dict[int, int] = {}

        # Interaction data
        self.num_users: int = 0
        self.num_items: int = 0
        self.ratings_df: Optional[pd.DataFrame] = None
        self.trust_df: Optional[pd.DataFrame] = None

        # Train and Test sets
        self.train_ratings: Dict[int, Dict[int, float]] = {}  # u -> {i: rating}
        self.test_ratings: Dict[int, Dict[int, float]] = {}   # u -> {i: rating}
        self.train_item_ratings: Dict[int, Dict[int, float]] = {} # i -> {u: rating}

        # Network structures (Directed)
        # In Epinions: u -> v means u trusts v.
        # Recommendation influence: u seeks advice from v.
        # Information diffusion flow: v (influencer) shares/adopts item -> u (follower) is exposed.
        self.trusts: Dict[int, Set[int]] = {}     # u -> set of users trusted by u (out-edges)
        self.trusted_by: Dict[int, Set[int]] = {} # v -> set of users who trust v (in-edges / followers)

        # Inferred influence edge weights
        # weight[(v, u)]: propagation probability from v to u during diffusion
        self.edge_influence_weights: Dict[Tuple[int, int], float] = {}
        # normalized social regularizer matrix: S[u, v]
        self.social_trust_weights: Dict[int, Dict[int, float]] = {}

        # User and Item mean ratings
        self.global_mean: float = 3.99
        self.user_means: np.ndarray = np.array([])
        self.item_means: np.ndarray = np.array([])

    def load_from_zips(self, ratings_zip_path: str, trust_zip_path: str,
                       sample_users: Optional[int] = None,
                       min_user_ratings: int = 5,
                       min_item_ratings: int = 3,
                       train_ratio: float = 0.8) -> None:
        """
        Loads ratings and trust from zip archives, filters/samples if specified,
        maps IDs to continuous indices, and builds network/interaction matrices.
        """
        print("[Environment] Loading raw ratings from zip...")
        with zipfile.ZipFile(ratings_zip_path, 'r') as z:
            with z.open('epinions/ratings_data.txt') as f:
                raw_ratings = pd.read_csv(f, sep=r'\s+', header=None,
                                          names=['user_id', 'item_id', 'rating'])

        print("[Environment] Loading raw trust network from zip...")
        with zipfile.ZipFile(trust_zip_path, 'r') as z:
            with z.open('epinions/trust_data.txt') as f:
                raw_trust = pd.read_csv(f, sep=r'\s+', header=None,
                                        names=['source_user_id', 'target_user_id', 'trust'])

        self._process_data(raw_ratings, raw_trust, sample_users,
                           min_user_ratings, min_item_ratings, train_ratio)

    def load_from_csvs(self, ratings_csv_path: str, trust_csv_path: str,
                       train_ratio: float = 0.8) -> None:
        """
        Loads pre-extracted or sampled CSV files.
        """
        print(f"[Environment] Loading ratings from CSV: {ratings_csv_path}")
        raw_ratings = pd.read_csv(ratings_csv_path)
        print(f"[Environment] Loading trust from CSV: {trust_csv_path}")
        raw_trust = pd.read_csv(trust_csv_path)

        self._process_data(raw_ratings, raw_trust, sample_users=None,
                           min_user_ratings=1, min_item_ratings=1,
                           train_ratio=train_ratio)

    def _process_data(self, raw_ratings: pd.DataFrame, raw_trust: pd.DataFrame,
                      sample_users: Optional[int],
                      min_user_ratings: int,
                      min_item_ratings: int,
                      train_ratio: float) -> None:
        """
        Processes data, samples connected subnetwork if requested, builds indices and splits.
        """
        # Ensure correct column names
        raw_ratings.columns = ['user_id', 'item_id', 'rating']
        raw_trust.columns = ['source_user_id', 'target_user_id', 'trust']

        if sample_users is not None and sample_users < raw_ratings['user_id'].nunique():
            print(f"[Environment] Sampling connected subnetwork with top ~{sample_users} active users...")
            # Pick high-interaction users with active trust links to ensure a dense, realistic network
            user_counts = raw_ratings['user_id'].value_counts()
            trust_users = set(raw_trust['source_user_id']).union(set(raw_trust['target_user_id']))
            eligible_users = [u for u in user_counts.index if u in trust_users]

            selected_users = set(eligible_users[:sample_users])
            raw_ratings = raw_ratings[raw_ratings['user_id'].isin(selected_users)].copy()
            raw_trust = raw_trust[raw_trust['source_user_id'].isin(selected_users) &
                                  raw_trust['target_user_id'].isin(selected_users)].copy()

            # Filter items with min ratings
            item_counts = raw_ratings['item_id'].value_counts()
            active_items = set(item_counts[item_counts >= min_item_ratings].index)
            raw_ratings = raw_ratings[raw_ratings['item_id'].isin(active_items)].copy()

        # Build contiguous index maps
        unique_users = sorted(list(set(raw_ratings['user_id']).union(
            set(raw_trust['source_user_id'])).union(set(raw_trust['target_user_id']))))
        unique_items = sorted(raw_ratings['item_id'].unique().tolist())

        self.num_users = len(unique_users)
        self.num_items = len(unique_items)
        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.idx2user = {i: u for i, u in enumerate(unique_users)}
        self.item2idx = {it: i for i, it in enumerate(unique_items)}
        self.idx2item = {i: it for i, it in enumerate(unique_items)}

        # Remap dataframe columns
        raw_ratings['u_idx'] = raw_ratings['user_id'].map(self.user2idx)
        raw_ratings['i_idx'] = raw_ratings['item_id'].map(self.item2idx)
        raw_trust['src_idx'] = raw_trust['source_user_id'].map(self.user2idx)
        raw_trust['tgt_idx'] = raw_trust['target_user_id'].map(self.user2idx)

        # Drop any unmapped (safe fallback)
        raw_ratings = raw_ratings.dropna(subset=['u_idx', 'i_idx']).astype({'u_idx': int, 'i_idx': int})
        raw_trust = raw_trust.dropna(subset=['src_idx', 'tgt_idx']).astype({'src_idx': int, 'tgt_idx': int})

        self.ratings_df = raw_ratings
        self.trust_df = raw_trust

        # Build Trust Graph
        for u in range(self.num_users):
            self.trusts[u] = set()
            self.trusted_by[u] = set()

        for _, row in raw_trust.iterrows():
            src = int(row['src_idx'])
            tgt = int(row['tgt_idx'])
            self.trusts[src].add(tgt)      # src trusts tgt
            self.trusted_by[tgt].add(src)  # tgt is trusted by src (src follows tgt)

        # Split Ratings into Train / Test (80/20) with random seed
        np.random.seed(self.random_seed)
        shuffled_indices = np.random.permutation(len(raw_ratings))
        train_len = int(len(raw_ratings) * train_ratio)
        train_idx = shuffled_indices[:train_len]
        test_idx = shuffled_indices[train_len:]

        train_df = raw_ratings.iloc[train_idx]
        test_df = raw_ratings.iloc[test_idx]

        for u in range(self.num_users):
            self.train_ratings[u] = {}
            self.test_ratings[u] = {}
        for i in range(self.num_items):
            self.train_item_ratings[i] = {}

        for _, row in train_df.iterrows():
            u = int(row['u_idx'])
            i = int(row['i_idx'])
            r = float(row['rating'])
            self.train_ratings[u][i] = r
            self.train_item_ratings[i][u] = r

        for _, row in test_df.iterrows():
            u = int(row['u_idx'])
            i = int(row['i_idx'])
            r = float(row['rating'])
            self.test_ratings[u][i] = r

        # Calculate Mean Statistics
        all_train_ratings = [r for u_rats in self.train_ratings.values() for r in u_rats.values()]
        self.global_mean = float(np.mean(all_train_ratings)) if all_train_ratings else 3.99

        self.user_means = np.full(self.num_users, self.global_mean)
        for u, rats in self.train_ratings.items():
            if len(rats) > 0:
                self.user_means[u] = np.mean(list(rats.values()))

        self.item_means = np.full(self.num_items, self.global_mean)
        for i, rats in self.train_item_ratings.items():
            if len(rats) > 0:
                self.item_means[i] = np.mean(list(rats.values()))

        # Compute Inferred Strengths of Social Influence
        self._compute_social_weights()

        print(f"[Environment] Setup complete:")
        print(f"  Users: {self.num_users}, Items: {self.num_items}")
        print(f"  Train ratings: {len(train_df)}, Test ratings: {len(test_df)}")
        print(f"  Trust edges: {len(raw_trust)}")
        print(f"  Global Rating Mean: {self.global_mean:.3f}")

    def _compute_social_weights(self) -> None:
        """
        Computes varying strengths of social influence:
        1. Normalized social trust weights S[u, v] for Social Regularization:
           S[u, v] = 1 / |T(u)| where v in T(u)
        2. Edge diffusion propagation probability p(v -> u):
           If v is trusted by u, v can influence u.
           p(v -> u) combines degree-based attenuation and preference homophily (Jaccard similarity).
        """
        for u in range(self.num_users):
            trusted_friends = self.trusts[u]
            if len(trusted_friends) > 0:
                weight = 1.0 / len(trusted_friends)
                self.social_trust_weights[u] = {v: weight for v in trusted_friends}
            else:
                self.social_trust_weights[u] = {}

        # Diffusion influence weights: v -> u (influencer v to follower u)
        for v in range(self.num_users):
            followers = self.trusted_by[v]
            v_items = set(self.train_ratings[v].keys())
            for u in followers:
                u_items = set(self.train_ratings[u].keys())
                # Taste homophily via Jaccard index of co-rated items
                union_len = len(v_items.union(u_items))
                jaccard = len(v_items.intersection(u_items)) / union_len if union_len > 0 else 0.0

                # Base propagation probability attenuated by in-degree of follower + boosted by homophily
                in_deg = max(1, len(self.trusts[u]))
                base_prob = 1.0 / in_deg
                # Blend structural influence and taste homophily
                prop_prob = float(np.clip(0.5 * base_prob + 0.5 * jaccard + 0.1, 0.05, 0.95))
                self.edge_influence_weights[(v, u)] = prop_prob

    def get_sparsity(self) -> float:
        """Returns the rating matrix sparsity ratio."""
        total_possible = self.num_users * self.num_items
        observed = sum(len(rats) for rats in self.train_ratings.values()) + sum(len(rats) for rats in self.test_ratings.values())
        return 1.0 - (observed / total_possible if total_possible > 0 else 0.0)

    def get_user_degree(self, u: int) -> Tuple[int, int]:
        """Returns (out-degree, in-degree) in the trust graph."""
        return len(self.trusts[u]), len(self.trusted_by[u])

    def save_sample_csv(self, ratings_out: str, trust_out: str) -> None:
        """Saves current environment as CSVs for reproducible standalone demo."""
        os.makedirs(os.path.dirname(ratings_out), exist_ok=True)
        os.makedirs(os.path.dirname(trust_out), exist_ok=True)
        if self.ratings_df is not None:
            self.ratings_df[['user_id', 'item_id', 'rating']].to_csv(ratings_out, index=False)
        if self.trust_df is not None:
            self.trust_df[['source_user_id', 'target_user_id', 'trust']].to_csv(trust_out, index=False)
        print(f"[Environment] Saved sample datasets to {ratings_out} and {trust_out}")
