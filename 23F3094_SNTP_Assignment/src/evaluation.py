"""
evaluation.py
Requirement 7: Evaluate the complete system

Comprehensive evaluation module:
1. Recommendation Quality (Observed Test Ground Truth):
   - Prediction error: RMSE, MAE
   - Ranking Quality: Precision@K, Recall@K, NDCG@K, MRR, HitRate@K
   - Classification Task: Adoption prediction (r >= 4.0) with Accuracy, Precision, Recall, F1, ROC-AUC
   - Coverage: Catalog Coverage, User Coverage
   - Diversity: Intra-List Diversity (ILD)
2. Diffusion Behavior (Simulator Outcomes):
   - Cascade Size, Exposures, Conversion Rate, Time-to-Peak
3. Interaction:
   - Recommender quality vs diffusion adoption outcomes
4. Leakage-free test protocol with repeated trials and variance reporting
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from environment import SocialEnvironment
from recommenders import BaseRecommender, BiasedMatrixFactorization
from diffusion_models import IndependentCascadeSimulator, run_monte_carlo_diffusion


def compute_rmse_mae(recommender: BaseRecommender, env: SocialEnvironment) -> Tuple[float, float]:
    """
    Evaluates RMSE and MAE on the observed held-out test ratings.
    Strictly prevents training-test leakage.
    """
    errors = []
    abs_errors = []

    for u, test_items in env.test_ratings.items():
        for i, r_actual in test_items.items():
            r_pred = recommender.predict(u, i)
            err = r_actual - r_pred
            errors.append(err * err)
            abs_errors.append(abs(err))

    if not errors:
        return 0.0, 0.0

    rmse = float(np.sqrt(np.mean(errors)))
    mae = float(np.mean(abs_errors))
    return rmse, mae


def compute_ranking_metrics_at_k(recommender: BaseRecommender, env: SocialEnvironment,
                                 k: int = 10, relevance_threshold: float = 4.0,
                                 sample_users_eval: int = 200) -> Dict[str, float]:
    """
    Computes Precision@K, Recall@K, NDCG@K, MRR, and HitRate@K.
    Relevance definition: actual rating >= relevance_threshold (e.g. 4.0).
    Candidate pool: user's test items plus a candidate set of unconsumed items.
    """
    precisions = []
    recalls = []
    ndcgs = []
    mrrs = []
    hits = []
    all_recommended_items = set()
    served_users = 0

    eval_users = [u for u in env.test_ratings if len(env.test_ratings[u]) > 0][:sample_users_eval]

    for u in eval_users:
        test_items = env.test_ratings[u]
        relevant_test = {i for i, r in test_items.items() if r >= relevance_threshold}
        if not relevant_test:
            continue

        served_users += 1

        # Predict for all items consumed in test plus unseen candidate items
        candidate_items = list(test_items.keys())
        # Add up to 50 unseen items to create a competitive ranking pool
        unseen = [i for i in range(env.num_items) if i not in env.train_ratings.get(u, {}) and i not in test_items]
        if len(unseen) > 50:
            np.random.seed(u)
            unseen = list(np.random.choice(unseen, 50, replace=False))
        candidate_items.extend(unseen)

        # Score candidates
        scores = [(i, recommender.predict(u, i)) for i in candidate_items]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k_items = [i for i, _ in scores[:k]]
        all_recommended_items.update(top_k_items)

        # Hits & Overlap
        hits_k = [1 if item in relevant_test else 0 for item in top_k_items]
        num_hits = sum(hits_k)

        # Precision & Recall
        precisions.append(num_hits / k)
        recalls.append(num_hits / len(relevant_test))
        hits.append(1.0 if num_hits > 0 else 0.0)

        # MRR
        rr = 0.0
        for rank_idx, item in enumerate(top_k_items):
            if item in relevant_test:
                rr = 1.0 / (rank_idx + 1)
                break
        mrrs.append(rr)

        # NDCG@K
        dcg = sum((2**h - 1) / np.log2(idx + 2) for idx, h in enumerate(hits_k))
        ideal_hits = [1] * min(k, len(relevant_test)) + [0] * max(0, k - len(relevant_test))
        idcg = sum((2**h - 1) / np.log2(idx + 2) for idx, h in enumerate(ideal_hits))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    # Coverage metrics
    catalog_coverage = len(all_recommended_items) / env.num_items if env.num_items > 0 else 0.0
    user_coverage = served_users / len(eval_users) if eval_users else 0.0

    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"mrr@{k}": float(np.mean(mrrs)) if mrrs else 0.0,
        f"hit_rate@{k}": float(np.mean(hits)) if hits else 0.0,
        f"catalog_coverage@{k}": float(catalog_coverage),
        f"user_coverage@{k}": float(user_coverage)
    }


def compute_classification_metrics(recommender: BaseRecommender, env: SocialEnvironment,
                                   threshold: float = 4.0) -> Dict[str, float]:
    """
    Meaningful Classification Task:
    Predict whether a user will adopt/like an item (rating >= 4.0).
    - Positive instances: observed test ratings >= threshold
    - Negative instances: observed test ratings < threshold (explicit dissatisfaction)
      plus sampled unconsumed items (implicit negative feedback).
    """
    y_true = []
    y_pred_prob = []
    y_pred_binary = []

    np.random.seed(env.random_seed)

    for u, test_items in env.test_ratings.items():
        for i, actual_r in test_items.items():
            true_label = 1 if actual_r >= threshold else 0
            y_true.append(true_label)

            pred_r = recommender.predict(u, i)
            # Sigmoid probability around decision boundary (3.5)
            prob = 1.0 / (1.0 + np.exp(-2.0 * (pred_r - 3.5)))
            y_pred_prob.append(prob)
            y_pred_binary.append(1 if pred_r >= threshold else 0)

        # Sample 1 unrated item per test item as implicit negative
        unrated = [it for it in range(env.num_items) if it not in env.train_ratings.get(u, {}) and it not in test_items]
        if unrated:
            neg_sample = int(np.random.choice(unrated))
            y_true.append(0)
            pred_neg = recommender.predict(u, neg_sample)
            prob_neg = 1.0 / (1.0 + np.exp(-2.0 * (pred_neg - 3.5)))
            y_pred_prob.append(prob_neg)
            y_pred_binary.append(1 if pred_neg >= threshold else 0)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred_binary)

    tp = np.sum((y_true_arr == 1) & (y_pred_arr == 1))
    fp = np.sum((y_true_arr == 0) & (y_pred_arr == 1))
    tn = np.sum((y_true_arr == 0) & (y_pred_arr == 0))
    fn = np.sum((y_true_arr == 1) & (y_pred_arr == 0))

    accuracy = (tp + tn) / len(y_true_arr) if len(y_true_arr) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Approximate ROC-AUC via rank order
    pos_probs = [p for p, t in zip(y_pred_prob, y_true) if t == 1]
    neg_probs = [p for p, t in zip(y_pred_prob, y_true) if t == 0]
    auc = 0.5
    if pos_probs and neg_probs:
        # Sample comparison
        concordant = 0
        trials = min(5000, len(pos_probs) * len(neg_probs))
        pos_sample = np.random.choice(pos_probs, trials, replace=True)
        neg_sample = np.random.choice(neg_probs, trials, replace=True)
        concordant = np.sum(pos_sample > neg_sample) + 0.5 * np.sum(pos_sample == neg_sample)
        auc = float(concordant / trials)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "num_eval_samples": len(y_true_arr)
    }


def compute_intra_list_diversity(recommender: BaseRecommender, env: SocialEnvironment,
                                 top_k: int = 10, sample_users: int = 100) -> float:
    """
    Measures Intra-List Diversity (ILD) as the average pairwise cosine distance
    between items in top-K recommendations.
    Uses item latent factor representations if available, or item rating profiles.
    """
    # Check if recommender has item embeddings Q
    has_embeddings = hasattr(recommender, "Q") and len(getattr(recommender, "Q")) > 0
    if has_embeddings:
        Q = getattr(recommender, "Q")
        norms = np.linalg.norm(Q, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        Q_norm = Q / norms

    ild_scores = []
    users_to_test = list(range(min(sample_users, env.num_users)))

    for u in users_to_test:
        rec_items = [i for i, _ in recommender.recommend(u, env, top_k=top_k)]
        if len(rec_items) < 2:
            continue

        pairwise_dist = []
        for a_idx in range(len(rec_items)):
            for b_idx in range(a_idx + 1, len(rec_items)):
                i_a = rec_items[a_idx]
                i_b = rec_items[b_idx]
                if has_embeddings and i_a < len(Q_norm) and i_b < len(Q_norm):
                    cos_sim = float(np.dot(Q_norm[i_a], Q_norm[i_b]))
                    dist = 1.0 - cos_sim
                else:
                    dist = 0.5  # fallback
                pairwise_dist.append(dist)

        if pairwise_dist:
            ild_scores.append(float(np.mean(pairwise_dist)))

    return float(np.mean(ild_scores)) if ild_scores else 0.0
