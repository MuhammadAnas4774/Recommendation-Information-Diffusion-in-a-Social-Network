"""
data_preparation.py
Prepares data from raw zip files and generates sample datasets for quick demonstration.
"""

import os
import sys

# Ensure src is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from environment import SocialEnvironment


def prepare_datasets(ratings_zip: str, trust_zip: str,
                     sample_out_ratings: str, sample_out_trust: str,
                     sample_users: int = 1000) -> SocialEnvironment:
    """
    Extracts raw zip data, creates a dense connected sample for demonstration,
    and writes out sample CSVs.
    """
    print("=" * 60)
    print("DATA PREPARATION & SAMPLING")
    print("=" * 60)
    env = SocialEnvironment(random_seed=42)
    env.load_from_zips(ratings_zip, trust_zip, sample_users=sample_users,
                       min_user_ratings=5, min_item_ratings=3, train_ratio=0.8)

    env.save_sample_csv(sample_out_ratings, sample_out_trust)
    print(f"Sample dataset ready with {env.num_users} users and {env.num_items} items.")
    return env


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    r_zip = os.path.join(base_dir, "ratings_data.zip")
    t_zip = os.path.join(base_dir, "trust_data.zip")

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    out_r = os.path.join(data_dir, "sample_ratings.csv")
    out_t = os.path.join(data_dir, "sample_trust.csv")

    prepare_datasets(r_zip, t_zip, out_r, out_t, sample_users=1000)
