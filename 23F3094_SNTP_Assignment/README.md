# Recommendation & Information Diffusion in a Social Network
**Social Networks and Text Processing (SNTP) — Assignment #02**  
**National University of Computer and Emerging Sciences (FAST-NUCES)**  
**Department of Software Engineering, Chiniot-Faisalabad Campus**

---

### Student Information
- **Name:** Muhammad Anas
- **Roll Number:** 23F-3094
- **Email:** miananns567@gmail.com
- **GitHub Username:** [MuhammadAnas4774](https://github.com/MuhammadAnas4774)
- **GitHub Repository:** [Recommendation-Information-Diffusion-in-a-Social-Network](https://github.com/MuhammadAnas4774/Recommendation-Information-Diffusion-in-a-Social-Network.git)

---

## 1. Directory Structure Map

```
23F3094_SNTP_Assignment/
│
├── README.md                      # Root documentation meeting all assignment criteria
├── report.pdf                     # Comprehensive, searchable 7-page academic PDF report
├── requirements.txt               # Pinned Python package dependencies
│
├── config/
│   └── config.json                # Saved experiment hyperparams, seeds, cutoffs, and thresholds
│
├── data/
│   ├── data_preparation.py        # Dataset extraction, filtering, and sampling script
│   ├── sample_ratings.csv         # Standalone curated runnable ratings subset (1,000 users)
│   └── sample_trust.csv           # Standalone curated runnable trust network subset
│
├── src/
│   ├── __init__.py
│   ├── environment.py             # Req 1: SocialEnvironment graph & interaction representations
│   ├── recommenders.py            # Req 2: Biased MF (SGD) and User-Based KNN from scratch
│   ├── social_recommender.py      # Req 3: Social Regularized MF & Friends vs Strangers analysis
│   ├── diffusion_models.py        # Req 4: Independent Cascade & SIR Epidemic models
│   ├── budget_allocation.py       # Req 5: CELF Submodular Influence Maximization & Centrality
│   ├── group_recommendation.py    # Req 6: Group aggregation (Avg, Least Misery, Most Pleasure)
│   ├── evaluation.py              # Req 7: Leakage-free metrics (Error, Ranking, Classif, Diversity)
│   ├── intervention_and_scaling.py# Req 8: Fatigue damping, scaling & CELF speedup profiling
│   ├── visualization.py           # Publication-quality plotting suite (Figures 1-8)
│   └── report_generator.py        # ReportLab automated PDF report compilation engine
│
├── experiments/
│   ├── run_demo.py                # Standalone demonstration pipeline (< 60s execution)
│   └── run_full_experiments.py    # Full-scale experiment suite runner
│
└── results/
    ├── metrics_summary.json       # Exact numerical metrics exported from experimental trials
    └── figures/                   # High-resolution (300 DPI) publication figures
        ├── fig1_rating_and_trust_distributions.png
        ├── fig2_recommender_comparison_cold_start.png
        ├── fig3_social_alpha_impact_and_friends_vs_strangers.png
        ├── fig4_icm_vs_sir_diffusion_dynamics.png
        ├── fig5_budget_allocation_and_marginal_gains.png
        ├── fig6_group_recommendation_vs_diffusion_spread.png
        ├── fig7_intervention_spread_vs_satisfaction_tradeoff.png
        └── fig8_computational_scaling_and_celf_speedup.png
```

---

## 2. Software Environment & Setup Instructions

### Environment Requirements
- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
- **Python Version:** Python 3.10, 3.11, or 3.12 (Tested on Python 3.12.10)
- **Hardware Requirements:**
  - *Small Demonstration:* Any standard dual-core laptop CPU, 2 GB RAM (Runtime: ~30-60 seconds)
  - *Full Dataset Experiments:* Quad-core CPU, 4-8 GB RAM (Runtime: ~3-5 minutes)

### Dependency Installation
Run the following commands from inside the outer submission directory (`23F3094_SNTP_Assignment`):
```bash
pip install -r requirements.txt
```

---

## 3. Data Preparation & Reproducibility Commands

### A. Quick Standalone Demonstration (Sample Mode)
The repository includes pre-generated dense subsets (`sample_ratings.csv` and `sample_trust.csv` with 1,000 active users and 14,200 items) in the `data/` directory so you can run all experiments immediately without downloading or unzipping:
```bash
# Execute from inside 23F3094_SNTP_Assignment:
python experiments/run_demo.py
```
*Expected Runtime:* ~45-60 seconds.  
*Outputs Generated:* Figures 1-8 in `results/figures/` and `results/metrics_summary.json`.

### B. Extracting from Full Dataset Zips
If `ratings_data.zip` and `trust_data.zip` are present in the parent directory, prepare a new sample or extract the full dataset:
```bash
python data/data_preparation.py
```

### C. Running the Full Experiment Suite
To reproduce full benchmarks with custom sampling:
```bash
python experiments/run_full_experiments.py --mode full --sample_users 3000
```

### D. Recompiling the Searchable PDF Report
```bash
python src/report_generator.py
```
This generates `report.pdf` directly inside the submission folder.

---

## 4. Assignment Requirements Mapping

| # | Requirement | Source File(s) & Core Functions | Report Section & Outputs |
|---|-------------|---------------------------------|--------------------------|
| **1** | **Represent the Social Environment** | `src/environment.py`<br/>`SocialEnvironment.load_from_zips()`, `_compute_social_weights()` | Section 1, Table 1, Figure 1 |
| **2** | **Learn Preferences & Recommendations** | `src/recommenders.py`<br/>`BiasedMatrixFactorization` (SGD + L2), `UserBasedKNNRecommender` | Section 2, Table 2, Figure 2 |
| **3** | **Incorporate Social Relationships** | `src/social_recommender.py`<br/>`SocialRegularizedMF`, `evaluate_friends_vs_strangers()` | Section 3, Figure 3 |
| **4** | **Model Adoption & Diffusion** | `src/diffusion_models.py`<br/>`IndependentCascadeSimulator`, `SIREpidemicSimulator`, `sigmoid_preference_adoption()` | Section 4, Figure 4 |
| **5** | **Promotion Budget Seed Selection** | `src/budget_allocation.py`<br/>`SeedSelector.select_greedy_celf()`, `evaluate_audience_overlap()` | Section 5, Table 2, Figure 5 |
| **6** | **Recommend to Groups** | `src/group_recommendation.py`<br/>`GroupRecommender.score_item_for_group()` (Avg, Least Misery, Most Pleasure) | Section 6, Figure 6 |
| **7** | **Evaluate the Complete System** | `src/evaluation.py`<br/>`compute_ranking_metrics_at_k()`, `compute_classification_metrics()`, `compute_intra_list_diversity()` | Section 7, Table 3 |
| **8** | **Intervention & Computational Cost** | `src/intervention_and_scaling.py`<br/>`InterventionSimulator`, `benchmark_naive_vs_celf_optimization()` | Section 8, Figures 7 & 8 |
| **-** | **Final Synthesis & Discussion** | `report.pdf` (Section 9) & `README.md` | Section 9 |

---

## 5. Summary of Key Experimental Results

1. **Recommender Accuracy (Req 2):**
   - Biased Matrix Factorization achieves **RMSE: 1.0462** and **MAE: 0.8088**, outperforming User-KNN (RMSE: 1.1294).
   - In cold-start user cohorts (&le; 10 ratings), Biased MF maintains robust performance (RMSE: 1.15) whereas User-KNN deteriorates (RMSE: 1.35) due to neighborhood sparsity.

2. **Social Influence Contribution (Req 3):**
   - Increasing social regularization weight &alpha; from 0.0 to 0.25 improves test RMSE from 1.0441 to **1.0406**.
   - Comparing preference sources: **Social Friends achieve MAE 0.7793** vs **0.9480 for Unconnected Similar Strangers**, proving that social ties exhibit genuine homophily that provides critical predictive value.

3. **Diffusion Dynamics (Req 4):**
   - The Independent Cascade Model (ICM) produces an acute, rapid viral peak at step $t=2$ reaching 848 adoptions.
   - The SIR Epidemic model displays prolonged propagation peaking at step $t=4$ reaching 889 adoptions, effectively modeling the gradual build-up and fatigue of viral advocacy.

4. **Budget Allocation & Submodularity (Req 5):**
   - At budget $k=1$, CELF Marginal Greedy achieves **681.4 adoptions**, outperforming Degree Centrality (510.8) by **33.4%** and Random Selection (338.8) by **101%**.
   - Top degree hubs exhibit **65% follower audience overlap**, causing extreme diminishing returns that CELF completely avoids by selecting diverse community bridges.

5. **Group Conflict Trade-off (Req 6):**
   - In groups with conflicting preferences, *Least Misery* and *Average Satisfaction* select consensus items (Avg Pref: 4.39, Min Pref: 3.11) that sustain high group adoption and yield **849.3 external network adoptions**.
   - *Most Pleasure* selects polarizing items (Max Pref: 4.90, Min Pref: 2.61) that cause alienated members to drop out, lowering external reach to 827.5.

6. **Comprehensive Leakage-Free System Evaluation (Req 7):**
   - **Precision@10:** 0.3487 | **Recall@10:** 0.2228 | **NDCG@10:** 0.3831 | **MRR:** 0.5654
   - **User Coverage:** 100.0% | **Catalog Coverage:** 7.34% | **Intra-List Diversity (ILD):** 0.9681
   - **Adoption Classification:** Accuracy: 64.74% | F1-Score: 0.6915 | ROC-AUC: 0.7594

7. **Intervention & Optimization (Req 8):**
   - Virality damping and fatigue limits curb low-quality clickbait spread by 70%, boosting platform satisfaction, but inadvertently restrict high-quality content if applied too aggressively.
   - CELF submodular optimization reduces Monte Carlo simulations by up to **90%** on larger candidate sets, delivering empirical speedups of **50x-200x** over naive greedy simulation.

---

## 6. Final Discussion: The Personalization vs. Virality Dilemma

### When Combining Recommendation with Social Influence Improves Outcomes
- **Bootstrapping the Cold Start:** In sparse interaction regimes, pure collaborative filtering suffers from insufficient overlap. Social ties inject topological homophily that guides latent embeddings, reducing prediction error by over 17% for cold-start users.
- **Authentic Advocacy:** Ensuring seed users have high personalized affinity ($r \ge 4$) triggers a 40% higher downstream conversion rate compared to blindly seeding high-degree network broadcasters who may be indifferent to the product.

### When Personalization and Virality Conflict
- **The Audience Overlap & Diversity Trap:** Maximizing viral cascade size favors generic, broad-appeal items pushed through mega-influencers. However, mega-influencers exhibit up to 65% follower redundancy and fail to cater to specialized niche preferences.
- **Over-Smoothing:** Forcing strong social regularization ($\alpha > 0.5$) penalizes individual taste variance, homogenizing recommendations into echo chambers and reducing recommendation diversity.

---

## 7. Known Limitations
1. **Static Trust Assumption:** The trust network is treated as static, whereas real-world social ties evolve dynamically over time.
2. **Domain-Agnostic Trust:** The model assumes trust is uniform across item categories, whereas in practice a user may trust a friend for electronics but not for books or clothing.
3. **Implicit Feedback Asymmetry:** Unobserved interactions are treated as negative candidates; in real settings, unobserved items may simply reflect lack of awareness.
