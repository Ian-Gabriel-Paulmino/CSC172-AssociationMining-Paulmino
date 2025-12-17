# CSC172 Association Rule Mining Project Progress Report
**Student:** Ian Gabriel Paulmino, 2022-1729
**Date:** December 17, 2025  
**Repository:** https://github.com/Ian-Gabriel-Paulmino/CSC172-AssociationMining-Paulmino


## 📊 Current Status

| Milestone | Status | Notes |
|-----------|--------|-------|
| Dataset Preparation | Completed | 989,818 → 364,384 transactions (36.8% retention) |
| Data Preprocessing |  Completed | One-hot encoded 364,384 × 17 matrix |
| EDA & Visualization |  Completed | 8 charts + category frequencies + co-occurrence heatmap |
| Apriori Implementation |  Completed | 16 frequent itemsets discovered |
| Rule Evaluation |  Completed | 16 association rules with all metrics |
| Final Documentation |  In Progress | README.md & inline comments |


## 1. Dataset Progress

- **Raw sessions loaded:** 989,818
- **Valid transactions:** 364,384 (cleaned)
- **Unique page categories:** 17 (frontpage, news, tech, local, opinion, on-air, misc, weather, health, living, business, sports, summary, bbs, travel, msn-news, msn-sports)
- **Preprocessing applied:** Removed single-page sessions, frontpage-only sessions, consecutive duplicates

**Sample transactions:**
```
Transaction 1: [frontpage] → [business] → [news]
Transaction 2: [news] → [sports] → [frontpage]
Transaction 3: [local] → [misc] → [on-air]
```


## 2. EDA Progress

**Key Findings:**

| Metric | Value |
|--------|-------|
| Top category | frontpage (73.9% of sessions) |
| Avg pages/session | 3.55 |
| Median pages/session | 3 |
| Min pages/session | 2 |
| Max pages/session | 9 |
| Sessions with 2-3 items | 77.2% |

**Top 5 Categories:**
1. frontpage (269,455 sessions)
2. news (195,833)
3. business (101,543)
4. sports (95,623)
5. bbs (88,123)

## 3. Apriori Results

**Frequent Itemsets:** 16 discovered (min_support = 0.05)

**Association Rules:** 16 rules (min_confidence = 0.20)

**Top 5 Rules (by Lift):**

| Rank | Antecedent | Consequent | Confidence | Lift |
|------|-----------|-----------|-----------|------|
| 1 | on-air | misc | 27.7% | 1.486 |
| 2 | misc | on-air | 39.3% | 1.486 |
| 3 | local | misc | 25.6% | 1.372 |
| 4 | misc | local | 28.6% | 1.372 |
| 5 | business | frontpage | 64.2% | 1.272 |

**Rule Strength Distribution:**
- Strong (Lift ≥ 1.3): 4 rules
- Moderate (Lift 1.2-1.3): 1 rule
- Weak (Lift 1.0-1.2): 5 rules
- Negative (Lift < 1.0): 6 rules



## 4. Challenges & Solutions

| Issue | Status | Resolution |
|-------|--------|------------|
| Large dataset (989K sessions) |  Fixed | Filtered to valid transactions only |
| Single-page sessions noise |  Fixed | Removed sessions with <2 distinct categories |


## 4. Next Steps (Before Final Submission)

- [ ] Write comprehensive README.md with:
  - Project overview & objectives
  - Dataset description & source
  - Data preprocessing pipeline
  - How to run the notebook
  - Result interpretation guide
- [ ] Add inline code comments & documentation
- [ ] Export final rules to CSV format
- [ ] Submit final deliverables




