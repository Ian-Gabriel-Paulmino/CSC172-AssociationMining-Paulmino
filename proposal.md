# CSC172 Association Rule Mining Project Proposal

**Student:** Ian Gabriel Paulmino, 2022-1729  
**Date:** December 12, 2025

---

## 1. Project Title

**Website User Navigation Pattern Analysis: Association Rule Mining on MSNBC.com Clickstream Data**

---

## 2. Problem Statement

Content platforms need to understand how users navigate between different sections to optimize website design and content recommendations. This project analyzes real MSNBC.com clickstream data (388,434 user sessions) to discover which content sections users visit together, enabling data-driven decisions for UX improvement and content discovery.

Philippine news and content platforms (GMA News, ABS-CBN News, Rappler) face similar challenges in understanding user navigation behavior. The insights and methodologies developed in this analysis can be directly applied to improve content recommendation systems, website layouts, and user engagement on local news websites and content platforms.

---

## 3. Objectives

- Preprocess clickstream data (decompress, map categories, remove noise)
- Conduct exploratory data analysis (EDA) to characterize user navigation patterns
- Apply Apriori algorithm to discover frequent content category combinations and association rules
- Evaluate rules using support, confidence, and lift metrics
- Derive actionable recommendations for website optimization

---

## 4. Dataset Plan

| Aspect | Details |
|--------|---------|
| **Source** | UCI Machine Learning Repository - MSNBC.com Anonymous Web Data |
| **Size** | 388,434 user sessions (after filtering) |
| **Categories** | 17 content sections (news, tech, sports, business, weather, health, etc.) |
| **Format** | Compressed sequence file (.seq.gz) |
| **Time Period** | September 28, 1999 (24-hour snapshot) |
| **Acquisition** | Free download from UCI repository |

---

## 5. Technical Approach

| Phase | Method | Tools |
|-------|--------|-------|
| **Data Loading** | Decompress .gz, parse sequences | 
| **Preprocessing** | Map codes to names, deduplicate, filter | `pandas` |
| **EDA** | Session statistics, frequency distributions, visualizations | `pandas`, `matplotlib` |
| **Algorithm** | Apriori (min_support=0.05, min_confidence=0.20) | `mlxtend` |
| **Evaluation** | Calculate support, confidence, lift metrics | `mlxtend.association_rules` |
| **Visualization** | Scatter plots, histograms, charts | `matplotlib` |

---

## 6. Expected Challenges & Mitigations

| Challenge | Solution |
|-----------|----------|
| Large dataset (989K raw sessions) | Pre-filter to valid sessions (≥2 categories) |
| Sparse navigation data | Focus on 2-item rules; adjust min_support if needed |
| Duplicate consecutive visits | Deduplicate within sessions |
| Trivial rules | Apply lift threshold (>1.2) for meaningful patterns |
| Session outliers | Analyze log-transformed statistics |

---

