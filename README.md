# 🛒 Integrated Retail Analytics for Store Optimization and Demand Forecasting

> A complete end-to-end Machine Learning project on Walmart's multi-store retail dataset covering anomaly detection, store segmentation, market basket analysis, and weekly sales forecasting.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Project Pipeline](#project-pipeline)
- [Key Results](#key-results)
- [Charts and Visualizations](#charts-and-visualizations)
- [ML Models](#ml-models)
- [Strategic Recommendations](#strategic-recommendations)
- [How to Run](#how-to-run)
- [Conclusion](#conclusion)

---

## 📖 Project Overview

This project performs a full retail analytics study on **45 Walmart stores** across **3 years (2010–2012)**. The objective is to:

- Detect anomalous sales patterns and understand what drives them
- Segment stores by performance behaviour for targeted strategies
- Discover department-level cross-selling opportunities
- Forecast weekly sales accurately using machine learning
- Formulate actionable strategies for inventory, markdowns, and store optimization

**Project Type:** EDA + Regression + Clustering + Time-Series + Unsupervised

---

## 📂 Dataset

Three CSV files are used in this project:

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `sales_data-set.csv` | 421,570 | 5 | Weekly sales per store and department |
| `stores_data-set.csv` | 45 | 3 | Store type (A/B/C) and size in sq ft |
| `Features_data_set.csv` | 8,190 | 12 | Economic indicators — CPI, unemployment, fuel price, temperature, 5 MarkDown columns |

### Column Descriptions

**Sales Dataset**
| Column | Description |
|--------|-------------|
| Store | Store number (1–45) |
| Dept | Department number |
| Date | Week start date |
| Weekly_Sales | Sales for the given store/dept/week |
| IsHoliday | Whether the week is a holiday week |

**Stores Dataset**
| Column | Description |
|--------|-------------|
| Store | Store number |
| Type | Store type — A (large), B (medium), C (small) |
| Size | Store size in square feet |

**Features Dataset**
| Column | Description |
|--------|-------------|
| Store | Store number |
| Date | Week start date |
| Temperature | Regional temperature (°F) |
| Fuel_Price | Cost of fuel in the region |
| MarkDown1–5 | Promotional markdown data (NA if no promotion that week) |
| CPI | Consumer Price Index |
| Unemployment | Regional unemployment rate |
| IsHoliday | Holiday week flag |

---

## 📁 Project Structure

```
walmart-retail-analytics/
│
├── Walmart_ML_Project_v2.ipynb     # Main Google Colab notebook
├── sales_data-set.csv              # Sales dataset
├── stores_data-set.csv             # Stores dataset
├── Features_data_set.csv           # Features/economic indicators dataset
├── walmart_xgb_model.joblib        # Saved best model (generated on run)
└── README.md                       # Project documentation
```

---

## 🛠 Tech Stack

| Category | Libraries |
|----------|-----------|
| Data Processing | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn`, `xgboost` |
| Anomaly Detection | `scikit-learn` IsolationForest |
| Clustering | `scikit-learn` KMeans |
| Market Basket | `mlxtend` (Apriori, association_rules) |
| Explainability | `shap` |
| Model Persistence | `joblib` |
| Environment | Google Colab |

---

## 🔄 Project Pipeline

```
Raw Data (3 CSVs)
       │
       ▼
Data Cleaning & Merging
  - Parse dates
  - Fill MarkDown NaN with 0 (domain-driven)
  - Encode IsHoliday, Store Type
  - Remove negative sales
       │
       ▼
Feature Engineering
  - Temporal: Year, Month, Week, Quarter
  - Cyclical: WeekSin/Cos, MonthSin/Cos
  - Aggregate: TotalMarkDown, NumActiveMarkDown
  - Lag: Rolling4wkSales (4-week rolling average)
       │
       ▼
Exploratory Data Analysis (20 Charts — UBM Framework)
  - Univariate: distributions, store type counts
  - Bivariate: sales by type, holiday effect, trends, correlations
  - Multivariate: type x month heatmap, markdown vs sales, external factors
       │
       ├──────────────────────────────────────┐
       ▼                                      ▼
Anomaly Detection                    Store Segmentation
  - Isolation Forest                   - K-Means Clustering
  - Holiday spikes confirmed           - Elbow + Silhouette tuning
  as genuine demand events             - 3 performance tiers identified
       │                                      │
       └──────────────┬───────────────────────┘
                      ▼
           Market Basket Analysis
             - Apriori Algorithm
             - Department co-purchase proxy
             - Association rules ranked by lift
                      │
                      ▼
           Demand Forecasting (3 Models)
             - Ridge Regression (baseline)
             - Random Forest Regressor
             - XGBoost Regressor  ← Best
             - All tuned with RandomizedSearchCV
                      │
                      ▼
           Model Explainability (SHAP)
                      │
                      ▼
           Strategic Recommendations
             + Model saved with joblib
```

---

## 📊 Key Results

### Model Performance Comparison

| Model | RMSE | MAE | R² | MAPE% |
|-------|------|-----|----|-------|
| Ridge Regression | ~$5,200 | ~$3,100 | ~0.85 | ~18% |
| Ridge (Tuned) | ~$5,000 | ~$3,000 | ~0.86 | ~17% |
| Random Forest | ~$2,800 | ~$1,600 | ~0.93 | ~10% |
| Random Forest (Tuned) | ~$2,600 | ~$1,500 | ~0.94 | ~9% |
| XGBoost | ~$2,400 | ~$1,400 | ~0.95 | ~8% |
| **XGBoost (Tuned)** | **~$2,200** | **~$1,300** | **~0.96** | **~7%** |

> ✅ **XGBoost (Tuned) is the final production model** — lowest error on all metrics, R² above 0.95.

### Key Findings

- **Rolling4wkSales** is the strongest single predictor (SHAP confirmed, ~0.95 correlation with target)
- **Type A stores** generate ~3× the median weekly sales of Type C stores
- **December** is the peak sales month every year — inventory must be pre-positioned by October
- **MarkDown promotions** boost sales but show **diminishing returns beyond ~$50,000/week**
- **Holiday weeks** (~7% of records) consistently generate significantly higher and more volatile sales
- **CPI and Unemployment** have a measurable negative impact on sales — economically stressed markets need value positioning

---

## 📈 Charts and Visualizations

20 charts produced following the **UBM (Univariate → Bivariate → Multivariate)** framework:

| # | Chart | Type | Key Insight |
|---|-------|------|-------------|
| 1 | MarkDown Missing Values | Bar | 30–65% missing — filled with 0 (no active promotion) |
| 2 | Weekly Sales Distribution | Histogram | Heavy right skew — log-transform normalises it |
| 3 | Store Type Count + Size | Bar + Histogram | 53% Type A; size strongly separates store types |
| 4 | Median Sales by Store Type | Bar | Type A has 3× the median sales of Type C |
| 5 | Holiday vs Non-Holiday Sales | Boxplot | Holiday weeks show higher median and wider spread |
| 6 | Total Sales Over Time | Line | Clear Nov–Dec spike every year; holidays align with peaks |
| 7 | Average Sales by Month | Bar | December peak; August back-to-school secondary peak |
| 8 | Correlation Heatmap | Heatmap | Rolling4wkSales ~0.95 corr; CPI/Unemployment negative |
| 9 | Top 10 Departments | Bar | Pareto pattern — top 10 depts drive majority of revenue |
| 10 | Sales by Store Type × Month | Heatmap | Seasonality is universal; Type A always the highest |
| 11 | Sales by Store Type × Holiday | Violin | Holiday uplift effect visible across all store types |
| 12 | MarkDown vs Weekly Sales | Scatter | Positive effect with clear diminishing returns |
| 13 | External Factors vs Sales | Dual-axis | Higher CPI and unemployment reduce sales |
| 14 | Quarterly Total Sales | Bar | Q4 consistently strongest; modest year-over-year growth |
| 15 | Summary Statistics | Table | Full descriptive stats for all numeric features |
| 16 | K-Means Optimisation | Elbow + Silhouette | Optimal k selected with silhouette score above 0.4 |
| 17 | RF Feature Importances | Horizontal Bar | Rolling4wkSales #1; structural features beat economic ones |
| 18 | Model Comparison | Bar (3 metrics) | XGBoost Tuned wins on all three metrics |
| 19 | SHAP Summary | SHAP Bar | Confirms feature ranking with directional impact |
| 20 | Actual vs Predicted | Scatter | XGBoost closely tracks actual; slight peak underestimation |

---

## 🤖 ML Models

### Model 1 — Ridge Regression (Baseline)
- Regularised linear model with L2 penalty
- Handles multicollinear economic features better than plain OLS
- Tuned: `alpha` via RandomizedSearchCV with 5-fold cross-validation

### Model 2 — Random Forest Regressor
- Ensemble of decision trees trained on bootstrapped data subsets
- Naturally captures non-linear interactions (holiday × markdown × store type)
- Tuned: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` via RandomizedSearchCV with 3-fold CV

### Model 3 — XGBoost Regressor ✅ Final Model
- Sequential gradient-boosted trees — each tree corrects residuals of the previous
- Built-in L1 and L2 regularisation handles correlated features
- Tuned: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight` via RandomizedSearchCV with 3-fold CV

### Why Temporal Split Instead of Random Split?

| | Temporal Split | Random Split |
|---|---|---|
| Train set | 2010–2011 | Random 67% |
| Test set | 2012 | Random 33% |
| Data leakage | ❌ None | ✅ Future weeks in training |
| Real-world validity | ✅ Mirrors production | ❌ Impossible in practice |

### Evaluation Metrics and Business Meaning

| Metric | Business Meaning |
|--------|-----------------|
| RMSE | Penalises large errors heavily — critical for high-value departments |
| MAE | Average dollar error per weekly prediction — direct cost exposure |
| R² | Percentage of sales variance explained by the model |
| MAPE% | Scale-independent — compare accuracy across all department sizes |

---

## 💡 Strategic Recommendations

### Inventory Management
- Increase inventory 25–35% entering Q4 (October) using per-cluster XGBoost forecasts
- High-volume cluster stores → 3–4 weeks safety stock
- Low-volume cluster stores → 1–2 weeks safety stock to minimise carrying costs
- Auto-trigger reorder review when 4-week rolling sales drops more than 15%

### Markdown and Promotions
- Deploy MarkDown1 two weeks before holiday weeks to drive anticipatory demand
- Cap total markdown spend at ~$50,000 per store per week — clear diminishing returns beyond this point
- Use Apriori-derived department affinity rules to design cross-department promotional bundles

### Store-Type Strategy

| Store Type | Strategy |
|------------|----------|
| Type A (Large) | Premium assortment, maintain 95%+ in-stock on top 10 departments |
| Type B (Medium) | EDLP focus, value bundles in high-unemployment markets |
| Type C (Small) | JIT replenishment, curated top-SKU-only assortment |

### Real-World Challenges
- **Data latency:** Rolling4wkSales requires real-time POS data feeds
- **Model drift:** CPI and unemployment change — quarterly retraining recommended
- **Scale:** 45 stores × 81 departments = 3,645 predictions per week — manageable with batch inference

---

## ▶️ How to Run

### Option 1 — Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `Walmart_ML_Project_v2.ipynb`
3. Upload the three CSV files when prompted:
   - `sales_data-set.csv`
   - `stores_data-set.csv`
   - `Features_data_set.csv`
4. Uncomment the file upload cell at the top of the notebook:
   ```python
   from google.colab import files
   files.upload()
   ```
5. Click **Runtime → Run All**

> ⚠️ The notebook runs end-to-end without errors. Estimated runtime: 10–15 minutes on Colab free tier.

### Option 2 — Local Jupyter

```bash
# Clone the repository
git clone https://github.com/your-username/walmart-retail-analytics.git
cd walmart-retail-analytics

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost mlxtend shap joblib

# Launch notebook
jupyter notebook Walmart_ML_Project_v2.ipynb
```

### Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
mlxtend
shap
joblib
```

---

## ✅ Conclusion

This project delivered a production-grade retail analytics pipeline covering:

- ✔️ Complete data cleaning with domain-driven imputation strategy
- ✔️ 15+ engineered features including cyclical encoding and lag features
- ✔️ 20 EDA visualizations following the UBM framework
- ✔️ Multi-dimensional anomaly detection using Isolation Forest
- ✔️ Store segmentation with validated cluster quality using K-Means and Silhouette scoring
- ✔️ Department affinity mining using Apriori market basket analysis
- ✔️ Three ML models with hyperparameter tuning and cross-validation
- ✔️ SHAP-based model explainability for business stakeholders
- ✔️ Deployment-ready model saved and validated on unseen data

**Final Model: XGBoost (Tuned) — R² > 0.95 | RMSE ~$2,200 | MAPE ~7%**

---

## 📄 License

This project is for educational purposes as part of a Machine Learning Capstone.
Dataset sourced from the Walmart Store Sales Forecasting challenge.

---

*Built with Python 3 on Google Colab*
