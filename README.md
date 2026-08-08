# California Property Close Price Prediction

An Automated Valuation Model (AVM) for predicting the final sale price of single-family residential properties in California, built using CRMLS (California Regional Multiple Listing Service) data.

---

## Project Overview

This project was developed as part of a 12-week internship at IDXExchange. The goal is to build a machine learning model that can estimate the close price of any single-family residential property in California based on its physical characteristics and location — regardless of whether the property is currently listed for sale.

---

## Dataset

- **Source:** CRMLS (California Regional Multiple Listing Service) via IDXExchange FTP server
- **Coverage:** December 2025 — May 2026 (6 months)
- **Filter:** `PropertyType = Residential` and `PropertySubType = SingleFamilyResidence`
- **Raw size:** ~61,727 rows, 78 columns
- **Target variable:** `ClosePrice` — the final sale price paid by the buyer

> **Note:** `ListPrice` and `OriginalListPrice` were intentionally excluded from all models to prevent target leakage and ensure the model generalizes to off-market properties.

---

## Project Structure

```
ca-price-prediction/
├── data/
│   ├── CRMLSSold202512.csv         ← Raw monthly files
│   ├── CRMLSSold202601.csv
│   ├── CRMLSSold202602.csv
│   ├── CRMLSSold202603.csv
│   ├── CRMLSSold202604.csv
│   ├── CRMLSSold202605.csv
│   ├── ca_school_districts.geojson ← CA school district boundaries
│   ├── train_data.csv              ← Cleaned training set (Dec 2025 – Apr 2026)
│   ├── test_data.csv               ← Cleaned test set (May 2026)
│   ├── train_enriched.csv          ← Enriched with engineered features
│   ├── test_enriched.csv           ← Enriched with engineered features
│   ├── best_model.joblib           ← Saved XGBoost model
│   ├── feature_cols.joblib         ← Saved feature column list
│   ├── metrics_summary.csv         ← Model evaluation results
│   └── price_band_analysis.csv     ← MdAPE by price band
├── notebooks/
│   ├── 01_exploration.ipynb        ← Week 2: EDA
│   ├── 02_preprocessing.ipynb      ← Week 3: Data cleaning
│   ├── 03_baseline_model.ipynb     ← Week 4: Linear Regression
│   ├── 04_model_comparison.ipynb   ← Week 5: Decision Tree & Random Forest
│   ├── 05_feature_engineering.ipynb← Week 6: Feature engineering
│   ├── 05_advanced_models.ipynb    ← Week 7: XGBoost & LightGBM
│   └── 06_evaluation.ipynb         ← Week 8: Full evaluation
├── app.py                          ← Streamlit prediction app
└── README.md
```

---

## Preprocessing Steps

1. **Filtering** — kept only `PropertyType = Residential` and `PropertySubType = SingleFamilyResidence`
2. **Column selection** — dropped 53 columns including:
   - Fully empty columns (100% missing)
   - Columns with >60% missing values
   - Administrative columns (agent names, emails, IDs)
   - Redundant columns (duplicate location fields)
3. **Missing value imputation:**
   - Numerical columns → median
   - Boolean YN columns → `False`
   - `AssociationFee`, `GarageSpaces` → `0`
   - Latitude/Longitude → county-specific median
   - Categorical columns → `"Unknown"`
4. **Encoding** — Label Encoding for `City`, `PostalCode`, `CountyOrParish`, `HighSchoolDistrict`
5. **Outlier removal** — kept only properties between $100K and $10M
6. **Train/test split** — most recent month (May 2026) as test set, preceding 5 months as training set

---

## Feature Engineering (Week 6)

| Feature | Description |
|---|---|
| `PropertyAge` | `2025 - YearBuilt`, clipped at 0 |
| `BedBathRatio` | `BedroomsTotal / BathroomsTotalInteger` |
| `TotalRooms` | `BedroomsTotal + BathroomsTotalInteger` |
| `AreaPerBedroom` | `LivingArea / BedroomsTotal` |
| `HasHOA` | Binary flag: `AssociationFee > 0` |
| `LotLivingRatio` | `LotSizeSquareFeet / LivingArea`, capped at 99th percentile |
| `UnifiedDistrict` | Spatially joined from CA School District boundaries (2025-26) |

The school district spatial join achieved a **76% match rate** across all properties. Unmatched properties were assigned `"Unknown"`.

---

## Models Tested

| Model | R² | MAE | MAPE | MdAPE |
|---|---|---|---|---|
| Linear Regression | 0.4796 | $464,001 | 44.99% | 31.06% |
| Decision Tree | 0.6954 | $268,633 | 19.47% | 12.16% |
| Random Forest | 0.8581 | $186,208 | 13.87% | 8.69% |
| XGBoost (tuned) | 0.8641 | $200,572 | 16.06% | 10.97% |
| LightGBM (tuned) | 0.8542 | $208,947 | 16.79% | 11.62% |

**Best model: XGBoost** with `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`

---

## Price Band Performance (MdAPE %)

| Price Band | Count | Random Forest | XGBoost |
|---|---|---|---|
| < $300K | 265 | 23.23% | 27.89% |
| $300K – $500K | 1,416 | 8.15% | 12.73% |
| $500K – $750K | 2,527 | 6.53% | 9.92% |
| $750K – $1M | 2,372 | 6.97% | 8.88% |
| $1M – $2M | 3,752 | 9.49% | 10.89% |
| > $2M | 1,633 | 13.98% | 14.60% |

Models perform best in the **$500K – $1M** range and struggle most with properties below $300K and above $2M.

---

## How to Re-run the Code

### Requirements

Install all dependencies:
```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm geopandas joblib streamlit jupyter
```

### Step-by-step

**1. Download raw data**
- Connect to FTP server via FileZilla
- Download all `CRMLSSold*.csv` files into `data/`

**2. Run notebooks in order**
```
cd ca-price-prediction
jupyter notebook
```
Then run each notebook in order:
- `01_exploration.ipynb`
- `02_preprocessing.ipynb`
- `03_baseline_model.ipynb`
- `04_model_comparison.ipynb`
- `05_feature_engineering.ipynb`
- `05_advanced_models.ipynb`
- `06_evaluation.ipynb`

**3. Launch the Streamlit app**
```
cd ca-price-prediction
streamlit run app.py
```
The app opens automatically at `http://localhost:8501`

---

## Streamlit App

The prediction app allows users to input property characteristics and receive an estimated close price instantly.

**Inputs:**
- Living area, bedrooms, bathrooms, lot size
- Year built, stories, garage spaces
- Location (latitude, longitude)
- Boolean features: pool, fireplace, view, HOA, new construction

**Output:**
- Estimated close price
- ±10% confidence range

---

## Key Findings

- Tree-based models (Random Forest, XGBoost) significantly outperform Linear Regression for real estate pricing
- Location features (`Latitude`, `Longitude`, `PostalCode_encoded`) are among the most important predictors
- Engineered features did not significantly improve model performance, suggesting the base features already capture most of the signal
- The model performs best for mid-range California homes ($500K–$1M) and struggles with ultra-luxury and very low-priced properties

---

## Team

Internship project at **IDXExchange**
Data source: **CRMLS** via Trestle Property API
