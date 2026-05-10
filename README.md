# A Multi-Stage Data Mining Analysis of Fare Variability in NYC Yellow Taxi Data


---

## Overview

This repository contains the complete implementation of a multi-stage data mining pipeline applied to NYC Yellow Taxi trip records from January 2026. The study examines fare variability, pricing consistency, and service distribution patterns across 263 TLC taxi zones using a combination of supervised and unsupervised machine learning methods.

The analysis is structured around four analytical stages:

1. **Fare Prediction Model** - Random Forest regression to estimate expected fares from observable trip characteristics
2. **Zone-Level Residual Analysis** - Spatial profiling of deviations between observed and predicted fares
3. **K-Means Clustering** - Identification of distinct zone service typologies
4. **Isolation Forest Anomaly Detection** - Detection of statistically unusual trips

---

## Key Results

| Metric | Value |
|---|---|
| Total trips analysed | 697,821 |
| Global fare prediction R² | 0.9852 |
| Manhattan R² | 0.9868 |
| Bronx R² | 0.8441 |
| Anomalous trips detected | 34,891 (5.0%) |
| Underserved zones identified | 55 |
| Most overpriced zone | Jackson Heights, Queens (+$0.32) |
| Most underpriced zone | Bensonhurst West, Brooklyn (−$0.45) |

---

## Dataset

The dataset used in this project is the **NYC Yellow Taxi Trip Records for January 2026**, sourced from the NYC Taxi and Limousine Commission (TLC) open data portal.

**Source:** https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

The raw Parquet file is processed through a 12-step cleaning pipeline before analysis. The final cleaned dataset (`final_taxi_data.csv`) contains 697,821 trips and 37 features including:

- 20 original TLC fields
- 11 engineered temporal and financial features
- 6 geographic enrichment fields from the TLC Taxi Zone Lookup table

Due to file size the processed CSV is not included in this repository. Download the raw January 2026 Parquet file from the TLC portal and run the preprocessing steps in the notebook to reproduce it, or contact the authors for access to the cleaned file.

**TLC Taxi Zone Lookup table** is also required for geographic enrichment and is available at the same TLC portal link above.

---

## Requirements

This project was developed and run on **Google Colab**. All dependencies are standard Python data science libraries.

```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
pyarrow
```

Install all dependencies with:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn pyarrow
```

If running on Google Colab all libraries except `imbalanced-learn` are pre-installed. Install the remaining one with:

```bash
!pip install imbalanced-learn
```

---

## How to Run

### On Google Colab (Recommended)

1. Upload `final_data.ipynb` to Google Colab
2. Upload `final_taxi_data.csv` to your Colab session storage or mount your Google Drive
3. Update the file path in the data loading cell if needed:
   ```python
   df = pd.read_csv('final_taxi_data.csv')
   ```
4. Run all cells in order - the notebook is structured sequentially and each section builds on the previous one

### Locally

1. Clone this repository
2. Install dependencies
3. Place `final_taxi_data.csv` in the same directory as the notebook
4. Launch Jupyter and run `final_data.ipynb`

```bash
git clone https://github.com/yourusername/nyc-taxi-fare-analysis
cd nyc-taxi-fare-analysis
pip install -r requirements.txt
jupyter notebook final_data.ipynb
```

---

## Notebook Structure

The notebook is organised into the following sections:

### 1. Data Loading and Validation
- Load the cleaned CSV dataset
- Validate row and column counts
- Fix any remaining null values in engineered feature columns

### 2. Exploratory Data Analysis (EDA)
- Univariate distributions: fare amount, trip distance, duration, speed
- Temporal patterns: trip volume by hour and day of week
- Spatial patterns: trip volume by borough
- Fare-per-mile distribution across boroughs

### 3. Fare Prediction Model 
- Random Forest Regressor trained on trip features
- Stratified 80/20 train-test split by borough (PU_Borough)
- Performance evaluation: R², RMSE, MAE
- Fairness residual computation for all 697,821 trips

### 4. Zone-Level Residual Analysis
- Aggregation of residuals by PULocationID
- Identification of top 10 overpriced and underpriced zones
- EWR zone filtering for stability
- Borough-level fairness summary
- One-way ANOVA significance testing (p < 0.001)

### 5. Borough-Level Model Comparison 
- Separate Random Forest models trained for Manhattan, Brooklyn, Queens, and Bronx
- Performance gradient analysis across boroughs
- Comparison table: R², RMSE, MAE per borough

### 6. K-Means Clustering
- Zone-level feature vector construction (6 features)
- StandardScaler normalisation
- Elbow Method and Silhouette Score for optimal k selection
- 4-cluster typology: Manhattan Core, Outer Borough Low-Demand, Airport Corridors, Mixed Peripheral

### 7. Isolation Forest Anomaly Detection
- Contamination parameter: 0.05
- Features: fare, distance, duration, speed, fare_per_mile, tip_percentage, total_amount
- Spatial profiling: anomaly rates by borough
- Temporal profiling: anomaly distribution by pickup hour

---

## Feature Description

### Original TLC Features (20)
Key fields used in modelling: `fare_amount`, `trip_distance`, `RatecodeID`, `PULocationID`, `DOLocationID`, `congestion_surcharge`, `cbd_congestion_fee`, `passenger_count`

### Engineered Features (11)
`trip_duration_min`, `speed_mph`, `pickup_hour`, `pickup_day`, `time_of_day`, `is_peak_hour`, `is_weekend`, `fare_per_mile`, `tip_percentage`

### Geographic Enrichment (6)
`PU_Borough`, `PU_Zone`, `PU_service_zone`, `DO_Borough`, `DO_Zone`, `DO_service_zone`


---

## Preprocessing Pipeline

The 12-step cleaning protocol applied before analysis:

| Step | Operation |
|---|---|
| 1 | Duplicate removal |
| 2 | Datetime parsing and invalid timestamp removal |
| 3 | Passenger count filter (retain 1–6) |
| 4 | Trip distance filter (retain 0.01–100 miles) |
| 5 | Fare amount filter (retain $0.01–$500) |
| 6 | Negative value removal |
| 7 | Trip duration filter (retain 1–180 minutes) |
| 8 | Speed filter (retain 1–80 mph) |
| 9 | Location ID validation (retain 1–265) |
| 10 | Null removal on critical fields |
| 11 | IQR outlier removal (1.5×IQR on fare, distance, total amount) |
| 12 | TLC Zone Lookup table join |

Raw dataset: ~1.9M records → Cleaned dataset: 697,821 records (36.7% retained)

---

## Authors

| Name | Institution | Email |
|---|---|---|
| Anjal P Dijo | IIIT Kottayam | anjalpdijo.2026emai10010@iiitkottayam.ac.in |
| Dr. Sheela S | IIIT Kottayam | sheelas.2026emai10055@iiitkottayam.ac.in |
| Seethu Elza Thomas | IIIT Kottayam | seethuelzathomas.2026emai10052@iiitkottayam.ac.in |

---

## Data Source Citation

```
NYC Taxi and Limousine Commission, "Yellow Taxi Trip Records — January 2026," 
2026. [Online]. Available: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
```

---

## Notes

- The notebook was developed on Google Colab with a GPU runtime - training the Random Forest models on the full dataset takes approximately 10–15 minutes on CPU and 3–5 minutes with GPU acceleration
- The `anomaly` column in the final dataset has one NaN value (one trip did not receive an Isolation Forest score due to a missing feature value) — this is handled in the anomaly rate calculations
- January 2026 is the first full month of NYC's CBD congestion pricing regime — the `cbd_congestion_fee` column captures this policy's effect on trip pricing
