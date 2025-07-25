# Real Estate Pricing Analysis

This project explores the **California Housing dataset**, a set of census‑block‑level observations compiled from the 1990 U.S. census.  Each record describes socio‑economic and geographic characteristics of a California census block group and includes a target variable representing the **median house value** for that block group.  The dataset contains **20,640 observations** and **eight numeric, predictive attributes**【60937101457875†L891-L934】.  The raw data can be obtained from the StatLib repository and consists of the following fields in order: longitude, latitude, housing median age, total rooms, total bedrooms, population, households, median income and median house value【32946114621828†L12-L29】.

## Repository contents

| File | Description |
|---|---|
| `real_estate_pricing_analysis.py` | Main analysis script.  Performs data loading, exploratory analysis, visualization, and predictive modeling (linear regression and random forest). |
| `figures/` | Contains generated plots: histograms of each feature and a correlation heatmap. |
| `README.md` | This documentation file summarizing the project and dataset. |

## Data loading and feature engineering

The analysis script first attempts to load the California housing dataset through scikit‑learn’s `fetch_california_housing`.  If the download fails (e.g. due to network restrictions), the script falls back to a local copy of `cal_housing.data` extracted from the StatLib repository.  When loading from the raw file, additional features are engineered to match scikit‑learn’s version of the dataset:

* **Average rooms per household** = `totalRooms / households` (denoted *AveRooms*)
* **Average bedrooms per household** = `totalBedrooms / households` (denoted *AveBedrms*)
* **Average occupants per household** = `population / households` (denoted *AveOccup*)
* **Median house value in $100,000s** = `medianHouseValue / 100000.0` (denoted *MedHouseVal*)

These engineered features align the raw data with the eight predictor variables and one target variable described in scikit‑learn’s documentation【60937101457875†L891-L934】.

## Exploratory data analysis

The script prints basic information about the dataset (shape, descriptive statistics) and generates histograms for each numeric feature.  It also computes a correlation matrix and visualizes it as a heatmap.  Key observations include:

* **Median income (MedInc)** and **house age (HouseAge)** are positively correlated with median house value.
* **Average bedrooms per household (AveBedrms)** shows a moderate negative correlation with price, reflecting that areas with larger households often have lower median values.
* **Longitude and latitude** capture geographic variation; properties closer to the coast (higher latitude and lower longitude) tend to have higher values.

## Predictive modeling

Two regression models were trained to predict median house values:

1. **Linear Regression:** A simple linear model trained on all features.  It achieved an R² of **~0.58** on the held‑out test set, with a mean absolute error (MAE) of **~0.53** and a root mean squared error (RMSE) of **~0.75**.
2. **Random Forest Regressor:** An ensemble of 200 decision trees.  This model captured non‑linear relationships and interactions, achieving a higher R² of **~0.81**, a MAE of **~0.33**, and a RMSE of **~0.50**.

These results demonstrate that non‑linear models can provide substantially better predictive performance on real estate pricing data compared with a simple linear approach.

## How to run

1. Ensure Python ≥3.8 is installed along with the packages specified in `real_estate_pricing_analysis.py`.
2. If network access is available, the script will automatically download the dataset using scikit‑learn.  Otherwise, download `cal_housing.tgz` from the StatLib repository, extract it so that the `CaliforniaHousing` folder sits next to the script, and then run:

```
python3 real_estate_pricing_analysis.py
```

The script will output summary statistics to the terminal and save plots into the `figures/` directory.

## References

* Scikit‑learn documentation for the California housing dataset, which describes the attributes and target variable【60937101457875†L891-L934】.
* StatLib repository description of the raw dataset fields【32946114621828†L12-L29】.
