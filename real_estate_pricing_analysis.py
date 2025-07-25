"""
Real Estate Pricing Analysis
============================

This script performs exploratory data analysis (EDA) and basic predictive modeling
on the California housing dataset available through scikit‑learn.  The dataset
contains 20,640 samples derived from the 1990 U.S. census, with eight numeric
features describing socio‑economic and geographic characteristics of
California census block groups【60937101457875†L891-L933】.  The target
variable is the median house value for each block group, expressed in
hundreds of thousands of dollars【60937101457875†L928-L934】.

The script performs the following steps:

1. Load the dataset and convert it to a pandas DataFrame.
2. Display basic information and summary statistics.
3. Create visualizations to explore feature distributions and relationships.
4. Compute a correlation matrix and plot it as a heatmap.
5. Train a simple linear regression model and a random forest regressor to
   predict median house values.
6. Evaluate the models using R², mean absolute error (MAE) and root mean
   squared error (RMSE) on a held‑out test set.

The resulting figures are saved to the `figures/` directory, and key results
are printed to the console.  This code is intended as a starting point for
further exploration of real estate pricing and predictive modeling.

Note: Running this script requires the following Python packages:
```
numpy
pandas
matplotlib
seaborn
scikit‑learn
```

To install any missing dependencies, run:

```bash
pip install numpy pandas matplotlib seaborn scikit‑learn
```

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def load_data(as_frame: bool = True) -> pd.DataFrame:
    """Load the California housing dataset and return a pandas DataFrame.

    Parameters
    ----------
    as_frame : bool, default True
        If True, return the data and target as a pandas DataFrame and Series.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing feature columns and a target column named
        `MedHouseVal`.
    """
    """
    Attempt to load the California housing dataset.  If the dataset cannot
    be downloaded via scikit‑learn (e.g., due to network restrictions), the
    function will look for a local copy of the `cal_housing.data` file
    extracted from the StatLib repository.  When loading from the local
    file, additional features are computed to mirror those provided by the
    scikit‑learn version of the dataset.

    The original `cal_housing.data` file contains the following columns in
    order【32946114621828†L12-L29】:

    1. longitude
    2. latitude
    3. housingMedianAge
    4. totalRooms
    5. totalBedrooms
    6. population
    7. households
    8. medianIncome
    9. medianHouseValue

    From these raw variables we compute average rooms per household,
    average bedrooms per household and average occupants per household to
    align with scikit‑learn's feature set.  The target value is scaled to
    represent median house value in units of $100,000, consistent with
    scikit‑learn【60937101457875†L928-L933】.
    """
    try:
        # Try loading via scikit‑learn (will download if available)
        dataset = fetch_california_housing(as_frame=as_frame)
        data = dataset.data
        target = dataset.target
        df = data.copy()
        df["MedHouseVal"] = target
        return df
    except Exception:
        # Fall back to local file
        data_path = os.path.join(os.path.dirname(__file__), "CaliforniaHousing", "cal_housing.data")
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                "cal_housing.data not found. Please place the extracted CaliforniaHousing directory in the script directory."
            )
        # Load raw data
        col_names = [
            "Longitude",
            "Latitude",
            "HouseAge",
            "TotalRooms",
            "TotalBedrooms",
            "Population",
            "Households",
            "MedInc_raw",
            "MedHouseVal_raw",
        ]
        raw_df = pd.read_csv(data_path, header=None, names=col_names)
        # Compute derived features to match scikit‑learn dataset
        df = pd.DataFrame()
        df["MedInc"] = raw_df["MedInc_raw"]
        df["HouseAge"] = raw_df["HouseAge"]
        df["AveRooms"] = raw_df["TotalRooms"] / raw_df["Households"]
        df["AveBedrms"] = raw_df["TotalBedrooms"] / raw_df["Households"]
        df["Population"] = raw_df["Population"]
        df["AveOccup"] = raw_df["Population"] / raw_df["Households"]
        df["Latitude"] = raw_df["Latitude"]
        df["Longitude"] = raw_df["Longitude"]
        # Scale median house value to hundreds of thousands
        df["MedHouseVal"] = raw_df["MedHouseVal_raw"] / 100000.0
        return df


def basic_info(df: pd.DataFrame) -> None:
    """Print basic information about the DataFrame.

    This function displays the first few rows, shape, and descriptive
    statistics of the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The housing data DataFrame.
    """
    print("=== First five rows ===")
    print(df.head())
    print("\n=== Data shape ===")
    print(df.shape)
    print("\n=== Descriptive statistics ===")
    print(df.describe().T)


def plot_distributions(df: pd.DataFrame, output_dir: str) -> None:
    """Create histograms for each numerical feature in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The housing data DataFrame.
    output_dir : str
        Directory where figures will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    numeric_cols = df.columns
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=30, color="steelblue")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        filename = os.path.join(output_dir, f"hist_{col}.png")
        plt.savefig(filename)
        plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    """Compute and plot a correlation heatmap of the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The housing data DataFrame.
    output_dir : str
        Directory where the heatmap figure will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix for California Housing Features")
    plt.tight_layout()
    filename = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(filename)
    plt.close()


def build_models(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Train linear regression and random forest models on the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The housing data DataFrame with features and target column `MedHouseVal`.
    test_size : float, default 0.2
        Fraction of data to reserve for testing.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    results : dict
        Dictionary containing trained models and evaluation metrics.
    """
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Linear Regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lr = lin_reg.predict(X_test)
    metrics_lr = {
        "R2": r2_score(y_test, y_pred_lr),
        "MAE": mean_absolute_error(y_test, y_pred_lr),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lr)),
    }

    # Random Forest Regressor
    rf_reg = RandomForestRegressor(
        n_estimators=200, random_state=random_state, n_jobs=-1
    )
    rf_reg.fit(X_train, y_train)
    y_pred_rf = rf_reg.predict(X_test)
    metrics_rf = {
        "R2": r2_score(y_test, y_pred_rf),
        "MAE": mean_absolute_error(y_test, y_pred_rf),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    }

    results = {
        "linear_regression": {
            "model": lin_reg,
            "metrics": metrics_lr,
        },
        "random_forest": {
            "model": rf_reg,
            "metrics": metrics_rf,
        },
    }
    return results


def print_model_metrics(results: dict) -> None:
    """Print the evaluation metrics for each model in a neat format.

    Parameters
    ----------
    results : dict
        Output of `build_models` containing models and their metrics.
    """
    print("\n=== Model Evaluation Metrics ===")
    for model_name, content in results.items():
        metrics = content["metrics"]
        print(f"\nModel: {model_name.replace('_', ' ').title()}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")


def main():
    # Load data
    df = load_data()
    # Display basic information
    basic_info(df)
    # Create output directory for figures
    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    # Plot distributions and correlation heatmap
    plot_distributions(df, fig_dir)
    plot_correlation_heatmap(df, fig_dir)
    # Train models and compute metrics
    results = build_models(df)
    # Print metrics
    print_model_metrics(results)


if __name__ == "__main__":
    main()