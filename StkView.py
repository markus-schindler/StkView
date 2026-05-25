#"/usr/bin/python
# -*- coding: utf-8 -*-

"""StkView.py: End-to-end quantitative analysis of stock price data.

The script loads CSV data, cleans it, performs statistical analysis,
creates a portfolio, computes risk metrics, visualizes correlations,
technical indicators and builds an LSTM forecast.
"""

__author__ = "Markus Schindler"
__copyright__ = "Copyright 2026"

__license__ = "Unlicense"
__version__ = "0.1.1"
__maintainer__ = "Markus Schindler"
__email__ = "schindlerdrmarkus@gmail.com"
__status__ = "Education"

# -------------------------- #
# Built-in / Generic Imports #
# -------------------------- #

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# --------------------- #
# Logging configuration #
# --------------------- #

log = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
log.addHandler(handler)
log.setLevel(logging.INFO)

# ---------------------------- #
# Data loading & preprocessing #
# ---------------------------- #

def load_data(csv_path: Path, sep: str = ";") -> pd.DataFrame:
    log.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path, sep = sep, low_memory = False)
    log.info("Loaded %d rows and %d columns", df.shape[0], df.shape[1])
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Starting data cleaning")
    #df = df.copy()

    # Change date format
    df["Date"] = pd.to_datetime(df["Date"], dayfirst = True, yearfirst = False)
    df.set_index("Date", inplace = True)

    # Format numbers and convert strings to float values
    df.replace({r"\.": r""}, regex = True, inplace = True)
    df.replace({r",": r"."}, regex = True, inplace = True)
    df = df.astype(float)

    # Some values are not correctly imported - likely a unit error
    df[df > 2000] /= 1000

    # Data imputation by replacing the NaN values by the column mean
    for col in df.columns:
        mean_val = df[col].mean()
        df.fillna({col: mean_val}, inplace = True)
    
    # Create list of columns that contains missing values or NaN
    cols_to_drop = df.columns[df.isna().sum() > 0].tolist()
    
    # Drop columns that are not part of the core analysis
    df.drop(columns = [c for c in cols_to_drop if c in df.columns], inplace = True)

    # Ensure proper ordering
    df = df.sort_index()

    log.info("Data cleaning completed - %d rows and %d columns remaining", df.shape[0], df.shape[1])
    return df

# ------------------ #
# Summary statistics #
# ------------------ #

def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Computing descriptive statistics")
    stats = df.describe().T[["mean", "std", "min", "max"]]
    stats.columns = ["Mean", "Std Dev", "Min", "Max"]
    log.info("Descriptive statistics computed")
    return stats

# -------------------- #
# Portfolio management #
# -------------------- #

def portfolio_selection(Path):
    try:
        selection = pd.read_csv(Path, sep = ";")
    except ValueError:
        log.error("Portfolio.csv could not be loaded!")
    weights = selection["Weights"].to_list()
    data = selection["Stock"].to_list()
    log.info("Portfolio.csv was loaded")
    return data, weights

def compute_portfolio_return(
        data_portfolio: pd.DataFrame, weights: List[float]
    )-> pd.Series:
    if data_portfolio.shape[1] != len(weights):
        log.error("Number of assets (%d) does not match number of weights (%d)",
            data_portfolio.shape[1], len(weights))
        raise ValueError("Mismatch between assets and weights")
    daily_changes = data_portfolio.pct_change().dropna()
    return (daily_changes * weights).sum(axis = 1)

def compute_portfolio_risk(
        data_portfolio: pd.DataFrame, confidence: float = 0.95
        ) -> Dict[str, float]:                                                                 
    log.info("Computing portfolio risk metrics")
    daily_return = data_portfolio.pct_change().dropna()
    volatility = daily_return.std()

    alpha = 1 - confidence
    # Value at Risk (VaR)
    var = daily_return.quantile(alpha)
 
    return {"Volatility (Std Dev)": volatility, "Value at Risk (VaR)": var}

# ------------------- #
# Correlation heatmap #
# ------------------- #

def plot_correlation_heatmap(
        corr: pd.DataFrame, output_path: Path
    ) -> None:
    log.info("Creating correlation heatmap")
    corr_daily_return = corr.corr()
    sns.set_theme(style = "white")
    mask = np.triu(np.ones_like(corr, dtype = bool))
    plt.figure(figsize = (8, 6), dpi = 100)
    cmap = sns.diverging_palette(170, 30, as_cmap = True)
    sns.heatmap(
        corr_daily_return,
        mask = mask,
        cmap = cmap,
        vmax = 1,
        center = 0,
        square = True,
        linewidths = 0.5,
        cbar_kws={"shrink": 0.5},
        )
    plt.savefig(output_path, dpi = 300, bbox_inches = "tight")
    plt.close()
    log.info("Heatmap saved to %s", output_path)

# ------------------------------ #
# Technical indicators (MA, RSI) #
# ------------------------------ #

def add_moving_averages(
        df: pd.DataFrame, price_column: str, windows: Tuple[int, ...] = (5, 20)
    ) -> pd.DataFrame:
    # Append moving average columns for the given price column
    for w in windows:
        col_name = f"{price_column}_{w}d_MA"
        df[col_name] = df[price_column].rolling(window = w).mean()
    return df
  
def calculate_rsi(
        series: pd.Series, window: int = 14
    ) -> pd.Series:
    # Return the Relative Strength Index (RSI) for a price series.
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window = window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window = window).mean()
    relative_strength = gain / loss
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi

def plot_technical_indicators(
        df: pd.DataFrame, price_column: str, output_path: Path,
    ) -> None:
    # Plot price, moving averages and RSI; save figures. 
    log.info("Plotting technical indicators for %s", price_column)  
    # Price + MAs
    plt.figure(figsize=(8, 6), dpi = 100)
    plt.plot(df.index, df[price_column], label = "Price", linewidth = 1, color = "maroon")
    for w in (5, 20):
        ma_col = f"{price_column}_{w}d_MA"
        plt.plot(df.index, df[ma_col], label=f"{w}d MA", linewidth = 1)
    plt.title(f"{price_column} – Moving Averages", weight = "bold")
    plt.xlabel("Date")
    plt.ylabel("Price (EUR)")
    plt.grid(True)
    plt.legend(loc = "upper right")
    plt.tight_layout()
    plt.savefig(output_path / f"{price_column}-MA.png", dpi = 300)
    plt.close()
    log.info("Moving Average plot has been saved to %s", output_path)

    # RSI
    rsi = calculate_rsi(df[price_column])
    plt.figure(figsize=(8, 6), dpi = 100)
    plt.plot(df.index, rsi, label = "RSI", color = "maroon", linewidth = 1)
    plt.axhline(30, linestyle = "--", color = "seagreen", linewidth = 1, label = "Oversold")
    plt.axhline(70, linestyle = "--", color = "dodgerblue", linewidth = 1, label = "Overbought")
    plt.title(f"RSI for {price_column}", weight = "bold")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend(loc = "lower right")
    plt.tight_layout()
    plt.savefig(output_path / f"{price_column}-RSI.png", dpi = 300)
    plt.close()
    log.info("Relative Strength Index has been saved to %s", output_path)

def sharpe_ratio(
        portfolio: pd.DataFrame, columns: list, risk_free_rate: float,
    ) -> pd.DataFrame:
    log.info("Calculating Sharpe Ratios")
    sharpe_ratio = []
    for i in columns:
        mean_return = portfolio[i].pct_change().mean()
        volatility = portfolio[i].pct_change().std()
        sharpe_ratio.append((mean_return - risk_free_rate) / volatility)
    table_data = pd.DataFrame({
        "Stock": columns,
        "Sharpe Ratio": sharpe_ratio
        })
    return table_data

def plot_sharpe_ratio(
        table_data: pd.DataFrame, output_path: Path,
    ) -> None:
    plt.figure(figsize = (8, 6), dpi = 100)
    bars = plt.bar(table_data["Stock"], table_data["Sharpe Ratio"], color = "mediumseagreen")
    plt.title("Sharpe Ratios for individual portfolio stocks", fontsize = 14, weight = "bold")
    #plt.xlabel("Stock", fontsize = 12)
    plt.ylabel("Sharpe Ratio", fontsize = 12)
    plt.xticks(rotation = 45)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.3f}',
                 ha = "center", va = "bottom", fontsize = 10)
    plt.tight_layout()
    plt.savefig(output_path, dpi = 300)
    plt.close()
    log.info("Sharpe Ratio bar chart saved to %s", output_path)

# ---------------------- #
# Monte Carlo simulation #
# ---------------------- #

def monte_carlo_simulation(
        last_price: float, volatility: float, days: int = 250,
        runs: int = 1000, drift: float = 0.0,
    ) -> np.ndarray:
    log.info("Running Monte Carlo simulation (%d runs, %d days)", runs, days)
    price_paths = np.zeros((runs, days))
    for i in range(runs):
        price_paths[i, 0] = last_price
        for j in range(1, days):
            price_paths[i, j] = price_paths[i, j-1] * np.exp(
                    np.random.normal(0, volatility)
            )
    log.info("Monte Carlo simulation completed")
    return price_paths

def plot_monte_carlo(
        paths: np.ndarray, price_column: str, output_path: Path, price_range: Tuple[float, float] = (0, 700)
    ) -> None:
    x = np.arange(0, paths.shape[1], 1)
    plt.figure(figsize = (8, 6), dpi = 100)
    plt.title(f"{price_column} – Monte Carlo Simulation", weight = "bold")
    plt.xlabel("Days")
    plt.ylabel("Price (EUR)")
    plt.grid(True)
    plt.xlim(0, paths.shape[1])
    plt.ylim(*price_range)
    plt.tight_layout()
    for i in range(paths.shape[0]):
        plt.plot(x, paths[i], color= "dodgerblue", linewidth=0.1, alpha=0.3)
    plt.savefig(output_path, dpi = 300)
    plt.close
    log.info("Monte Carlo plot saved to %s", output_path)
  
# ---------- #
# LSTM model #
# -----------#

class LSTM_Model(nn.Module):
    # LSTM based regression model.
    def __init__(self, input_size: int = 1, hidden_size: int = 50, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out
 
def create_dataset(
        dataset: np.ndarray, lookback: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    # Transform a time‑series into input‑target pairs.
    X, y = [], []
    for i in range(len(dataset) - lookback):
        X.append(dataset[i : i + lookback])
        y.append(dataset[i + 1: i + lookback + 1])
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
  
def train_lstm(
        timeseries: np.ndarray, output_path: Path, epochs: int = 1000,
        batch_size: int = 8, lr: float = 1e-3, lookback: int = 4,
    ) -> Tuple[LSTM_Model, float, float]:
    
    # Train the LSTM and return the model plus test RMSE.
    train_size = int(len(timeseries) * 0.80)
    test_size = len(timeseries) - train_size
    train_series, test_series = timeseries[:train_size], timeseries[train_size:]
    X_train, y_train = create_dataset(train_series, lookback)
    X_test, y_test = create_dataset(test_series, lookback)

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training LSTM on %s", device)

    model = LSTM_Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    criterion = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle = True, batch_size = batch_size)
 
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            y_pred.to(device)
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation every epoch (could be less frequent)
        if epoch % 100 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = sqrt(criterion(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = sqrt(criterion(y_pred, y_test))
        log.info("Epoch %d: train RMSE %4f, test RMSE %.4f", epoch, train_rmse, test_rmse)

    # Plot training and test predictions
    with torch.no_grad():
        train_plot = np.ones_like(timeseries) * np.nan
        y_pred_train = model(X_train)
        y_pred_train = y_pred_train[:, -1, :]
        for i in range(lookback, train_size):
            train_plot[i] = float(y_pred_train[i - lookback])
        test_plot = np.ones_like(timeseries) * np.nan
        y_pred_test = model(X_test)
        y_pred_test = y_pred_test[:, -1, :]
        for i in range(lookback, test_size):
            test_plot[i + train_size] = float(y_pred_test[i - lookback])
    
    plt.figure(figsize = (8, 6), dpi = 100)
    plt.title("LSTM Forecast", weight = "bold")
    plt.plot(timeseries, linewidth = 0, marker = "o", markersize = 3, color = "dodgerblue", label = "Data")
    plt.plot(train_plot, linewidth = 1, color = "maroon", label = "Train Prediction")
    plt.plot(test_plot, linewidth = 1, color = "seagreen", label = "Prediction")
    plt.xlim(0, )
    plt.xlabel("Days")
    plt.ylabel("Price (EUR)")
    plt.grid(True)
    plt.legend(loc = "upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi = 300)
    plt.close()
    log.info("LSTM plot saved to %s", output_path)

    log.info("Training completed - final test RMSE: %.4f", test_rmse)
    return model, test_rmse

# ------------------------------- #
# Argument parsing & main routine #
# ------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Quantitative analysis of stock price data.")
    parser.add_argument(
        "csv_path",
        type = Path,
        help = "Path to the CSV file containing raw stock data.",
    )
    parser.add_argument(
        "--price-column",
        default = "Advanced Micro Devices (EUR)",
        help = "Column to use for time-series modelling (default: Advanced Micro Devices (EUR).",
    )
    parser.add_argument(
        "--confidence",
        type = float,
        default = 0.95,
        help = "Confidence level for VaR (default: 0.95).",
    )
    parser.add_argument(
        "--output-dir",
        type = Path,
        default = Path("output"),
        help = "Directory where plots and intermediate files will be stored (default: output).",
    )
    parser.add_argument(
        "--risk-free-rate",
        type = float,
        default = 0.04 / 250,
        help = "Annual risk-free rate (default: 0.04 / 250).",
    )
    parser.add_argument(
        "--monte-carlo-days",
        type = int,
        default = 250,
        help = "Days to simulate in Monte Carlo (default: 250).",
    )
    parser.add_argument(
        "--monte-carlo-runs",                                                          
        type = int,
        default = 1000,
        help = "Number of Monte Carlo simulation paths (default: 1000).",
    )
    parser.add_argument(
        "--epochs",
        type = int,
        default = 1000,
        help = "Maximum number of LSTM training epochs (default: 1000).",
    )
    parser.add_argument(
        "--batch-size",
        type = int,
        default = 8,
        help = "Batch size for LSTM training (default: 8).",      
    )
    parser.add_argument(
        "--learning-rate",
        type = float,
        default = 0.001,
        help = "Learning rate for LSTM training (default: 0.001).",
    )
    parser.add_argument(
        "--lookback",
        type = int,
        default = 4,
        help = "Look‑back window for LSTM dataset creation (default: 4).",
    )
    return parser.parse_args()

# ------------ #
# Main Routine #
# ------------ #

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents = True, exist_ok = True)

    # 1. Load & clean data
    raw_df = load_data(args.csv_path)
    df = clean_data(raw_df)

    # 2. Descriptive statistics
    stats = descriptive_statistics(df)
    print(stats)

    # 3. Portfolio analysis (example subset)
    columns, weights = portfolio_selection("input/portfolio.csv")
    portfolio_data = df[columns]

    portfolio_ret = compute_portfolio_return(portfolio_data, weights)
    risk_metrics = compute_portfolio_risk(portfolio_data, confidence = args.confidence)
    print(risk_metrics)

    # 4. Correlation heatmap
    corr = portfolio_data.pct_change().dropna().corr()
    plot_correlation_heatmap(corr, args.output_dir / "Heatmap.png")
    
    # 5. Technical indicators for a chosen ticker
    add_moving_averages(df, args.price_column, windows = (5, 20))
    plot_technical_indicators(df, args.price_column, args.output_dir)

    # 6. Sharpe ratios
    sharpe_ratios = sharpe_ratio(portfolio_data, columns, args.risk_free_rate)
    plot_sharpe_ratio(sharpe_ratios, args.output_dir / "Sharpe-Ratios.png")

    # 7. Monte Carlo simulation
    last_price = df[args.price_column].iloc[-1]
    volatility = df[args.price_column].pct_change().std()
    mc_paths = monte_carlo_simulation(
        last_price = last_price,
        volatility = volatility,
        days = args.monte_carlo_days,
        runs = args.monte_carlo_runs,
    )
    plot_monte_carlo(mc_paths, args.price_column, args.output_dir / "Monte-Carlo.png")

    # 7. LSTM forecasting                                                              
    timeseries = df[[args.price_column]].values.astype("float32")

    train_model, test_rmse = train_lstm(
        timeseries,
        args.output_dir / "LSTM.png",
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.lookback,
    )

if __name__ == "__main__":
     main()
