from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score

RANDOM_SEED = 42
TEST_RATIO = 0.05
N_SPLITS = 100
TOP_N = 10
DATA_BASE_PATH = Path("./data")


gold_file_path = DATA_BASE_PATH / "final_gold_data.csv"

gold = pd.read_csv(gold_file_path, sep=";", encoding="utf-8", parse_dates=["timestamp"])


def treat_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(by="timestamp", ascending=True)


def shift_and_clean_nan(
    df: pd.DataFrame, column: str, shift: int = 1, rename_to: str | None = None
) -> pd.DataFrame:
    if not rename_to:
        rename_to = f"shift_{column}_{shift}"
    df[rename_to] = df[column].shift(shift)
    df = df.dropna(subset=[rename_to])  # Remove last row with NaN

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_variation"] = df["open"] - df["close"]
    df["max_diff"] = df["high"] - df["low"]

    df["return"] = (df["tomorrow_close"] - df["close"]) / df["close"]

    return df


type ColType = Literal["month", "year", "day", "hour", "minute", "second"]


def access_time_series(series: pd.Series, col_type: ColType) -> pd.Series:
    if col_type not in ["month", "year", "day", "hour", "minute", "second"]:
        raise TypeError("Invalid option")

    return getattr(series.dt, col_type)


def encode_time_features(
    df: pd.DataFrame, time_col: str, col_type: ColType | list[ColType]
) -> pd.DataFrame:
    if isinstance(col_type, str):
        col_type = [col_type]

    for c_type in col_type:
        df[c_type] = access_time_series(df[time_col], c_type)

    return pd.get_dummies(df, columns=col_type, drop_first=True, dtype=int)


access_time_series(gold["timestamp"], "month")


gold["month"] = gold["timestamp"].dt.month
gold["year"] = gold["timestamp"].dt.year
gold = pd.get_dummies(gold, columns=["month", "year"], drop_first=True, dtype=int)

vectorizer = TfidfVectorizer(
    stop_words="english", max_features=1000, ngram_range=(1, 2), min_df=5, max_df=0.8
)
tfidf_features = vectorizer.fit_transform(gold["headlines"])
TfidfMatrix = pd.DataFrame(
    tfidf_features.toarray(), columns=vectorizer.get_feature_names_out()
)  # type: ignore

sum_words = TfidfMatrix.sum(axis=0)
words_freq = [
    (word, sum_words.iloc[idx]) for word, idx in vectorizer.vocabulary_.items()
]
sorted_words = sorted(words_freq, key=lambda x: x[1], reverse=True)

words = [word for word, _ in sorted_words[:TOP_N]]
frequencies = [freq for _, freq in sorted_words[:TOP_N]]

tfidf_df = (
    pd.DataFrame({"word": words, "frequency": frequencies})
    .set_index("word")
    .sort_values(by="frequency", ascending=False)
)

print("Top words:")
print(tfidf_df)
print("\n\n")


def contain_word(series: pd.Series, word: str) -> pd.Series:
    """
    Check if a word is contained in the headlines.
    """
    return series.str.contains(word, case=False, na=False)


def log_normalize(series: pd.Series) -> pd.Series:
    """
    Log normalize a series.
    """
    return np.log(series.abs() + 1)


for word in words:
    gold[word] = contain_word(gold["headlines"], word).astype(int)

gold["volume"] = log_normalize(gold["volume"])
gold["headlines_count"] = (
    gold["headlines"].apply(lambda x: len(x.split("/"))).astype(int)
)

target = "tomorrow_close"
to_drop = [
    "headlines",
    "timestamp",
    # "open",
    "high",
    "low",
    "close",
    "unit",
    "currency",
    # "volume",
    # "day_variation",
    # "max_diff",
    "tomorrow_close",
    "return",
]

X = gold.drop(to_drop, axis=1)
y = gold[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_RATIO, shuffle=False
)

pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", Ridge()),
    ]
)

cv = TimeSeriesSplit(n_splits=N_SPLITS)

gs = GridSearchCV(
    pipeline,
    param_grid={
        "model__alpha": [0.01, 0.1, 1, 10, 100],
        "model__solver": ["auto", "saga"],
        "model__random_state": [RANDOM_SEED],
    },
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
)

gs.fit(X_train, y_train)

print("Best parameters:", gs.best_params_)
print("Best score:", gs.best_score_)
print("Grid Search Results:")

results = pd.DataFrame(gs.cv_results_)
results = results.sort_values(by="rank_test_score")
results = results[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
results["mean_test_score"] = -results["mean_test_score"]
results["std_test_score"] = results["std_test_score"].abs()
results["params"] = results["params"].apply(
    lambda x: {k: v for k, v in x.items() if k != "model__"}
)
results = results.rename(
    columns={
        "mean_test_score": "RMSE",
        "std_test_score": "RMSE_std",
        "rank_test_score": "Rank",
    }
)
results = results.set_index("Rank")

# Calculate R² for the best model
best_model = gs.best_estimator_
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R² for the best model: {r2:.4f}")

# results.to_csv("grid_search_results.csv", sep=";", index=True)
print(results)


def print_95_percentile_ci(results) -> None:
    """
    Print the 95% confidence interval for the predictions.
    """

    rmse = results["RMSE"].values

    rmse_95th = np.percentile(rmse, [0.025, 0.975])

    print(f"95% CI for RMSE: {rmse_95th}")


print_95_percentile_ci(results)
