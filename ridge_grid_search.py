from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score

RANDOM_SEED = 42
TEST_RATIO = 0.2
TOP_N = 20
DATA_BASE_PATH = Path("./data")


gold_file_path = DATA_BASE_PATH / "final_gold_data.csv"

gold = pd.read_csv(gold_file_path, sep=";", encoding="utf-8", parse_dates=["timestamp"])

gold["day_variation"] = gold["open"] - gold["close"]
gold["max_diff"] = gold["high"] - gold["low"]

gold["tomorrow_close"] = gold["close"].shift(-1)
gold = gold.dropna(subset=["tomorrow_close"])  # Remove last row with NaN


gold["return"] = (gold["tomorrow_close"] - gold["close"]) / gold[
    "close"
]  # return might be our target


gold["month"] = gold["timestamp"].dt.month
gold["year"] = gold["timestamp"].dt.year
gold = pd.get_dummies(gold, columns=["month", "year"], drop_first=True)

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


def contain_word(series: pd.Series, word: str) -> pd.Series:
    """
    Check if a word is contained in the headlines.
    """
    return series.str.contains(word, case=False, na=False)


for word in words:
    gold[word] = contain_word(gold["headlines"], word)


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

pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("model", Ridge()),
    ]
)

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

gs = GridSearchCV(
    pipeline,
    param_grid={
        "model__alpha": np.arange(7000, 10001, 1000),
        "model__solver": ["auto", "saga"],
    },
    scoring="neg_root_mean_squared_error",
    cv=kf,
    n_jobs=-1,
)

gs.fit(X, y)

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
y_pred = best_model.predict(X)
r2 = r2_score(y, y_pred)
print(f"R² for the best model: {r2:.4f}")

# results.to_csv("grid_search_results.csv", sep=";", index=True)
print(results)
