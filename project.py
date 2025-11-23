import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time

CSV_PATH = "C:\\Users\\karth\\OneDrive\\Desktop\\PROJECTS\\ML_Project\\movies_cleaned.csv"
df = pd.read_csv(CSV_PATH)

title_col = None
for c in ["title", "movie_name", "name", "original_title"]:
    if c in df.columns:
        title_col = c
        break

df[title_col] = df[title_col].astype(str).str.strip()

if "original_language" in df.columns:
    df["original_language"] = (
        df["original_language"].astype(str).str.lower().str.strip().replace(["-", "nan", ""], "unknown")
    )
    df = df[df["original_language"] != "unknown"]

def clean_genres(x):
    if pd.isna(x):
        return []
    return [g.strip().lower() for g in str(x).split("|") if g.strip()]

df["genres"] = df["genres"].apply(clean_genres)

num_cols = ["popularity", "budget", "revenue", "runtime", "vote_average", "vote_count"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[num_cols] = df[num_cols].fillna(df[num_cols].median())

if "release_date" in df.columns:
    d1 = pd.to_datetime(df["release_date"], format="%d-%m-%Y", errors="coerce")
    d2 = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_date_parsed"] = d1.fillna(d2)
    df["year"] = df["release_date_parsed"].dt.year
else:
    df["year"] = np.nan

if df["year"].notna().any():
    df["year"] = df["year"].fillna(int(df["year"].median()))
else:
    df["year"] = 2000

df["year"] = df["year"].astype(int)
df = df.reset_index(drop=True)

scale_cols = num_cols + ["year"]
scaler = MinMaxScaler()
num_features = scaler.fit_transform(df[scale_cols])

def genre_sim(a, b):
    s1, s2 = set(a), set(b)
    if not s1 and not s2:
        return 0
    return len(s1 & s2) / len(s1 | s2)

def lang_sim(a, b):
    return 1 if str(a).lower() == str(b).lower() else 0

def cosine_sim(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n1, n2 = np.linalg.norm(a), np.linalg.norm(b)
    if n1 == 0 or n2 == 0:
        return 0
    return np.dot(a, b) / (n1 * n2)

def get_recommendations(movie, k=10):
    if movie not in df[title_col].values:
        return None
    idx = int(df.index[df[title_col] == movie][0])
    tg = df.loc[idx, "genres"]
    tl = df.loc[idx, "original_language"]
    tn = num_features[idx]
    sims = []
    for i in df.index:
        if i == idx:
            continue
        g = genre_sim(tg, df.loc[i, "genres"])
        l = lang_sim(tl, df.loc[i, "original_language"])
        n = cosine_sim(tn, num_features[i])
        s = 0.5 * g + 0.2 * l + 0.3 * n
        sims.append((i, s))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]

def recommend(movie, k=10):
    r = get_recommendations(movie, k)
    if r is None:
        print("Movie not found.\n")
        return
    print(f"\nTop {k} recommendations for: {movie}\n")
    for i, s in r:
        print(df.iloc[i][title_col], " (Similarity:", round(s, 3), ")")
    print()

def precision_at_k(movie, k=10):
    r = get_recommendations(movie, k)
    if r is None:
        return None
    idx = int(df.index[df[title_col] == movie][0])
    tg = set(df.loc[idx, "genres"])
    rel = sum(1 for i, _ in r if len(tg & set(df.loc[i, "genres"])) > 0)
    return rel / k

def mean_jaccard(movie):
    r = get_recommendations(movie, 10)
    if r is None:
        return None
    idx = int(df.index[df[title_col] == movie][0])
    tg = set(df.loc[idx, "genres"])
    vals = [genre_sim(tg, df.loc[i, "genres"]) for i, _ in r]
    return float(np.mean(vals))

def language_match_rate(movie):
    r = get_recommendations(movie, 10)
    if r is None:
        return None
    idx = int(df.index[df[title_col] == movie][0])
    lang = df.loc[idx, "original_language"]
    return sum(1 for i, _ in r if df.loc[i, "original_language"] == lang) / 10

def avg_similarity_score(movie):
    r = get_recommendations(movie, 10)
    if r is None:
        return None
    idx = int(df.index[df[title_col] == movie][0])
    vals = []
    for i, _ in r:
        g = genre_sim(df.loc[idx, "genres"], df.loc[i, "genres"])
        l = lang_sim(df.loc[idx, "original_language"], df.loc[i, "original_language"])
        n = cosine_sim(num_features[idx], num_features[i])
        vals.append(0.5 * g + 0.2 * l + 0.3 * n)
    return float(np.mean(vals))

def response_time(movie):
    t0 = time.time()
    get_recommendations(movie)
    return time.time() - t0

def plot_language_distribution():
    c = df["original_language"].value_counts().head(10)
    plt.figure(figsize=(8, 8))
    plt.pie(c, labels=c.index, autopct="%1.1f%%")
    plt.title("Top Languages")
    plt.savefig("language_distribution.png", dpi=300)
    plt.show()

def plot_genre_frequency():
    g = df["genres"].explode().value_counts().head(15)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=g.index, y=g.values)
    plt.xticks(rotation=45)
    plt.title("Top Genres")
    plt.savefig("genre_frequency.png", dpi=300)
    plt.show()

def plot_popularity_vs_rating():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df["popularity"], y=df["vote_average"], alpha=0.3)
    plt.title("Popularity vs Rating")
    plt.savefig("popularity_vs_rating.png", dpi=300)
    plt.show()

def plot_movies_per_year():
    y = df["year"]
    y = y[(y > 1800) & (y < 2100)]
    c = y.value_counts().sort_index()
    plt.figure(figsize=(14, 6))
    plt.plot(c.index, c.values, marker="o")
    plt.title("Movies Per Year")
    plt.savefig("movies_per_year.png", dpi=300)
    plt.show()

def plot_budget_revenue_roi():
    d = df.copy()
    d["budget_safe"] = d["budget"].replace(0, np.nan)
    d["roi"] = (d["revenue"] - d["budget"]) / d["budget_safe"]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["budget_safe", "revenue"])
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=d["budget_safe"], y=d["revenue"], hue=d["roi"], palette="coolwarm", alpha=0.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Budget vs Revenue")
    plt.savefig("budget_vs_revenue_roi.png", dpi=300)
    plt.show()

def plot_votes_vs_rating():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df["vote_count"], y=df["vote_average"], alpha=0.3)
    plt.xscale("log")
    plt.title("Votes vs Rating")
    plt.savefig("votes_vs_rating.png", dpi=300)
    plt.show()

def plot_performance(movie):
    if movie not in df[title_col].values:
        movie = df.iloc[0][title_col]
    vals = [
        precision_at_k(movie),
        mean_jaccard(movie),
        language_match_rate(movie),
        avg_similarity_score(movie),
        response_time(movie)
    ]
    names = ["Precision@10", "Jaccard", "Lang Match", "Avg Sim", "Time"]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=names, y=vals)
    plt.xticks(rotation=45)
    plt.title(f"Performance: {movie}")
    plt.savefig("performance_metrics.png", dpi=300)
    plt.show()

def run_all():
    plot_language_distribution()
    plot_genre_frequency()
    plot_popularity_vs_rating()
    plot_movies_per_year()
    plot_budget_revenue_roi()
    plot_votes_vs_rating()

    sample = "RRR" if "RRR" in df[title_col].values else df.iloc[0][title_col]
    plot_performance(sample)

    print("\nRecommendation system ready.\n")
    while True:
        name = input("Enter movie name (or exit): ").strip()
        if name.lower() in ["exit", "quit", "q"]:
            break
        recommend(name)

if __name__ == "__main__":
    run_all()
