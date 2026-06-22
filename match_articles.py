"""
Step 2: Embed all articles with sentence-transformers, cluster by cosine
similarity, store clusters in the database.

Only articles from different countries can form a cluster (no point
comparing a country to itself). A cluster must span >= 2 countries.

Run: python match_articles.py
"""
import sqlite3
import json
import numpy as np
from datetime import datetime

DB_PATH = "news.db"
MODEL_NAME = "all-MiniLM-L6-v2"   # fast, lightweight, good quality
SIM_THRESHOLD = 0.45               # cosine similarity cutoff
MIN_COUNTRIES = 2                  # cluster must cover at least this many countries


def get_articles(conn):
    rows = conn.execute("""
        SELECT id, country_code, title, description
        FROM articles
        ORDER BY id
    """).fetchall()
    return rows


def build_texts(articles):
    texts = []
    for _, _, title, desc in articles:
        text = title.strip()
        if desc:
            text += ". " + desc[:300]
        texts.append(text)
    return texts


def embed(texts, model):
    print(f"  Embedding {len(texts)} articles...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True,
                               convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def cosine_sim_matrix(embeddings):
    # embeddings are already L2-normalised, so dot product == cosine sim
    return embeddings @ embeddings.T


def greedy_cluster(articles, sim_matrix, threshold, min_countries):
    n = len(articles)
    assigned = [False] * n
    clusters = []

    # Sort candidate pairs by similarity desc
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                pairs.append((sim_matrix[i, j], i, j))
    pairs.sort(reverse=True)

    # Union-find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for sim, i, j in pairs:
        union(i, j)

    # Group by root
    from collections import defaultdict
    groups = defaultdict(list)
    for idx in range(n):
        groups[find(idx)].append(idx)

    # Keep only groups that span >= min_countries AND have > 1 member
    for root, members in groups.items():
        if len(members) < 2:
            continue
        countries_in_group = {articles[m][1] for m in members}
        if len(countries_in_group) < min_countries:
            continue
        clusters.append(members)

    return clusters


def save_clusters(conn, articles, clusters, sim_matrix):
    # Clear old clusters
    conn.execute("DELETE FROM cluster_members")
    conn.execute("DELETE FROM clusters")
    conn.commit()

    now = datetime.utcnow().isoformat()
    saved = 0
    for members in clusters:
        # Label = title of the article with highest average similarity to others
        avg_sims = []
        for m in members:
            sims = [sim_matrix[m, other] for other in members if other != m]
            avg_sims.append(np.mean(sims) if sims else 0)
        center_idx = members[int(np.argmax(avg_sims))]
        label = articles[center_idx][2][:120]  # title

        cur = conn.execute(
            "INSERT INTO clusters (created_at, label) VALUES (?, ?)", (now, label))
        cluster_id = cur.lastrowid

        for m in members:
            article_id = articles[m][0]
            sims = [sim_matrix[m, other] for other in members if other != m]
            avg_sim = float(np.mean(sims)) if sims else 0.0
            conn.execute("""
                INSERT OR IGNORE INTO cluster_members (cluster_id, article_id, similarity)
                VALUES (?, ?, ?)
            """, (cluster_id, article_id, avg_sim))

        conn.commit()
        saved += 1

    return saved


def main():
    print(f"\n{'='*50}")
    print(f"  Article Matching  (model: {MODEL_NAME})")
    print(f"{'='*50}\n")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    conn = sqlite3.connect(DB_PATH)

    articles = get_articles(conn)
    print(f"  Loaded {len(articles)} articles from DB\n")

    texts = build_texts(articles)
    embeddings = embed(texts, model)

    print("\n  Computing similarity matrix...")
    sim_matrix = cosine_sim_matrix(embeddings)

    print(f"  Clustering (threshold={SIM_THRESHOLD}, min_countries={MIN_COUNTRIES})...")
    clusters = greedy_cluster(articles, sim_matrix, SIM_THRESHOLD, MIN_COUNTRIES)
    print(f"  Found {len(clusters)} cross-country clusters\n")

    n_saved = save_clusters(conn, articles, clusters, sim_matrix)

    # Summary
    print("  Top clusters by size:")
    rows = conn.execute("""
        SELECT c.id, c.label, COUNT(cm.article_id) as n,
               GROUP_CONCAT(DISTINCT a.country_code) as countries
        FROM clusters c
        JOIN cluster_members cm ON cm.cluster_id = c.id
        JOIN articles a ON a.id = cm.article_id
        GROUP BY c.id
        ORDER BY n DESC
        LIMIT 10
    """).fetchall()
    for cid, label, n, countries in rows:
        print(f"    [{cid}] {n} articles | {countries}")
        print(f"         {label[:80]}")

    conn.close()

    print(f"\n{'='*50}")
    print(f"  Done. {n_saved} clusters saved to {DB_PATH}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
