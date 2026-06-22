"""
Step 4: CLI report — top 5 cross-country story clusters with bias scores.

Run: python report.py
     python report.py --clusters 10      # show more clusters
     python report.py --min-countries 3  # only clusters with 3+ countries
"""
import sqlite3
import json
import argparse
import sys
from datetime import datetime

DB_PATH = "news.db"

COUNTRY_FLAGS = {
    "us": "🇺🇸", "gb": "🇬🇧", "es": "🇪🇸", "fr": "🇫🇷",
    "ru": "🇷🇺", "cn": "🇨🇳", "il": "🇮🇱", "br": "🇧🇷",
    "in": "🇮🇳", "za": "🇿🇦", "de": "🇩🇪", "mx": "🇲🇽",
}

COUNTRY_NAMES = {
    "us": "United States", "gb": "United Kingdom", "es": "Spain",
    "fr": "France",        "ru": "Russia",         "cn": "China",
    "il": "Israel",        "br": "Brazil",         "in": "India",
    "za": "South Africa",  "de": "Germany",        "mx": "Mexico",
}

SENTIMENT_BAR_LEN = 20


def bar(value, lo=-1.0, hi=1.0, width=20):
    """Render a signed bar: negative fills left, positive fills right."""
    mid = width // 2
    norm = (value - lo) / (hi - lo)
    pos = int(norm * width)
    pos = max(0, min(width - 1, pos))
    chars = ["-"] * width
    if pos < mid:
        for i in range(pos, mid):
            chars[i] = "◀"
    elif pos > mid:
        for i in range(mid, pos):
            chars[i] = "▶"
    chars[mid] = "|"
    return "".join(chars)


def sentiment_label(pol):
    if pol <= -0.3:
        return "very negative"
    if pol <= -0.1:
        return "negative"
    if pol < 0.1:
        return "neutral"
    if pol < 0.3:
        return "positive"
    return "very positive"


def subjectivity_label(subj):
    if subj < 0.2:
        return "factual"
    if subj < 0.4:
        return "mostly factual"
    if subj < 0.6:
        return "mixed"
    if subj < 0.8:
        return "opinionated"
    return "highly opinionated"


def get_top_clusters(conn, n, min_countries):
    rows = conn.execute("""
        SELECT c.id, c.label,
               COUNT(DISTINCT cm.article_id) as n_articles,
               COUNT(DISTINCT a.country_code) as n_countries
        FROM clusters c
        JOIN cluster_members cm ON cm.cluster_id = c.id
        JOIN articles a ON a.id = cm.article_id
        GROUP BY c.id
        HAVING n_countries >= ?
        ORDER BY n_countries DESC, n_articles DESC
        LIMIT ?
    """, (min_countries, n)).fetchall()
    return rows


def get_cluster_articles(conn, cluster_id):
    rows = conn.execute("""
        SELECT a.id, a.country_code, a.country_name, a.title,
               a.description, a.source, a.published,
               bs.sentiment_polarity, bs.sentiment_subjectivity,
               bs.emotional_word_count, bs.emotional_word_ratio,
               bs.entity_count, bs.entities_json,
               cm.similarity
        FROM cluster_members cm
        JOIN articles a ON a.id = cm.article_id
        LEFT JOIN bias_scores bs ON bs.article_id = a.id
        WHERE cm.cluster_id = ?
        ORDER BY a.country_code
    """, (cluster_id,)).fetchall()
    return rows


def top_entities(entities_json, n=3):
    try:
        ents = json.loads(entities_json or "[]")
    except Exception:
        return []
    seen = set()
    out = []
    for e in ents:
        key = e["text"].lower()
        if key not in seen:
            seen.add(key)
            out.append(f"{e['text']} ({e['label']})")
        if len(out) >= n:
            break
    return out


def print_cluster(rank, cluster_id, label, articles, use_emoji=True):
    n_countries = len({r[1] for r in articles})
    print()
    print(f"  {'─'*70}")
    print(f"  #{rank}  CLUSTER {cluster_id}  ·  {len(articles)} articles across {n_countries} countries")
    print(f"  {'─'*70}")
    print(f"  STORY: {label[:80]}")
    print()

    # Per-country breakdown
    print(f"  {'COUNTRY':<18} {'SENTIMENT':^22} {'SUBJ':>6}  {'EMO':>4}  {'ENTITIES'}")
    print(f"  {'·'*18} {'·'*22} {'·'*6}  {'·'*4}  {'·'*30}")

    for row in articles:
        (art_id, cc, cname, title, desc, source, published,
         pol, subj, emo_cnt, emo_ratio, ent_cnt, ents_json, sim) = row

        pol = pol or 0.0
        subj = subj or 0.0
        emo_cnt = emo_cnt or 0
        ents_json = ents_json or "[]"

        flag = COUNTRY_FLAGS.get(cc, "  ") if use_emoji else ""
        country_label = f"{flag} {COUNTRY_NAMES.get(cc, cname)}"[:18]
        sentiment_bar = bar(pol, -1, 1, 20)
        entities = top_entities(ents_json)
        ent_str = ", ".join(entities) if entities else "—"

        print(f"  {country_label:<18} [{sentiment_bar}] {subj:5.2f}  {emo_cnt:4d}  {ent_str[:40]}")
        print(f"  {'':18}   {sentiment_label(pol):<20}  {subjectivity_label(subj)}")
        # Headline
        headline = title[:75] + "…" if len(title) > 75 else title
        print(f"  {'':18}   \"{headline}\"")
        print(f"  {'':18}   [{source}]  {(published or '')[:10]}")
        print()


def bias_summary(articles):
    """Return a one-line framing divergence note."""
    pol_by_country = {}
    for row in articles:
        cc = row[1]
        pol = row[7] or 0.0
        pol_by_country.setdefault(cc, []).append(pol)
    avgs = {cc: sum(v)/len(v) for cc, v in pol_by_country.items()}
    if len(avgs) < 2:
        return None
    most_pos = max(avgs, key=avgs.get)
    most_neg = min(avgs, key=avgs.get)
    if avgs[most_pos] - avgs[most_neg] < 0.1:
        return "Framing is broadly aligned across countries."
    return (f"Most positive framing: {COUNTRY_NAMES.get(most_pos, most_pos)} "
            f"({avgs[most_pos]:+.2f})  |  "
            f"Most negative framing: {COUNTRY_NAMES.get(most_neg, most_neg)} "
            f"({avgs[most_neg]:+.2f})")


def main():
    parser = argparse.ArgumentParser(description="World News Bias Report")
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters to show")
    parser.add_argument("--min-countries", type=int, default=2,
                        help="Minimum countries per cluster (default 2)")
    parser.add_argument("--no-emoji", action="store_true", help="Disable flag emojis")
    args = parser.parse_args()

    try:
        conn = sqlite3.connect(DB_PATH)
    except Exception as e:
        print(f"Cannot open {DB_PATH}: {e}")
        sys.exit(1)

    # Check DB is populated
    n_art = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    n_clust = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
    n_scores = conn.execute("SELECT COUNT(*) FROM bias_scores").fetchone()[0]

    if n_art == 0:
        print("No articles found. Run: python init_db.py")
        sys.exit(1)
    if n_clust == 0:
        print("No clusters found. Run: python match_articles.py")
        sys.exit(1)
    if n_scores == 0:
        print("No bias scores found. Run: python score_bias.py")
        sys.exit(1)

    use_emoji = not args.no_emoji

    print()
    print(f"  {'═'*70}")
    print(f"  WORLD NEWS BIAS ANALYZER  ·  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  {n_art} articles  |  {n_clust} clusters  |  {n_scores} scored")
    print(f"  {'═'*70}")

    clusters = get_top_clusters(conn, args.clusters, args.min_countries)

    if not clusters:
        print(f"\n  No clusters with >= {args.min_countries} countries found.")
        print("  Try --min-countries 2")
        conn.close()
        return

    print(f"\n  TOP {len(clusters)} STORY CLUSTERS  "
          f"(min {args.min_countries} countries, ranked by country coverage)\n")

    for rank, (cid, label, n_art_c, n_cntry) in enumerate(clusters, 1):
        articles = get_cluster_articles(conn, cid)
        print_cluster(rank, cid, label, articles, use_emoji=use_emoji)
        summary = bias_summary(articles)
        if summary:
            print(f"  ► FRAMING DIVERGENCE: {summary}")

    print()
    print(f"  {'═'*70}")
    print(f"  SENTIMENT SCALE:  ◀◀◀ negative  |  neutral  |  positive ▶▶▶")
    print(f"  SUBJECTIVITY:     0.0 = purely factual  →  1.0 = highly opinionated")
    print(f"  EMO:              emotional word count in title + description")
    print(f"  {'═'*70}")
    print()

    conn.close()


if __name__ == "__main__":
    main()
