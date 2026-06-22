"""
Step 5: AllSides-based bias classifier.

1. Fuzzy-match DB article sources → AllSides ratings (ground truth)
2. Engineer features: TF-IDF + sentiment + emotional words + stylistic signals
3. Train LogisticRegression, print precision/recall/F1 + confusion matrix
4. Predict bias label + confidence for ALL articles
5. Save predictions to news.db

Run: python classify_bias.py
"""
import os, re, sqlite3, json
import pandas as pd
import numpy as np
from rapidfuzz import process as fuzz_process, fuzz

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "news.db")
ALLSIDES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "allsides_ratings.csv")

# ── Manual source → AllSides name overrides ───────────────────────────────────
# Keys are news.db source_ids; values are exact AllSides news_source strings.
MANUAL_MAP = {
    "bbc":                        "BBC News",
    "yahoo":                      "Yahoo! News",
    "washingtontimes":            "Washington Times",
    "latimes":                    "Los Angeles Times",
    "nypost":                     "New York Post",
    "usnews":                     "U.S. News & World Report",
    "dailymailuk":                "Daily Mail",
    "independentuk":              "The Independent",
    "thesun":                     "The Sun (UK)",
    "jpost":                      "The Jerusalem Post",
    "timesofisrael":              "Times of Israel",
    "euronews":                   "Euronews",
    "si":                         "Sports Illustrated",
    "lemonde":                    "Le Monde",
    "hindustantimes":             "Hindustan Times",
    "indianexpress":              "The Indian Express",
    "upi":                        "UPI",
    "clickondetroit":             "Local TV News",
    "benzinga":                   "Benzinga",
    "dallasnews":                 "The Dallas Morning News",
    "expresscouk":                "Daily Express",
}

# AllSides rating → canonical label
RATING_MAP = {
    "left":          "Left",
    "left-center":   "Lean-Left",
    "center":        "Center",
    "right-center":  "Lean-Right",
    "right":         "Right",
}
LABEL_ORDER = ["Left", "Lean-Left", "Center", "Lean-Right", "Right"]

# Emotional word lexicon (same as score_bias.py)
EMOTIONAL_WORDS = {
    "attack","attacks","attacked","attacking","kill","kills","killed","killing",
    "bomb","bombs","bombed","bombing","terror","terrorist","terrorism",
    "war","wars","warfare","crisis","crises","disaster","catastrophe","catastrophic",
    "dead","death","deaths","deadly","brutal","brutality","violence","violent",
    "threat","threats","threatening","danger","dangerous","illegal","illegally",
    "extremist","extremism","radical","propaganda","corrupt","corruption","scandal",
    "chaos","chaotic","destroy","destroyed","destruction","collapse","collapsing",
    "panic","panicking","alarming","alarmed","outrage","outraged","outrageous",
    "horrific","horrifying","horrified","tragic","tragedy","flee","fleeing","fled",
    "slaughter","massacre","invasion","invade","invaded","triumph","triumphant",
    "victory","victorious","hero","heroic","heroes","liberation","liberate","liberated",
    "historic","historical","breakthrough","miracle","regime","dictator","dictatorship",
    "oppression","oppressive","freedom","liberty","innocent","victims","refugees",
    "displaced","occupation","occupied","resistance","provocation","aggression",
    "aggressive","hostage","hostages","rebel","rebels","rebellion","coup","blockade",
    "sanctions","retaliation","escalation","escalate","ceasefire","genocide",
    "atrocity","atrocities",
}

PASSIVE_RE = re.compile(
    r'\b(was|were|is|are|has been|have been|had been|being)\s+\w+(?:ed|en)\b',
    re.IGNORECASE
)


# ── 1. Load AllSides ──────────────────────────────────────────────────────────

def load_allsides():
    df = pd.read_csv(ALLSIDES_PATH)
    df = df[df["rating"].isin(RATING_MAP.keys())].copy()
    df["canonical"] = df["rating"].map(RATING_MAP)
    df["norm_name"] = df["news_source"].str.lower().str.strip()
    return df


# ── 2. Fuzzy source matching ──────────────────────────────────────────────────

def normalize_source(s):
    s = s.lower().strip()
    for prefix in ("the ", "a ", "an "):
        if s.startswith(prefix):
            s = s[len(prefix):]
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def match_sources(db_sources, allsides_df):
    """Return dict: source_id → {'allsides_name', 'rating', 'method', 'score'}"""
    as_names   = allsides_df["news_source"].tolist()
    as_norms   = allsides_df["norm_name"].tolist()
    as_ratings = allsides_df["canonical"].tolist()

    results = {}
    for src in db_sources:
        # 1) Manual override
        if src in MANUAL_MAP:
            target = MANUAL_MAP[src]
            if target in as_names:
                idx = as_names.index(target)
                results[src] = {
                    "allsides_name": target,
                    "rating": as_ratings[idx],
                    "method": "manual",
                    "score": 100,
                }
                continue

        # 2) Fuzzy match on normalized name
        norm = normalize_source(src.replace("_", " "))
        match = fuzz_process.extractOne(
            norm, as_norms,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=72,
        )
        if match:
            matched_name, score, idx = match
            results[src] = {
                "allsides_name": as_names[idx],
                "rating": as_ratings[idx],
                "method": "fuzzy",
                "score": score,
            }

    return results


# ── 3. Feature engineering ────────────────────────────────────────────────────

def build_text(title, description):
    t = (title or "").strip()
    d = (description or "").strip()
    return t + (" " + d[:400] if d else "")


def handcrafted_features(title, description):
    text  = build_text(title, description)
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    n_tok  = max(len(tokens), 1)

    emo_ratio    = sum(1 for t in tokens if t in EMOTIONAL_WORDS) / n_tok
    passive_cnt  = len(PASSIVE_RE.findall(text))
    exclaim      = text.count("!")
    question     = text.count("?")
    title_len    = len(title or "")
    quote_cnt    = text.count('"') + text.count('"') + text.count('"')
    upper_words  = sum(1 for t in (title or "").split()
                       if len(t) > 2 and t.isupper()) / max(len((title or "").split()), 1)

    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        pol  = blob.sentiment.polarity
        subj = blob.sentiment.subjectivity
    except Exception:
        pol, subj = 0.0, 0.0

    return [pol, subj, emo_ratio, passive_cnt, exclaim, question,
            title_len, quote_cnt, upper_words]

HC_FEATURE_NAMES = [
    "sentiment_polarity", "sentiment_subjectivity", "emotional_ratio",
    "passive_count", "exclamation_marks", "question_marks",
    "title_length", "quote_count", "uppercase_word_ratio",
]


# ── 4. Train & evaluate ───────────────────────────────────────────────────────

def train(labeled_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.preprocessing import StandardScaler
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.metrics import classification_report, confusion_matrix
    import scipy.sparse as sp

    texts  = labeled_df.apply(lambda r: build_text(r["title"], r["description"]), axis=1).tolist()
    labels = labeled_df["label"].tolist()
    hc     = np.array([handcrafted_features(r["title"], r["description"])
                       for _, r in labeled_df.iterrows()])

    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    X_tfidf = tfidf.fit_transform(texts)

    # Scale handcrafted features
    scaler = StandardScaler()
    X_hc = scaler.fit_transform(hc)

    X = sp.hstack([X_tfidf, sp.csr_matrix(X_hc)])
    y = np.array(labels)

    # Cross-val (only if enough data)
    n_per_class = pd.Series(labels).value_counts().min()
    cv_folds    = min(5, n_per_class) if n_per_class >= 2 else None

    clf = LogisticRegression(
        max_iter=1000,
        C=0.5,
        class_weight="balanced",
        solver="lbfgs",
    )

    if cv_folds and cv_folds >= 2:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_res = cross_validate(clf, X, y, cv=cv,
                                scoring=["accuracy", "f1_weighted"],
                                return_train_score=False)
        print(f"\n  Cross-val ({cv_folds}-fold):")
        print(f"    accuracy:    {cv_res['test_accuracy'].mean():.3f} ± {cv_res['test_accuracy'].std():.3f}")
        print(f"    f1-weighted: {cv_res['test_f1_weighted'].mean():.3f} ± {cv_res['test_f1_weighted'].std():.3f}")
    else:
        print(f"\n  (Too few samples per class for cross-validation — training on full set)")

    clf.fit(X, y)

    # In-sample evaluation
    y_pred = clf.predict(X)
    present_labels = sorted(set(labels), key=LABEL_ORDER.index)
    print("\n  Classification report (in-sample):")
    print(classification_report(y, y_pred, labels=present_labels, zero_division=0))

    print("  Confusion matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(y, y_pred, labels=present_labels)
    header = "  " + "  ".join(f"{l[:8]:>8}" for l in present_labels)
    print(header)
    for label, row in zip(present_labels, cm):
        print(f"  {label[:8]:>8}  " + "  ".join(f"{v:>8}" for v in row))

    return clf, tfidf, scaler


def predict_all(clf, tfidf, scaler, all_df):
    import scipy.sparse as sp

    texts = all_df.apply(lambda r: build_text(r["title"], r["description"]), axis=1).tolist()
    hc    = np.array([handcrafted_features(r["title"], r["description"])
                      for _, r in all_df.iterrows()])

    X_tfidf = tfidf.transform(texts)
    X_hc    = scaler.transform(hc)
    X       = sp.hstack([X_tfidf, sp.csr_matrix(X_hc)])

    preds   = clf.predict(X)
    probas  = clf.predict_proba(X)
    confs   = probas.max(axis=1)

    return preds, confs


# ── 5. DB helpers ─────────────────────────────────────────────────────────────

def ensure_columns(conn):
    existing = {row[1] for row in conn.execute("PRAGMA table_info(articles)")}
    for col, typ in [("allsides_rating", "TEXT"),
                     ("predicted_bias",  "TEXT"),
                     ("bias_confidence", "REAL")]:
        if col not in existing:
            conn.execute(f"ALTER TABLE articles ADD COLUMN {col} {typ}")
    conn.commit()


def save_ground_truth(conn, source_matches):
    for src, info in source_matches.items():
        conn.execute(
            "UPDATE articles SET allsides_rating = ? WHERE source = ?",
            (info["rating"], src)
        )
    conn.commit()


def save_predictions(conn, article_ids, preds, confs):
    for aid, pred, conf in zip(article_ids, preds, confs):
        conn.execute(
            "UPDATE articles SET predicted_bias = ?, bias_confidence = ? WHERE id = ?",
            (pred, float(conf), int(aid))
        )
    conn.commit()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*56}")
    print(f"  Bias Classifier")
    print(f"{'='*56}\n")

    # Load AllSides
    allsides_df = load_allsides()
    print(f"  AllSides: {len(allsides_df)} rated sources loaded")

    # Load DB sources
    conn = sqlite3.connect(DB_PATH)
    ensure_columns(conn)

    db_sources = [r[0] for r in conn.execute(
        "SELECT DISTINCT source FROM articles WHERE source IS NOT NULL"
    ).fetchall()]
    print(f"  DB sources: {len(db_sources)} unique\n")

    # Fuzzy match
    matches = match_sources(db_sources, allsides_df)
    print(f"  Source matching results:")
    for src, info in sorted(matches.items()):
        print(f"    {src:35s} → {info['allsides_name']:30s}  [{info['rating']:10s}]  ({info['method']}, {info['score']:.0f})")

    print(f"\n  Matched {len(matches)}/{len(db_sources)} sources")

    # Save ground truth
    save_ground_truth(conn, matches)

    # Count labeled articles
    labeled = conn.execute("""
        SELECT a.id, a.title, a.description, a.allsides_rating as label
        FROM articles a
        WHERE a.allsides_rating IS NOT NULL
    """).fetchall()
    labeled_df = pd.DataFrame(labeled, columns=["id", "title", "description", "label"])

    matched_sources = conn.execute("""
        SELECT source, allsides_rating, COUNT(*) as n
        FROM articles
        WHERE allsides_rating IS NOT NULL
        GROUP BY source, allsides_rating
        ORDER BY n DESC
    """).fetchall()
    print(f"\n  Labeled articles by source:")
    for src, rating, n in matched_sources:
        print(f"    {src:30s}  {rating:10s}  ({n} articles)")
    print(f"\n  Total labeled: {len(labeled_df)} articles")
    print(f"  Label distribution:\n{labeled_df['label'].value_counts().to_string()}")

    if len(labeled_df) < 4:
        print("\n  ⚠ Too few labeled articles to train a meaningful classifier.")
        print("    Falling back to AllSides rating where available + neutral default.")
        # Apply ground truth as prediction where available, center otherwise
        all_arts = conn.execute(
            "SELECT id, allsides_rating FROM articles"
        ).fetchall()
        for aid, rating in all_arts:
            pred = rating if rating else "Center"
            conn.execute(
                "UPDATE articles SET predicted_bias = ?, bias_confidence = ? WHERE id = ?",
                (pred, 1.0 if rating else 0.5, aid)
            )
        conn.commit()
        conn.close()
        print("\n  Done (fallback mode — fetch more English news for better results)")
        return

    # Train
    print(f"\n  Training classifier...")
    clf, tfidf, scaler = train(labeled_df)

    # Predict all
    all_arts = conn.execute(
        "SELECT id, title, description FROM articles"
    ).fetchall()
    all_df = pd.DataFrame(all_arts, columns=["id", "title", "description"])
    preds, confs = predict_all(clf, tfidf, scaler, all_df)

    # For labeled articles, override with ground truth at full confidence
    pred_list = list(preds)
    conf_list = list(confs)
    id_to_idx = {row["id"]: i for i, row in all_df.iterrows()}
    for _, row in labeled_df.iterrows():
        idx = id_to_idx.get(row["id"])
        if idx is not None:
            pred_list[idx] = row["label"]
            conf_list[idx] = 1.0

    save_predictions(conn, all_df["id"].tolist(), pred_list, conf_list)
    conn.close()

    # Summary
    pred_series = pd.Series(pred_list)
    print(f"\n  Predictions across all {len(pred_list)} articles:")
    for label in LABEL_ORDER:
        n = (pred_series == label).sum()
        bar = "█" * n
        print(f"    {label:12s}  {n:3d}  {bar}")

    print(f"\n{'='*56}")
    print(f"  Done. Predictions saved to news.db")
    print(f"{'='*56}\n")


if __name__ == "__main__":
    main()
