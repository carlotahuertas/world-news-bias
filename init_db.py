"""
Step 1: Create news.db and load all data/*.json files into it.
Run: python init_db.py
"""
import sqlite3
import json
import os
import glob
import re
from datetime import datetime

DB_PATH = "news.db"
DATA_DIR = "data"


def strip_html(text):
    if not text:
        return ""
    return re.sub(r"<[^>]+>", " ", text).strip()


def create_schema(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS articles (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            country_code    TEXT NOT NULL,
            country_name    TEXT NOT NULL,
            title           TEXT NOT NULL,
            description     TEXT,
            source          TEXT,
            url             TEXT,
            published       TEXT,
            category        TEXT,
            fetched_at      TEXT,
            fetch_file      TEXT,
            UNIQUE(url)
        );

        CREATE TABLE IF NOT EXISTS clusters (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at      TEXT NOT NULL,
            label           TEXT
        );

        CREATE TABLE IF NOT EXISTS cluster_members (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id      INTEGER NOT NULL REFERENCES clusters(id),
            article_id      INTEGER NOT NULL REFERENCES articles(id),
            similarity      REAL,
            UNIQUE(cluster_id, article_id)
        );

        CREATE TABLE IF NOT EXISTS bias_scores (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id          INTEGER NOT NULL REFERENCES articles(id),
            sentiment_polarity  REAL,
            sentiment_subjectivity REAL,
            emotional_word_count INTEGER,
            emotional_word_ratio REAL,
            entity_count        INTEGER,
            entities_json       TEXT,
            scored_at           TEXT,
            UNIQUE(article_id)
        );

        CREATE INDEX IF NOT EXISTS idx_articles_country  ON articles(country_code);
        CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published);
        CREATE INDEX IF NOT EXISTS idx_cluster_members_cluster ON cluster_members(cluster_id);
    """)
    conn.commit()


def load_json_file(conn, filepath):
    filename = os.path.basename(filepath)
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    inserted = 0
    skipped = 0
    countries = data.get("countries", {})

    for country_code, articles in countries.items():
        for art in articles:
            url = art.get("url", "").strip()
            if not url:
                skipped += 1
                continue

            title = art.get("title", "").strip()
            description = strip_html(art.get("description", ""))
            source = art.get("source", "")
            published = art.get("published", "")
            category = json.dumps(art.get("category", []))
            fetched_at = art.get("fetched_at", "")
            country_name = art.get("country_name", "")

            try:
                conn.execute("""
                    INSERT OR IGNORE INTO articles
                        (country_code, country_name, title, description,
                         source, url, published, category, fetched_at, fetch_file)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (country_code, country_name, title, description,
                      source, url, published, category, fetched_at, filename))
                if conn.execute("SELECT changes()").fetchone()[0]:
                    inserted += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"  Warning: could not insert article: {e}")
                skipped += 1

    conn.commit()
    return inserted, skipped


def main():
    json_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    if not json_files:
        print(f"No JSON files found in {DATA_DIR}/")
        return

    print(f"\n{'='*50}")
    print(f"  Initialising {DB_PATH}")
    print(f"{'='*50}\n")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    create_schema(conn)
    print("Schema created.\n")

    total_inserted = 0
    total_skipped = 0
    for path in json_files:
        ins, skp = load_json_file(conn, path)
        total_inserted += ins
        total_skipped += skp
        print(f"  {os.path.basename(path):35s}  +{ins} new  ({skp} skipped/duplicate)")

    conn.close()

    print(f"\n{'='*50}")
    print(f"  Done. {total_inserted} articles loaded, {total_skipped} skipped.")
    print(f"  Database: {os.path.abspath(DB_PATH)}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
