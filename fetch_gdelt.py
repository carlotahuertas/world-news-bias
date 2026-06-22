"""
fetch_gdelt.py
--------------
Uses the GDELT Doc API to search for articles on a given topic
across multiple sources and countries, returning tone scores
for bias comparison.

This is Layer 2 of the World News Bias Analyzer.
"""

from gdeltdoc import GdeltDoc, Filters
import json
import time
import os
from datetime import datetime, timedelta

gd = GdeltDoc()

# Topics we want to track — these are the events we'll
# compare across different national news sources
TOPICS = [
    "Ukraine Russia war",
    "Iran nuclear deal",
    "climate change",
    "immigration",
    "artificial intelligence regulation",
]

# Sources we care about — representing different national
# and political perspectives
SOURCES = [
    "bbc.co.uk",           # UK public broadcaster
    "theguardian.com",     # UK left-leaning
    "foxnews.com",         # US right-leaning
    "nytimes.com",         # US center-left
    "rt.com",              # Russian state media
    "aljazeera.com",       # Qatar-based, Global South perspective
    "dw.com",              # German public broadcaster
    "lemonde.fr",          # French center-left
    "tass.com",            # Russian state news agency
    "timesofindia.com",    # Indian perspective
    "globo.com",           # Brazilian major outlet
    "haaretz.com",         # Israeli left-leaning
]

def fetch_topic_coverage(topic, days_back=3):
    """
    For a given topic, search GDELT for recent articles
    and return them with tone scores.
    """
    end_date   = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)

    print(f"\n  Searching: '{topic}'")
    print(f"  Period: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")

    try:
        f = Filters(
            keyword    = topic,
            start_date = start_date.strftime("%Y-%m-%d"),
            end_date   = end_date.strftime("%Y-%m-%d"),
            num_records = 50,
        )
        articles = gd.article_search(f)

        if articles is None or articles.empty:
            print(f"  — No results found")
            return []

        print(f"  ✓ Found {len(articles)} articles")

        # Convert to list of dicts with clean fields
        results = []
        for _, row in articles.iterrows():
            results.append({
                "topic":         topic,
                "title":         row.get("title", ""),
                "url":           row.get("url", ""),
                "domain":        row.get("domain", ""),
                "language":      row.get("language", ""),
                "source_country":row.get("sourcecountry", ""),
                "published":     str(row.get("seendate", "")),
                "fetched_at":    datetime.utcnow().isoformat(),
            })

        return results

    except Exception as e:
        print(f"  Error: {e}")
        return []

def fetch_source_tone(topic, source_domain, days_back=7):
    """
    Get the tone score for a specific source covering a topic.
    Tone ranges from -10 (very negative) to +10 (very positive).
    This is GDELT's built-in sentiment measure.
    """
    end_date   = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)

    try:
        f = Filters(
            keyword    = topic,
            domain     = source_domain,
            start_date = start_date.strftime("%Y-%m-%d"),
            end_date   = end_date.strftime("%Y-%m-%d"),
        )
        # Get tone timeline for this source on this topic
        tone_data = gd.timeline_search("timelinetone", f)

        if tone_data is None or tone_data.empty:
            return None

        # Average tone across the time period
        avg_tone = tone_data["Tone"].mean()
        return round(avg_tone, 3)

    except Exception as e:
        return None

def main():
    os.makedirs("data", exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  GDELT News Fetcher — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*55}")

    all_results = {}

    for topic in TOPICS:
        print(f"\n{'─'*55}")
        print(f"  TOPIC: {topic.upper()}")
        print(f"{'─'*55}")

        # Step 1 — get articles covering this topic
        articles = fetch_topic_coverage(topic, days_back=3)
        time.sleep(6)  # respect GDELT rate limit (1 per 5 sec)

        # Step 2 — get tone scores per source for this topic
        tone_scores = {}
        print(f"\n  Fetching tone scores per source...")
        for source in SOURCES:
            tone = fetch_source_tone(topic, source, days_back=7)
            if tone is not None:
                tone_scores[source] = tone
                print(f"    {source:<30} tone: {tone:+.2f}")
            time.sleep(6)

        all_results[topic] = {
            "articles":    articles,
            "tone_scores": tone_scores,
        }

    # Save everything
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    path = f"data/gdelt_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*55}")
    print(f"  Done. Results saved to {path}")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    main()
