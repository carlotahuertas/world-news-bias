import requests
import json
import os
from datetime import datetime

API_KEY = "pub_5d00406494564761a2484bb8d1963692"

# Country codes + readable names + priority local sources
COUNTRIES = {
    "us": {"name": "United States", "lang": "en"},
    "gb": {"name": "United Kingdom", "lang": "en"},
    "es": {"name": "Spain",          "lang": "es"},
    "fr": {"name": "France",         "lang": "fr"},
    "ru": {"name": "Russia",         "lang": "en"},
    "cn": {"name": "China",          "lang": "zh"},
    "il": {"name": "Israel",         "lang": "en"},
    "br": {"name": "Brazil",         "lang": "pt"},
    "in": {"name": "India",          "lang": "en"},
    "za": {"name": "South Africa",   "lang": "en"},
    "de": {"name": "Germany",        "lang": "de"},
    "mx": {"name": "Mexico",         "lang": "es"},
}

# Low-quality aggregators to filter out
BLOCKLIST = {
    "menafn", "mixvale", "bignewsnetwork",
    "investing_za", "investing_in", "investing_uk",
    "investing_de", "investing_fr",
    "bitcoinworld", "news247plus",
    "ign_za", "ign_me", "ign_latam",  # gaming, not news
    "businesswire", "prnewswire_apac",  # press releases
}

def fetch_country_news(country_code, lang, size=8):
    """Fetch top headlines for a country, filtered by language."""
    url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": API_KEY,
        "country": country_code,
        "language": lang,
        "size": size,
        "prioritydomain": "top",  # prefer top-tier sources
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        articles = data.get("results", [])

        # Filter out blocklisted sources
        filtered = [a for a in articles if a.get("source_id") not in BLOCKLIST]
        return filtered

    except Exception as e:
        print(f"  Error fetching {country_code}: {e}")
        return []

def clean_article(article, country_code, country_name):
    """Extract only the fields we care about."""
    return {
        "country_code": country_code,
        "country_name": country_name,
        "title": article.get("title", ""),
        "description": article.get("description", ""),
        "source": article.get("source_id", ""),
        "url": article.get("link", ""),
        "published": article.get("pubDate", ""),
        "category": article.get("category", []),
        "fetched_at": datetime.utcnow().isoformat(),
    }

def main():
    os.makedirs("data", exist_ok=True)
    all_articles = {}
    total = 0

    print(f"\n{'='*50}")
    print(f"  World News Fetcher — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*50}\n")

    for code, meta in COUNTRIES.items():
        print(f"Fetching {meta['name']} ({code})...")
        articles = fetch_country_news(code, meta["lang"])
        cleaned = [clean_article(a, code, meta["name"]) for a in articles]
        all_articles[code] = cleaned
        total += len(cleaned)

        for a in cleaned:
            print(f"  ✓ [{a['source']}] {a['title'][:80]}")

        if not cleaned:
            print(f"  — No results")
        print()

    # Save to JSON
    output = {
        "fetched_at": datetime.utcnow().isoformat(),
        "total_articles": total,
        "countries": all_articles
    }
    path = f"data/news_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"  Done. {total} articles saved to {path}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()