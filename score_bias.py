"""
Step 3: NLP bias scoring for every article.
- TextBlob: sentiment polarity & subjectivity
- spaCy: named entity extraction
- Emotional word frequency (hand-crafted lexicon)

Run: python score_bias.py
"""
import sqlite3
import json
import re
from datetime import datetime

DB_PATH = "news.db"

# Curated emotional/charged language lexicon
EMOTIONAL_WORDS = {
    # strongly negative
    "attack", "attacks", "attacked", "attacking",
    "kill", "kills", "killed", "killing",
    "bomb", "bombs", "bombed", "bombing",
    "terror", "terrorist", "terrorism",
    "war", "wars", "warfare",
    "crisis", "crises",
    "disaster", "catastrophe", "catastrophic",
    "dead", "death", "deaths", "deadly",
    "brutal", "brutality",
    "violence", "violent",
    "threat", "threats", "threatening",
    "danger", "dangerous",
    "illegal", "illegally",
    "extremist", "extremism", "radical",
    "propaganda",
    "corrupt", "corruption",
    "scandal",
    "chaos", "chaotic",
    "destroy", "destroyed", "destruction",
    "collapse", "collapsing",
    "panic", "panicking",
    "alarming", "alarmed",
    "outrage", "outraged", "outrageous",
    "horrific", "horrifying", "horrified",
    "tragic", "tragedy",
    "flee", "fleeing", "fled",
    "slaughter",
    "massacre",
    "invasion", "invade", "invaded",
    # strongly positive (can also indicate framing bias)
    "triumph", "triumphant",
    "victory", "victorious",
    "hero", "heroic", "heroes",
    "liberation", "liberate", "liberated",
    "historic", "historical",
    "breakthrough",
    "miracle",
    # emotive / loaded
    "regime", "dictator", "dictatorship",
    "oppression", "oppressive",
    "freedom", "liberty",
    "innocent", "victims",
    "refugees", "displaced",
    "occupation", "occupied",
    "resistance",
    "provocation",
    "aggression", "aggressive",
    "hostage", "hostages",
    "rebel", "rebels", "rebellion",
    "coup",
    "blockade",
    "sanctions",
    "retaliation",
    "escalation", "escalate",
    "ceasefire",
    "genocide",
    "atrocity", "atrocities",
}


def tokenize(text):
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def emotional_score(text):
    tokens = tokenize(text)
    if not tokens:
        return 0, 0.0
    hits = sum(1 for t in tokens if t in EMOTIONAL_WORDS)
    return hits, hits / len(tokens)


def score_article(article_id, title, description, nlp, blob_cls):
    text = title
    if description:
        text += " " + description

    # TextBlob sentiment (English-only; non-English will get 0.0)
    try:
        blob = blob_cls(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
    except Exception:
        polarity, subjectivity = 0.0, 0.0

    # spaCy entities (English model — skip if text too short)
    entities = []
    try:
        doc = nlp(text[:1000])
        entities = [{"text": ent.text, "label": ent.label_}
                    for ent in doc.ents
                    if ent.label_ in ("PERSON", "ORG", "GPE", "NORP", "EVENT", "LAW")]
    except Exception:
        pass

    emo_count, emo_ratio = emotional_score(text)

    return {
        "article_id": article_id,
        "sentiment_polarity": round(polarity, 4),
        "sentiment_subjectivity": round(subjectivity, 4),
        "emotional_word_count": emo_count,
        "emotional_word_ratio": round(emo_ratio, 4),
        "entity_count": len(entities),
        "entities_json": json.dumps(entities),
        "scored_at": datetime.utcnow().isoformat(),
    }


def main():
    print(f"\n{'='*50}")
    print(f"  Bias Scoring")
    print(f"{'='*50}\n")

    import spacy
    from textblob import TextBlob

    print("  Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    conn = sqlite3.connect(DB_PATH)

    articles = conn.execute(
        "SELECT id, title, description FROM articles ORDER BY id"
    ).fetchall()
    print(f"  Scoring {len(articles)} articles...\n")

    inserted = 0
    for article_id, title, description in articles:
        s = score_article(article_id, title or "", description or "", nlp, TextBlob)
        conn.execute("""
            INSERT OR REPLACE INTO bias_scores
                (article_id, sentiment_polarity, sentiment_subjectivity,
                 emotional_word_count, emotional_word_ratio,
                 entity_count, entities_json, scored_at)
            VALUES (:article_id, :sentiment_polarity, :sentiment_subjectivity,
                    :emotional_word_count, :emotional_word_ratio,
                    :entity_count, :entities_json, :scored_at)
        """, s)
        inserted += 1

    conn.commit()

    # Quick sanity check
    sample = conn.execute("""
        SELECT a.country_code, a.title, bs.sentiment_polarity,
               bs.sentiment_subjectivity, bs.emotional_word_count, bs.entity_count
        FROM bias_scores bs
        JOIN articles a ON a.id = bs.article_id
        WHERE bs.emotional_word_count > 0
        ORDER BY bs.emotional_word_count DESC
        LIMIT 8
    """).fetchall()

    print("  Most emotionally loaded articles:")
    for cc, title, pol, subj, emo, ents in sample:
        print(f"    [{cc}] emo={emo:2d} pol={pol:+.2f} subj={subj:.2f}  {title[:70]}")

    conn.close()

    print(f"\n{'='*50}")
    print(f"  Done. {inserted} articles scored in {DB_PATH}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
