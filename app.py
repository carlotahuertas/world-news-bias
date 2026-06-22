"""
World News Bias Analyzer — Premium editorial dashboard.
Run: python app.py → http://localhost:8050
"""
import os, sqlite3, json, webbrowser, threading
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, ALL, callback_context

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "news.db")

COUNTRY_ISO3  = {"us":"USA","gb":"GBR","es":"ESP","fr":"FRA","ru":"RUS","cn":"CHN",
                 "il":"ISR","br":"BRA","in":"IND","za":"ZAF","de":"DEU","mx":"MEX"}
COUNTRY_NAMES = {"us":"United States","gb":"United Kingdom","es":"Spain","fr":"France",
                 "ru":"Russia","cn":"China","il":"Israel","br":"Brazil","in":"India",
                 "za":"South Africa","de":"Germany","mx":"Mexico"}
COUNTRY_FLAGS = {"us":"🇺🇸","gb":"🇬🇧","es":"🇪🇸","fr":"🇫🇷","ru":"🇷🇺","cn":"🇨🇳",
                 "il":"🇮🇱","br":"🇧🇷","in":"🇮🇳","za":"🇿🇦","de":"🇩🇪","mx":"🇲🇽"}
ISO3_TO_CODE  = {v: k for k, v in COUNTRY_ISO3.items()}
LABEL_ORDER   = ["Left", "Lean-Left", "Center", "Lean-Right", "Right"]

# ── Design tokens ─────────────────────────────────────────────────────────────
BG    = "#0a0a0a"
SFC   = "#141414"
SFC2  = "#1c1c1c"
BD    = "rgba(255,255,255,0.06)"
BDB   = "rgba(255,255,255,0.12)"
T1    = "#f0ede8"
T2    = "#9a9990"
T3    = "#52524e"
GOLD  = "#c4b89a"
GOLDD = "rgba(196,184,154,0.12)"

BIAS_COLORS = {
    "Left":       "#e63946",
    "Lean-Left":  "#f4a261",
    "Center":     "#6b7280",
    "Lean-Right": "#457b9d",
    "Right":      "#1d3557",
}
BIAS_FG = {
    "Left": "#fff", "Lean-Left": "#200800",
    "Center": "#fff", "Lean-Right": "#fff", "Right": "#b8d4f0",
}

FF_SERIF = "'Playfair Display', Georgia, serif"
FF_SANS  = "'Inter', -apple-system, sans-serif"
FF_MONO  = "'DM Mono', 'Fira Code', monospace"

# ── Injected CSS ──────────────────────────────────────────────────────────────
_CSS = """
*, *::before, *::after { box-sizing: border-box; }
html, body { height: 100%; overflow: hidden; background: #0a0a0a; }
#react-entry-point, ._dash-loading-callback { height: 100%; }

::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.07); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.15); }

input::placeholder { color: #3a3a38; font-family: inherit; }
input:focus { outline: none; }
a { text-decoration: none; }
a:hover { opacity: 0.82; }

.story-card {
    cursor: pointer;
    transition: transform 0.14s ease, box-shadow 0.14s ease;
    animation: cardIn 0.2s ease both;
}
.story-card:hover { transform: translateY(-1px); box-shadow: 0 6px 28px rgba(0,0,0,0.6); }

.div-card {
    animation: cardIn 0.2s ease both;
    transition: transform 0.14s ease, box-shadow 0.14s ease;
}
.div-card:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(0,0,0,0.5); }

@keyframes cardIn {
    from { opacity: 0; transform: translateY(7px); }
    to   { opacity: 1; transform: translateY(0); }
}

.filter-pill { cursor: pointer; user-select: none; transition: opacity 0.12s; }
.filter-pill:hover { opacity: 0.75; }
"""

# ── DB ─────────────────────────────────────────────────────────────────────────

def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def load_stats():
    with _conn() as c:
        n_art     = c.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        n_clust   = c.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        n_cc      = c.execute("SELECT COUNT(DISTINCT country_code) FROM articles").fetchone()[0]
        n_labeled = c.execute(
            "SELECT COUNT(*) FROM articles WHERE allsides_rating IS NOT NULL"
        ).fetchone()[0]
    return n_art, n_clust, n_cc, n_labeled


def load_country_stats():
    with _conn() as c:
        rows = c.execute("""
            SELECT a.country_code,
                   COUNT(*) AS n_articles,
                   AVG(COALESCE(bs.sentiment_polarity,0)) AS avg_pol,
                   AVG(COALESCE(bs.sentiment_subjectivity,0)) AS avg_subj
            FROM articles a
            LEFT JOIN bias_scores bs ON bs.article_id = a.id
            GROUP BY a.country_code
        """).fetchall()
    return [dict(r) for r in rows]


def load_country_articles(cc, bias_filter=None, search=None):
    q = """
        SELECT a.id, a.title, a.description, a.source, a.published, a.url,
               COALESCE(bs.sentiment_polarity,0)     AS sentiment_polarity,
               COALESCE(bs.sentiment_subjectivity,0) AS sentiment_subjectivity,
               COALESCE(bs.emotional_word_count,0)   AS emotional_word_count,
               a.predicted_bias, a.bias_confidence, a.allsides_rating
        FROM articles a LEFT JOIN bias_scores bs ON bs.article_id = a.id
        WHERE a.country_code = ?
    """
    p = [cc]
    if bias_filter and bias_filter != "all":
        q += " AND a.predicted_bias = ?"
        p.append(bias_filter)
    if search and len(search.strip()) >= 2:
        q += " AND (LOWER(a.title) LIKE ? OR LOWER(a.description) LIKE ?)"
        s = f"%{search.lower().strip()}%"
        p += [s, s]
    q += " ORDER BY a.published DESC"
    with _conn() as c:
        return [dict(r) for r in c.execute(q, p).fetchall()]


def search_all_articles(query, limit=40):
    s = f"%{query.lower().strip()}%"
    with _conn() as c:
        rows = c.execute("""
            SELECT a.id, a.title, a.description, a.source, a.published, a.url,
                   a.country_code, a.country_name,
                   COALESCE(bs.sentiment_polarity,0)     AS sentiment_polarity,
                   COALESCE(bs.sentiment_subjectivity,0) AS sentiment_subjectivity,
                   COALESCE(bs.emotional_word_count,0)   AS emotional_word_count,
                   a.predicted_bias, a.bias_confidence, a.allsides_rating
            FROM articles a LEFT JOIN bias_scores bs ON bs.article_id = a.id
            WHERE LOWER(a.title) LIKE ? OR LOWER(a.description) LIKE ?
            ORDER BY a.published DESC LIMIT ?
        """, (s, s, limit)).fetchall()
    return [dict(r) for r in rows]


def load_clusters(n=10, min_countries=2):
    with _conn() as c:
        rows = c.execute("""
            SELECT c.id, c.label,
                   COUNT(DISTINCT cm.article_id)  AS n_articles,
                   COUNT(DISTINCT a.country_code) AS n_countries
            FROM clusters c
            JOIN cluster_members cm ON cm.cluster_id = c.id
            JOIN articles a ON a.id = cm.article_id
            GROUP BY c.id HAVING n_countries >= ?
            ORDER BY n_countries DESC, n_articles DESC LIMIT ?
        """, (min_countries, n)).fetchall()
    return [dict(r) for r in rows]


def load_cluster_articles(cluster_id):
    with _conn() as c:
        rows = c.execute("""
            SELECT a.id, a.country_code, a.country_name, a.title, a.source,
                   a.published, a.url,
                   COALESCE(bs.sentiment_polarity,0)     AS sentiment_polarity,
                   COALESCE(bs.sentiment_subjectivity,0) AS sentiment_subjectivity,
                   COALESCE(bs.emotional_word_count,0)   AS emotional_word_count,
                   cm.similarity, a.predicted_bias, a.bias_confidence, a.allsides_rating
            FROM cluster_members cm
            JOIN articles a ON a.id = cm.article_id
            LEFT JOIN bias_scores bs ON bs.article_id = a.id
            WHERE cm.cluster_id = ? ORDER BY a.country_code
        """, (cluster_id,)).fetchall()
    return [dict(r) for r in rows]


def get_article_cluster_peers(article_id):
    with _conn() as c:
        cm = c.execute(
            "SELECT cluster_id FROM cluster_members WHERE article_id = ?",
            (article_id,)
        ).fetchone()
        if not cm:
            return []
        rows = c.execute("""
            SELECT a.id, a.country_code, a.country_name, a.title, a.source,
                   a.published, a.url,
                   COALESCE(bs.sentiment_polarity,0) AS sentiment_polarity,
                   a.predicted_bias, a.bias_confidence, a.allsides_rating
            FROM cluster_members cm
            JOIN articles a ON a.id = cm.article_id
            LEFT JOIN bias_scores bs ON bs.article_id = a.id
            WHERE cm.cluster_id = ? AND cm.article_id != ?
            ORDER BY a.country_code LIMIT 4
        """, (cm[0], article_id)).fetchall()
    return [dict(r) for r in rows]


def get_divergence_clusters(n=3):
    clusters = load_clusters(n=20, min_countries=1)
    scored = []
    for c in clusters:
        arts = load_cluster_articles(c["id"])
        df = pd.DataFrame(arts)
        if df.empty:
            continue
        left_df  = df[df["predicted_bias"].isin(["Left", "Lean-Left"])]
        right_df = df[df["predicted_bias"].isin(["Lean-Right", "Right"])]
        if not left_df.empty and not right_df.empty:
            lp = float(left_df["sentiment_polarity"].mean())
            rp = float(right_df["sentiment_polarity"].mean())
            sp = abs(rp - lp)
        else:
            by_cc = df.groupby("country_code")["sentiment_polarity"].mean()
            lp = float(by_cc.min()) if len(by_cc) >= 1 else 0.0
            rp = float(by_cc.max()) if len(by_cc) >= 2 else 0.0
            sp = abs(rp - lp)
        scored.append({**c, "left_pol": lp, "right_pol": rp, "lr_spread": sp,
                       "left_n": len(left_df), "right_n": len(right_df)})
    scored.sort(key=lambda x: x["lr_spread"], reverse=True)
    return scored[:n]


# ── Visual helpers ─────────────────────────────────────────────────────────────

def sentiment_label(pol):
    if pol is None: return "—"
    if pol <= -0.3:  return "very negative"
    if pol <= -0.1:  return "negative"
    if pol <   0.1:  return "neutral"
    if pol <   0.3:  return "positive"
    return "very positive"


def sent_color(pol):
    if pol is None or abs(pol) < 0.05: return T3
    return "#df5c6a" if pol < 0 else "#4cad80"


def bias_pill(label, conf=None, gt=None):
    c = BIAS_COLORS.get(label, "#555")
    f = BIAS_FG.get(label, "#fff")
    suffix = " ✓" if gt else (f" {conf*100:.0f}%" if conf else "")
    return html.Span(f"{label}{suffix}", style={
        "display": "inline-block",
        "padding": "2px 9px",
        "borderRadius": "100px",
        "fontSize": "10px",
        "fontFamily": FF_MONO,
        "background": c,
        "color": f,
        "flexShrink": "0",
        "letterSpacing": "0.01em",
    })


def sent_bar(pol):
    pol = pol or 0.0
    pct = abs(pol) * 50
    color = sent_color(pol)
    left  = f"{50 - pct:.1f}%" if pol < 0 else "50%"
    width = f"{pct:.1f}%"
    return html.Div([
        html.Div(style={
            "position": "relative", "height": "3px",
            "background": "rgba(255,255,255,0.05)", "borderRadius": "2px",
        }, children=[
            html.Div(style={
                "position": "absolute", "left": "50%", "top": "0",
                "width": "1px", "height": "100%",
                "background": "rgba(255,255,255,0.1)",
            }),
            html.Div(style={
                "position": "absolute", "left": left, "width": width,
                "height": "100%", "background": color, "borderRadius": "1px",
            }),
        ]),
        html.Div(sentiment_label(pol), style={
            "fontSize": "10px", "color": color,
            "marginTop": "3px", "fontFamily": FF_MONO,
        }),
    ], style={"marginTop": "8px"})


def build_world_map(country_stats):
    df = pd.DataFrame(country_stats)
    df["iso3"] = df["country_code"].map(COUNTRY_ISO3)
    df["name"] = df["country_code"].map(COUNTRY_NAMES)
    df["flag"] = df["country_code"].map(COUNTRY_FLAGS)
    df["hover"] = df.apply(lambda r: (
        f"<b>{r['flag']} {r['name']}</b><br>"
        f"<b>{r['n_articles']}</b> articles<br>"
        f"avg sentiment: {r['avg_pol']:+.2f}"
    ), axis=1)
    max_n = max(df["n_articles"].max(), 1)
    fig = go.Figure(go.Choropleth(
        locations=df["iso3"],
        z=df["n_articles"],
        text=df["hover"],
        customdata=df["country_code"],
        hovertemplate="%{text}<extra></extra>",
        colorscale=[
            [0.00, "#0c0a07"],
            [0.20, "#1e1a10"],
            [0.50, "#5e4416"],
            [0.80, "#a8822e"],
            [1.00, "#c4b89a"],
        ],
        zmin=0, zmax=max_n, showscale=False,
        marker_line_color="rgba(255,255,255,0.04)",
        marker_line_width=0.5,
    ))
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,  coastlinecolor="#1c180f",
            showland=True,        landcolor="#0c0a07",
            showocean=True,       oceancolor="#070707",
            showlakes=False,
            showcountries=True,   countrycolor="#1a1710",
            projection_type="natural earth",
            bgcolor=BG,
        ),
        paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(l=0, r=0, t=0, b=0),
        height=None, autosize=True,
        clickmode="event+select", dragmode=False,
        font=dict(color=T2),
    )
    return fig


# ── Feed components ────────────────────────────────────────────────────────────

def make_story_card(art, expanded_id=None):
    aid   = art["id"]
    title = (art.get("title") or "Untitled")
    desc  = (art.get("description") or "")
    src   = (art.get("source") or "—").upper()
    date  = (art.get("published") or "")[:10]
    url   = art.get("url") or "#"
    pol   = art.get("sentiment_polarity") or 0.0
    bias  = art.get("predicted_bias")
    conf  = art.get("bias_confidence")
    gt    = art.get("allsides_rating")

    is_exp      = (expanded_id is not None and str(expanded_id) == str(aid))
    bdr_color   = BIAS_COLORS.get(bias, "rgba(255,255,255,0.07)")
    toggle_hint = "▴ collapse" if is_exp else "▾ how others covered this"

    header_row = html.Div([
        html.Span(src, style={
            "fontSize": "10px", "fontFamily": FF_MONO,
            "color": GOLD, "letterSpacing": "0.07em",
        }),
        html.Span(f" · {date}", style={
            "fontSize": "10px", "fontFamily": FF_MONO,
            "color": T3, "marginRight": "auto",
        }),
        bias_pill(bias, conf, gt) if bias else html.Span(),
    ], style={"display": "flex", "alignItems": "center",
              "gap": "4px", "marginBottom": "8px"})

    headline = html.Div(
        title[:122] + ("…" if len(title) > 122 else ""),
        style={
            "fontSize": "15px", "fontFamily": FF_SERIF,
            "fontWeight": "500", "color": T1,
            "lineHeight": "1.45", "marginBottom": "5px",
        }
    )

    snippet = html.Div(
        desc[:145] + ("…" if len(desc) > 145 else ""),
        style={
            "fontSize": "12px", "color": T2,
            "lineHeight": "1.6", "fontFamily": FF_SANS,
        }
    ) if desc else html.Span()

    hint = html.Div(toggle_hint, style={
        "fontSize": "10px", "color": T3,
        "marginTop": "7px", "fontFamily": FF_MONO,
        "letterSpacing": "0.04em",
    })

    collapsed_inner = html.Div(
        [header_row, headline, snippet, sent_bar(pol), hint],
        id={"type": "card-btn", "index": aid},
        n_clicks=0,
        style={"cursor": "pointer"},
    )

    expanded_inner = html.Span()
    if is_exp:
        peers = get_article_cluster_peers(aid)

        if gt:
            bias_expl = f"AllSides ground truth: {bias} ✓"
        elif conf:
            bias_expl = f"Classifier: {bias}  ({conf*100:.0f}% confidence)"
        else:
            bias_expl = f"Bias label: {bias or 'Unknown'}"

        framing_note = None
        if peers:
            df_all = pd.DataFrame([art] + peers)
            ldf = df_all[df_all["predicted_bias"].isin(["Left", "Lean-Left"])]
            rdf = df_all[df_all["predicted_bias"].isin(["Lean-Right", "Right"])]
            if not ldf.empty and not rdf.empty:
                lp2 = float(ldf["sentiment_polarity"].mean())
                rp2 = float(rdf["sentiment_polarity"].mean())
                if abs(rp2 - lp2) > 0.15:
                    framing_note = (
                        f"Left-leaning sources frame this story {sentiment_label(lp2)}. "
                        f"Right-leaning sources frame this {sentiment_label(rp2)}."
                    )

        peer_nodes = []
        for p in peers[:3]:
            flag  = COUNTRY_FLAGS.get(p["country_code"], "")
            cname = p.get("country_name") or p.get("country_code", "")
            ptit  = p.get("title") or ""
            pbias = p.get("predicted_bias")
            ppol  = p.get("sentiment_polarity") or 0.0
            peer_nodes.append(html.Div([
                html.Div([
                    html.Span(f"{flag} {cname}", style={
                        "fontSize": "11px", "color": T2,
                        "minWidth": "96px", "marginRight": "8px",
                    }),
                    bias_pill(pbias) if pbias else html.Span(),
                    html.Span(f" {sentiment_label(ppol)}", style={
                        "fontSize": "10px", "color": sent_color(ppol),
                        "marginLeft": "6px", "fontFamily": FF_MONO,
                    }),
                ], style={"display": "flex", "alignItems": "center",
                          "marginBottom": "3px"}),
                html.A(
                    ptit[:82] + ("…" if len(ptit) > 82 else ""),
                    href=p.get("url") or "#", target="_blank",
                    style={
                        "fontSize": "12px", "color": T2,
                        "fontStyle": "italic", "lineHeight": "1.4",
                        "display": "block",
                    }
                ),
            ], style={
                "borderLeft": f"2px solid {BIAS_COLORS.get(pbias, BD)}",
                "paddingLeft": "10px", "marginBottom": "10px",
            }))

        expanded_inner = html.Div([
            html.Div(style={"height": "1px",
                            "background": "rgba(255,255,255,0.06)",
                            "margin": "12px 0"}),

            html.Div(bias_expl, style={
                "fontSize": "11px", "fontFamily": FF_MONO,
                "color": BIAS_COLORS.get(bias, T3),
                "letterSpacing": "0.03em", "marginBottom": "12px",
            }),

            html.Div([
                html.Span("⚠  ", style={"color": "#f4a261"}),
                html.Span(framing_note, style={"color": T2}),
            ], style={
                "fontSize": "11px", "lineHeight": "1.5",
                "background": "rgba(244,162,97,0.07)",
                "border": "1px solid rgba(244,162,97,0.2)",
                "borderRadius": "4px", "padding": "8px 10px",
                "marginBottom": "12px",
            }) if framing_note else html.Span(),

            html.Div([
                html.Div("HOW OTHERS COVERED THIS", style={
                    "fontSize": "9px", "fontFamily": FF_MONO,
                    "color": T3, "letterSpacing": "0.13em", "marginBottom": "10px",
                }),
                *peer_nodes,
            ]) if peer_nodes else html.Div(
                "No cross-country coverage found for this story.",
                style={"fontSize": "11px", "color": T3, "fontFamily": FF_MONO}
            ),

            html.Div([
                html.A("Read original  ↗", href=url, target="_blank", style={
                    "fontSize": "11px", "color": GOLD, "fontFamily": FF_MONO,
                    "letterSpacing": "0.05em", "padding": "5px 12px",
                    "border": f"1px solid {GOLDD}",
                    "borderRadius": "4px", "display": "inline-block",
                }),
            ], style={"marginTop": "12px"}),
        ])

    return html.Div(
        [collapsed_inner, expanded_inner],
        className="story-card",
        style={
            "background": SFC,
            "borderLeft": f"3px solid {bdr_color}",
            "borderRadius": "0 6px 6px 0",
            "padding": "14px 16px",
            "marginBottom": "8px",
        },
    )


def make_divergence_card(c, rank):
    label  = c.get("label") or "Untitled story"
    lp     = c.get("left_pol")
    rp     = c.get("right_pol")
    spread = c.get("lr_spread", 0)
    n_art  = c.get("n_articles", 0)
    n_cc   = c.get("n_countries", 0)
    high   = spread > 0.25

    def _bar(pol, color):
        if pol is None:
            return html.Div(style={"flex": "1"})
        pct   = abs(pol) * 50
        left  = f"{50 - pct:.1f}%" if pol < 0 else "50%"
        width = f"{pct:.1f}%"
        return html.Div(style={
            "position": "relative", "height": "5px", "flex": "1",
            "background": "rgba(255,255,255,0.04)", "borderRadius": "3px",
        }, children=[
            html.Div(style={
                "position": "absolute", "left": "50%", "top": "0",
                "width": "1px", "height": "100%",
                "background": "rgba(255,255,255,0.1)",
            }),
            html.Div(style={
                "position": "absolute", "left": left, "width": width,
                "height": "100%", "background": color, "borderRadius": "3px",
            }),
        ])

    def _row(pol, color, side):
        if pol is None:
            return html.Div()
        return html.Div([
            html.Span(side, style={
                "fontSize": "9px", "fontFamily": FF_MONO,
                "color": color, "letterSpacing": "0.07em",
                "minWidth": "82px",
            }),
            _bar(pol, color),
            html.Span(f"{pol:+.2f}", style={
                "fontSize": "10px", "color": color,
                "fontFamily": FF_MONO, "minWidth": "36px",
                "textAlign": "right", "marginLeft": "8px",
            }),
            html.Span(sentiment_label(pol), style={
                "fontSize": "9px", "color": T3,
                "fontFamily": FF_MONO, "marginLeft": "8px",
                "minWidth": "68px",
            }),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"})

    return html.Div([
        html.Div([
            html.Span(f"#{rank}", style={
                "fontSize": "11px", "fontFamily": FF_MONO,
                "color": GOLD, "marginRight": "10px",
            }),
            html.Span(f"{n_cc} countries · {n_art} articles", style={
                "fontSize": "10px", "color": T3,
                "fontFamily": FF_MONO, "marginRight": "auto",
            }),
            html.Span("HIGH DIVERGENCE", style={
                "fontSize": "9px", "fontFamily": FF_MONO, "color": "#f4a261",
                "background": "rgba(244,162,97,0.1)",
                "border": "1px solid rgba(244,162,97,0.22)",
                "borderRadius": "3px", "padding": "1px 7px",
                "letterSpacing": "0.07em",
            }) if high else html.Span(f"spread {spread:.2f}", style={
                "fontSize": "10px", "color": T3, "fontFamily": FF_MONO,
            }),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),

        html.Div(
            label[:100] + ("…" if len(label) > 100 else ""),
            style={
                "fontSize": "14px", "fontFamily": FF_SERIF,
                "fontWeight": "500", "color": T1,
                "lineHeight": "1.45", "marginBottom": "12px",
            }
        ),

        _row(lp, BIAS_COLORS["Left"], "LEFT SAYS"),
        _row(rp, BIAS_COLORS["Lean-Right"], "RIGHT SAYS"),
    ], className="div-card", style={
        "background": SFC, "border": f"1px solid {BD}",
        "borderRadius": "6px", "padding": "14px 16px", "marginBottom": "8px",
    })


def render_filter_pills(active):
    pills = []
    for lbl, val in [("All", "all")] + [(l, l) for l in LABEL_ORDER]:
        on     = val == active
        color  = BIAS_COLORS.get(val, GOLD) if val != "all" else GOLD
        pills.append(html.Span(lbl, className="filter-pill",
            id={"type": "filter-pill", "index": val},
            n_clicks=0,
            style={
                "display": "inline-block",
                "padding": "4px 12px", "borderRadius": "100px",
                "fontSize": "10px", "fontFamily": FF_MONO,
                "letterSpacing": "0.04em",
                "border": f"1px solid {color if on else 'rgba(255,255,255,0.09)'}",
                "background": color if on else "transparent",
                "color": "#fff" if on else T3,
                "marginRight": "5px",
            }
        ))
    return pills


def _empty(msg):
    return html.Div(msg, style={
        "color": T2, "fontSize": "13px", "textAlign": "center",
        "marginTop": "64px", "fontFamily": FF_SANS,
    })


# ── App ────────────────────────────────────────────────────────────────────────

_GF = ("https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@"
       "0,500;0,600;0,700;1,500&family=Inter:wght@300;400;500&family=DM+Mono:"
       "wght@400;500&display=swap")

app = dash.Dash(
    __name__,
    external_stylesheets=[_GF],
    suppress_callback_exceptions=True,
)
app.title = "World News Bias Analyzer"

app.index_string = (
    "<!DOCTYPE html>\n<html>\n<head>\n{%metas%}\n<title>{%title%}</title>\n"
    "{%favicon%}\n{%css%}\n<style>" + _CSS + "</style>\n</head>\n<body>\n"
    "{%app_entry%}\n<footer>\n{%config%}\n{%scripts%}\n{%renderer%}\n"
    "</footer>\n</body>\n</html>"
)

_cs   = load_country_stats()
_stat = load_stats()
_mfig = build_world_map(_cs)
n_art, n_clust, n_cc, n_labeled = _stat

app.layout = html.Div([

    dcc.Store(id="selected-country",  data=None),
    dcc.Store(id="expanded-card",     data=None),
    dcc.Store(id="bias-filter-store", data="all"),
    dcc.Interval(id="clock-tick",     interval=1000, n_intervals=0),

    # ── Top bar ────────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("◈ ", style={"color": GOLD, "fontSize": "13px"}),
            html.Span("WORLD NEWS BIAS", style={
                "fontSize": "12px", "fontWeight": "600",
                "color": T1, "letterSpacing": "0.14em", "fontFamily": FF_MONO,
            }),
        ], style={"display": "flex", "alignItems": "center"}),

        html.Div([
            *[html.Div([
                html.Span(str(v), style={
                    "fontSize": "17px", "fontWeight": "600", "color": GOLD,
                    "display": "block", "lineHeight": "1.1", "fontFamily": FF_MONO,
                }),
                html.Span(lbl, style={
                    "fontSize": "8px", "color": T3, "letterSpacing": "0.1em",
                    "textTransform": "uppercase", "fontFamily": FF_MONO,
                }),
            ], style={
                "textAlign": "center", "padding": "0 16px",
                "borderLeft": f"1px solid {BD}",
            }) for v, lbl in [
                (n_art, "articles"), (n_cc, "countries"),
                (n_clust, "clusters"), (n_labeled, "labeled"),
            ]],
        ], style={"display": "flex", "alignItems": "center"}),

        html.Div(id="clock-display", style={
            "fontSize": "11px", "fontFamily": FF_MONO,
            "color": T3, "letterSpacing": "0.07em",
        }),
    ], style={
        "height": "52px", "background": SFC,
        "borderBottom": f"1px solid {BD}",
        "display": "flex", "alignItems": "center",
        "padding": "0 20px", "justifyContent": "space-between",
        "flexShrink": "0",
    }),

    # ── Main split ─────────────────────────────────────────────────────────────
    html.Div([

        # ── Map (55%) ──────────────────────────────────────────────────────────
        html.Div([
            dcc.Graph(
                id="world-map",
                figure=_mfig,
                responsive=True,
                config={"displayModeBar": False, "scrollZoom": False},
                style={"height": "100%", "width": "100%"},
            ),
        ], style={
            "width": "55%", "height": "100%",
            "borderRight": f"1px solid {BD}", "overflow": "hidden",
        }),

        # ── Feed panel (45%) ───────────────────────────────────────────────────
        html.Div([

            # Search bar
            html.Div([
                html.Span("⌕", style={
                    "position": "absolute", "left": "12px", "top": "50%",
                    "transform": "translateY(-50%)",
                    "color": T3, "fontSize": "16px", "pointerEvents": "none",
                }),
                dcc.Input(
                    id="search-input",
                    type="text",
                    placeholder="Search articles, topics, or countries…",
                    debounce=True,
                    style={
                        "width": "100%",
                        "background": "rgba(255,255,255,0.025)",
                        "color": T1,
                        "border": f"1px solid {BD}",
                        "borderRadius": "6px",
                        "padding": "9px 12px 9px 34px",
                        "fontSize": "12px",
                        "fontFamily": FF_SANS,
                    },
                ),
            ], style={"position": "relative", "margin": "14px 14px 10px 14px"}),

            # Filter pills
            html.Div(
                id="filter-pill-row",
                children=render_filter_pills("all"),
                style={
                    "padding": "0 14px 12px 14px",
                    "borderBottom": f"1px solid {BD}",
                    "display": "flex", "flexWrap": "wrap", "gap": "2px",
                },
            ),

            # Feed label
            html.Div(
                id="feed-label",
                style={
                    "padding": "9px 14px 4px 14px",
                    "fontSize": "9px", "fontFamily": FF_MONO,
                    "color": T3, "letterSpacing": "0.13em",
                    "textTransform": "uppercase",
                },
            ),

            # Feed scroll area
            html.Div(
                id="feed-content",
                style={
                    "flex": "1", "overflowY": "auto",
                    "padding": "0 14px 14px 14px", "minHeight": "0",
                },
            ),

        ], style={
            "width": "45%", "height": "100%",
            "display": "flex", "flexDirection": "column",
            "background": BG,
        }),

    ], style={
        "display": "flex", "flex": "1",
        "height": "calc(100vh - 52px)", "overflow": "hidden",
    }),

], style={
    "background": BG, "height": "100vh", "overflow": "hidden",
    "fontFamily": FF_SANS, "color": T1,
    "display": "flex", "flexDirection": "column",
})


# ═══ CALLBACKS ════════════════════════════════════════════════════════════════

@app.callback(
    Output("clock-display", "children"),
    Input("clock-tick", "n_intervals"),
)
def update_clock(_):
    return datetime.utcnow().strftime("%a %b %d %Y  %H:%M:%S UTC")


@app.callback(
    Output("selected-country", "data"),
    Input("world-map", "clickData"),
    State("selected-country", "data"),
)
def store_country(click_data, current):
    if not click_data:
        return current
    pt = click_data["points"][0]
    cc = pt.get("customdata") or ISO3_TO_CODE.get(pt.get("location", ""))
    return cc or current


@app.callback(
    Output("expanded-card", "data"),
    Input({"type": "card-btn", "index": ALL}, "n_clicks"),
    State("expanded-card", "data"),
)
def toggle_card(n_clicks_list, current):
    triggered = callback_context.triggered
    if not triggered or not any(n for n in n_clicks_list if n):
        return current
    prop_id = triggered[0]["prop_id"]
    try:
        id_dict = json.loads(prop_id.split(".")[0])
        card_id = id_dict["index"]
    except Exception:
        return current
    return None if str(current) == str(card_id) else card_id


@app.callback(
    Output("bias-filter-store", "data"),
    Output("filter-pill-row", "children"),
    Input({"type": "filter-pill", "index": ALL}, "n_clicks"),
    State("bias-filter-store", "data"),
)
def update_filter(n_clicks_list, current):
    triggered = callback_context.triggered
    if not triggered or not any(n for n in n_clicks_list if n):
        return current, render_filter_pills(current)
    prop_id = triggered[0]["prop_id"]
    try:
        id_dict = json.loads(prop_id.split(".")[0])
        new_val = id_dict["index"]
    except Exception:
        return current, render_filter_pills(current)
    return new_val, render_filter_pills(new_val)


@app.callback(
    Output("feed-content", "children"),
    Output("feed-label", "children"),
    Input("selected-country", "data"),
    Input("search-input", "value"),
    Input("bias-filter-store", "data"),
    Input("expanded-card", "data"),
)
def update_feed(country, search, bias_filter, expanded_id):
    search = (search or "").strip()
    has_search = len(search) >= 2

    if has_search:
        sq = search.lower()
        matched_cc = None
        for cc, cname in COUNTRY_NAMES.items():
            if sq == cname.lower() or sq == cc or sq in cname.lower():
                matched_cc = cc
                break
        if matched_cc:
            arts = load_country_articles(matched_cc, bias_filter)
            flag = COUNTRY_FLAGS.get(matched_cc, "")
            name = COUNTRY_NAMES.get(matched_cc, matched_cc)
            lbl  = f"{flag} {name}  ·  matched \"{search}\""
            if not arts:
                return [_empty(f"No articles for {name}.")], lbl
            return [make_story_card(a, expanded_id) for a in arts], lbl
        else:
            results = search_all_articles(search)
            lbl = f"search: \"{search}\"  ·  {len(results)} results"
            if not results:
                return [_empty(f"No results for \"{search}\".")], lbl
            return [make_story_card(a, expanded_id) for a in results], lbl

    if country:
        arts = load_country_articles(country, bias_filter)
        flag = COUNTRY_FLAGS.get(country, "")
        name = COUNTRY_NAMES.get(country, country)
        if not arts:
            return [_empty(f"No articles for {name}.")], f"{flag} {name}"
        avg_pol = sum(a["sentiment_polarity"] for a in arts) / len(arts)
        lbl = f"{flag} {name}  ·  {len(arts)} articles  ·  avg {avg_pol:+.2f}"
        if bias_filter and bias_filter != "all":
            lbl += f"  ·  {bias_filter} only"
        return [make_story_card(a, expanded_id) for a in arts], lbl

    # Default: Breaking Divergence
    divs = get_divergence_clusters(n=3)
    if not divs:
        return [_empty("No divergence data — run match_articles.py")], "Breaking Divergence"
    return [make_divergence_card(c, i + 1) for i, c in enumerate(divs)], "Breaking Divergence"


# ═══ ENTRY POINT ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    threading.Timer(1.2, lambda: webbrowser.open("http://localhost:8050")).start()
    app.run(debug=False, port=8050, host="0.0.0.0")
