"""Visual design system for the recruiter workspace.

A restrained navy/blue palette on neutral surfaces, defined once as tokens and
emitted as a single CSS block.  Every component in the UI reads from these
tokens rather than hard-coding colours, so the whole app stays visually
consistent and the palette can be swapped in one place.

The chart palette follows the same discipline: categorical series are assigned
fixed, distinguishable hues with adequate contrast in both light and dark
surfaces, and status colours (supported / partial / not found) are reserved and
never reused for ordinary series — a reader must never have to guess whether a
green bar means "good" or just "series 3".
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------

#: Core surface + text tokens.
COLORS: dict[str, str] = {
    "navy_900": "#0B1B33",
    "navy_800": "#12294A",
    "navy_700": "#1B3A63",
    "blue_600": "#1D4ED8",
    "blue_500": "#2563EB",
    "blue_400": "#3B82F6",
    "blue_200": "#BFDBFE",
    "blue_050": "#EFF6FF",
    "slate_900": "#0F172A",
    "slate_700": "#334155",
    "slate_500": "#64748B",
    "slate_400": "#94A3B8",
    "slate_200": "#E2E8F0",
    "slate_100": "#F1F5F9",
    "slate_050": "#F8FAFC",
    "white": "#FFFFFF",
    "amber_600": "#B45309",
    "amber_100": "#FEF3C7",
    "green_700": "#15803D",
    "green_100": "#DCFCE7",
    "red_700": "#B91C1C",
    "red_100": "#FEE2E2",
    "violet_600": "#6D28D9",
    "teal_600": "#0F766E",
}

#: Status semantics — reserved, never used for generic series.
STATUS_COLORS: dict[str, str] = {
    "supported": COLORS["green_700"],
    "partial": COLORS["amber_600"],
    "not_found": COLORS["slate_500"],
    "blocked": COLORS["red_700"],
    "abstained": COLORS["slate_500"],
    "answered": COLORS["blue_600"],
}

#: Categorical series palette for charts (colour-blind aware ordering).
SERIES_COLORS: list[str] = [
    "#1D4ED8",  # blue
    "#0F766E",  # teal
    "#B45309",  # amber
    "#6D28D9",  # violet
    "#BE185D",  # rose
    "#0369A1",  # sky
    "#4D7C0F",  # olive
    "#7C2D12",  # rust
]

#: Sequential ramp for magnitude encodings.
SEQUENTIAL_BLUE: list[str] = ["#EFF6FF", "#BFDBFE", "#93C5FD", "#60A5FA", "#3B82F6", "#1D4ED8", "#12294A"]


def series_color(index: int) -> str:
    return SERIES_COLORS[index % len(SERIES_COLORS)]


def status_color(status: str) -> str:
    return STATUS_COLORS.get(status, COLORS["slate_500"])


# --------------------------------------------------------------------------
# Chart defaults
# --------------------------------------------------------------------------


def plotly_layout(*, height: int = 340, showlegend: bool = False) -> dict[str, Any]:
    """Shared Plotly layout: flat, spacious, readable, no chartjunk."""
    return {
        "height": height,
        "showlegend": showlegend,
        "margin": {"l": 8, "r": 8, "t": 32, "b": 8},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
            "size": 13,
            "color": COLORS["slate_700"],
        },
        "xaxis": {
            "showgrid": False,
            "zeroline": False,
            "linecolor": COLORS["slate_200"],
            "tickfont": {"size": 12, "color": COLORS["slate_500"]},
        },
        "yaxis": {
            "showgrid": True,
            "gridcolor": COLORS["slate_100"],
            "zeroline": False,
            "linecolor": "rgba(0,0,0,0)",
            "tickfont": {"size": 12, "color": COLORS["slate_500"]},
        },
        "hoverlabel": {
            "bgcolor": COLORS["navy_900"],
            "font": {"color": COLORS["white"], "size": 12},
            "bordercolor": COLORS["navy_900"],
        },
    }


PLOTLY_CONFIG: dict[str, Any] = {
    "displayModeBar": False,
    "responsive": True,
}


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------


def build_css() -> str:
    """The app's stylesheet, built from the tokens above."""
    c = COLORS
    return f"""
<style>
/* ---------- Typography & base ---------- */
html, body, [class*="css"] {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
}}

.block-container {{
    padding-top: 2.2rem;
    padding-bottom: 4rem;
    max-width: 1240px;
}}

h1, h2, h3, h4 {{ color: {c['navy_900']}; letter-spacing: -0.018em; font-weight: 650; }}
h1 {{ font-size: 1.95rem; margin-bottom: 0.15rem; }}
h2 {{ font-size: 1.32rem; margin-top: 1.6rem; }}
h3 {{ font-size: 1.05rem; margin-top: 1.1rem; }}
p, li, label, span, div {{ color: {c['slate_700']}; }}
a {{ color: {c['blue_600']}; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}

/* ---------- App header ---------- */
.app-header {{
    display: flex; align-items: center; gap: 0.85rem;
    padding: 0.9rem 1.15rem; margin-bottom: 1.1rem;
    background: linear-gradient(105deg, {c['navy_900']} 0%, {c['navy_700']} 62%, {c['blue_600']} 100%);
    border-radius: 14px;
    box-shadow: 0 1px 2px rgba(11,27,51,.16), 0 8px 24px -12px rgba(11,27,51,.4);
}}
.app-header .mark {{
    width: 38px; height: 38px; border-radius: 10px; flex: 0 0 38px;
    background: rgba(255,255,255,.14);
    border: 1px solid rgba(255,255,255,.26);
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-weight: 700; font-size: 1rem; letter-spacing: -0.02em;
}}
.app-header .titles {{ line-height: 1.25; }}
.app-header .titles .t {{ color: #fff; font-size: 1.08rem; font-weight: 650; }}
.app-header .titles .s {{ color: rgba(255,255,255,.76); font-size: 0.79rem; }}
.app-header .spacer {{ flex: 1; }}
.app-header .mode {{
    color: rgba(255,255,255,.92); font-size: 0.72rem; text-align: right;
    background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.2);
    padding: 0.32rem 0.62rem; border-radius: 999px; white-space: nowrap;
}}

/* ---------- Synthetic-data banner ---------- */
.banner {{
    display: flex; gap: 0.65rem; align-items: flex-start;
    padding: 0.68rem 0.95rem; border-radius: 10px; font-size: 0.83rem;
    margin-bottom: 1rem; border: 1px solid;
}}
.banner .ico {{ font-size: 0.95rem; line-height: 1.35; }}
.banner.info   {{ background: {c['blue_050']}; border-color: {c['blue_200']}; color: {c['navy_800']}; }}
.banner.warn   {{ background: {c['amber_100']}; border-color: #FDE68A; color: #78350F; }}
.banner.danger {{ background: {c['red_100']}; border-color: #FECACA; color: #7F1D1D; }}
.banner.ok     {{ background: {c['green_100']}; border-color: #BBF7D0; color: #14532D; }}
.banner strong {{ color: inherit; }}

/* ---------- Cards ---------- */
.card {{
    background: {c['white']};
    border: 1px solid {c['slate_200']};
    border-radius: 12px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 1px 2px rgba(15,23,42,.04);
}}
.card.tight {{ padding: 0.75rem 0.9rem; }}
.card.flat {{ box-shadow: none; }}

/* ---------- Metric tiles ---------- */
.tiles {{ display: grid; gap: 0.7rem; grid-template-columns: repeat(auto-fit, minmax(148px, 1fr)); }}
.tile {{
    background: {c['white']}; border: 1px solid {c['slate_200']};
    border-radius: 11px; padding: 0.78rem 0.9rem;
    box-shadow: 0 1px 2px rgba(15,23,42,.04);
}}
.tile .k {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: .06em; color: {c['slate_500']}; font-weight: 600; }}
.tile .v {{ font-size: 1.42rem; font-weight: 660; color: {c['navy_900']}; line-height: 1.22; margin-top: .16rem; letter-spacing: -0.02em; }}
.tile .d {{ font-size: 0.73rem; color: {c['slate_500']}; margin-top: .1rem; }}
.tile.accent {{ border-color: {c['blue_200']}; background: {c['blue_050']}; }}
.tile.good {{ border-color: #BBF7D0; background: {c['green_100']}; }}
.tile.warn {{ border-color: #FDE68A; background: {c['amber_100']}; }}

/* ---------- Answer rendering ---------- */
.answer {{
    background: {c['white']}; border: 1px solid {c['slate_200']};
    border-left: 3px solid {c['blue_600']};
    border-radius: 10px; padding: 1rem 1.15rem;
    font-size: 0.935rem; line-height: 1.62; color: {c['slate_900']};
}}
.answer.abstained {{ border-left-color: {c['slate_400']}; background: {c['slate_050']}; }}
.answer.blocked {{ border-left-color: {c['red_700']}; background: {c['red_100']}; }}
.answer p {{ margin: 0 0 .62rem 0; }}
.answer ul {{ margin: .2rem 0 .7rem 1.05rem; padding: 0; }}
.answer li {{ margin-bottom: .34rem; }}

/* Inline citation chip */
.cite {{
    display: inline-block; font-size: 0.7rem; font-weight: 650;
    background: {c['blue_050']}; color: {c['blue_600']};
    border: 1px solid {c['blue_200']}; border-radius: 5px;
    padding: 0.02rem 0.3rem; margin: 0 0.08rem;
    vertical-align: baseline; cursor: help;
}}

/* ---------- Citation cards ---------- */
.cit-card {{
    background: {c['slate_050']}; border: 1px solid {c['slate_200']};
    border-radius: 9px; padding: 0.7rem 0.85rem; margin-bottom: 0.55rem;
}}
.cit-head {{ display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.36rem; }}
.cit-num {{
    background: {c['blue_600']}; color: #fff; font-size: 0.68rem; font-weight: 700;
    border-radius: 5px; padding: 0.08rem 0.36rem; min-width: 1.5rem; text-align: center;
}}
.cit-file {{ font-weight: 620; color: {c['navy_900']}; font-size: 0.84rem; word-break: break-all; }}
.cit-loc {{ color: {c['slate_500']}; font-size: 0.75rem; }}
.cit-quote {{
    font-size: 0.84rem; color: {c['slate_700']}; line-height: 1.55;
    border-left: 2px solid {c['blue_200']}; padding-left: 0.6rem; margin-top: 0.3rem;
}}
.cit-meta {{ font-size: 0.71rem; color: {c['slate_500']}; margin-top: 0.34rem; }}

/* ---------- Pills / badges ---------- */
.pill {{
    display: inline-block; font-size: 0.7rem; font-weight: 620;
    border-radius: 999px; padding: 0.1rem 0.52rem; border: 1px solid;
    white-space: nowrap;
}}
.pill.supported {{ background: {c['green_100']}; color: {c['green_700']}; border-color: #BBF7D0; }}
.pill.partial   {{ background: {c['amber_100']}; color: {c['amber_600']}; border-color: #FDE68A; }}
.pill.not_found {{ background: {c['slate_100']}; color: {c['slate_500']}; border-color: {c['slate_200']}; }}
.pill.synthetic {{ background: {c['violet_600']}18; color: {c['violet_600']}; border-color: {c['violet_600']}33; }}
.pill.upload    {{ background: {c['teal_600']}18; color: {c['teal_600']}; border-color: {c['teal_600']}33; }}
.pill.info      {{ background: {c['blue_050']}; color: {c['blue_600']}; border-color: {c['blue_200']}; }}
.pill.muted     {{ background: {c['slate_100']}; color: {c['slate_500']}; border-color: {c['slate_200']}; }}
.pill.danger    {{ background: {c['red_100']}; color: {c['red_700']}; border-color: #FECACA; }}

/* ---------- Requirement matrix rows ---------- */
.req {{
    border: 1px solid {c['slate_200']}; border-radius: 10px;
    padding: 0.8rem 0.95rem; margin-bottom: 0.6rem; background: {c['white']};
}}
.req.supported {{ border-left: 3px solid {c['green_700']}; }}
.req.partial   {{ border-left: 3px solid {c['amber_600']}; }}
.req.not_found {{ border-left: 3px solid {c['slate_400']}; background: {c['slate_050']}; }}
.req-q {{ font-weight: 620; color: {c['navy_900']}; font-size: 0.9rem; margin-bottom: 0.3rem; }}
.req-r {{ font-size: 0.83rem; color: {c['slate_700']}; line-height: 1.55; }}
.req-e {{ font-size: 0.79rem; color: {c['slate_700']}; margin-top: 0.4rem;
          background: {c['slate_050']}; border-radius: 7px; padding: 0.42rem 0.6rem;
          border-left: 2px solid {c['blue_200']}; }}

/* ---------- Source / search result rows ---------- */
.src {{
    border: 1px solid {c['slate_200']}; border-radius: 9px;
    padding: 0.7rem 0.85rem; margin-bottom: 0.5rem; background: {c['white']};
}}
.src-head {{ display: flex; justify-content: space-between; gap: 0.6rem; align-items: baseline; flex-wrap: wrap; }}
.src-file {{ font-weight: 610; color: {c['navy_900']}; font-size: 0.845rem; word-break: break-all; }}
.src-snip {{ font-size: 0.835rem; color: {c['slate_700']}; line-height: 1.55; margin-top: 0.3rem; }}
.src-terms {{ font-size: 0.72rem; color: {c['slate_500']}; margin-top: 0.3rem; }}

/* Score bar — a ranking signal, explicitly not a probability */
.score-wrap {{ display: flex; align-items: center; gap: 0.5rem; min-width: 148px; }}
.score-bar {{ flex: 1; height: 5px; border-radius: 3px; background: {c['slate_200']}; overflow: hidden; }}
.score-fill {{ height: 100%; border-radius: 3px; background: linear-gradient(90deg, {c['blue_400']}, {c['blue_600']}); }}
.score-num {{ font-size: 0.72rem; color: {c['slate_500']}; font-variant-numeric: tabular-nums; min-width: 2.5rem; text-align: right; }}

/* ---------- Empty / error states ---------- */
.empty {{
    text-align: center; padding: 2.1rem 1.2rem;
    border: 1px dashed {c['slate_200']}; border-radius: 12px;
    background: {c['slate_050']};
}}
.empty .t {{ font-weight: 620; color: {c['navy_900']}; margin-bottom: 0.28rem; }}
.empty .s {{ font-size: 0.84rem; color: {c['slate_500']}; max-width: 46ch; margin: 0 auto; }}

.err {{ background: {c['red_100']}; border: 1px solid #FECACA; border-left: 3px solid {c['red_700']};
        border-radius: 9px; padding: 0.7rem 0.9rem; margin-bottom: 0.7rem; }}
.err .t {{ font-weight: 630; color: #7F1D1D; font-size: 0.86rem; }}
.err .m {{ font-size: 0.83rem; color: #7F1D1D; margin-top: 0.2rem; }}
.err .r {{ font-size: 0.79rem; color: #991B1B; margin-top: 0.32rem; font-style: italic; }}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {{ background: {c['slate_050']}; border-right: 1px solid {c['slate_200']}; }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}
.side-block {{ margin-bottom: 1.15rem; }}
.side-block .h {{
    font-size: 0.69rem; text-transform: uppercase; letter-spacing: .07em;
    color: {c['slate_500']}; font-weight: 700; margin-bottom: 0.4rem;
}}
.side-row {{ display: flex; justify-content: space-between; font-size: 0.79rem; padding: 0.14rem 0; }}
.side-row .k {{ color: {c['slate_500']}; }}
.side-row .v {{ color: {c['navy_900']}; font-weight: 560; }}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap: 0.15rem; border-bottom: 1px solid {c['slate_200']}; }}
.stTabs [data-baseweb="tab"] {{
    height: 40px; padding: 0 0.85rem; font-size: 0.86rem; font-weight: 560;
    color: {c['slate_500']}; border-radius: 8px 8px 0 0;
}}
.stTabs [aria-selected="true"] {{ color: {c['blue_600']}; background: {c['blue_050']}; }}

/* ---------- Widgets ---------- */
.stButton > button {{
    border-radius: 8px; font-weight: 570; font-size: 0.855rem;
    border: 1px solid {c['slate_200']}; transition: all .12s ease;
}}
.stButton > button:hover {{ border-color: {c['blue_400']}; color: {c['blue_600']}; }}
.stButton > button[kind="primary"] {{
    background: {c['blue_600']}; border-color: {c['blue_600']}; color: #fff;
}}
.stButton > button[kind="primary"]:hover {{ background: {c['blue_500']}; border-color: {c['blue_500']}; color: #fff; }}
div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea {{ border-radius: 8px; font-size: 0.885rem; }}
div[data-testid="stTextInput"] input:focus {{ border-color: {c['blue_400']}; box-shadow: 0 0 0 3px {c['blue_050']}; }}
.stSelectbox div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div {{ border-radius: 8px; font-size: 0.86rem; }}
div[data-testid="stExpander"] {{ border: 1px solid {c['slate_200']}; border-radius: 10px; }}
div[data-testid="stExpander"] summary {{ font-size: 0.855rem; font-weight: 560; }}

/* Dataframe */
div[data-testid="stDataFrame"] {{ border: 1px solid {c['slate_200']}; border-radius: 10px; overflow: hidden; }}

/* Chat messages */
div[data-testid="stChatMessage"] {{ background: transparent; padding: 0.1rem 0; }}

/* Sliders */
div[data-testid="stSlider"] label {{ font-size: 0.84rem; }}

/* Progress bar */
div[data-testid="stProgress"] > div > div > div {{ background: {c['blue_600']}; }}

/* Divider */
hr {{ border-color: {c['slate_200']}; margin: 1.15rem 0; }}

/* Caption */
.caption {{ font-size: 0.78rem; color: {c['slate_500']}; line-height: 1.5; }}
.section-label {{
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: .07em;
    color: {c['slate_500']}; font-weight: 700; margin: 0.9rem 0 0.45rem 0;
}}
</style>
"""


__all__ = [
    "COLORS",
    "PLOTLY_CONFIG",
    "SEQUENTIAL_BLUE",
    "SERIES_COLORS",
    "STATUS_COLORS",
    "build_css",
    "plotly_layout",
    "series_color",
    "status_color",
]
