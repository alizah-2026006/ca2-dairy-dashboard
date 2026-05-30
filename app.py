#  Imports and environment setup

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


# Project paths
# app.py is stored at the main CA-2 folder level.
# The deployed app reads from CSV files inside data/processed.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"


# Create Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=False
)

# Required for deployment with: gunicorn app:server
server = app.server

# Hide Plotly's modebar for a cleaner farmer-facing dashboard.
graph_config = {"displayModeBar": False}


# Centralising colours and styles to prevents repeated hard-coded styling for all sliders.
IRELAND_COLOR = "#1f7a5c"
PEER_COLOR = "#6b8ba4"
POSITIVE_COLOR = "#1f7a5c"
NEGATIVE_COLOR = "#c44e52"
NEUTRAL_COLOR = "#b8a26c"
SOFT_GREY = "#d9dee3"
DARK_TEXT = "#001f3f"
MUTED_TEXT = "#4f5b62"
SOFT_BORDER = "#e1e5e8"
LIGHT_GREEN = "#eef7f3"
LIGHT_BG = "#fbfcfb"

COUNTRY_COLOR_MAP = {
    "Ireland": IRELAND_COLOR,
    "France": "#4e79a7",
    "Germany": "#9c755f",
    "Italy": "#c44e52",
    "Netherlands": "#f2b447",
    "Poland": "#8172b3"
}

SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]
SENTIMENT_COLOR_MAP = {
    "Negative": NEGATIVE_COLOR,
    "Neutral": NEUTRAL_COLOR,
    "Positive": POSITIVE_COLOR
}

PAGE_STYLE = {"maxWidth": "1320px"}

CHART_HEADING_STYLE = {
    "fontWeight": "600",
    "fontSize": "1.25rem",
    "color": DARK_TEXT,
    "marginBottom": "1rem"
}

MAIN_HEADING_STYLE = {
    "fontWeight": "700",
    "fontSize": "1.65rem",
    "color": DARK_TEXT,
    "textAlign": "center",
    "marginTop": "1.3rem",
    "marginBottom": "1.6rem"
}

ACTION_MAIN_HEADING_STYLE = {
    "fontWeight": "600",
    "fontSize": "1.25rem",
    "color": DARK_TEXT,
    "textAlign": "center",
    "marginTop": "1.2rem",
    "marginBottom": "1.4rem"
}

ACTION_CARD_TITLE_STYLE = {
    "fontWeight": "600",
    "fontSize": "1.05rem",
    "lineHeight": "1.3",
    "color": DARK_TEXT,
    "marginBottom": "0.45rem"
}

SECTION_HEADING_STYLE = {
    "fontWeight": "650",
    "fontSize": "1.18rem",
    "color": DARK_TEXT,
    "marginBottom": "0.8rem"
}

CARD_VALUE_STYLE = {
    "fontWeight": "600",
    "fontSize": "1.35rem",
    "lineHeight": "1.25",
    "color": DARK_TEXT
}

CARD_TITLE_STYLE = {
    "fontWeight": "650",
    "fontSize": "1.12rem",
    "lineHeight": "1.25",
    "color": DARK_TEXT,
    "marginBottom": "0.45rem"
}

CARD_LABEL_STYLE = {
    "fontSize": "0.9rem",
    "color": "#263238",
    "marginBottom": "0"
}

CARD_TEXT_STYLE = {
    "fontSize": "0.9rem",
    "color": MUTED_TEXT,
    "lineHeight": "1.45",
    "marginBottom": "0"
}

NOTE_LABEL_STYLE = {
    "display": "block",
    "fontSize": "0.75rem",
    "fontWeight": "700",
    "letterSpacing": "0.04em",
    "textTransform": "uppercase",
    "color": IRELAND_COLOR,
    "marginBottom": "4px"
}

TUFTE_NOTE_STYLE = {
    "backgroundColor": LIGHT_BG,
    "border": f"1px solid {SOFT_BORDER}",
    "borderLeft": f"4px solid {IRELAND_COLOR}",
    "borderRadius": "6px",
    "padding": "14px 16px",
    "fontSize": "0.95rem",
    "lineHeight": "1.6",
    "color": "#263238",
    "boxShadow": "0 1px 2px rgba(0, 0, 0, 0.03)"
}

PANEL_LABEL_STYLE = {
    "display": "block",
    "fontSize": "0.72rem",
    "fontWeight": "700",
    "letterSpacing": "0.04em",
    "textTransform": "uppercase",
    "color": IRELAND_COLOR,
    "marginBottom": "0.5rem"
}

BADGE_STYLE = {
    "display": "inline-block",
    "backgroundColor": LIGHT_GREEN,
    "color": IRELAND_COLOR,
    "fontSize": "0.72rem",
    "fontWeight": "700",
    "letterSpacing": "0.04em",
    "textTransform": "uppercase",
    "padding": "5px 10px",
    "borderRadius": "999px",
    "marginBottom": "0.8rem"
}

LIMITATION_TEXT_STYLE = {
    "fontSize": "0.84rem",
    "color": "#6c757d",
    "marginTop": "1rem",
    "lineHeight": "1.5"
}

# Create functions for each tasks to avaid repetitions
def tufte_note(label, text):
    """Create a compact evidence note following Tufte-style minimal explanation."""
    return html.Div(
        [html.Span(label, style=NOTE_LABEL_STYLE), html.Span(text)],
        style=TUFTE_NOTE_STYLE,
        className="mt-3"
    )


def limitation_note(text):
    """Create a small italic limitation note for responsible interpretation."""
    return html.P(html.Em(text), style=LIMITATION_TEXT_STYLE)


def metric_card(value, label):
    """Create a reusable KPI card."""
    return dbc.Card(
        dbc.CardBody([
            html.H4(value, style=CARD_VALUE_STYLE),
            html.P(label, style=CARD_LABEL_STYLE)
        ]),
        className="border-0 shadow-sm h-100"
    )


def evidence_card(title, text):
    """Create a short evidence card for the action-plan tab."""
    return dbc.Card(
        dbc.CardBody([
            html.H4(title, style=ACTION_CARD_TITLE_STYLE),
            html.P(text, style=CARD_TEXT_STYLE)
        ]),
        className="border-0 shadow-sm h-100"
    )


def bullet_list(items):
    """Create a readable bullet list for recommendation panels."""
    return html.Ul(
        [html.Li(item) for item in items],
        style={
            "fontSize": "0.92rem",
            "color": MUTED_TEXT,
            "lineHeight": "1.6",
            "paddingLeft": "1.15rem",
            "marginBottom": "0"
        }
    )


def info_box(label, items):
    """Create one column of the farmer recommendation panel."""
    return dbc.Card(
        dbc.CardBody([
            html.Span(label, style=PANEL_LABEL_STYLE),
            bullet_list(items)
        ]),
        className="border-0 h-100",
        style={
            "backgroundColor": "#ffffff",
            "border": f"1px solid {SOFT_BORDER}",
            "borderRadius": "8px"
        }
    )


def make_empty_figure(message):
    """Return a blank Plotly figure with a clear message when data is unavailable."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        height=430,
        margin=dict(t=30, b=40, l=40, r=40),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14, color="#6c757d")
            )
        ]
    )
    return fig


def apply_clean_layout(fig, height=430, margin=None, hovermode=None, showlegend=False):
    """Apply a consistent visual style to Plotly figures."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=margin or dict(t=20, b=40, l=60, r=40),
        hovermode=hovermode,
        showlegend=showlegend,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=13, color=DARK_TEXT),
        xaxis=dict(
            showline=True,
            linecolor=SOFT_GREY,
            gridcolor="#edf2f5",
            zeroline=False
        ),
        yaxis=dict(
            showline=True,
            linecolor=SOFT_GREY,
            gridcolor="#edf2f5",
            zeroline=False
        )
    )
    return fig

# Some more generally used functions with doc string

def clean_feature_name(feature):
    """Convert model/database feature names into farmer-readable labels."""
    feature_labels = {
        "lag_export_value": "Previous export value",
        "output_per_labour": "Output per labour",
        "milk_per_cow": "Milk per cow",
        "milk_prod_1000t": "Milk production",
        "butter_prod_1000t": "Butter production",
        "cheese_prod_1000t": "Cheese production",
        "milk_powder_prod_1000t": "Milk powder production",
        "milk_delivered_to_dairies_1000t": "Milk delivered to dairies",
        "dairy_cow_share": "Dairy cow share",
        "dairy_cow_pop_1000": "Dairy cow population",
        "agri_output_basic_price_million_eur": "Agricultural output",
        "cap_direct_payments_million_eur": "CAP direct payments",
        "milk_prod_growth_pct": "Milk production growth",
        "export_growth_pct": "Export growth"
    }
    return feature_labels.get(feature, str(feature).replace("_", " ").title())


def breakdown_label(value):
    """Create a readable label for sentiment breakdown columns."""
    return {"country": "Country", "perspective": "Perspective"}.get(
        value,
        str(value).replace("_", " ").title()
    )


def ordinal(number):
    """Return ordinal text such as 1st, 2nd, 3rd."""
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def safe_pct_change(new_value, old_value):
    """Safely calculate percentage change while avoiding divide-by-zero issues."""
    if pd.isna(new_value) or pd.isna(old_value) or old_value == 0:
        return pd.NA
    return ((new_value - old_value) / old_value) * 100


def format_growth(value):
    """Format a percentage growth value for cards and recommendation text."""
    if pd.isna(value):
        return "not available"
    return f"{value:.1f}%"

# Load data for all four sliders

def load_processed_csv(file_name):
    """
    The deployed dashboard reads CSV files instead of MySQL because
    Render cannot access a local MySQL database running on my laptop.
    """
    file_path = DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find '{file_name}'. Expected location: {file_path}"
        )

    return pd.read_csv(file_path)


# Load the final processed datasets used by the dashboard.
predict_df = load_processed_csv("predict_df.csv")
sentiment_df = load_processed_csv("sentiment_df.csv")
ml_feature_importance = load_processed_csv("ml_feature_importance.csv")


# Standardise one known column-name issue before creating dashboard datasets.
predict_df = predict_df.rename(columns={"milk_delivered to dairies_1000t": "milk_delivered_to_dairies_1000t"})

# Ireland export trend data.
ireland_df = (predict_df[predict_df["country"] == "Ireland"][["year", "total_dairy_export_value_1000USD"]].sort_values("year").copy())

# Ireland product-level export data.
product_export_columns = {
    "Raw milk of cattle_Export value_1000 USD": "milk_export_value_1000USD",
    "Butter of cow milk_Export value_1000 USD": "butter_export_value_1000USD",
    "Cheese from whole cow milk_Export value_1000 USD": "cheese_export_value_1000USD",
    "Whole milk powder_Export value_1000 USD": "milk_powder_export_value_1000USD"
}

required_product_columns = ["year"] + list(product_export_columns.keys())
missing_product_columns = [column for column in required_product_columnsif column not in predict_df.columns]

if missing_product_columns:
    raise KeyError(
        "The following product export columns are missing from predict_df.csv: "
        f"{missing_product_columns}"
    )

product_export_df = (
    predict_df[predict_df["country"] == "Ireland"]
    [required_product_columns]
    .rename(columns=product_export_columns)
    .sort_values("year")
    .copy()
)


# Wider EU comparison data for benchmarking and policy/output evidence.
comparison_countries = ["Ireland","France","Germany","Italy","Netherlands","Poland"]
comparison_df = (
    predict_df[predict_df["country"].isin(comparison_countries)]
    .sort_values(["country", "year"])
    .copy()
)

# Clean and prepare the data
# Some SQL column names contain spaces. Rename once so the rest of the code wil be easier to maintain.
comparison_df = comparison_df.rename(
    columns={"milk_delivered to dairies_1000t": "milk_delivered_to_dairies_1000t"}
)

# Convert key numeric columns safely. This prevents chart/callback errors caused by string values.
comparison_numeric_columns = [
    "year",
    "milk_prod_1000t",
    "butter_prod_1000t",
    "cheese_prod_1000t",
    "milk_powder_prod_1000t",
    "milk_delivered_to_dairies_1000t",
    "milk_per_cow",
    "dairy_cow_pop_1000",
    "output_per_labour",
    "agri_output_basic_price_million_eur",
    "dairy_cow_share",
    "cap_direct_payments_million_eur",
    "milk_prod_growth_pct",
    "export_growth_pct"
]

for column in comparison_numeric_columns:
    if column in comparison_df.columns:
        comparison_df[column] = pd.to_numeric(comparison_df[column], errors="coerce")

comparison_df = comparison_df.dropna(subset=["country", "year"]).copy()
comparison_df["year"] = comparison_df["year"].astype(int)

# Prepare Irish export trend.
ireland_df["year"] = pd.to_numeric(ireland_df["year"], errors="coerce")
ireland_df["total_dairy_export_value_1000USD"] = pd.to_numeric(
    ireland_df["total_dairy_export_value_1000USD"],
    errors="coerce"
)
ireland_df = ireland_df.dropna(subset=["year", "total_dairy_export_value_1000USD"]).copy()
ireland_df["year"] = ireland_df["year"].astype(int)

# Values are stored in thousand USD. Dividing by 1,000,000 gives USD billions.
ireland_df["export_value_bn"] = ireland_df["total_dairy_export_value_1000USD"] / 1_000_000

latest_year = int(ireland_df["year"].max())
first_year = int(ireland_df["year"].min())
latest_export = ireland_df.loc[ireland_df["year"] == latest_year, "total_dairy_export_value_1000USD"].iloc[0]
first_export = ireland_df.loc[ireland_df["year"] == first_year, "total_dairy_export_value_1000USD"].iloc[0]
latest_export_bn = latest_export / 1_000_000
export_growth = safe_pct_change(latest_export, first_export)

if 2023 in ireland_df["year"].values:
    export_2023 = ireland_df.loc[ireland_df["year"] == 2023, "total_dairy_export_value_1000USD"].iloc[0]
    recovery_growth = safe_pct_change(latest_export, export_2023)
else:
    recovery_growth = pd.NA

peak_year = int(ireland_df.loc[ireland_df["total_dairy_export_value_1000USD"].idxmax(), "year"])

# Prepare product export data in both wide and long forms.
product_export_df["year"] = pd.to_numeric(product_export_df["year"], errors="coerce")
product_export_df = product_export_df.dropna(subset=["year"]).copy()
product_export_df["year"] = product_export_df["year"].astype(int)

product_value_columns = [
    "butter_export_value_1000USD",
    "cheese_export_value_1000USD",
    "milk_powder_export_value_1000USD",
    "milk_export_value_1000USD"
]

for column in product_value_columns:
    product_export_df[column] = pd.to_numeric(product_export_df[column], errors="coerce")

product_label_map = {
    "butter_export_value_1000USD": "Butter",
    "cheese_export_value_1000USD": "Cheese",
    "milk_powder_export_value_1000USD": "Milk powder",
    "milk_export_value_1000USD": "Raw milk"
}

product_long = product_export_df.melt(
    id_vars="year",
    value_vars=product_value_columns,
    var_name="Product",
    value_name="Export Value 1000USD"
)
product_long["Product"] = product_long["Product"].map(product_label_map)
product_long = product_long.dropna(subset=["Export Value 1000USD"]).copy()
latest_product_year = int(product_export_df["year"].max())

# Prepare ML feature importance.
ml_feature_importance["Coefficient"] = pd.to_numeric(
    ml_feature_importance["Coefficient"],
    errors="coerce"
)
ml_feature_importance["Feature Label"] = ml_feature_importance["Feature"].apply(clean_feature_name)
ml_feature_importance["Absolute Coefficient"] = ml_feature_importance["Coefficient"].abs()

top_driver_df = (
    ml_feature_importance
    .dropna(subset=["Coefficient"])
    .sort_values("Absolute Coefficient", ascending=False)
    .head(8)
    .copy()
)

if top_driver_df.empty:
    top_driver_df = pd.DataFrame(
        {
            "Feature": ["lag_export_value"],
            "Feature Label": ["Previous export value"],
            "Coefficient": [1.0],
            "Absolute Coefficient": [1.0]
        }
    )

max_abs_coef = top_driver_df["Absolute Coefficient"].max()
top_driver_df["Relative Signal"] = np.where(
    max_abs_coef == 0,
    0,
    (top_driver_df["Coefficient"] / max_abs_coef) * 100
)
top_driver_df["Direction"] = np.where(
    top_driver_df["Relative Signal"] >= 0,
    "Same direction as exports",
    "Opposite direction to exports"
)
top_driver_df = top_driver_df.sort_values("Relative Signal", ascending=True)

top_signal_row = top_driver_df.sort_values("Absolute Coefficient", ascending=False).iloc[0]
non_lag_driver_df = top_driver_df[top_driver_df["Feature"] != "lag_export_value"].copy()
top_non_lag_row = (
    non_lag_driver_df.sort_values("Absolute Coefficient", ascending=False).iloc[0]
    if not non_lag_driver_df.empty
    else top_signal_row
)

# Prepare sentiment data.
sentiment_df["sentiment_label"] = (
    sentiment_df["sentiment_label"].astype(str).str.strip().str.title()
)
sentiment_df["year"] = pd.to_numeric(sentiment_df.get("year"), errors="coerce")

if "country" in sentiment_df.columns:
    sentiment_df["country"] = sentiment_df["country"].astype(str).str.strip()

if "perspective" in sentiment_df.columns:
    sentiment_df["perspective"] = sentiment_df["perspective"].astype(str).str.strip().str.title()

sentiment_counts = (
    sentiment_df["sentiment_label"]
    .value_counts()
    .reindex(SENTIMENT_ORDER)
    .fillna(0)
    .astype(int)
)
total_posts = int(sentiment_counts.sum())
sentiment_shares = (sentiment_counts / total_posts * 100).round(1) if total_posts > 0 else pd.Series({s: 0 for s in SENTIMENT_ORDER})
negative_share = sentiment_shares["Negative"]
neutral_share = sentiment_shares["Neutral"]
positive_share = sentiment_shares["Positive"]
overall_tone = "Slightly negative" if negative_share > positive_share else "More positive" if positive_share > negative_share else "Mixed"

# Dropdown options are generated from available sentiment columns.
breakdown_options = []
if "country" in sentiment_df.columns:
    breakdown_options.append({"label": "By country", "value": "country"})
if "perspective" in sentiment_df.columns:
    breakdown_options.append({"label": "By perspective", "value": "perspective"})
default_breakdown = "country" if "country" in sentiment_df.columns else breakdown_options[0]["value"]


# Benchmark Feature Definitions

FEATURE_OPTIONS = {
    "milk_prod_1000t": {
        "label": "Milk production",
        "unit": "million tonnes",
        "source_unit": "1000 tonnes",
        "ranking_title": "Which country produced the most milk",
        "trend_title": "How has milk production changed over time?"
    },
    "butter_prod_1000t": {
        "label": "Butter production",
        "unit": "million tonnes",
        "source_unit": "1000 tonnes",
        "ranking_title": "Which country produced the most butter",
        "trend_title": "How has butter production changed over time?"
    },
    "cheese_prod_1000t": {
        "label": "Cheese production",
        "unit": "million tonnes",
        "source_unit": "1000 tonnes",
        "ranking_title": "Which country produced the most cheese",
        "trend_title": "How has cheese production changed over time?"
    },
    "milk_powder_prod_1000t": {
        "label": "Milk powder production",
        "unit": "million tonnes",
        "source_unit": "1000 tonnes",
        "ranking_title": "Which country produced the most milk powder",
        "trend_title": "How has milk powder production changed over time?"
    },
    "milk_delivered_to_dairies_1000t": {
        "label": "Milk delivered to dairies",
        "unit": "million tonnes",
        "source_unit": "1000 tonnes",
        "ranking_title": "Which country delivered the most milk to dairies",
        "trend_title": "How has milk delivered to dairies changed over time?"
    },
    "milk_per_cow": {
        "label": "Milk per cow",
        "unit": "tonnes per cow",
        "source_unit": "tonnes per cow",
        "ranking_title": "Which country produced the most milk per cow",
        "trend_title": "How has milk per cow changed over time?"
    },
    "dairy_cow_pop_1000": {
        "label": "Dairy cow population",
        "unit": "million cows",
        "source_unit": "1000 cows",
        "ranking_title": "Which country had the largest dairy cow population",
        "trend_title": "How has the dairy cow population changed over time?"
    },
    "output_per_labour": {
        "label": "Output per labour",
        "unit": "€ per AWU",
        "source_unit": "€ thousand per AWU",
        "ranking_title": "Which country had the highest output per labour unit",
        "trend_title": "How has output per labour changed over time?"
    },
    "agri_output_basic_price_million_eur": {
        "label": "Agricultural output",
        "unit": "€ million",
        "source_unit": "€ million",
        "ranking_title": "Which country had the highest agricultural output",
        "trend_title": "How has agricultural output changed over time?"
    },
    "dairy_cow_share": {
        "label": "Dairy cow share",
        "unit": "%",
        "source_unit": "%",
        "ranking_title": "Which country had the highest dairy cow share",
        "trend_title": "How has dairy cow share changed over time?"
    },
    "cap_direct_payments_million_eur": {
        "label": "CAP direct payments",
        "unit": "€ million",
        "source_unit": "€ million",
        "ranking_title": "Which country received the most CAP direct payments",
        "trend_title": "How have CAP direct payments changed over time?"
    },
    "milk_prod_growth_pct": {
        "label": "Milk production growth",
        "unit": "%",
        "source_unit": "%",
        "ranking_title": "Which country had the highest milk production growth",
        "trend_title": "How has milk production growth changed over time?"
    },
    "export_growth_pct": {
        "label": "Export growth",
        "unit": "%",
        "source_unit": "%",
        "ranking_title": "Which country had the highest export growth",
        "trend_title": "How has export growth changed over time?"
    }
}

# Only keep feature options that actually exist in the loaded dataframe.
FEATURE_OPTIONS = {
    feature: info
    for feature, info in FEATURE_OPTIONS.items()
    if feature in comparison_df.columns
}


def display_value(value, feature):
    """Convert source units into farmer-readable display units."""
    if pd.isna(value):
        return value

    source_unit = FEATURE_OPTIONS[feature]["source_unit"]

    if source_unit in ["1000 tonnes", "1000 cows"]:
        return value / 1000
    if feature == "output_per_labour":
        return value * 1000
    return value


def format_value(value, feature):
    """Format a value with the correct unit for cards and explanations."""
    if value is None or pd.isna(value):
        return "N/A"

    unit = FEATURE_OPTIONS[feature]["unit"]
    value = display_value(value, feature)

    if unit == "%":
        return f"{value:.1f}%"
    if unit == "million tonnes":
        return f"{value:,.2f} million tonnes"
    if unit == "million cows":
        return f"{value:,.2f} million cows"
    if unit == "tonnes per cow":
        return f"{value:.2f} tonnes per cow"
    if unit == "€ per AWU":
        return f"€{value:,.0f} per AWU"
    if unit == "€ million":
        return f"€{value:,.0f}m"
    return f"{value:,.0f}"


def format_chart_label(value, feature):
    """
    Create compact labels for chart annotations.

    The axis title already gives the full unit, but the bar labels also include
    short units so farmers can read the chart without looking back and forth.
    """
    if value is None or pd.isna(value):
        return "N/A"

    unit = FEATURE_OPTIONS[feature]["unit"]
    value = display_value(value, feature)

    if unit == "%":
        return f"{value:.1f}%"
    if unit == "million tonnes":
        return f"{value:,.2f}m t"
    if unit == "million cows":
        return f"{value:,.2f}m cows"
    if unit == "tonnes per cow":
        return f"{value:.2f} t/cow"
    if unit == "€ per AWU":
        return f"€{value:,.0f}/AWU"
    if unit == "€ million":
        return f"€{value:,.0f}m"
    return f"{value:,.0f}"


def describe_change(start_value, end_value, feature):
    """Create a natural-language trend description for a selected feature."""
    if pd.isna(start_value) or pd.isna(end_value):
        return "had incomplete trend data"

    change = end_value - start_value
    unit = FEATURE_OPTIONS[feature]["unit"]

    if unit == "%":
        if change > 0:
            return f"increased by {abs(change):.1f} percentage points"
        if change < 0:
            return f"decreased by {abs(change):.1f} percentage points"
        return "stayed unchanged"

    pct_change = safe_pct_change(end_value, start_value)

    if pd.isna(pct_change):
        return f"changed from {format_value(start_value, feature)} to {format_value(end_value, feature)}"
    if change > 0:
        return f"increased by {pct_change:.1f}%"
    if change < 0:
        return f"decreased by {abs(pct_change):.1f}%"
    return "stayed unchanged"


# Static Figures and Evidence Text

# Export trend figure.
export_fig = px.line(
    ireland_df,
    x="year",
    y="export_value_bn",
    markers=True,
    labels={"year": "Year", "export_value_bn": "Export Value (USD Bn)"}
)
export_fig.update_traces(line=dict(width=3, color=IRELAND_COLOR), marker=dict(size=7, color=IRELAND_COLOR))
apply_clean_layout(export_fig, height=430, margin=dict(t=20, b=40, l=60, r=40), hovermode="x unified")
export_fig.update_layout(showlegend=False)
export_fig.update_xaxes(tickmode="linear", dtick=1)

# ML driver figure.
def make_driver_figure():
    """Create a lollipop chart showing the strongest model signals."""
    fig = go.Figure()

    for _, row in top_driver_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[0, row["Relative Signal"]],
                y=[row["Feature Label"], row["Feature Label"]],
                mode="lines",
                line=dict(color=SOFT_GREY, width=2),
                hoverinfo="skip",
                showlegend=False
            )
        )

    for direction, colour in [
        ("Same direction as exports", POSITIVE_COLOR),
        ("Opposite direction to exports", NEGATIVE_COLOR)
    ]:
        direction_df = top_driver_df[top_driver_df["Direction"] == direction]
        fig.add_trace(
            go.Scatter(
                x=direction_df["Relative Signal"],
                y=direction_df["Feature Label"],
                mode="markers",
                name=direction,
                marker=dict(size=13, color=colour),
                customdata=direction_df[["Feature", "Coefficient"]].to_numpy(),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Notebook feature: %{customdata[0]}<br>"
                    "Relative model signal: %{x:.1f}%<br>"
                    "Ridge coefficient: %{customdata[1]:,.2f}"
                    "<extra></extra>"
                )
            )
        )

    fig.add_vline(x=0, line_width=1, line_color="#9aa5aa")
    fig.update_layout(
        template="plotly_white",
        height=430,
        margin=dict(t=50, b=40, l=180, r=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, title=""),
        font=dict(family="Arial", size=13, color=DARK_TEXT),
        xaxis=dict(
            title="Relative model signal, strongest = 100%",
            range=[-110, 110],
            tickvals=[-100, -50, 0, 50, 100],
            ticktext=["-100%", "-50%", "0", "50%", "100%"],
            showline=True,
            linecolor=SOFT_GREY,
            gridcolor="#edf2f5",
            zeroline=False
        ),
        yaxis=dict(title="", showline=False, gridcolor="#ffffff", zeroline=False)
    )
    return fig


driver_fig = make_driver_figure()

# Sentiment trend figure.
year_sentiment = (
    sentiment_df
    .dropna(subset=["year", "sentiment_label"])
    .groupby(["year", "sentiment_label"])
    .size()
    .reset_index(name="Posts")
)

if year_sentiment.empty:
    sentiment_trend_fig = make_empty_figure("No yearly sentiment data available.")
else:
    sentiment_trend_fig = px.line(
        year_sentiment,
        x="year",
        y="Posts",
        color="sentiment_label",
        markers=True,
        color_discrete_map=SENTIMENT_COLOR_MAP,
        category_orders={"sentiment_label": SENTIMENT_ORDER},
        labels={"year": "Year", "Posts": "Number of posts", "sentiment_label": "Sentiment"}
    )
    sentiment_trend_fig.update_traces(line=dict(width=3), marker=dict(size=7))
    apply_clean_layout(sentiment_trend_fig, height=430, margin=dict(t=20, b=40, l=60, r=40), hovermode="x unified", showlegend=True)
    sentiment_trend_fig.update_layout(legend_title="")
    sentiment_trend_fig.update_xaxes(tickmode="linear", dtick=1)

# Product evidence for recommendation tab.
product_totals = (
    product_long.groupby("Product", as_index=False)["Export Value 1000USD"]
    .sum()
    .sort_values("Export Value 1000USD", ascending=False)
)
top_two_over_years = product_totals.head(2)["Product"].tolist()
top_two_products_over_years = " and ".join(top_two_over_years) if len(top_two_over_years) >= 2 else "processed dairy products"

latest_product_values = (
    product_long[product_long["year"] == latest_product_year]
    .groupby("Product", as_index=False)["Export Value 1000USD"]
    .sum()
)
latest_total_selected = latest_product_values["Export Value 1000USD"].sum()
latest_product_values["Share (%)"] = np.where(
    latest_total_selected == 0,
    0,
    (latest_product_values["Export Value 1000USD"] / latest_total_selected) * 100
)
latest_product_values = latest_product_values.sort_values("Export Value 1000USD", ascending=False)
top_product = latest_product_values.iloc[0]
top_two_share_latest = latest_product_values.head(2)["Share (%)"].sum()

# Evidence text used in cards and farmer action recommendations.
top_signal_label = clean_feature_name(top_signal_row["Feature"])
farm_efficiency_signal = clean_feature_name(top_non_lag_row["Feature"])

ireland_sentiment_df = sentiment_df.copy()
if "country" in ireland_sentiment_df.columns:
    ireland_sentiment_df = ireland_sentiment_df[ireland_sentiment_df["country"].str.lower() == "ireland"].copy()

if not ireland_sentiment_df.empty:
    ireland_top_sentiment = ireland_sentiment_df["sentiment_label"].value_counts().idxmax().lower()
    ireland_sentiment_text = f"Ireland showed a mainly {ireland_top_sentiment} discussion pattern within the country-level sentiment view."
else:
    ireland_sentiment_text = "Ireland-specific sentiment records were not available after filtering, so the sentiment view should be read only as wider market mood context."

if "perspective" in sentiment_df.columns:
    perspective_counts = sentiment_df["perspective"].value_counts()
    if "Producer" in perspective_counts.index and "Consumer" in perspective_counts.index and perspective_counts["Producer"] > perspective_counts["Consumer"]:
        perspective_balance_text = "Producer posts were more common than consumer posts, so the sentiment view may reflect producer discussion more strongly than consumer demand."
    else:
        perspective_balance_text = "The sentiment view should still be read cautiously because online discussion does not represent the full market."
else:
    perspective_balance_text = "The sentiment view should be read cautiously because online discussion does not represent the full market."

# Policy/output evidence for recommendations.
def ireland_change_text(column, value_label):
    """Create evidence text for Ireland's first-to-latest change in a selected column."""
    temp_df = comparison_df[(comparison_df["country"] == "Ireland") & comparison_df[column].notna()].copy()
    if temp_df.empty:
        return pd.NA, f"Ireland's {value_label.lower()} data was not available after filtering."

    first_obs_year = int(temp_df["year"].min())
    latest_obs_year = int(temp_df["year"].max())
    first_obs = temp_df.loc[temp_df["year"] == first_obs_year, column].iloc[0]
    latest_obs = temp_df.loc[temp_df["year"] == latest_obs_year, column].iloc[0]
    pct_change = safe_pct_change(latest_obs, first_obs)

    text = f"Ireland’s {value_label.lower()} changed by {pct_change:.1f}% from {first_obs_year} to {latest_obs_year}."
    return pct_change, text


ireland_output_growth, ireland_output_text = ireland_change_text("agri_output_basic_price_million_eur", "agricultural output")

ireland_cap_df = comparison_df[(comparison_df["country"] == "Ireland") & comparison_df["cap_direct_payments_million_eur"].notna()].copy()
if not ireland_cap_df.empty:
    ireland_cap_first_year = int(ireland_cap_df["year"].min())
    ireland_cap_latest_year = int(ireland_cap_df["year"].max())
    ireland_cap_first = ireland_cap_df.loc[ireland_cap_df["year"] == ireland_cap_first_year, "cap_direct_payments_million_eur"].iloc[0]
    ireland_cap_latest = ireland_cap_df.loc[ireland_cap_df["year"] == ireland_cap_latest_year, "cap_direct_payments_million_eur"].iloc[0]
    ireland_cap_change = safe_pct_change(ireland_cap_latest, ireland_cap_first)
    cap_change_text = (
        f"CAP direct payments changed by {ireland_cap_change:.1f}% from {ireland_cap_first_year} to {ireland_cap_latest_year}, "
        f"reaching €{ireland_cap_latest:,.0f}m in the latest available CAP year."
    )
    cap_interpretation_text = (
        "CAP support stayed broadly stable compared with output growth, so Ireland’s agricultural growth "
        "should not be interpreted as being driven by rising payments alone."
    )
else:
    cap_change_text = "Latest CAP payment data was not available after filtering."
    cap_interpretation_text = "Policy support should still be monitored because payment and regulation changes can affect planning."

# Recommendation Data

RECOMMENDATION_DATA = {
    "export_growth": {
        "dropdown_label": "Export growth",
        "badge": "Monitor",
        "title": "Protect export momentum before expanding.",
        "summary": "A strong export year is positive, but expansion should be based on repeated demand signals, not only one good year.",
        "data": [
            f"Irish dairy exports reached ${latest_export_bn:.2f}Bn in {latest_year}.",
            f"Export value increased by {export_growth:.1f}% from {first_year} to {latest_year}.",
            f"{top_signal_label} was the strongest signal in the analysis, showing that recent export performance is an important early warning measure."
        ],
        "actions": [
            "Compare each year’s export value with the previous year before increasing scale.",
            "Check processor demand before committing to major production expansion.",
            "Keep a cautious cash-flow plan if export value weakens after a strong year."
        ],
        "watch": [
            "Year-on-year export movement.",
            "Processor demand signals.",
            "Sudden drops after a high export year."
        ]
    },
    "farm_productivity": {
        "dropdown_label": "Farm productivity",
        "badge": "Act first",
        "title": "Improve productivity, not only production scale.",
        "summary": "The clearest controllable signal is efficiency. Producing more is useful only if labour, herd and input use are also managed well.",
        "data": [
            f"{farm_efficiency_signal} is the strongest farm-efficiency indicator for improved export value.",
            "This links competitiveness with how efficiently labour and resources are turned into output.",
            "Productivity should be monitored alongside export value, because export growth is stronger when farms stay efficient."
        ],
        "actions": [
            "Track output per labour input, not only total milk volume.",
            "Review labour-heavy jobs that could be improved through planning or technology.",
            "Use herd, yield and labour records to identify where efficiency is being lost."
        ],
        "watch": [
            "Labour pressure during busy production periods.",
            "Production growth that comes only from more scale.",
            "Whether productivity improves when production increases."
        ]
    },
    "processed_dairy": {
        "dropdown_label": "Processed dairy demand",
        "badge": "Act",
        "title": "Align milk quality with processed dairy demand.",
        "summary": "Farmers do not directly control final export products, but consistent milk quality supports processors producing higher-value dairy exports.",
        "data": [
            f"Across the years shown, {top_two_products_over_years} led selected Irish dairy export value.",
            f"In {latest_product_year}, {top_product['Product']} alone accounted for {top_product['Share (%)']:.1f}%, while the top two products together contributed {top_two_share_latest:.1f}% of selected product export value."
        ],
        "actions": [
            "Pay attention to processor signals around butter, cheese and milk solids.",
            "Prioritise consistent milk quality, not only raw milk volume.",
            "Use product demand as context when planning herd and production decisions."
        ],
        "watch": [
            "Changes in butter and cheese export value.",
            "Processor quality requirements.",
            "Whether raw milk growth is matched by processed dairy demand."
        ]
    },
    "market_mood": {
        "dropdown_label": "Market mood",
        "badge": "Watch",
        "title": "Use sentiment as a caution signal, not proof of demand.",
        "summary": "Ireland’s sentiment picture is useful context, but it should still be read carefully because online discussion does not represent every farmer, processor or consumer.",
        "data": [
            ireland_sentiment_text,
            perspective_balance_text,
            "Sentiment is useful for understanding discussion tone, but it should not replace export, productivity or product-demand evidence."
        ],
        "actions": [
            "Use positive Ireland discussion as a confidence signal, not as proof of guaranteed demand.",
            "Watch repeated concerns around price, sustainability, subsidies and farming pressure.",
            "Separate producer concerns from consumer demand when interpreting sentiment."
        ],
        "watch": [
            "Whether Ireland’s positive discussion remains stable over time.",
            "Whether negative themes increase in consumer-facing discussions.",
            "Whether public discussion weakens while export performance also weakens."
        ]
    },
    "policy_risk": {
        "dropdown_label": "Policy and subsidy risk",
        "badge": "Watch closely",
        "title": "Plan for policy changes, but do not rely on support alone.",
        "summary": "Ireland’s agricultural output increased strongly, while CAP direct payments did not rise at the same pace. Growth should therefore be read as sector performance supported by policy backing, not as payment growth alone.",
        "data": [
            ireland_output_text,
            cap_change_text,
            cap_interpretation_text
        ],
        "actions": [
            "Use output growth as a positive signal, but avoid assuming support levels will always increase with production.",
            "Check CAP/payment updates before making long-term investment decisions.",
            "Build farm plans that can handle policy, payment or regulation changes."
        ],
        "watch": [
            "CAP payment changes.",
            "Environmental or production-related rules.",
            "Investment decisions that depend heavily on current support levels."
        ]
    }
}

# Building layout for each tab
def build_overview_tab():
    """Overview tab: Irish export trend and product demand by year."""
    default_product_year = int(product_export_df["year"].max())

    return dbc.Tab(
        label="Overview",
        children=[
            html.Br(),
            dbc.Row([
                dbc.Col(metric_card(f"${latest_export_bn:,.2f}Bn", "Latest Export Value"), xs=12, md=3),
                dbc.Col(metric_card(f"{export_growth:.1f}%", f"Long-Term Growth ({first_year}–{latest_year})"), xs=12, md=3),
                dbc.Col(metric_card(f"{peak_year}", "Peak Export Year"), xs=12, md=3),
                dbc.Col(metric_card(format_growth(recovery_growth), "Recovery from 2023"), xs=12, md=3)
            ], className="mb-4 g-3"),

            html.H4("Export Performance Overview", className="mb-2"),
            html.P(
                f"Irish dairy exports reached ${latest_export_bn:.2f}Bn in {latest_year}. "
                "Butter and cheese dominate selected dairy export value, showing the importance of higher-value processed dairy demand.",
                className="text-muted mb-4"
            ),

            dbc.Row([
                dbc.Col([
                    html.H5("Ireland Dairy Export Value Trend", style=CHART_HEADING_STYLE),
                    dcc.Graph(figure=export_fig, config=graph_config),
                    tufte_note(
                        "Market Direction",
                        "Most of the export growth came after 2020. The 2023 setback was short-lived, with export value returning to a new high in 2024."
                    )
                ], xs=12, lg=6),

                dbc.Col([
                    html.H5(id="product-chart-heading", children=f"Top Irish Dairy Exports ({default_product_year})", style=CHART_HEADING_STYLE),
                    dcc.Dropdown(
                        id="product-year-filter",
                        options=[{"label": int(y), "value": int(y)} for y in sorted(product_export_df["year"].unique())],
                        value=default_product_year,
                        clearable=False,
                        className="mb-3"
                    ),
                    dcc.Graph(id="product-export-chart", config=graph_config),
                    html.Div(id="product-export-insight")
                ], xs=12, lg=6)
            ], className="g-4")
        ]
    )


def build_benchmarking_tab():
    """Benchmarking tab: compare Ireland with other EU dairy countries."""
    default_feature = "milk_prod_1000t" if "milk_prod_1000t" in FEATURE_OPTIONS else list(FEATURE_OPTIONS.keys())[0]

    return dbc.Tab(
        label="Efficiency Benchmarking",
        children=[
            html.Br(),
            html.H4("How does Ireland compare to top dairy-producing countries in Europe?", className="mb-2"),
            html.P(
                "Choose a year, measure and comparison countries. Ireland is highlighted in green to keep the visual comparison clear.",
                className="text-muted mb-4"
            ),

            dbc.Row([
                dbc.Col([
                    html.Label("Year"),
                    dcc.Dropdown(
                        id="comparison-year-filter",
                        options=[{"label": int(y), "value": int(y)} for y in sorted(comparison_df["year"].unique())],
                        value=int(comparison_df["year"].max()),
                        clearable=False
                    )
                ], xs=12, md=2),
                dbc.Col([
                    html.Label("Measure"),
                    dcc.Dropdown(
                        id="comparison-feature-filter",
                        options=[{"label": info["label"], "value": feature} for feature, info in FEATURE_OPTIONS.items()],
                        value=default_feature,
                        clearable=False
                    )
                ], xs=12, md=4),
                dbc.Col([
                    html.Label("Comparison countries"),
                    dcc.Dropdown(
                        id="comparison-country-filter",
                        options=[{"label": c, "value": c} for c in sorted(comparison_df["country"].unique()) if c != "Ireland"],
                        value=["Netherlands", "Germany", "France"],
                        multi=True,
                        clearable=False
                    )
                ], xs=12, md=6)
            ], className="mb-4 g-3"),

            dbc.Row([
                dbc.Col(metric_card(html.Span(id="ireland-value-card"), "Ireland Value"), xs=12, md=3),
                dbc.Col(metric_card(html.Span(id="highest-peer-card"), "Top Peer Country"), xs=12, md=3),
                dbc.Col(metric_card(html.Span(id="peer-average-card"), "Comparison Average"), xs=12, md=3),
                dbc.Col(metric_card(html.Span(id="gap-card"), "Ireland vs Comparison Avg"), xs=12, md=3)
            ], className="mb-4 g-3"),

            dbc.Row([
                dbc.Col([
                    html.H5(id="ranking-heading", style=CHART_HEADING_STYLE),
                    dcc.Graph(id="ranking-chart", config=graph_config, style={"height": "520px"})
                ], xs=12, lg=7),

                # The explanation is kept closer to the chart to reduce empty space
                # and make the ranking + interpretation feel like one visual unit.
                dbc.Col(
                    html.Div(id="ranking-chart-description"),
                    xs=12,
                    lg=5,
                    style={"display": "flex", "alignItems": "center"}
                )
            ], className="g-2 mb-4 align-items-center"),

            dbc.Row([
                dbc.Col(html.Div(id="trend-chart-description"), xs=12, lg=4),
                dbc.Col([
                    html.H5(id="trend-heading", style=CHART_HEADING_STYLE),
                    dcc.Graph(id="trend-chart", config=graph_config, style={"height": "520px"})
                ], xs=12, lg=8)
            ], className="g-4 align-items-center")
        ]
    )


def build_sentiment_tab():
    """Sentiment and ML driver tab."""
    return dbc.Tab(
        label="Market Sentiment & Export Drivers",
        children=[
            html.Br(),
            dbc.Row([
                dbc.Col(metric_card(clean_feature_name(top_signal_row["Feature"]), "Strongest model signal"), xs=12, md=4),
                dbc.Col(metric_card(clean_feature_name(top_non_lag_row["Feature"]), "Strongest non-lag signal"), xs=12, md=4),
                dbc.Col(metric_card(f"{negative_share:.1f}% negative", "Overall Reddit tone"), xs=12, md=4)
            ], className="mb-4 g-3"),

            html.H5("What signals should farmers watch?", style=CHART_HEADING_STYLE),
            dcc.Graph(figure=driver_fig, config=graph_config),
            tufte_note(
                "Export Drivers",
                "Previous export value was the strongest signal found by the model. The strongest non-lag signal was output per labour. This suggests farmers should watch both recent export performance and productivity-related indicators."
            ),
            limitation_note(
                "The findings come from the analysis of limited dataset, so they should be used as supporting evidence rather than as a direct forecast."
            ),

            dbc.Row([
                dbc.Col([
                    html.H5("How has dairy discussion changed over time?", style=CHART_HEADING_STYLE),
                    dcc.Graph(figure=sentiment_trend_fig, config=graph_config),
                    tufte_note(
                        "Sentiment Over Time",
                        "The trend shows how many positive, neutral and negative posts appeared each year. It helps identify whether dairy discussion is becoming more active, more positive or more negative over time."
                    )
                ], xs=12, lg=6),

                dbc.Col([
                    html.H5("Where is sentiment strongest?", style=CHART_HEADING_STYLE),
                    dcc.Dropdown(
                        id="sentiment-breakdown-filter",
                        options=breakdown_options,
                        value=default_breakdown,
                        clearable=False,
                        className="mb-3"
                    ),
                    dcc.Graph(id="sentiment-breakdown-chart", config=graph_config),
                    html.Div(id="sentiment-breakdown-note")
                ], xs=12, lg=6)
            ], className="g-4 mt-4"),

            limitation_note(
                "Reddit sentiment may be biased by social media users, language differences and uneven post volumes, so it should be treated as a market mood indicator rather than a direct measure of demand."
            )
        ]
    )


def make_recommendation_view(selected_focus):
    """Create the recommendation panel for the selected decision focus."""
    selected = RECOMMENDATION_DATA[selected_focus]

    return dbc.Card(
        dbc.CardBody([
            html.Div(selected["badge"], style=BADGE_STYLE),
            html.H3(
                selected["title"],
                style={
                    "fontWeight": "700",
                    "fontSize": "1.55rem",
                    "lineHeight": "1.25",
                    "color": DARK_TEXT,
                    "marginBottom": "0.75rem"
                }
            ),
            html.P(
                selected["summary"],
                style={
                    "fontSize": "0.98rem",
                    "color": MUTED_TEXT,
                    "lineHeight": "1.6",
                    "marginBottom": "1.2rem"
                }
            ),
            dbc.Row([
                dbc.Col(info_box("What the data shows", selected["data"]), xs=12, lg=4),
                dbc.Col(info_box("Recommended action", selected["actions"]), xs=12, lg=4),
                dbc.Col(info_box("What to watch", selected["watch"]), xs=12, lg=4)
            ], className="g-3")
        ]),
        className="border-0 shadow-sm",
        style={
            "backgroundColor": LIGHT_BG,
            "borderRadius": "8px",
            "marginTop": "1rem"
        }
    )


def build_action_plan_tab():
    """Farmer action-plan tab: turns evidence into clear decision guidance."""
    return dbc.Tab(
        label="Farmer Action Plan",
        children=[
            html.Br(),
            
            html.H3("What should farmers focus on next?", 
                        style={
                            **CHART_HEADING_STYLE,
                            "textAlign": "center",
                            "fontSize": "1.75rem",
                            "fontWeight":"600",
                            "marginTop": "1.4rem",
                            "marginBottom": "1.6rem"
                        }
                   ),

            dbc.Row([
                dbc.Col(
                    evidence_card(
                        "Export momentum",
                        f"${latest_export_bn:.2f}Bn export value in {latest_year}; {format_growth(recovery_growth)} recovery from 2023."
                    ),
                    xs=12, md=6, lg=3
                ),
                dbc.Col(
                    evidence_card("Farm efficiency", f"{farm_efficiency_signal} identified as the strongest farm-efficiency signal."),
                    xs=12, md=6, lg=3
                ),
                dbc.Col(
                    evidence_card("Processed dairy demand", f"{top_two_products_over_years} led selected Irish dairy export value."),
                    xs=12, md=6, lg=3
                ),
                dbc.Col(
                    evidence_card("Market mood", "Overall positive sentiment is identified for Irish dairy, but it should be read cautiously alongside export and productivity data."),
                    xs=12, md=6, lg=3
                )
            ], className="mb-4 g-3"),

            dbc.Row([
                dbc.Col([
                    html.H5("Select decision focus", style={**SECTION_HEADING_STYLE, "textAlign": "center"}),
                    dcc.Dropdown(
                        id="decision-focus-filter",
                        options=[
                            {"label": value["dropdown_label"], "value": key}
                            for key, value in RECOMMENDATION_DATA.items()
                        ],
                        value="farm_productivity",
                        clearable=False
                    )
                ], xs=12, md=8, lg=5)
            ], justify="center", className="mb-3"),

            html.Div(id="recommendation-output"),
            limitation_note(
                "These recommendations are based on the analysis of limited dataset and should guide farmer judgement rather than be treated as guaranteed outcomes."
            )
        ]
    )

# Callbacks

@app.callback(
    Output("product-export-chart", "figure"),
    Output("product-export-insight", "children"),
    Output("product-chart-heading", "children"),
    Input("product-year-filter", "value")
)
def update_product_export_chart(selected_year):
    """Update product export chart when the selected year changes."""
    selected_df = product_export_df[product_export_df["year"] == selected_year]
    previous_df = product_export_df[product_export_df["year"] == selected_year - 1]

    if selected_df.empty:
        return (
            make_empty_figure("No product export data available for this year."),
            tufte_note("Export Concentration", "No product export evidence is available for the selected year."),
            f"Top Irish Dairy Exports ({selected_year})"
        )

    product_values = pd.DataFrame({
        "Product": ["Butter", "Cheese", "Milk powder", "Raw milk"],
        "Export Value (USD Bn)": [
            selected_df["butter_export_value_1000USD"].iloc[0] / 1_000_000,
            selected_df["cheese_export_value_1000USD"].iloc[0] / 1_000_000,
            selected_df["milk_powder_export_value_1000USD"].iloc[0] / 1_000_000,
            selected_df["milk_export_value_1000USD"].iloc[0] / 1_000_000
        ]
    })

    if not previous_df.empty:
        previous_values = {
            "Butter": previous_df["butter_export_value_1000USD"].iloc[0] / 1_000_000,
            "Cheese": previous_df["cheese_export_value_1000USD"].iloc[0] / 1_000_000,
            "Milk powder": previous_df["milk_powder_export_value_1000USD"].iloc[0] / 1_000_000,
            "Raw milk": previous_df["milk_export_value_1000USD"].iloc[0] / 1_000_000
        }
        product_values["Previous Value"] = product_values["Product"].map(previous_values)
        product_values["Growth (%)"] = product_values.apply(
            lambda row: safe_pct_change(row["Export Value (USD Bn)"], row["Previous Value"]),
            axis=1
        )
    else:
        product_values["Growth (%)"] = 0

    total_value = product_values["Export Value (USD Bn)"].sum()
    product_values["Share (%)"] = np.where(
        total_value == 0,
        0,
        (product_values["Export Value (USD Bn)"] / total_value) * 100
    )
    product_values = product_values.sort_values("Export Value (USD Bn)", ascending=False)
    product_values["Label"] = product_values["Export Value (USD Bn)"].apply(lambda value: f"${value:.2f}Bn")

    fig = px.bar(
        product_values,
        x="Product",
        y="Export Value (USD Bn)",
        text="Label",
        labels={"Product": "", "Export Value (USD Bn)": "Export Value (USD Bn)"},
        hover_data={"Export Value (USD Bn)": ":.2f", "Share (%)": ":.1f", "Growth (%)": ":.1f"}
    )
    fig.update_traces(marker_color=IRELAND_COLOR, textposition="outside")
    apply_clean_layout(fig, height=430, margin=dict(t=20, b=40, l=60, r=40))
    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[0, product_values["Export Value (USD Bn)"].max() * 1.22])

    top_product_year = product_values.iloc[0]
    top_two_share = product_values.head(2)["Share (%)"].sum()
    insight = tufte_note(
        "Export Concentration",
        f"{top_product_year['Product']} led selected dairy exports in {selected_year}, accounting for {top_product_year['Share (%)']:.1f}% of value. The top two selected products together accounted for {top_two_share:.1f}%."
    )

    return fig, insight, f"Top Irish Dairy Exports ({selected_year})"


@app.callback(
    Output("ireland-value-card", "children"),
    Output("highest-peer-card", "children"),
    Output("peer-average-card", "children"),
    Output("gap-card", "children"),
    Output("ranking-chart", "figure"),
    Output("trend-chart", "figure"),
    Output("ranking-heading", "children"),
    Output("trend-heading", "children"),
    Output("ranking-chart-description", "children"),
    Output("trend-chart-description", "children"),
    Input("comparison-year-filter", "value"),
    Input("comparison-feature-filter", "value"),
    Input("comparison-country-filter", "value")
)
def update_country_comparison(selected_year, selected_feature, selected_countries):
    """Update country benchmarking cards, ranking chart and trend chart."""
    if selected_feature not in FEATURE_OPTIONS:
        selected_feature = list(FEATURE_OPTIONS.keys())[0]

    if not selected_countries:
        selected_countries = ["Netherlands", "Germany", "France"]

    selected_countries = list(dict.fromkeys(selected_countries + ["Ireland"]))
    feature_info = FEATURE_OPTIONS[selected_feature]
    feature_label = feature_info["label"]
    unit = feature_info["unit"]
    chart_value_column = "chart_value"

    year_df = comparison_df[
        (comparison_df["year"] == selected_year) &
        (comparison_df["country"].isin(selected_countries))
    ].copy()
    trend_df = comparison_df[comparison_df["country"].isin(selected_countries)].copy()

    year_df = year_df.dropna(subset=[selected_feature])
    trend_df = trend_df.dropna(subset=[selected_feature])

    if year_df.empty:
        empty_fig = make_empty_figure("No complete data available for this selection.")
        return (
            "N/A", "N/A", "N/A", "N/A",
            empty_fig, empty_fig,
            f"{feature_info['ranking_title']} in {selected_year}?",
            feature_info["trend_title"],
            tufte_note("Data Availability", "No complete ranking data is available for the selected year and measure."),
            tufte_note("Data Availability", "No complete trend data is available for the selected measure.")
        )

    year_df[chart_value_column] = year_df[selected_feature].apply(lambda value: display_value(value, selected_feature))
    trend_df[chart_value_column] = trend_df[selected_feature].apply(lambda value: display_value(value, selected_feature))

    ireland_row = year_df[year_df["country"] == "Ireland"]
    ireland_value = None if ireland_row.empty else ireland_row[selected_feature].iloc[0]

    peer_df = year_df[year_df["country"] != "Ireland"]
    if peer_df.empty:
        comparison_average = None
        highest_peer_card = "N/A"
    else:
        comparison_average = peer_df[selected_feature].mean()
        highest_peer_row = peer_df.sort_values(selected_feature, ascending=False).iloc[0]
        highest_peer_card = f"{highest_peer_row['country']}: {format_value(highest_peer_row[selected_feature], selected_feature)}"

    gap_pct = None if ireland_value is None or comparison_average is None or pd.isna(comparison_average) or comparison_average == 0 else safe_pct_change(ireland_value, comparison_average)
    gap_card = "N/A" if gap_pct is None or pd.isna(gap_pct) else f"+{gap_pct:.1f}%" if gap_pct >= 0 else f"{gap_pct:.1f}%"

    # Ranking chart.
    year_df = year_df.sort_values(selected_feature, ascending=True)
    year_df["Display Value"] = year_df[selected_feature].apply(lambda value: format_chart_label(value, selected_feature))

    ranking_fig = px.bar(
        year_df,
        x=chart_value_column,
        y="country",
        orientation="h",
        text="Display Value",
        labels={chart_value_column: f"{feature_label} ({unit})", "country": ""}
    )
    ranking_fig.update_traces(
        marker_color=[COUNTRY_COLOR_MAP.get(country, PEER_COLOR) for country in year_df["country"]],
        textposition="outside",
        cliponaxis=False
    )
    apply_clean_layout(ranking_fig, height=520, margin=dict(t=20, b=50, l=105, r=120))
    ranking_fig.update_layout(showlegend=False)
    ranking_fig.update_xaxes(title=f"{feature_label} ({unit})")

    # Trend chart.
    trend_fig = px.line(
        trend_df,
        x="year",
        y=chart_value_column,
        color="country",
        markers=True,
        color_discrete_map=COUNTRY_COLOR_MAP,
        labels={"year": "Year", chart_value_column: f"{feature_label} ({unit})", "country": "Country"}
    )

    for trace in trend_fig.data:
        trace.line.color = COUNTRY_COLOR_MAP.get(trace.name, PEER_COLOR)
        trace.marker.color = COUNTRY_COLOR_MAP.get(trace.name, PEER_COLOR)
        if trace.name == "Ireland":
            trace.line.width = 4
            trace.marker.size = 8
            trace.opacity = 1
        else:
            trace.line.width = 2.5
            trace.marker.size = 6
            trace.opacity = 0.85

    apply_clean_layout(trend_fig, height=520, margin=dict(t=65, b=50, l=80, r=30), hovermode="closest", showlegend=True)
    trend_fig.update_layout(
        legend_title="Country",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    trend_fig.update_xaxes(tickmode="linear", dtick=1)
    trend_fig.update_yaxes(title=f"{feature_label} ({unit})")

    # Dynamic descriptions.
    # The ranking note answers "Where does Ireland stand now?"
    # The trend note answers "Is Ireland improving over time?"
    ranked_df = year_df.sort_values(selected_feature, ascending=False).reset_index(drop=True)
    top_row = ranked_df.iloc[0]
    top_country = top_row["country"]
    top_value = top_row[selected_feature]
    ireland_rank_row = ranked_df[ranked_df["country"] == "Ireland"]

    if ireland_rank_row.empty or ireland_value is None:
        ranking_description_text = (
            f"{top_country} had the highest {feature_label.lower()} among the selected countries in {selected_year}, "
            f"at {format_value(top_value, selected_feature)}. Ireland does not have complete data for this measure in the selected year."
        )
    else:
        ireland_rank = int(ireland_rank_row.index[0]) + 1
        countries_above_ireland = ranked_df.loc[
            ranked_df.index < ireland_rank - 1,
            "country"
        ].tolist()

        if countries_above_ireland:
            countries_above_text = ", ".join(countries_above_ireland)
            position_text = f"below {countries_above_text}"
        else:
            position_text = "at the top of the selected comparison group"

        ranking_description_text = (
            f"{top_country} had the highest {feature_label.lower()} among the selected countries in {selected_year}, "
            f"at {format_value(top_value, selected_feature)}. Ireland placed {ordinal(ireland_rank)}, "
            f"{position_text}, at {format_value(ireland_value, selected_feature)}. "
            "This chart is a current-scale comparison, not a full judgement of competitiveness."
        )

    ireland_trend = trend_df[trend_df["country"] == "Ireland"].sort_values("year")

    if ireland_trend.empty or len(ireland_trend) < 2:
        trend_description_text = f"Ireland does not have enough data to describe how {feature_label.lower()} changed over time."
    else:
        ireland_start_year = int(ireland_trend["year"].iloc[0])
        ireland_end_year = int(ireland_trend["year"].iloc[-1])
        ireland_start_value = ireland_trend[selected_feature].iloc[0]
        ireland_end_value = ireland_trend[selected_feature].iloc[-1]
        ireland_change_text = describe_change(ireland_start_value, ireland_end_value, selected_feature)

        # Compare Ireland's movement with the average movement of the selected peer group.
        peer_trend = trend_df[trend_df["country"] != "Ireland"].copy()
        peer_start_df = peer_trend[peer_trend["year"] == ireland_start_year]
        peer_end_df = peer_trend[peer_trend["year"] == ireland_end_year]

        if not peer_start_df.empty and not peer_end_df.empty:
            peer_start_avg = peer_start_df[selected_feature].mean()
            peer_end_avg = peer_end_df[selected_feature].mean()
            peer_change_text = describe_change(peer_start_avg, peer_end_avg, selected_feature)

            trend_description_text = (
                f"Ireland's {feature_label.lower()} {ireland_change_text} between {ireland_start_year} and {ireland_end_year}, "
                f"moving from {format_value(ireland_start_value, selected_feature)} to {format_value(ireland_end_value, selected_feature)}. "
                f"Over the same period, the selected peer average {peer_change_text}. "
                "This line chart is therefore used to read momentum, not only the latest ranking."
            )
        else:
            trend_description_text = (
                f"Ireland's {feature_label.lower()} {ireland_change_text} between {ireland_start_year} and {ireland_end_year}, "
                f"moving from {format_value(ireland_start_value, selected_feature)} to {format_value(ireland_end_value, selected_feature)}. "
                "This line chart is used to read Ireland's direction of travel rather than repeat the latest-year ranking."
            )

    return (
        format_value(ireland_value, selected_feature),
        highest_peer_card,
        format_value(comparison_average, selected_feature),
        gap_card,
        ranking_fig,
        trend_fig,
        f"{feature_info['ranking_title']} in {selected_year}?",
        feature_info["trend_title"],
        tufte_note("Current Position", ranking_description_text),
        tufte_note("Change Over Time", trend_description_text)
    )


@app.callback(
    Output("sentiment-breakdown-chart", "figure"),
    Output("sentiment-breakdown-note", "children"),
    Input("sentiment-breakdown-filter", "value")
)
def update_sentiment_breakdown(selected_breakdown):
    """Update the sentiment breakdown chart by country or perspective."""
    if selected_breakdown not in sentiment_df.columns:
        return (
            make_empty_figure("No sentiment breakdown data available."),
            tufte_note("Sentiment Evidence", "The selected sentiment breakdown is not available in the notebook output.")
        )

    breakdown_df = (
        sentiment_df
        .dropna(subset=[selected_breakdown, "sentiment_label"])
        .groupby([selected_breakdown, "sentiment_label"])
        .size()
        .reset_index(name="Posts")
    )

    if breakdown_df.empty:
        return (
            make_empty_figure("No sentiment breakdown data available."),
            tufte_note("Sentiment Evidence", "No sentiment data is available for this view.")
        )

    fig = px.bar(
        breakdown_df,
        x=selected_breakdown,
        y="Posts",
        color="sentiment_label",
        barmode="group",
        color_discrete_map=SENTIMENT_COLOR_MAP,
        category_orders={"sentiment_label": SENTIMENT_ORDER},
        labels={selected_breakdown: breakdown_label(selected_breakdown), "Posts": "Number of posts", "sentiment_label": "Sentiment"}
    )
    apply_clean_layout(fig, height=430, margin=dict(t=20, b=70, l=60, r=40), showlegend=True)
    fig.update_layout(legend_title="")
    fig.update_xaxes(title=breakdown_label(selected_breakdown), gridcolor="#ffffff")
    fig.update_yaxes(title="Number of posts")

    if selected_breakdown == "country":
        note_text = (
            "This view compares the tone of dairy discussion by country. A higher number of positive posts suggests a more favourable discussion tone, "
            "while a higher number of negative posts suggests more concern or criticism."
        )
    else:
        note_text = (
            "This view compares sentiment by perspective. It helps show whether consumer and producer discussions carry different tones in the sample."
        )

    return fig, tufte_note("Sentiment Evidence", note_text)


@app.callback(
    Output("recommendation-output", "children"),
    Input("decision-focus-filter", "value")
)
def update_recommendation_view(selected_focus):
    """Update the farmer recommendation panel."""
    if selected_focus not in RECOMMENDATION_DATA:
        selected_focus = "farm_productivity"
    return make_recommendation_view(selected_focus)

# Main app layout

app.layout = dbc.Container([
    html.H1(
        "Ireland Agricultural Decision Intelligence Dashboard",
        className="text-center my-4"
    ),

    html.H4(
        "Farmer-Focused Export Demand, Production Efficiency, Market Outlook and Decision Signals",
        className="text-center text-muted mb-3"
    ),

    html.P(
        f"Source: FAOSTAT / Eurostat / Reddit sentiment sample. Dashboard updated: {datetime.today().strftime('%d %b %Y')}.",
        className="text-center text-muted small mb-4"
    ),

    dbc.Tabs([
        build_overview_tab(),
        build_benchmarking_tab(),
        build_sentiment_tab(),
        build_action_plan_tab()
    ])

], fluid=False, style=PAGE_STYLE)


# Render will use: gunicorn app:server
# This block only runs when we test the app locally with: python app.py
if __name__ == "__main__":
    if app.layout is None:
        raise RuntimeError(
            "The Dash layout has not been created yet. "
            "Check that the app.layout section exists before this run block."
        )

    app.run(
        debug=True,
        dev_tools_ui=False,
        port=8051,
        use_reloader=False
    )






































































































































































