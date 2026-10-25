"""
Practical 10 - Sales Analytics Web Application (Streamlit)

Features
1. Upload + validation (CSV / Excel / JSON)
2. Preprocessing (missing values, duplicates, inconsistent text, dates)
3. Reusable statistical-summary functions
4. Interactive sidebar filters + 5 live KPIs
5. Interactive Plotly charts, tabbed layout, error handling

Run locally:  streamlit run app.py
"""

import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
SAMPLE_FILE = "exp10.csv"
REQUIRED_COLUMNS = ["Order ID", "Order Date", "Sales", "Quantity", "Profit"]
PLACEHOLDERS = ["", "?", "unknown", "not available", "n/a", "na", "nan",
                "none", "null", "-", "not-a-date"]
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
# lower-case / no-space key  ->  canonical label
SYNONYMS = {
    "sameday": "Same Day",
    "standardclass": "Standard Class",
    "firstclass": "First Class",
    "secondclass": "Second Class",
    "officesupply": "Office Supplies",
    "officesupplies": "Office Supplies",
    "unknownsegment": "Unknown",
}
TEXT_COLUMNS = ["Ship Mode", "Segment", "Country", "City", "State", "Region",
                "Category", "Sub-Category", "Product Name"]
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y", "%d-%m-%Y",
                "%m/%d/%Y", "%Y/%m/%d", "%d-%b-%y", "%b %d, %Y"]
MAX_REASONABLE_QTY = 1000

st.set_page_config(page_title="Sales Analytics Dashboard", page_icon="📊",
                   layout="wide")


# ----------------------------------------------------------------------------
# 1. LOADING + VALIDATION
# ----------------------------------------------------------------------------
def load_data(source) -> pd.DataFrame:
    """Read CSV / XLSX / JSON from an uploaded file object or a file path."""
    name = source if isinstance(source, str) else source.name
    ext = os.path.splitext(name)[1].lower()

    if ext == ".csv":
        try:
            return pd.read_csv(source, dtype=str)
        except UnicodeDecodeError:
            if not isinstance(source, str):
                source.seek(0)
            return pd.read_csv(source, dtype=str, encoding="latin1")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(source, dtype=str)
    if ext == ".json":
        try:
            return pd.read_json(source, dtype=False)
        except ValueError:  # maybe JSON-lines
            if not isinstance(source, str):
                source.seek(0)
            return pd.read_json(source, lines=True, dtype=False)
    raise ValueError(f"Unsupported file type '{ext}'. Use .csv, .xlsx or .json")


def standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip spaces and match column names to the expected ones (case-insensitive)."""
    expected = {c.lower(): c for c in REQUIRED_COLUMNS + TEXT_COLUMNS +
                ["Ship Date", "Discount", "Customer ID", "Product ID"]}
    df = df.copy()
    df.columns = [expected.get(str(c).strip().lower(), str(c).strip())
                  for c in df.columns]
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """Return dimensions, column info, missing values and duplicate count."""
    info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(t) for t in df.dtypes],
        "Missing": df.isna().sum().values,
        "Missing %": (df.isna().mean() * 100).round(2).values,
        "Unique Values": df.nunique().values,
    })
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "info": info,
        "total_missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "missing_required": [c for c in REQUIRED_COLUMNS if c not in df.columns],
    }


# ----------------------------------------------------------------------------
# 2. PREPROCESSING (each step is a small reusable function)
# ----------------------------------------------------------------------------
def replace_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and turn junk tokens ('?', 'unknown', ...) into NaN."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith(("str", "string")):
            s = df[col].astype("object").where(df[col].notna(), None)
            s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
            s = s.map(lambda v: np.nan if isinstance(v, str)
                      and v.lower() in PLACEHOLDERS else v)
            df[col] = s
    return df


def standardise_text(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Fix inconsistent casing / spelling: 'consumer' -> 'Consumer', 'SameDay' -> 'Same Day'."""
    df = df.copy()
    fixed = 0
    for col in [c for c in TEXT_COLUMNS if c in df.columns]:
        before = df[col].copy()

        def _fix(v):
            if not isinstance(v, str):
                return v
            key = re.sub(r"\s+", "", v.lower())
            return SYNONYMS.get(key, " ".join(v.split()).title())

        df[col] = df[col].map(_fix)
        fixed += int((before.fillna("") != df[col].fillna("")).sum())
    return df, fixed


def to_number(value):
    """'five' -> 5, 'ten percent' -> 0.1, '12.5' -> 12.5, junk -> NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip().lower().replace(",", "")
    is_percent = "percent" in text or text.endswith("%")
    text = text.replace("percent", "").replace("%", "").strip()
    if text in NUMBER_WORDS:
        number = float(NUMBER_WORDS[text])
    else:
        try:
            number = float(text)
        except ValueError:
            return np.nan
    if not np.isfinite(number):  # 'inf', 'nan' strings
        return np.nan
    return number / 100 if is_percent else number


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse a column that mixes several date formats into one datetime column."""
    series = series.astype("object")
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:
        remaining = result.isna() & series.notna()
        if not remaining.any():
            break
        parsed = pd.to_datetime(series[remaining], format=fmt, errors="coerce")
        result.loc[remaining] = parsed
    return result


def clean_data(df: pd.DataFrame, fill_strategy: str = "Median") -> Tuple[pd.DataFrame, List[str]]:
    """Full cleaning pipeline. Returns (clean_df, human-readable log)."""
    log: List[str] = []
    df = replace_placeholders(df)
    start = len(df)

    # --- duplicates
    dups = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    log.append(f"Removed {dups} duplicate rows.")

    # --- inconsistent text
    df, n_fixed = standardise_text(df)
    log.append(f"Standardised {n_fixed} inconsistent text values "
               "(case, spelling, e.g. 'SameDay' -> 'Same Day').")

    # --- numeric columns
    for col in ["Sales", "Quantity", "Profit", "Discount"]:
        if col in df.columns:
            df[col] = df[col].map(to_number)

    bad_qty = (df["Quantity"] <= 0) | (df["Quantity"] > MAX_REASONABLE_QTY)
    log.append(f"Marked {int(bad_qty.sum())} invalid quantities "
               f"(<=0 or >{MAX_REASONABLE_QTY}) as missing.")
    df.loc[bad_qty, "Quantity"] = np.nan

    if "Discount" in df.columns:
        bad_disc = (df["Discount"] < 0) | (df["Discount"] > 1)
        log.append(f"Marked {int(bad_disc.sum())} invalid discounts (outside 0-1) as missing.")
        df.loc[bad_disc, "Discount"] = np.nan

    # negative sales and extreme outliers (e.g. 9,999,999.99 typo) are treated as invalid
    q1, q3 = df["Sales"].quantile([0.25, 0.75])
    upper = q3 + 20 * (q3 - q1)
    bad_sales = (df["Sales"] < 0) | (df["Sales"] > upper)
    log.append(f"Marked {int(bad_sales.sum())} invalid Sales values (negative or extreme "
               f"outliers > {upper:,.0f}) as missing.")
    df.loc[bad_sales, "Sales"] = np.nan

    # --- dates
    for col in ["Order Date", "Ship Date"]:
        if col in df.columns:
            df[col] = parse_dates(df[col])
    if "Ship Date" in df.columns:
        bad_ship = df["Ship Date"] < df["Order Date"]
        df.loc[bad_ship, "Ship Date"] = pd.NaT
        log.append(f"Cleared {int(bad_ship.sum())} ship dates earlier than order dates.")
    log.append("Converted Order Date / Ship Date to datetime (mixed formats handled).")

    # --- rows we cannot repair: no ID, no date, no Sales/Profit
    before = len(df)
    df = df.dropna(subset=["Order ID", "Order Date", "Sales", "Profit"]).reset_index(drop=True)
    log.append(f"Dropped {before - len(df)} rows missing Order ID, Order Date, Sales or Profit "
               "(money values are never guessed).")

    # --- fill the remaining numeric gaps
    for col in [c for c in ["Quantity", "Discount"] if c in df.columns]:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        if fill_strategy == "Drop rows":
            df = df.dropna(subset=[col]).reset_index(drop=True)
            log.append(f"Dropped {n_missing} rows with missing {col}.")
        else:
            fill = {"Median": df[col].median(), "Mean": df[col].mean(), "Zero": 0}[fill_strategy]
            if col == "Quantity":
                fill = round(fill)
            df[col] = df[col].fillna(fill)
            log.append(f"Filled {n_missing} missing {col} values with {fill_strategy.lower()} ({fill:.2f}).")

    # --- fill remaining text gaps
    text_cols = [c for c in df.columns
                 if not pd.api.types.is_numeric_dtype(df[c])
                 and not pd.api.types.is_datetime64_any_dtype(df[c])]
    for col in text_cols:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            df[col] = df[col].fillna("Unknown")
            log.append(f"Filled {n_missing} missing text values in '{col}' with 'Unknown'.")

    df["Quantity"] = df["Quantity"].astype(int)

    # --- derived columns
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    if "Ship Date" in df.columns:
        df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    log.append(f"Final dataset: {len(df)} rows (from {start}).")
    return df, log


# ----------------------------------------------------------------------------
# 3. STATISTICS (reusable)
# ----------------------------------------------------------------------------
def numeric_summary(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Mean, median, std, variance, min, max, IQR, skewness for numeric columns."""
    out = df[cols].describe().T
    out["median"] = df[cols].median()
    out["variance"] = df[cols].var()
    out["IQR"] = out["75%"] - out["25%"]
    out["skewness"] = df[cols].skew()
    return out[["count", "mean", "median", "std", "variance", "min", "25%", "75%",
                "max", "IQR", "skewness"]].round(2)


def categorical_summary(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Count, share of orders, sales and profit for each category value."""
    g = df.groupby(col).agg(Rows=("Order ID", "size"), Sales=("Sales", "sum"),
                            Profit=("Profit", "sum")).reset_index()
    g["Profit Margin (%)"] = (g["Profit"] / g["Sales"] * 100).round(2)
    return g.sort_values("Sales", ascending=False)


def calculate_kpis(df: pd.DataFrame) -> dict:
    """The five required KPIs."""
    total_sales = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    return {
        "Total Sales": total_sales,
        "Total Profit": total_profit,
        "Total Orders": df["Order ID"].nunique(),
        "Total Quantity": int(df["Quantity"].sum()),
        "Profit Margin (%)": (total_profit / total_sales * 100) if total_sales else 0.0,
    }


# ----------------------------------------------------------------------------
# 4. FILTERS
# ----------------------------------------------------------------------------
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Draw the sidebar filters and return the filtered dataframe."""
    st.sidebar.header("🔎 Filters")
    if st.sidebar.button("↺ Reset all filters", width="stretch"):
        for key in [k for k in st.session_state if str(k).startswith("f_")]:
            del st.session_state[key]
        st.rerun()

    min_d, max_d = df["Order Date"].min().date(), df["Order Date"].max().date()
    date_range = st.sidebar.date_input("Order date range", (min_d, max_d), key="f_date",
                                       min_value=min_d, max_value=max_d)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
    else:
        start, end = min_d, max_d

    filtered = df[(df["Order Date"].dt.date >= start) & (df["Order Date"].dt.date <= end)]

    for col in ["Region", "Category", "Segment", "Ship Mode", "State"]:
        if col in filtered.columns:
            options = sorted(df[col].unique())
            chosen = st.sidebar.multiselect(col, options, default=options, key=f"f_{col}")
            filtered = filtered[filtered[col].isin(chosen)]
    return filtered


# ----------------------------------------------------------------------------
# 5. CHARTS
# ----------------------------------------------------------------------------
PALETTES = {
    "Ocean": ["#22d3ee", "#6366f1", "#f59e0b", "#10b981", "#f43f5e", "#a78bfa", "#38bdf8"],
    "Sunset": ["#fb7185", "#f59e0b", "#a855f7", "#f97316", "#facc15", "#ec4899", "#fb923c"],
    "Forest": ["#34d399", "#a3e635", "#2dd4bf", "#fbbf24", "#60a5fa", "#86efac", "#14b8a6"],
}
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
.hero { padding: 26px 30px; border-radius: 18px; margin-bottom: 18px;
  background: linear-gradient(120deg,#0e7490 0%,#4338ca 55%,#be185d 100%); }
.hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; color: #fff; }
.hero p { margin: 4px 0 0; color: #e0e7ff; }
.kpi { background: #111a2e; border: 1px solid #1f2a44; border-left: 5px solid var(--c);
  border-radius: 14px; padding: 14px 16px; transition: transform .2s, box-shadow .2s; }
.kpi:hover { transform: translateY(-4px); box-shadow: 0 10px 24px rgba(0,0,0,.45); }
.kpi .l { font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; color: #94a3b8; }
.kpi .v { font-size: 1.75rem; font-weight: 800; color: #fff; }
.kpi .d { font-size: .8rem; color: var(--c); }
.insight { background: #111a2e; border: 1px solid #1f2a44; border-radius: 12px;
  padding: 10px 14px; margin-bottom: 8px; }
</style>
"""


def style(fig, height=390):
    """Apply the shared dark, transparent look to every chart."""
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=height,
                      margin=dict(l=10, r=10, t=55, b=10),
                      font=dict(family="Manrope"), legend_title_text="")
    return fig


def kpi_card(label, value, note, color):
    st.markdown(f'<div class="kpi" style="--c:{color}"><div class="l">{label}</div>'
                f'<div class="v">{value}</div><div class="d">{note}</div></div>',
                unsafe_allow_html=True)


def chart_trend(df, metric, grain, kind, show_avg):
    d = df.copy()
    d["Period"] = d["Order Date"].dt.to_period("M" if grain == "Month" else "Q").dt.to_timestamp()
    g = d.groupby("Period")[metric].sum().reset_index()
    fig = {"Line": lambda: px.line(g, x="Period", y=metric, markers=True),
           "Area": lambda: px.area(g, x="Period", y=metric),
           "Bar": lambda: px.bar(g, x="Period", y=metric)}[kind]()
    if show_avg:
        g["avg"] = g[metric].rolling(3, min_periods=1).mean()
        fig.add_scatter(x=g["Period"], y=g["avg"], name="3-period average",
                        line=dict(dash="dash", color="#f59e0b"))
    fig.update_layout(title=f"{metric} by {grain}", hovermode="x unified")
    return style(fig)


def chart_yoy(df):
    d = df.copy()
    d["Mon"] = d["Order Date"].dt.month
    g = d.groupby(["Year", "Mon"])["Sales"].sum().reset_index()
    g["Cumulative Sales"] = g.groupby("Year")["Sales"].cumsum()
    g["Year"] = g["Year"].astype(str)
    fig = px.line(g, x="Mon", y="Cumulative Sales", color="Year", markers=True,
                  title="Cumulative Sales - Year over Year")
    fig.update_xaxes(dtick=1, title="Month")
    return style(fig)


def chart_treemap(df):
    g = df.groupby(["Category", "Sub-Category"]).agg(Sales=("Sales", "sum"),
                                                     Profit=("Profit", "sum")).reset_index()
    g["Margin %"] = (g["Profit"] / g["Sales"] * 100).round(1)
    fig = px.treemap(g, path=["Category", "Sub-Category"], values="Sales", color="Margin %",
                     color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                     title="Sales Mix (size = sales, colour = profit margin %)")
    return style(fig, 430)


def chart_heatmap(df):
    g = df.groupby(["Region", "Category"]).agg(S=("Sales", "sum"), P=("Profit", "sum")).reset_index()
    g["Margin %"] = g["P"] / g["S"] * 100
    p = g.pivot(index="Region", columns="Category", values="Margin %")
    fig = px.imshow(p, text_auto=".1f", color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0, aspect="auto",
                    title="Profit Margin % - Region x Category")
    return style(fig)


def chart_pareto(df, n):
    g = df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).reset_index()
    g["Cum %"] = g["Sales"].cumsum() / g["Sales"].sum() * 100
    g = g.head(n)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=g["Product Name"], y=g["Sales"], name="Sales", marker_color="#22d3ee")
    fig.add_scatter(x=g["Product Name"], y=g["Cum %"], name="Cumulative %", mode="lines+markers",
                    line=dict(color="#f59e0b"), secondary_y=True)
    fig.update_yaxes(range=[0, 105], secondary_y=True, title="Cumulative %")
    fig.update_layout(title=f"Pareto - Top {n} Products")
    return style(fig)


def chart_waterfall(df):
    g = df.groupby("Sub-Category")["Profit"].sum().sort_values(ascending=False)
    fig = go.Figure(go.Waterfall(x=list(g.index) + ["TOTAL"], y=list(g.values) + [0],
                                 measure=["relative"] * len(g) + ["total"],
                                 increasing=dict(marker_color="#10b981"),
                                 decreasing=dict(marker_color="#f43f5e"),
                                 totals=dict(marker_color="#6366f1")))
    fig.update_layout(title="Profit Contribution by Sub-Category (Waterfall)")
    return style(fig)


def chart_discount(df):
    g = df.assign(Discount=(df["Discount"] * 100).round().astype(int).astype(str) + "%") \
          .groupby("Discount")["Profit"].agg(["mean", "size"]).reset_index()
    g = g.sort_values("Discount", key=lambda s: s.str.rstrip("%").astype(int))
    fig = px.bar(g, x="Discount", y="mean", color="mean", text="size",
                 color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                 labels={"mean": "Avg Profit per row"},
                 title="Does Discounting Hurt Profit? (labels = # rows)")
    return style(fig)


def chart_scatter(df):
    fig = px.scatter(df, x="Sales", y="Profit", color="Category", size="Quantity",
                     hover_data=["Order ID", "Product Name", "Discount"], opacity=.8,
                     title="Sales vs Profit (bubble = quantity; below 0 = loss)")
    fig.add_hline(y=0, line_dash="dot", line_color="#f43f5e")
    return style(fig, 430)


def chart_ship(df):
    d = df.dropna(subset=["Shipping Days"])
    return style(px.box(d, x="Ship Mode", y="Shipping Days", color="Ship Mode", points="all",
                        title="Delivery Time by Ship Mode (days)"))


def chart_dimension(df, dim, metric, n):
    g = df.groupby(dim)[metric].sum().nlargest(n).reset_index()
    bar = px.bar(g, x=metric, y=dim, orientation="h", color=metric,
                 color_continuous_scale="Tealgrn", title=f"{metric} by {dim}")
    bar.update_yaxes(autorange="reversed")
    donut = px.pie(g[g[metric] > 0], names=dim, values=metric, hole=.55,
                   title=f"{dim} share of {metric}")
    return style(bar), style(donut)


def build_insights(df):
    """Auto-generated, data-driven talking points for the filtered data."""
    out = []
    cat = df.groupby("Category")["Sales"].sum()
    out.append(f"🏆 **{cat.idxmax()}** is the top category with ${cat.max():,.0f} "
               f"({cat.max() / cat.sum():.0%} of sales).")
    sub = df.groupby("Sub-Category").agg(S=("Sales", "sum"), P=("Profit", "sum"))
    sub["m"] = sub["P"] / sub["S"] * 100
    out.append(f"⚠️ Weakest margin: **{sub['m'].idxmin()}** at {sub['m'].min():.1f}%; "
               f"best: **{sub['m'].idxmax()}** at {sub['m'].max():.1f}%.")
    loss = (df["Profit"] < 0).mean()
    out.append(f"📉 **{loss:.0%}** of rows are loss-making, totalling "
               f"${df.loc[df['Profit'] < 0, 'Profit'].sum():,.0f}.")
    month = df.groupby("Month")["Sales"].sum()
    out.append(f"📅 Best month: **{month.idxmax():%b %Y}** (${month.max():,.0f}).")
    corr = df["Discount"].corr(df["Profit"])
    if pd.notna(corr):
        out.append(f"🏷️ Discount vs profit correlation is **{corr:.2f}** "
                   f"({'higher discounts tend to reduce profit' if corr < -0.1 else 'weak link'}).")
    return out


def render_dashboard(f, full):
    """Dashboard tab: animated KPI cards + 5 analytical sub-tabs."""
    st.sidebar.header("🎨 Display")
    px.defaults.color_discrete_sequence = PALETTES[st.sidebar.selectbox("Colour palette", list(PALETTES))]
    top_n = st.sidebar.slider("Top-N items", 3, 15, 8)

    k, base = calculate_kpis(f), calculate_kpis(full)
    pct = lambda a, b: f"{a / b:.0%} of all data" if b else ""
    c = st.columns(5)
    with c[0]: kpi_card("💰 Total Sales", f"${k['Total Sales']:,.0f}", pct(k["Total Sales"], base["Total Sales"]), "#22d3ee")
    with c[1]: kpi_card("📈 Total Profit", f"${k['Total Profit']:,.0f}", pct(k["Total Profit"], base["Total Profit"]), "#10b981")
    with c[2]: kpi_card("🧾 Total Orders", f"{k['Total Orders']:,}", pct(k["Total Orders"], base["Total Orders"]), "#a78bfa")
    with c[3]: kpi_card("📦 Quantity Sold", f"{k['Total Quantity']:,}", pct(k["Total Quantity"], base["Total Quantity"]), "#f59e0b")
    diff = k["Profit Margin (%)"] - base["Profit Margin (%)"]
    with c[4]: kpi_card("% Profit Margin", f"{k['Profit Margin (%)']:.2f}%",
                        f"{'▲' if diff >= 0 else '▼'} {abs(diff):.2f} pp vs overall", "#f43f5e")
    st.write("")

    with st.expander("💡 Auto-generated insights", expanded=True):
        for line in build_insights(f):
            st.markdown(f'<div class="insight">{line}</div>', unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs(["📈 Trends", "🛍️ Products", "🗺️ Regions & Segments",
                                  "💹 Profitability", "🔬 Deep Dive"])
    with t1:
        a, b, c2, d = st.columns(4)
        metric = a.selectbox("Metric", ["Sales", "Profit", "Quantity"])
        grain = b.radio("Group by", ["Month", "Quarter"], horizontal=True)
        kind = c2.radio("Chart type", ["Line", "Area", "Bar"], horizontal=True)
        avg = d.toggle("Show moving average", True)
        st.plotly_chart(chart_trend(f, metric, grain, kind, avg), width="stretch")
        st.plotly_chart(chart_yoy(f), width="stretch")
    with t2:
        st.plotly_chart(chart_treemap(f), width="stretch")
        st.plotly_chart(chart_pareto(f, top_n), width="stretch")
    with t3:
        dim = st.radio("Analyse by", ["Region", "State", "Segment", "Category", "Ship Mode", "City"],
                       horizontal=True)
        m = st.radio("Measure", ["Sales", "Profit", "Quantity"], horizontal=True)
        bar, donut = chart_dimension(f, dim, m, top_n)
        l, r = st.columns([3, 2])
        l.plotly_chart(bar, width="stretch")
        r.plotly_chart(donut, width="stretch")
        st.plotly_chart(chart_heatmap(f), width="stretch")
    with t4:
        st.plotly_chart(chart_waterfall(f), width="stretch")
        l, r = st.columns(2)
        l.plotly_chart(chart_discount(f), width="stretch")
        r.plotly_chart(chart_ship(f), width="stretch")
    with t5:
        st.plotly_chart(chart_scatter(f), width="stretch")
        st.subheader("🔻 Loss-making orders (worst first)")
        st.dataframe(f[f["Profit"] < 0].nsmallest(15, "Profit")[
            ["Order ID", "Order Date", "Product Name", "Region", "Sales", "Discount", "Profit"]],
            width="stretch", hide_index=True)
        st.download_button("⬇️ Download filtered CSV", f.to_csv(index=False).encode("utf-8"),
                           "filtered_data.csv", "text/csv")


# ----------------------------------------------------------------------------
# 6. UI
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_clean(df: pd.DataFrame, strategy: str):
    return clean_data(df, strategy)


def get_raw_dataframe():
    """Sidebar upload widget (+ optional bundled sample). Returns (df, name) or (None, None)."""
    st.sidebar.header("📁 Data")
    uploaded = st.sidebar.file_uploader("Upload CSV, Excel or JSON",
                                        type=["csv", "xlsx", "xls", "json"])
    use_sample = False
    if os.path.exists(SAMPLE_FILE):
        use_sample = st.sidebar.checkbox(f"Use sample dataset ({SAMPLE_FILE})",
                                         value=uploaded is None)
    try:
        if uploaded is not None:
            return load_data(uploaded), uploaded.name
        if use_sample:
            return load_data(SAMPLE_FILE), SAMPLE_FILE
    except Exception as exc:  # corrupt file, wrong format, missing openpyxl, ...
        st.error(f"❌ Could not read the file: {exc}")
    return None, None


def show_validation(raw: pd.DataFrame, report: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", report["rows"])
    c2.metric("Columns", report["cols"])
    c3.metric("Missing values", report["total_missing"])
    c4.metric("Duplicate rows", report["duplicates"])
    st.subheader("Columns, data types & missing values")
    st.dataframe(report["info"], width="stretch", hide_index=True)
    st.subheader("Preview (first 10 rows)")
    st.dataframe(raw.head(10), width="stretch")
    if report["duplicates"]:
        with st.expander("Show duplicate records"):
            st.dataframe(raw[raw.duplicated(keep=False)], width="stretch")


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>📊 Sales Analytics Dashboard</h1>'
                '<p>Upload → Validate → Clean → Filter → Explore</p></div>',
                unsafe_allow_html=True)

    raw, name = get_raw_dataframe()
    if raw is None:
        st.info("👈 Upload a CSV, Excel or JSON file in the sidebar to begin.")
        st.stop()

    raw = standardise_column_names(raw)
    report = validate_data(raw)
    if report["missing_required"]:
        st.error(f"Missing required column(s): {', '.join(report['missing_required'])}")
        st.stop()

    st.sidebar.header("🧹 Cleaning")
    strategy = st.sidebar.selectbox("Fill missing Quantity / Discount with",
                                    ["Median", "Mean", "Zero", "Drop rows"])
    try:
        df, log = cached_clean(raw, strategy)
    except Exception as exc:
        st.error(f"❌ Cleaning failed: {exc}")
        st.stop()
    if df.empty:
        st.warning("No rows left after cleaning.")
        st.stop()

    filtered = apply_filters(df)

    tab_data, tab_clean, tab_stats, tab_dash = st.tabs(
        ["1️⃣ Data Validation", "2️⃣ Cleaning", "3️⃣ Statistics", "4️⃣ Dashboard"])

    with tab_data:
        st.success(f"Loaded **{name}**")
        show_validation(raw, report)

    with tab_clean:
        st.subheader("Cleaning log")
        for line in log:
            st.write("✅ " + line)
        st.subheader("Cleaned data validation")
        after = validate_data(df)
        a1, a2, a3 = st.columns(3)
        a1.metric("Rows", after["rows"])
        a2.metric("Missing values", after["total_missing"])
        a3.metric("Duplicates", after["duplicates"])
        st.dataframe(df.head(20), width="stretch")
        st.download_button("⬇️ Download cleaned CSV",
                           df.to_csv(index=False).encode("utf-8"),
                           "cleaned_data.csv", "text/csv")

    with tab_stats:
        st.subheader("Numeric summary")
        st.dataframe(numeric_summary(df, ["Sales", "Quantity", "Profit", "Discount"]),
                     width="stretch")
        col = st.selectbox("Group by", ["Category", "Sub-Category", "Region", "Segment",
                                        "Ship Mode", "State"])
        st.dataframe(categorical_summary(df, col), width="stretch", hide_index=True)
        st.subheader("Correlation")
        st.plotly_chart(px.imshow(df[["Sales", "Quantity", "Profit", "Discount"]].corr().round(2),
                                  text_auto=True, color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1), width="stretch")

    with tab_dash:
        if filtered.empty:
            st.warning("No data matches the current filters. Adjust them in the sidebar.")
        else:
            st.caption(f"Showing {len(filtered):,} of {len(df):,} cleaned rows")
            render_dashboard(filtered, df)


if __name__ == "__main__":
    main()