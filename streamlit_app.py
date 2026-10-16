from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- Page setup and look ----------
st.set_page_config(page_title="Retail Revenue Forecasting", page_icon="🛒", layout="wide")

PALETTE = ["#F97316", "#38BDF8", "#A78BFA", "#34D399", "#FBBF24", "#F472B6"]
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = PALETTE

st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    h1 {font-weight: 800; letter-spacing: -0.5px;}
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #111B2E, #16233B);
        border: 1px solid #243352;
        border-radius: 14px;
        padding: 14px 18px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
    }
    [data-testid="stMetricLabel"] {opacity: 0.75;}
    [data-testid="stMetricValue"] {color: #F97316;}
    button[data-baseweb="tab"] {font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


def show(fig):
    """Apply one consistent chart style and render."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text=""),
        font=dict(family="Source Sans Pro, sans-serif"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    st.plotly_chart(fig, width="stretch")


st.title("🛒 Retail Revenue Forecasting & Discount Analytics")
st.caption("Explore historical retail revenue, discount patterns, and short-horizon sales forecasts.")

# ---------- Data location (no upload; files ship with the repo) ----------
APP_DIR = Path(__file__).resolve().parent
DATA_DIRS = [APP_DIR / "data" / "raw", APP_DIR / "data", APP_DIR]


def find_sources():
    sources = {}
    for folder in DATA_DIRS:
        if folder.is_dir():
            for path in folder.glob("*.csv"):
                sources.setdefault(path.name.lower(), path)
    return sources


def read_csv(source, wanted_columns):
    header = pd.read_csv(source, nrows=0)
    actual = {column.lower(): column for column in header.columns}
    selected = [actual[name.lower()] for name in wanted_columns if name.lower() in actual]
    return pd.read_csv(source, usecols=selected, low_memory=False)


def forecast_weekly_sales(data, horizon=8):
    weekly = data.groupby("week_no")["sales_value"].sum().sort_index()
    first_week, last_week = int(weekly.index.min()), int(weekly.index.max())
    history = weekly.reindex(range(first_week, last_week + 1), fill_value=0).astype(float)
    if len(history) < 12:
        return None, None, "At least 12 weeks are needed for a useful forecast backtest."

    holdout_size = min(8, max(1, len(history) // 5))
    split_at = len(history) - holdout_size
    train = history.iloc[:split_at]
    actual = history.iloc[split_at:]
    moving_predictions = []
    moving_history = train.tolist()
    for actual_value in actual:
        moving_predictions.append(sum(moving_history[-4:]) / min(4, len(moving_history)))
        moving_history.append(float(actual_value))

    candidates = {"4-week moving average": moving_predictions}
    if len(train) >= 52:
        candidates["52-week seasonal naïve"] = [
            float(history.iloc[split_at + offset - 52]) for offset in range(holdout_size)
        ]
    scores = {
        name: sum(abs(float(actual.iloc[i]) - prediction) for i, prediction in enumerate(predictions)) / holdout_size
        for name, predictions in candidates.items()
    }
    model_name = min(scores, key=scores.get)
    selected = candidates[model_name]
    actual_values = actual.tolist()
    denominator = sum(abs(value) for value in actual_values)
    wmape = sum(abs(actual_values[i] - selected[i]) for i in range(holdout_size)) / denominator if denominator else 0
    backtest = pd.DataFrame({"week_no": actual.index, "actual_sales": actual_values, "forecast_sales": selected})

    if model_name == "52-week seasonal naïve":
        future_values = [float(history.iloc[len(history) + offset - 52]) for offset in range(horizon)]
    else:
        moving_history = history.tolist()
        future_values = []
        for _ in range(horizon):
            prediction = sum(moving_history[-4:]) / min(4, len(moving_history))
            future_values.append(prediction)
            moving_history.append(prediction)
    forecast = pd.DataFrame({"week_no": range(last_week + 1, last_week + horizon + 1), "forecast_sales": future_values})
    return forecast, backtest, {"model": model_name, "mae": scores[model_name], "wmape": wmape}


@st.cache_data(show_spinner="Reading transaction data…")
def load_data():
    sources = find_sources()
    if "transaction_data.csv" not in sources:
        searched = ", ".join(str(p) for p in DATA_DIRS)
        found = ", ".join(sorted(sources)) or "no CSV files"
        return None, {}, f"transaction_data.csv was not found. Searched: {searched}. Found: {found}."

    columns = ["household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "QUANTITY",
               "SALES_VALUE", "STORE_ID", "RETAIL_DISC", "WEEK_NO",
               "COUPON_DISC", "COUPON_MATCH_DISC"]
    data = read_csv(sources["transaction_data.csv"], columns)
    data.columns = [name.lower() for name in data.columns]
    required = {"sales_value", "quantity", "week_no", "product_id"}
    if not required.issubset(data.columns):
        return None, {}, "The transaction file is missing required fields."

    for name in ["retail_disc", "coupon_disc", "coupon_match_disc"]:
        if name not in data:
            data[name] = 0
        data[name] = pd.to_numeric(data[name], errors="coerce").fillna(0).abs()
    for name in ["sales_value", "quantity", "week_no", "product_id"]:
        data[name] = pd.to_numeric(data[name], errors="coerce")
    data = data.dropna(subset=["sales_value", "quantity", "week_no", "product_id"])
    data["discount_total"] = data[["retail_disc", "coupon_disc", "coupon_match_disc"]].sum(axis=1)
    data["gross_sales"] = data["sales_value"] + data["discount_total"]
    data["discounted"] = data["discount_total"] > 0

    optional = {}
    optional_columns = {
        "product.csv": ["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC"],
        "campaign_desc.csv": ["CAMPAIGN", "DESCRIPTION", "START_DAY", "END_DAY"],
        "campaign_table.csv": ["household_key", "CAMPAIGN"],
        "coupon_redempt.csv": ["household_key", "DAY", "COUPON_UPC", "CAMPAIGN"],
    }
    for filename, wanted in optional_columns.items():
        if filename in sources:
            optional[filename] = read_csv(sources[filename], wanted)

    # Keep campaign tables consistent with the sampled households in the sales data
    if "household_key" in data.columns:
        households = set(data["household_key"].unique())
        for filename in ["campaign_table.csv", "coupon_redempt.csv"]:
            if filename in optional:
                frame = optional[filename]
                key = next((c for c in frame.columns if c.lower() == "household_key"), None)
                if key:
                    optional[filename] = frame[frame[key].isin(households)]

    if "product.csv" in optional:
        products = optional["product.csv"]
        products.columns = [name.lower() for name in products.columns]
        category = "department" if "department" in products else "commodity_desc"
        products = products[["product_id", category]].drop_duplicates("product_id")
        data = data.merge(products, on="product_id", how="left")
        # Blank or missing names become "Unknown" (fixes the empty tag in the filter)
        cleaned = data[category].astype("string").str.strip()
        data[category] = cleaned.mask(cleaned.isna() | (cleaned == ""), "Unknown").astype(str)
        data = data.rename(columns={category: "category"})
    else:
        data["category"] = "All products (product.csv not supplied)"
    return data, optional, None


data, optional, error = load_data()
if error:
    st.error(error)
    st.caption("Add the CSV files to the data/raw folder of the repository and reboot the app.")
    st.stop()
if data.empty:
    st.error("No usable transaction rows were found.")
    st.stop()


@st.cache_data(max_entries=4, show_spinner=False)
def filtered_csv(week_range, categories):
    subset = data[data.week_no.between(*week_range) & data.category.astype(str).isin(categories)]
    return subset.to_csv(index=False).encode("utf-8")


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Filters")
    weeks = sorted(data.week_no.dropna().unique().tolist())
    week_range = st.slider("Transaction week", int(min(weeks)), int(max(weeks)), (int(min(weeks)), int(max(weeks))))
    category_options = sorted(data.category.dropna().astype(str).unique())
    picked = st.multiselect("Category", category_options, default=[], placeholder="All categories")
    selected_categories = picked if picked else category_options
    view = data[data.week_no.between(*week_range) & data.category.astype(str).isin(selected_categories)]
    st.caption(f"{len(selected_categories)} of {len(category_options)} categories included")
    st.divider()
    n_households = data["household_key"].nunique() if "household_key" in data.columns else None
    sample_note = f" Sample of {n_households:,} households." if n_households else ""
    st.caption(f"Source: dunnhumby, The Complete Journey.{sample_note} "
               "Figures describe this sample, not the full retailer.")

if view.empty:
    st.warning("No rows match these filters.")
    st.stop()

# ---------- KPIs ----------
net = view.sales_value.sum()
discounts = view.discount_total.sum()
gross = view.gross_sales.sum()
if "basket_id" in view.columns:
    volume_label, volume_value = "Baskets", view.basket_id.nunique()
else:
    volume_label, volume_value = "Line items", len(view)
kpis = st.columns(5)
kpis[0].metric("Net sales", f"${net:,.0f}")
kpis[1].metric("Recorded discounts", f"${discounts:,.0f}")
kpis[2].metric("Discount rate", f"{discounts / gross if gross else 0:.1%}")
kpis[3].metric(volume_label, f"{volume_value:,.0f}")
kpis[4].metric("Discounted line items", f"{view.discounted.mean():.1%}")
st.caption("Discount rate = recorded discounts ÷ (net sales + recorded discounts). Gross sales is derived, not a source field.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Forecast", "Discounts", "Categories", "Campaigns"])
with tab1:
    weekly = view.groupby("week_no", as_index=False).agg(
        net_sales=("sales_value", "sum"), discounts=("discount_total", "sum"))
    weekly = weekly.rename(columns={"net_sales": "Net sales", "discounts": "Discounts"})
    show(px.line(weekly, x="week_no", y=["Net sales", "Discounts"], markers=True,
                 title="Weekly net sales and recorded discounts",
                 labels={"week_no": "Week", "value": "Dollars ($)"}))
    st.caption("Weeks are a relative index in the source data, not calendar dates. "
               "Sales climb steeply in the earliest weeks; read that as how this dataset starts, not as a trend to extrapolate.")
    with st.expander("Data quality notes"):
        q = view["quantity"]
        notes = pd.DataFrame({
            "Measure": ["Line items", "Median quantity per line", "Mean quantity per line",
                        "99th percentile quantity", "Maximum quantity", "Lines with negative sales (returns)"],
            "Value": [f"{len(view):,}", f"{q.median():,.1f}", f"{q.mean():,.1f}",
                      f"{q.quantile(0.99):,.1f}", f"{q.max():,.0f}", f"{(view.sales_value < 0).mean():.2%}"],
        })
        st.dataframe(notes, hide_index=True, width="stretch")
        st.caption("Total quantity is not shown as a headline figure: in this dataset it is heavily skewed, "
                   "so a raw sum is not a reliable unit count.")
with tab2:
    forecast, backtest, result = forecast_weekly_sales(view)
    if forecast is None:
        st.info(result)
    else:
        score1, score2, score3 = st.columns(3)
        score1.metric("Selected baseline", result["model"])
        score2.metric("Holdout MAE", f"${result['mae']:,.0f}")
        score3.metric("Holdout WMAPE", f"{result['wmape']:.1%}")
        history = view.groupby("week_no", as_index=False)["sales_value"].sum().rename(columns={"sales_value": "sales"})
        backtest_chart = backtest.melt(id_vars="week_no", value_vars=["actual_sales", "forecast_sales"],
                                       var_name="series", value_name="sales")
        backtest_chart["series"] = backtest_chart["series"].map({"actual_sales": "Actual", "forecast_sales": "Backtest forecast"})
        future_chart = forecast.rename(columns={"forecast_sales": "sales"}).assign(series="Forecast")
        historical_chart = history.assign(series="Historical sales")
        chart_data = pd.concat([historical_chart, backtest_chart, future_chart], ignore_index=True)
        show(px.line(chart_data, x="week_no", y="sales", color="series", markers=True,
                     title="Historical sales, holdout backtest, and 8-week forecast",
                     labels={"week_no": "Week", "sales": "Sales ($)"}))
        forecast_table = forecast.rename(columns={"week_no": "Week", "forecast_sales": "Forecast sales ($)"})
        st.dataframe(forecast_table, width="stretch", hide_index=True)
        st.caption("The app selects between a 4-week moving average and a 52-week seasonal naïve baseline using holdout MAE. "
                   "A moving-average forecast is flat by design and shows no trend or seasonality. "
                   "This forecast is an estimate, not a guaranteed outcome.")
with tab3:
    amounts = pd.DataFrame({"Discount type": ["Retailer", "Coupon", "Retailer coupon match"],
        "Amount": [view.retail_disc.sum(), view.coupon_disc.sum(), view.coupon_match_disc.sum()]})
    show(px.bar(amounts, x="Discount type", y="Amount", color="Discount type",
                title="Recorded discount amounts", labels={"Amount": "Amount ($)"}))
    by_status = view.groupby("discounted", as_index=False).agg(net_sales=("sales_value", "sum"))
    by_status["Status"] = by_status.discounted.map({True: "Discount recorded", False: "No discount"})
    show(px.bar(by_status, x="Status", y="net_sales", color="Status",
                title="Net sales by observed discount status", labels={"net_sales": "Net sales ($)"}))
with tab4:
    summary = view.groupby("category", as_index=False).agg(
        net_sales=("sales_value", "sum"), units=("quantity", "sum"),
        discounts=("discount_total", "sum"), line_items=("sales_value", "size"))
    summary = summary.sort_values("net_sales", ascending=False)
    top = summary.head(20).sort_values("net_sales")
    show(px.bar(top, x="net_sales", y="category", orientation="h",
                title="Top categories by net sales", labels={"net_sales": "Net sales ($)", "category": ""}))
    table = summary.rename(columns={"category": "Category", "net_sales": "Net sales ($)",
                                    "units": "Quantity (raw)", "discounts": "Discounts ($)",
                                    "line_items": "Line items"})
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption("Quantity is the raw source field and may mix units, weights and bulk codes; compare categories by sales, not quantity.")
with tab5:
    redemptions = optional.get("coupon_redempt.csv")
    campaigns = optional.get("campaign_table.csv")
    descriptions = optional.get("campaign_desc.csv")
    if redemptions is None:
        st.info("Add coupon_redempt.csv and campaign_table.csv to data/raw for reach and redemption counts.")
    else:
        redemptions.columns = [name.lower() for name in redemptions.columns]
        a, b = st.columns(2)
        a.metric("Coupon redemption events", f"{len(redemptions):,}")
        b.metric("Households with a redemption", f"{redemptions.household_key.nunique():,}")
        if campaigns is not None:
            campaigns.columns = [name.lower() for name in campaigns.columns]
            reached = campaigns.groupby("campaign", as_index=False).household_key.nunique().rename(columns={"household_key": "households_reached"})
            redeemed = redemptions.groupby("campaign", as_index=False).size().rename(columns={"size": "redemption_events"})
            summary = reached.merge(redeemed, on="campaign", how="left").fillna(0)
            summary["redemption_rate"] = summary.redemption_events / summary.households_reached
            if descriptions is not None:
                descriptions.columns = [name.lower() for name in descriptions.columns]
                summary = summary.merge(descriptions, on="campaign", how="left")
            st.dataframe(summary.sort_values("redemption_rate", ascending=False), width="stretch", hide_index=True)
            st.caption("Descriptive event rate for the sampled households; a household may redeem more than once. "
                       "Campaign figures ignore the week and category filters.")

st.download_button("Download filtered transactions", filtered_csv(week_range, tuple(selected_categories)),
                   "filtered_transactions.csv", "text/csv")