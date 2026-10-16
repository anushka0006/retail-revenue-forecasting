from pathlib import Path
from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Retail Revenue Forecasting", page_icon="🛒", layout="wide")
st.title("Retail Revenue Forecasting & Discount Analytics")
st.caption("Explore historical retail revenue, discount patterns, and short-horizon sales forecasts.")

with st.sidebar:
    st.header("Load dataset")
    uploads = st.file_uploader(
        "Upload a ZIP of dunnhumby CSVs, or select CSV files",
        type=["zip", "csv"], accept_multiple_files=True,
        help="At minimum include transaction_data.csv. Add product and campaign files for richer views.",
    )
    st.markdown("[Official dataset source](https://www.dunnhumby.com/source-files/)")


def read_csv(source, wanted_columns):
    if hasattr(source, "seek"):
        source.seek(0)
    header = pd.read_csv(source, nrows=0)
    actual = {column.lower(): column for column in header.columns}
    selected = [actual[name.lower()] for name in wanted_columns if name.lower() in actual]
    if hasattr(source, "seek"):
        source.seek(0)
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


@st.cache_data(show_spinner="Reading transaction data…", max_entries=2)
def load_data(uploaded_payloads, signatures):
    del signatures
    sources = {path.name.lower(): path for path in Path("data/raw").glob("*.csv")}
    archives = []
    for filename, content in uploaded_payloads:
        if filename.lower().endswith(".zip"):
            archive = ZipFile(BytesIO(content))
            archives.append(archive)
            for member in archive.namelist():
                name = Path(member).name.lower()
                if name.endswith(".csv"):
                    sources[name] = archive.open(member)
        elif filename.lower().endswith(".csv"):
            sources[Path(filename).name.lower()] = BytesIO(content)

    if "transaction_data.csv" not in sources:
        return None, {}, "Upload transaction_data.csv or a ZIP containing it."

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
    if "product.csv" in optional:
        products = optional["product.csv"]
        products.columns = [name.lower() for name in products.columns]
        category = "department" if "department" in products else "commodity_desc"
        products = products[["product_id", category]].drop_duplicates("product_id")
        data = data.merge(products, on="product_id", how="left")
        data[category] = data[category].fillna("Unknown")
        data = data.rename(columns={category: "category"})
    else:
        data["category"] = "All products (product.csv not supplied)"
    return data, optional, None


payloads = [(item.name, item.getvalue()) for item in uploads] if uploads else []
signatures = [(name, len(content)) for name, content in payloads]
data, optional, error = load_data(payloads, signatures)
if error:
    st.info("Upload the dataset to begin. Raw data is not bundled with the app.")
    st.markdown("Download The Complete Journey from the official source page, then upload its ZIP or select transaction_data.csv and product.csv. Add campaign CSVs to enable the campaign view.")
    st.caption("This is dunnhumby transaction data, not generated sample data. See README for source and metric limitations.")
    st.stop()
if data.empty:
    st.error("No usable transaction rows were found.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    weeks = sorted(data.week_no.dropna().unique().tolist())
    week_range = st.slider("Transaction week", int(min(weeks)), int(max(weeks)), (int(min(weeks)), int(max(weeks))))
    category_options = sorted(data.category.dropna().astype(str).unique())
    selected_categories = st.multiselect("Category", category_options, default=category_options)
    view = data[data.week_no.between(*week_range) & data.category.astype(str).isin(selected_categories)]

if view.empty:
    st.warning("No rows match these filters.")
    st.stop()

net = view.sales_value.sum()
discounts = view.discount_total.sum()
gross = view.gross_sales.sum()
kpis = st.columns(5)
kpis[0].metric("Net sales", f"${net:,.0f}")
kpis[1].metric("Recorded discounts", f"${discounts:,.0f}")
kpis[2].metric("Discount rate", f"{discounts / gross if gross else 0:.1%}")
kpis[3].metric("Recorded quantity", f"{view.quantity.sum():,.0f}")
kpis[4].metric("Discounted line items", f"{view.discounted.mean():.1%}")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Forecast", "Discounts", "Categories", "Campaigns"])
with tab1:
    weekly = view.groupby("week_no", as_index=False).agg(
        net_sales=("sales_value", "sum"), discounts=("discount_total", "sum"))
    st.plotly_chart(px.line(weekly, x="week_no", y=["net_sales", "discounts"], markers=True,
                            title="Weekly net sales and recorded discounts"), width="stretch")
    st.caption("Weeks are a relative index in the source data, not calendar dates.")
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
        st.plotly_chart(px.line(chart_data, x="week_no", y="sales", color="series", markers=True,
                                title="Historical sales, holdout backtest, and 8-week forecast"), width="stretch")
        st.dataframe(forecast, width="stretch", hide_index=True)
        st.caption("The app selects between a 4-week moving average and a 52-week seasonal naïve baseline using holdout MAE. This forecast is an estimate, not a guaranteed outcome.")
with tab3:
    amounts = pd.DataFrame({"Discount type": ["Retailer", "Coupon", "Retailer coupon match"],
        "Amount": [view.retail_disc.sum(), view.coupon_disc.sum(), view.coupon_match_disc.sum()]})
    st.plotly_chart(px.bar(amounts, x="Discount type", y="Amount", color="Discount type",
                           title="Recorded discount amounts"), width="stretch")
    by_status = view.groupby("discounted", as_index=False).agg(net_sales=("sales_value", "sum"))
    by_status["Status"] = by_status.discounted.map({True: "Discount recorded", False: "No discount"})
    st.plotly_chart(px.bar(by_status, x="Status", y="net_sales", color="Status",
                           title="Net sales by observed discount status"), width="stretch")
with tab4:
    summary = view.groupby("category", as_index=False).agg(
        net_sales=("sales_value", "sum"), units=("quantity", "sum"),
        discounts=("discount_total", "sum"), line_items=("sales_value", "size"))
    summary = summary.sort_values("net_sales", ascending=False)
    st.plotly_chart(px.bar(summary.head(20), x="net_sales", y="category", orientation="h",
                           title="Top categories by net sales"), width="stretch")
    st.dataframe(summary, width="stretch", hide_index=True)
with tab5:
    redemptions = optional.get("coupon_redempt.csv")
    campaigns = optional.get("campaign_table.csv")
    descriptions = optional.get("campaign_desc.csv")
    if redemptions is None:
        st.info("Add coupon_redempt.csv and campaign_table.csv for reach and redemption counts.")
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
            st.caption("Descriptive event rate; a household may redeem more than once.")

st.download_button("Download filtered transactions", view.to_csv(index=False).encode("utf-8"),
                   "filtered_transactions.csv", "text/csv")

