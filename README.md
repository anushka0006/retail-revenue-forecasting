# Retail Revenue Forecasting & Discount Analytics

An end-to-end Streamlit project forecasting weekly retail net sales and monitoring discounts, categories, and coupon activity using dunnhumby's The Complete Journey dataset.

The official [dunnhumby Source Files page](https://www.dunnhumby.com/source-files/) describes household transaction data from 2,500 frequent shoppers over two years, plus marketing history. It is supplied as downloadable files, not a documented public API. No source data is included here.

## What it measures and limitations

The dashboard compares a 4-week moving average with a 52-week seasonal naïve forecast when enough data is available. It selects a baseline using a final-period holdout and reports MAE and WMAPE. It also shows net sales, recorded discounts, reconstructed gross sales, discount rate, category trends, and coupon redemption counts. The data does not include product costs, so the project focuses on revenue forecasting and discount monitoring rather than profit or promotion ROI. Redemption comparisons are descriptive.

## Data setup

1. Download The Complete Journey from the official source page.
2. Upload a ZIP containing its CSV files in the dashboard, or place CSVs in data/raw/. At minimum include transaction_data.csv. Add product.csv for category views and campaign_desc.csv, campaign_table.csv, coupon_redempt.csv for campaign metrics.
3. Raw data is excluded by .gitignore; do not commit it to GitHub.

## Run locally

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    streamlit run streamlit_app.py

On macOS or Linux, activate with source .venv/bin/activate.

## Deploy

Push this project to a GitHub repository. In Streamlit Community Cloud, create an app from that repository and select streamlit_app.py. Upload the source ZIP or CSVs after deployment. No API keys are needed.

## Resume wording

After you have run the dashboard, reviewed the backtest metrics, and checked the forecast:

> Built an interactive Streamlit retail revenue dashboard using dunnhumby's 2.5M+ transaction-line dataset; engineered weekly sales and discount metrics, compared moving-average and seasonal-naïve forecasts using a holdout MAE/WMAPE backtest, and added category and coupon-redemption analysis.

Add public app and repository links after deployment. Cite dunnhumby as the dataset source.

