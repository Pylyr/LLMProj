#!/bin/bash

# ---------------------------------------------------------
# Financial Data Processing Script
# ---------------------------------------------------------
# Source: Kaggle Reported Financials Dataset
# Download from: https://www.kaggle.com/datasets/finnhub/reported-financials?resource=download
# After downloading, place the `reported-financials.zip` file inside the `raw/` folder
# ---------------------------------------------------------

# Create necessary directories
mkdir -p raw/financial_reports
mkdir -p preprocessed/financial_reports

unzip -o raw/reported-financials.zip -d raw/financial_reports/

PYTHONPATH=. python process_financial_data.py raw/financial_reports preprocessed/financial_reports/
