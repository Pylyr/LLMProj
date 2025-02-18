#!/bin/bash

# ---------------------------------------------------------
# MIMIC-III Data Processing Script
# ---------------------------------------------------------
# Source: Kaggle MIMIC-III Dataset
# Download from: https://www.kaggle.com/datasets/asjad99/mimiciii
# After downloading, place the `mimiciii.zip` file inside the `raw/` folder
# ---------------------------------------------------------

# Create necessary directories
mkdir -p raw/mimiciii
mkdir -p preprocessed/mimiciii

# Unzip dataset
unzip -o /content/drive/MyDrive/mimiciii.zip -d data_preprocessing/raw/mimiciii/

# Process data
PYTHONPATH=. python data_preprocessing/process_mimic_data.py data_preprocessing/raw/mimiciii preprocessed/mimiciii/
