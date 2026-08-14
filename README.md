# Fashion-Analytics-Pipeline
# Integrated Fashion Retail Analytics Pipeline

An end-to-end data analytics and machine learning solution tailored for the fashion retail industry. This repository combines **Marketing Mix Modeling**, **Customer Segmentation**, and **Category Demand Time-Series Forecasting**.

---

## Technical Overview

### 1. Marketing Mix Modeling
Quantifies incremental sales and Return on Investment across major apparel advertising channels while accounting for baseline demand, seasonal collection releases (Spring/Summer and Autumn/Winter), non-linear carryover (Adstock Decay), and diminishing marginal returns (Power Saturation).

* **Adstock Decay Transformation:**
  $$Adstocked\_Spend_t = Spend_t + \theta \cdot Adstocked\_Spend_{t-1}$$
* **Power Saturation Transformation:**
  $$Saturated\_Spend_t = (Adstocked\_Spend_t)^{\alpha}$$

### 2. Customer Behavioral Segmentation
Groups fashion customers using standardized metrics for Recency, Frequency, Monetary Value, and Discount Sensitivity using K-Means Clustering to identify distinct buyer personas ranging from full-price brand trendsetters to promotional buyers.

### 3. Category Demand Time-Series Forecasting
Predicts weekly category sales demand over a 26-week horizon using Ridge Regression incorporating harmonic seasonal terms, collection release indicators, and autoregressive lag variables.

---

## Directory Organization

```text
fashion-analytics-pipeline/
│
├── data/
│   ├── marketing_mix_data.csv
│   ├── customer_rfm_segments.csv
│   └── demand_forecasting_data.csv
│
├── src/
│   ├── generate_datasets.py
│   └── run_pipeline.py
│
├── .gitignore
├── requirements.txt
└── README.md
