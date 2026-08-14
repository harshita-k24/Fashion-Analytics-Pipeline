import os
import numpy as np
import pandas as pd

# Set fixed seed so every dataset generates identically
np.random.seed(42)
NUMBER_OF_WEEKS = 156  # 3 Full Years (156 Weeks)
NUMBER_OF_CUSTOMERS = 1000  # 1,000 Full Customer Records

def build_complete_datasets():
    print("Generating complete datasets...")

    # ==========================================
    # DATASET 1: MARKETING MIX DATA (156 ROWS)
    # ==========================================
    time_index = np.arange(NUMBER_OF_WEEKS)

    def generate_spending(mean_val, burst_prob, burst_mult):
        spending = np.random.gamma(shape=3, scale=mean_val/3, size=NUMBER_OF_WEEKS)
        spending[np.random.rand(NUMBER_OF_WEEKS) < burst_prob] *= burst_mult
        return spending

    influencer_spending = generate_spending(12000, 0.15, 2.2)
    search_spending = 0.45 * influencer_spending + generate_spending(6000, 0.10, 1.8)
    brand_spending = generate_spending(8000, 0.08, 2.5)
    markdown_spending = generate_spending(5000, 0.20, 2.0)

    # Base trend and seasonal fashion drops
    trend = 80000 + 35 * time_index
    seasonality = 1 + 0.15 * np.sin(2 * np.pi * time_index / 52 - np.pi/2)
    collection_drops = np.zeros(NUMBER_OF_WEEKS)
    collection_drops[10::26] = 0.25
    baseline_sales = trend * seasonality * (1 + collection_drops)

    def adstock_transformation(series, decay_rate):
        out = np.zeros(len(series))
        out[0] = series[0]
        for i in range(1, len(series)):
            out[i] = series[i] + decay_rate * out[i-1]
        return out

    def saturation_transformation(series, alpha_exponent):
        return np.power(np.clip(series, 0, None), alpha_exponent)

    influencer_contrib = 3.5 * saturation_transformation(adstock_transformation(influencer_spending, 0.60), 0.65)
    search_contrib = 2.2 * saturation_transformation(adstock_transformation(search_spending, 0.15), 0.85)
    brand_contrib = 1.8 * saturation_transformation(adstock_transformation(brand_spending, 0.70), 0.50)
    markdown_contrib = 4.2 * saturation_transformation(adstock_transformation(markdown_spending, 0.05), 0.90)

    noise = np.random.normal(0, 0.02 * baseline_sales.mean(), NUMBER_OF_WEEKS)
    total_sales = baseline_sales + influencer_contrib + search_contrib + brand_contrib + markdown_contrib + noise

    marketing_df = pd.DataFrame({
        'Week': time_index,
        'Influencer_Social_Spending': influencer_spending.round(2),
        'Paid_Search_Spending': search_spending.round(2),
        'Brand_Public_Relations_Out_Of_Home_Spending': brand_spending.round(2),
        'Markdown_Promotional_Spending': markdown_spending.round(2),
        'Total_Sales': total_sales.round(2)
    })

    # ==========================================
    # DATASET 2: CUSTOMER SEGMENTATION DATA (1,000 ROWS)
    # ==========================================
    customer_ids = [f"CUSTOMER_{i:04d}" for i in range(1, NUMBER_OF_CUSTOMERS + 1)]
    recency = np.random.exponential(scale=30, size=NUMBER_OF_CUSTOMERS).clip(1, 180).round(0)
    frequency = np.random.negative_binomial(n=3, p=0.3, size=NUMBER_OF_CUSTOMERS).clip(1, 20)
    monetary = (frequency * np.random.uniform(45, 180, size=NUMBER_OF_CUSTOMERS) + 
                np.random.normal(0, 20, NUMBER_OF_CUSTOMERS)).clip(20, 5000).round(2)
    discount_sensitivity = np.random.beta(a=2, b=5, size=NUMBER_OF_CUSTOMERS).round(2)

    customer_df = pd.DataFrame({
        'Customer_Identifier': customer_ids,
        'Recency_Days': recency,
        'Frequency_Orders': frequency,
        'Monetary_Value': monetary,
        'Discount_Sensitivity': discount_sensitivity
    })

    # ==========================================
    # DATASET 3: DEMAND FORECASTING DATA (156 ROWS)
    # ==========================================
    drop_indicators = np.zeros(NUMBER_OF_WEEKS)
    drop_indicators[10::26] = 1.0

    forecasting_df = pd.DataFrame({
        'Week': time_index,
        'Total_Sales': total_sales.round(2),
        'Trend': time_index,
        'Sine_52_Harmonic': np.sin(2 * np.pi * time_index / 52).round(4),
        'Cosine_52_Harmonic': np.cos(2 * np.pi * time_index / 52).round(4),
        'Collection_Drop_Indicator': drop_indicators,
        'Lag_1_Sales': pd.Series(total_sales).shift(1).bfill().round(2),
        'Lag_2_Sales': pd.Series(total_sales).shift(2).bfill().round(2)
    })

    # Export Comma-Separated Values Files
    os.makedirs("data", exist_ok=True)
    marketing_df.to_csv("data/marketing_mix_data.csv", index=False)
    customer_df.to_csv("data/customer_rfm_segments.csv", index=False)
    forecasting_df.to_csv("data/demand_forecasting_data.csv", index=False)

    print("Success! Generated the complete dataset files:")
    print(f" - data/marketing_mix_data.csv ({len(marketing_df)} rows)")
    print(f" - data/customer_rfm_segments.csv ({len(customer_df)} rows)")
    print(f" - data/demand_forecasting_data.csv ({len(forecasting_df)} rows)")

if __name__ == "__main__":
    build_complete_datasets()
