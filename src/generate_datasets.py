import os
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)
NUMBER_OF_WEEKS = 156  # Three years of weekly observations

def generate_marketing_mix_dataset():
    time_index = np.arange(NUMBER_OF_WEEKS)

    def generate_spending(mean_value, burst_probability, burst_multiplier):
        spending = np.random.gamma(shape=3, scale=mean_value/3, size=NUMBER_OF_WEEKS)
        spending[np.random.rand(NUMBER_OF_WEEKS) < burst_probability] *= burst_multiplier
        return spending

    influencer_social_spending = generate_spending(12000, 0.15, 2.2)
    paid_search_spending = 0.45 * influencer_social_spending + generate_spending(6000, 0.10, 1.8)
    brand_public_relations_out_of_home_spending = generate_spending(8000, 0.08, 2.5)
    markdown_promotional_spending = generate_spending(5000, 0.20, 2.0)

    # Base trend, seasonal patterns, and collection drops
    trend = 80000 + 35 * time_index
    seasonality = 1 + 0.15 * np.sin(2 * np.pi * time_index / 52 - np.pi/2)
    collection_drops = np.zeros(NUMBER_OF_WEEKS)
    collection_drops[10::26] = 0.25  # Spring/Summer and Autumn/Winter drops
    baseline_sales = trend * seasonality * (1 + collection_drops)

    # Channel transformations: Adstock Carryover and Power Saturation
    def adstock_transformation(series, decay_rate):
        output = np.zeros(len(series))
        output[0] = series[0]
        for i in range(1, len(series)):
            output[i] = series[i] + decay_rate * output[i-1]
        return output

    def saturation_transformation(series, alpha_exponent):
        return np.power(np.clip(series, 0, None), alpha_exponent)

    influencer_contrib = 3.5 * saturation_transformation(adstock_transformation(influencer_social_spending, 0.60), 0.65)
    search_contrib = 2.2 * saturation_transformation(adstock_transformation(paid_search_spending, 0.15), 0.85)
    brand_contrib = 1.8 * saturation_transformation(adstock_transformation(brand_public_relations_out_of_home_spending, 0.70), 0.50)
    markdown_contrib = 4.2 * saturation_transformation(adstock_transformation(markdown_promotional_spending, 0.05), 0.90)

    noise = np.random.normal(0, 0.02 * baseline_sales.mean(), NUMBER_OF_WEEKS)
    total_sales = baseline_sales + influencer_contrib + search_contrib + brand_contrib + markdown_contrib + noise

    marketing_dataframe = pd.DataFrame({
        'Week': time_index,
        'Influencer_Social_Spending': influencer_social_spending.round(2),
        'Paid_Search_Spending': paid_search_spending.round(2),
        'Brand_Public_Relations_Out_Of_Home_Spending': brand_public_relations_out_of_home_spending.round(2),
        'Markdown_Promotional_Spending': markdown_promotional_spending.round(2),
        'Total_Sales': total_sales.round(2)
    })
    return marketing_dataframe

def generate_customer_segmentation_dataset():
    NUMBER_OF_CUSTOMERS = 1000
    customer_identifiers = [f"CUSTOMER_{i:04d}" for i in range(1, NUMBER_OF_CUSTOMERS + 1)]

    recency_days = np.random.exponential(scale=30, size=NUMBER_OF_CUSTOMERS).clip(1, 180).round(0)
    frequency_orders = np.random.negative_binomial(n=3, p=0.3, size=NUMBER_OF_CUSTOMERS).clip(1, 20)
    monetary_value = (frequency_orders * np.random.uniform(45, 180, size=NUMBER_OF_CUSTOMERS) + 
                      np.random.normal(0, 20, NUMBER_OF_CUSTOMERS)).clip(20, 5000).round(2)
    discount_sensitivity = np.random.beta(a=2, b=5, size=NUMBER_OF_CUSTOMERS).round(2)

    customer_dataframe = pd.DataFrame({
        'Customer_Identifier': customer_identifiers,
        'Recency_Days': recency_days,
        'Frequency_Orders': frequency_orders,
        'Monetary_Value': monetary_value,
        'Discount_Sensitivity': discount_sensitivity
    })
    return customer_dataframe

def generate_demand_forecasting_dataset(marketing_dataframe):
    sales_series = marketing_dataframe['Total_Sales'].values
    time_index = marketing_dataframe['Week'].values

    collection_drops = np.zeros(NUMBER_OF_WEEKS)
    collection_drops[10::26] = 1.0

    forecasting_dataframe = pd.DataFrame({
        'Week': time_index,
        'Total_Sales': sales_series,
        'Trend': time_index,
        'Sine_52_Harmonic': np.sin(2 * np.pi * time_index / 52).round(4),
        'Cosine_52_Harmonic': np.cos(2 * np.pi * time_index / 52).round(4),
        'Collection_Drop_Indicator': collection_drops,
        'Lag_1_Sales': pd.Series(sales_series).shift(1).bfill().round(2),
        'Lag_2_Sales': pd.Series(sales_series).shift(2).bfill().round(2)
    })
    return forecasting_dataframe

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    marketing_df = generate_marketing_mix_dataset()
    customer_df = generate_customer_segmentation_dataset()
    forecasting_df = generate_demand_forecasting_dataset(marketing_df)

    marketing_df.to_csv("data/marketing_mix_data.csv", index=False)
    customer_df.to_csv("data/customer_rfm_segments.csv", index=False)
    forecasting_df.to_csv("data/demand_forecasting_data.csv", index=False)

    print("Data generation complete. Comma-Separated Values files saved in data/ directory.")
