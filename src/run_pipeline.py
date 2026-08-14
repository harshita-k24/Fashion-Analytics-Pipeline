import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, mean_absolute_percentage_error, mean_squared_error

def execute_marketing_mix_model():
    print("==================================================")
    print("1. Marketing Mix Modeling Execution")
    print("==================================================")
    
    dataframe = pd.read_csv("data/marketing_mix_data.csv")
    
    def adstock_transformation(series, decay_rate):
        output = np.zeros(len(series))
        output[0] = series[0]
        for i in range(1, len(series)):
            output[i] = series[i] + decay_rate * output[i-1]
        return output

    def saturation_transformation(series, alpha_exponent):
        return np.power(np.clip(series, 0, None), alpha_exponent)

    # Apply non-linear decay and diminishing returns transformations
    transformed_features = pd.DataFrame()
    transformed_features['Influencer_Social'] = saturation_transformation(
        adstock_transformation(dataframe['Influencer_Social_Spending'].values, 0.60), 0.65)
    transformed_features['Paid_Search'] = saturation_transformation(
        adstock_transformation(dataframe['Paid_Search_Spending'].values, 0.15), 0.85)
    transformed_features['Brand_Public_Relations'] = saturation_transformation(
        adstock_transformation(dataframe['Brand_Public_Relations_Out_Of_Home_Spending'].values, 0.70), 0.50)
    transformed_features['Markdown_Promotional'] = saturation_transformation(
        adstock_transformation(dataframe['Markdown_Promotional_Spending'].values, 0.05), 0.90)

    model = LinearRegression()
    model.fit(transformed_features, dataframe['Total_Sales'])
    
    coefficient_of_determination = model.score(transformed_features, dataframe['Total_Sales'])
    print(f"Marketing Mix Model Coefficient of Determination (R-Squared): {coefficient_of_determination:.4f}\n")

    channels = ['Influencer_Social_Spending', 'Paid_Search_Spending', 
                'Brand_Public_Relations_Out_Of_Home_Spending', 'Markdown_Promotional_Spending']
    
    print("Estimated Channel Metrics:")
    for idx, channel in enumerate(transformed_features.columns):
        estimated_contribution = model.coef_[idx] * transformed_features[channel].values
        total_spending = dataframe[channels[idx]].sum()
        return_on_investment = estimated_contribution.sum() / total_spending
        print(f" - Channel: {channel} | Estimated Return on Investment: {return_on_investment:.2f}")

def execute_customer_segmentation():
    print("\n==================================================")
    print("2. Customer Behavioral Segmentation (K-Means Clustering)")
    print("==================================================")
    
    customer_dataframe = pd.read_csv("data/customer_rfm_segments.csv")
    features = ['Recency_Days', 'Frequency_Orders', 'Monetary_Value', 'Discount_Sensitivity']
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(customer_dataframe[features])
    
    kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
    customer_dataframe['Cluster_Group'] = kmeans_model.fit_predict(scaled_features)
    
    silhouette_value = silhouette_score(scaled_features, customer_dataframe['Cluster_Group'])
    print(f"Clustering Silhouette Score Evaluation Metric: {silhouette_value:.4f}\n")
    
    profiles = customer_dataframe.groupby('Cluster_Group')[features].mean().round(2)
    profiles['Customer_Count'] = customer_dataframe['Cluster_Group'].value_counts()
    print("Segment Mean Characteristic Profiles:")
    print(profiles)

def execute_demand_forecasting():
    print("\n==================================================")
    print("3. Category Demand Time-Series Forecasting")
    print("==================================================")
    
    forecasting_dataframe = pd.read_csv("data/demand_forecasting_data.csv")
    
    train_size = 130  # Training period across 130 weeks
    train_set = forecasting_dataframe.iloc[:train_size]
    test_set = forecasting_dataframe.iloc[train_size:]
    
    predictor_columns = ['Trend', 'Sine_52_Harmonic', 'Cosine_52_Harmonic', 
                         'Collection_Drop_Indicator', 'Lag_1_Sales', 'Lag_2_Sales']
    
    forecasting_model = Ridge(alpha=1.0)
    forecasting_model.fit(train_set[predictor_columns], train_set['Total_Sales'])
    
    predictions = forecasting_model.predict(test_set[predictor_columns])
    
    mean_absolute_percentage_error_val = mean_absolute_percentage_error(test_set['Total_Sales'], predictions)
    root_mean_squared_error_val = np.sqrt(mean_squared_error(test_set['Total_Sales'], predictions))
    
    print("Out-of-Sample Evaluation Metrics (26-Week Forecast Horizon):")
    print(f" - Mean Absolute Percentage Error: {mean_absolute_percentage_error_val * 100:.2f}%")
    print(f" - Root Mean Squared Error: ${root_mean_squared_error_val:,.2f}")

if __name__ == "__main__":
    execute_marketing_mix_model()
    execute_customer_segmentation()
    execute_demand_forecasting()
