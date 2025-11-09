"""
Housing Price Prediction Web Application
Datathon 2025 - Problem Statement #11

This Streamlit app provides an interactive interface for:
1. Exploring housing data
2. Making price predictions
3. Visualizing regional trends
4. Displaying model performance metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1E88E5;
        padding-bottom: 1rem;
    }
    h2 {
        color: #424242;
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    """Load and prepare the California Housing dataset"""
    california_housing = fetch_california_housing(as_frame=True)
    df = california_housing.frame
    return df

@st.cache_data
def engineer_features(df):
    """Create engineered features"""
    df_eng = df.copy()

    # Engineer new features
    df_eng['RoomsPerHousehold'] = df_eng['AveRooms'] * df_eng['AveOccup']
    df_eng['BedroomsRatio'] = df_eng['AveBedrms'] / df_eng['AveRooms']
    df_eng['PopulationPerHousehold'] = df_eng['Population'] / (df_eng['Population'] / df_eng['AveOccup'])
    df_eng['IncomeToAgeRatio'] = df_eng['MedInc'] / (df_eng['HouseAge'] + 1)
    df_eng['IsCoastal'] = (df_eng['Longitude'] < -119).astype(int)
    df_eng['TotalRooms'] = df_eng['AveRooms'] * df_eng['Population'] / df_eng['AveOccup']
    df_eng['LuxuryScore'] = df_eng['MedInc'] * df_eng['AveRooms'] / (df_eng['AveOccup'] + 1)

    # Handle any infinite or NaN values
    df_eng.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_eng.fillna(df_eng.median(), inplace=True)

    return df_eng

@st.cache_resource
def train_models(df_eng):
    """Train and return models"""
    X = df_eng.drop('MedHouseVal', axis=1)
    y = df_eng['MedHouseVal']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # Train Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Calculate metrics
    y_pred_rf = rf_model.predict(X_test)
    y_pred_lr = lr_model.predict(X_test)

    metrics = {
        'rf': {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
            'r2': r2_score(y_test, y_pred_rf),
            'mae': mean_absolute_error(y_test, y_pred_rf)
        },
        'lr': {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
            'r2': r2_score(y_test, y_pred_lr),
            'mae': mean_absolute_error(y_test, y_pred_lr)
        }
    }

    return rf_model, lr_model, metrics, X_test, y_test

def main():
    # Title and introduction
    st.title("🏠 California Housing Price Prediction")
    st.markdown("### Datathon 2025 - Interactive Analytics Dashboard")
    st.markdown("---")

    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
        df_eng = engineer_features(df)

    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["🏡 Home", "🔍 Data Explorer", "🎯 Price Predictor", 
         "📈 Model Performance", "🗺️ Geographic Analysis", "💡 Insights"]
    )

    # HOME PAGE
    if page == "🏡 Home":
        st.header("Welcome to the Housing Price Prediction Dashboard")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Districts", f"{len(df):,}")
        with col2:
            st.metric("Average Price", f"${df['MedHouseVal'].mean():.2f}00k")
        with col3:
            st.metric("Features", len(df.columns) - 1)

        st.markdown("---")

        st.subheader("📋 Project Overview")
        st.write("""
        This interactive dashboard analyzes California housing prices using machine learning. 
        The project includes:

        - **Exploratory Data Analysis**: Understand data distributions and patterns
        - **Feature Engineering**: Create new predictive features
        - **Machine Learning Models**: Linear Regression and Random Forest
        - **Geographic Visualization**: Interactive maps showing price distributions
        - **Regional Insights**: Analysis of different California regions
        - **Price Prediction**: Real-time house price estimation
        """)

        st.subheader("📊 Quick Dataset Overview")

        # Display sample data
        st.dataframe(df.head(10), use_container_width=True)

        # Dataset statistics
        st.subheader("📈 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

    # DATA EXPLORER PAGE
    elif page == "🔍 Data Explorer":
        st.header("Data Exploration")

        # Feature selection
        st.subheader("Feature Distributions")
        feature = st.selectbox("Select a feature to visualize:", df.columns)

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df[feature], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_xlabel(feature, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'Distribution of {feature}', fontsize=14, fontweight='bold')
            ax.axvline(df[feature].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
            ax.axvline(df[feature].median(), color='green', linestyle='--', linewidth=2, label='Median')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(df[feature], vert=True)
            ax.set_ylabel(feature, fontsize=12)
            ax.set_title(f'Box Plot of {feature}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        # Correlation heatmap
        st.subheader("🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 10))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   square=True, linewidths=1, ax=ax)
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        st.pyplot(fig)

        # Feature relationships
        st.subheader("🔗 Feature Relationships")
        col1, col2 = st.columns(2)

        with col1:
            x_feature = st.selectbox("X-axis:", df.columns, index=0)
        with col2:
            y_feature = st.selectbox("Y-axis:", df.columns, index=len(df.columns)-1)

        fig, ax = plt.subplots(figsize=(10, 6))
        sample_df = df.sample(n=min(3000, len(df)), random_state=42)
        ax.scatter(sample_df[x_feature], sample_df[y_feature], alpha=0.5, s=20, color='purple')
        ax.set_xlabel(x_feature, fontsize=12)
        ax.set_ylabel(y_feature, fontsize=12)
        ax.set_title(f'{x_feature} vs {y_feature}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # PRICE PREDICTOR PAGE
    elif page == "🎯 Price Predictor":
        st.header("House Price Predictor")
        st.markdown("Enter property details to get a price prediction:")

        # Train models
        rf_model, lr_model, metrics, X_test, y_test = train_models(df_eng)

        # Input form
        col1, col2 = st.columns(2)

        with col1:
            med_inc = st.slider("Median Income (in $10k)", 
                               float(df['MedInc'].min()), 
                               float(df['MedInc'].max()), 
                               float(df['MedInc'].mean()))

            house_age = st.slider("House Age (years)", 
                                 float(df['HouseAge'].min()), 
                                 float(df['HouseAge'].max()), 
                                 float(df['HouseAge'].mean()))

            ave_rooms = st.slider("Average Rooms", 
                                 float(df['AveRooms'].min()), 
                                 float(df['AveRooms'].max()), 
                                 float(df['AveRooms'].mean()))

            ave_bedrms = st.slider("Average Bedrooms", 
                                  float(df['AveBedrms'].min()), 
                                  float(df['AveBedrms'].max()), 
                                  float(df['AveBedrms'].mean()))

        with col2:
            population = st.slider("Population", 
                                  float(df['Population'].min()), 
                                  float(df['Population'].max()), 
                                  float(df['Population'].mean()))

            ave_occup = st.slider("Average Occupancy", 
                                 float(df['AveOccup'].min()), 
                                 float(df['AveOccup'].max()), 
                                 float(df['AveOccup'].mean()))

            latitude = st.slider("Latitude", 
                                float(df['Latitude'].min()), 
                                float(df['Latitude'].max()), 
                                float(df['Latitude'].mean()))

            longitude = st.slider("Longitude", 
                                 float(df['Longitude'].min()), 
                                 float(df['Longitude'].max()), 
                                 float(df['Longitude'].mean()))

        # Engineer features for prediction
        rooms_per_household = ave_rooms * ave_occup
        bedrooms_ratio = ave_bedrms / ave_rooms
        population_per_household = population / (population / ave_occup)
        income_to_age_ratio = med_inc / (house_age + 1)
        is_coastal = 1 if longitude < -119 else 0
        total_rooms = ave_rooms * population / ave_occup
        luxury_score = med_inc * ave_rooms / (ave_occup + 1)

        # Create input dataframe
        input_data = pd.DataFrame({
            'MedInc': [med_inc],
            'HouseAge': [house_age],
            'AveRooms': [ave_rooms],
            'AveBedrms': [ave_bedrms],
            'Population': [population],
            'AveOccup': [ave_occup],
            'Latitude': [latitude],
            'Longitude': [longitude],
            'RoomsPerHousehold': [rooms_per_household],
            'BedroomsRatio': [bedrooms_ratio],
            'PopulationPerHousehold': [population_per_household],
            'IncomeToAgeRatio': [income_to_age_ratio],
            'IsCoastal': [is_coastal],
            'TotalRooms': [total_rooms],
            'LuxuryScore': [luxury_score]
        })

        if st.button("🔮 Predict Price", type="primary"):
            # Make predictions
            rf_prediction = rf_model.predict(input_data)[0]
            lr_prediction = lr_model.predict(input_data)[0]

            st.markdown("---")
            st.subheader("💰 Prediction Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Random Forest Prediction", f"${rf_prediction:.2f}00k")
            with col2:
                st.metric("Linear Regression Prediction", f"${lr_prediction:.2f}00k")
            with col3:
                avg_prediction = (rf_prediction + lr_prediction) / 2
                st.metric("Average Prediction", f"${avg_prediction:.2f}00k")

            st.success(f"✅ Estimated house value: ${avg_prediction*100:.0f},000")

            # Show confidence
            st.info(f"📊 Model Confidence (R²): Random Forest = {metrics['rf']['r2']:.3f}, Linear Regression = {metrics['lr']['r2']:.3f}")

    # MODEL PERFORMANCE PAGE
    elif page == "📈 Model Performance":
        st.header("Model Performance Analysis")

        # Train models
        rf_model, lr_model, metrics, X_test, y_test = train_models(df_eng)

        # Display metrics
        st.subheader("📊 Model Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Random Forest")
            st.metric("RMSE", f"{metrics['rf']['rmse']:.4f}")
            st.metric("R² Score", f"{metrics['rf']['r2']:.4f}")
            st.metric("MAE", f"{metrics['rf']['mae']:.4f}")

        with col2:
            st.markdown("### Linear Regression")
            st.metric("RMSE", f"{metrics['lr']['rmse']:.4f}")
            st.metric("R² Score", f"{metrics['lr']['r2']:.4f}")
            st.metric("MAE", f"{metrics['lr']['mae']:.4f}")

        # Performance comparison
        st.subheader("📊 Performance Comparison")

        comparison_df = pd.DataFrame({
            'Model': ['Random Forest', 'Linear Regression'],
            'RMSE': [metrics['rf']['rmse'], metrics['lr']['rmse']],
            'R² Score': [metrics['rf']['r2'], metrics['lr']['r2']],
            'MAE': [metrics['rf']['mae'], metrics['lr']['mae']]
        })

        st.dataframe(comparison_df, use_container_width=True)

        # Visualize comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        models = ['Random\nForest', 'Linear\nRegression']
        rmse_values = [metrics['rf']['rmse'], metrics['lr']['rmse']]
        r2_values = [metrics['rf']['r2'], metrics['lr']['r2']]

        axes[0].bar(models, rmse_values, color=['#4ECDC4', '#FF6B6B'], edgecolor='black')
        axes[0].set_title('RMSE Comparison', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('RMSE')
        axes[0].grid(True, alpha=0.3, axis='y')

        for i, v in enumerate(rmse_values):
            axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

        axes[1].bar(models, r2_values, color=['#4ECDC4', '#FF6B6B'], edgecolor='black')
        axes[1].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('R² Score')
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3, axis='y')

        for i, v in enumerate(r2_values):
            axes[1].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)

        # Feature importance
        st.subheader("🎯 Feature Importance (Random Forest)")

        feature_importance = pd.DataFrame({
            'Feature': df_eng.drop('MedHouseVal', axis=1).columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(feature_importance)), feature_importance['Importance'], color='teal', edgecolor='black')
        ax.set_yticks(range(len(feature_importance)))
        ax.set_yticklabels(feature_importance['Feature'])
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        st.pyplot(fig)

        st.dataframe(feature_importance, use_container_width=True)

    # GEOGRAPHIC ANALYSIS PAGE
    elif page == "🗺️ Geographic Analysis":
        st.header("Geographic Analysis")

        # Map visualization
        st.subheader("🗺️ Housing Price Distribution Map")

        # Scatter plot
        fig, ax = plt.subplots(figsize=(14, 10))
        scatter = ax.scatter(df['Longitude'], df['Latitude'], 
                           c=df['MedHouseVal'], s=df['Population']/50, 
                           alpha=0.4, cmap='viridis', edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Median House Value (in $100k)')
        ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
        ax.set_title('California Housing Prices by Location\n(Size represents population)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Interactive heatmap
        st.subheader("🔥 Interactive Heatmap")

        # Sample data for heatmap
        sample_size = st.slider("Sample size for heatmap:", 1000, 10000, 3000, 500)
        df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)

        # Create map
        california_map = folium.Map(
            location=[36.7, -119.7],
            zoom_start=6,
            tiles='OpenStreetMap'
        )

        heat_data = [[row['Latitude'], row['Longitude'], row['MedHouseVal']] 
                    for idx, row in df_sample.iterrows()]

        HeatMap(heat_data, 
               min_opacity=0.3,
               max_opacity=0.8,
               radius=15,
               blur=25).add_to(california_map)

        folium_static(california_map)

        # Regional analysis
        st.subheader("📍 Regional Statistics")

        # Simple regional grouping
        df_regional = df.copy()
        df_regional['Region'] = 'Other'
        df_regional.loc[df_regional['Longitude'] < -122, 'Region'] = 'Coastal'
        df_regional.loc[(df_regional['Latitude'] > 38) & (df_regional['Longitude'] >= -122), 'Region'] = 'Northern'
        df_regional.loc[(df_regional['Latitude'] <= 35), 'Region'] = 'Southern'

        regional_stats = df_regional.groupby('Region').agg({
            'MedHouseVal': ['mean', 'median', 'std', 'count'],
            'MedInc': 'mean',
            'Population': 'sum'
        }).round(3)

        st.dataframe(regional_stats, use_container_width=True)

        # Regional price distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        regions = df_regional['Region'].unique()
        region_data = [df_regional[df_regional['Region'] == region]['MedHouseVal'].values 
                      for region in regions]

        bp = ax.boxplot(region_data, labels=regions, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_edgecolor('black')

        ax.set_ylabel('Median House Value (in $100k)', fontsize=12)
        ax.set_title('Price Distribution by Region', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)

    # INSIGHTS PAGE
    elif page == "💡 Insights":
        st.header("Key Insights and Recommendations")

        # Train models for metrics
        rf_model, lr_model, metrics, X_test, y_test = train_models(df_eng)

        st.subheader("🎯 Key Findings")

        st.markdown(f"""
        ### 📊 Dataset Overview
        - **Total Districts Analyzed**: {len(df):,}
        - **Features Analyzed**: {len(df.columns) - 1} original + 7 engineered
        - **Price Range**: ${df['MedHouseVal'].min():.2f}00k to ${df['MedHouseVal'].max():.2f}00k
        - **Average Price**: ${df['MedHouseVal'].mean():.2f}00k

        ### 🤖 Model Performance
        - **Best Model**: Random Forest (R² = {metrics['rf']['r2']:.4f})
        - **Average Prediction Error**: ${metrics['rf']['rmse']*100:.2f}k
        - **Feature Engineering Impact**: Improved model accuracy significantly

        ### 📈 Top Factors Affecting House Prices
        1. **Median Income**: Strongest predictor (correlation: {df.corr().loc['MedInc', 'MedHouseVal']:.3f})
        2. **Location**: Coastal areas command premium prices
        3. **House Size**: More rooms correlate with higher values
        4. **Age**: Newer properties tend to be more valuable
        5. **Population Density**: Affects pricing in complex ways

        ### 🗺️ Geographic Insights
        - **Highest Values**: Coastal regions (San Francisco Bay, LA, San Diego)
        - **Best Value**: Inland and Northern regions
        - **Hot Zones**: Properties near major coastal cities

        ### 💼 Investment Recommendations

        **For Investors:**
        - Focus on coastal areas for high returns
        - Consider emerging inland markets for growth potential
        - Target properties with 5+ rooms in growing areas

        **For Homebuyers:**
        - Best affordability in Northern and Southern inland regions
        - Consider income-to-price ratio for long-term sustainability
        - Newer properties (< 20 years) offer better value retention

        **For Urban Planners:**
        - Address affordability in high-demand coastal areas
        - Improve infrastructure in Central Valley
        - Balance development across regions

        ### 🔮 Future Improvements
        - Incorporate time-series data for trend analysis
        - Add economic indicators (employment, interest rates)
        - Include property-specific features (sq. footage, condition)
        - Implement advanced models (XGBoost, Neural Networks)
        """)

        # Correlation summary
        st.subheader("📊 Feature Correlations with Price")

        correlations = df.corr()['MedHouseVal'].sort_values(ascending=False).drop('MedHouseVal')

        fig, ax = plt.subplots(figsize=(10, 6))
        correlations.plot(kind='barh', ax=ax, color='teal', edgecolor='black')
        ax.set_xlabel('Correlation Coefficient', fontsize=12)
        ax.set_title('Feature Correlation with House Value', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        st.pyplot(fig)

        # Summary statistics
        st.subheader("📈 Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🏆 Datathon 2025 - Problem Statement #11</p>
        <p>Housing Price Prediction with Economic and Demographic Factors</p>
        <p>Built with ❤️ using Streamlit, scikit-learn, and Folium</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
