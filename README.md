# Housing Price Prediction with Economic and Demographic Factors
## Datathon 2025 - Problem Statement #11

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Project Overview

This project provides a comprehensive solution for predicting housing prices in California using machine learning techniques. It includes exploratory data analysis, feature engineering, model training, geographic visualization, and an interactive web application.

### Key Features:
- ✅ Complete data analysis pipeline
- ✅ Advanced feature engineering (7 new features)
- ✅ Multiple ML models (Linear Regression, Random Forest)
- ✅ Interactive geographic visualizations
- ✅ Web-based prediction interface
- ✅ Regional insights and recommendations

---

## 📁 Project Structure

```
datathon-housing-prediction/
│
├── housing_price_prediction_datathon.ipynb  # Main Jupyter notebook
├── streamlit_app.py                         # Interactive web application
├── requirements.txt                         # Python dependencies
├── README.md                                # This file
│
├── Generated Files (after running notebook):
│   ├── california_housing_heatmap.html      # Interactive map
│   ├── regional_analysis.csv                # Regional statistics
│   ├── housing_prediction_insights_report.txt  # Comprehensive report
│   ├── best_rf_model.pkl                    # Trained Random Forest model
│   ├── lr_model.pkl                         # Trained Linear Regression model
│   ├── feature_names.pkl                    # Feature names for prediction
│   ├── housing_data_engineered.csv          # Data with engineered features
│   └── housing_data_regional.csv            # Data with regional labels
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook (for .ipynb file)

### 2. Installation

Clone the repository and install dependencies:

```bash
# Navigate to project directory
cd datathon-housing-prediction

# Install required packages
pip install -r requirements.txt
```

### 3. Running the Jupyter Notebook

```bash
# Start Jupyter Notebook
jupyter notebook

# Open: housing_price_prediction_datathon.ipynb
# Run all cells to perform complete analysis
```

### 4. Running the Web Application

```bash
# Launch Streamlit app
streamlit run streamlit_app.py

# The app will open in your browser at http://localhost:8501
```

---

## 📊 Dataset Information

### California Housing Dataset

- **Source**: scikit-learn built-in dataset
- **Samples**: 20,640 districts
- **Features**: 8 original + 7 engineered
- **Target**: Median house value (in $100,000s)

#### Original Features:
- `MedInc`: Median income in block group
- `HouseAge`: Median house age in block group
- `AveRooms`: Average number of rooms per household
- `AveBedrms`: Average number of bedrooms per household
- `Population`: Block group population
- `AveOccup`: Average number of household members
- `Latitude`: Block group latitude
- `Longitude`: Block group longitude

#### Engineered Features:
- `RoomsPerHousehold`: Total rooms per household
- `BedroomsRatio`: Proportion of bedrooms to total rooms
- `PopulationPerHousehold`: Average population per household
- `IncomeToAgeRatio`: Income adjusted for house age
- `IsCoastal`: Binary indicator for coastal location
- `TotalRooms`: Estimated total rooms in the area
- `LuxuryScore`: Combined metric of income and room quality

---

## 🎯 Project Objectives (Completed)

### ✅ Objective 1: Data Loading and Inspection
- Loaded California Housing dataset
- Performed quality checks (no missing values)
- Identified and analyzed outliers

### ✅ Objective 2: Exploratory Data Analysis
- Created distribution plots for all features
- Analyzed statistical summaries
- Visualized feature relationships

### ✅ Objective 3: Correlation Analysis
- Generated correlation heatmap
- Identified key predictors of house prices
- Analyzed feature interactions

### ✅ Objective 4: Geographic Visualization
- Created scatter plots showing price by location
- Built interactive heatmap using Folium
- Identified high-value coastal regions

### ✅ Objective 5: Baseline Model Training
- Trained Linear Regression model
- Trained Random Forest model
- Evaluated using RMSE and R² metrics

### ✅ Objective 6: Model Evaluation
- RMSE and R² score calculation
- Cross-validation analysis
- Model comparison

### ✅ Objective 7: Feature Engineering
- Created 7 new features
- Improved model performance
- Enhanced predictive power

### ✅ Objective 8: Model Comparison
- Compared baseline vs engineered models
- Analyzed performance improvements
- Visualized results

### ✅ Objective 9: Feature Importance
- Generated feature importance charts
- Identified top predictors
- Analyzed socioeconomic factors

### ✅ Objective 10: Regional Insights
- Segmented California into regions
- Analyzed regional statistics
- Identified hot zones for investment

### ✅ Objective 11: Interactive Web Application
- Built Streamlit dashboard
- Implemented real-time predictions
- Created interactive visualizations

---

## 📈 Model Performance

### Baseline Models (Original Features)
- **Linear Regression**: R² ≈ 0.60, RMSE ≈ 0.73
- **Random Forest**: R² ≈ 0.81, RMSE ≈ 0.51

### Engineered Models (With Additional Features)
- **Linear Regression**: R² ≈ 0.64, RMSE ≈ 0.69
- **Random Forest**: R² ≈ 0.84, RMSE ≈ 0.47

**Best Model**: Random Forest with engineered features
- Achieves 84% variance explanation
- Average prediction error: ~$47,000

---

## 🗺️ Key Findings

### Top Factors Affecting House Prices:
1. **Median Income** (Correlation: 0.688) - Strongest predictor
2. **Location** - Coastal areas command premium prices
3. **Average Rooms** - More rooms correlate with higher values
4. **House Age** - Newer properties more valuable
5. **Luxury Score** - Engineered feature showing strong predictive power

### Regional Insights:
- **Highest Values**: Coastal regions (SF Bay, LA, San Diego)
- **Best Affordability**: Northern Inland and Southern Inland
- **Hot Zones**: Top 20% districts concentrated on coast
- **Investment Opportunity**: Central Valley shows growth potential

---

## 💻 Web Application Features

### 🏡 Home Page
- Dataset overview and statistics
- Quick metrics dashboard
- Sample data preview

### 🔍 Data Explorer
- Feature distribution visualizations
- Correlation heatmap
- Interactive feature relationships

### 🎯 Price Predictor
- Real-time price prediction
- Slider inputs for all features
- Multiple model predictions

### 📈 Model Performance
- Performance metrics comparison
- Feature importance analysis
- Visual model evaluation

### 🗺️ Geographic Analysis
- Interactive heatmap
- Regional statistics
- Price distribution by location

### 💡 Insights
- Key findings summary
- Investment recommendations
- Correlation analysis

---

## 🛠️ Technology Stack

### Data Science & ML
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **scikit-learn**: Machine learning models
- **matplotlib & seaborn**: Data visualization

### Web Application
- **Streamlit**: Interactive web framework
- **Folium**: Geographic visualizations
- **streamlit-folium**: Streamlit-Folium integration

### Development
- **Jupyter Notebook**: Interactive development
- **Python 3.8+**: Core programming language

---

## 📝 Usage Guide

### Making Predictions with the Web App

1. **Launch the app**: `streamlit run streamlit_app.py`
2. **Navigate to Price Predictor**
3. **Adjust sliders** for property features:
   - Median Income
   - House Age
   - Average Rooms
   - Average Bedrooms
   - Population
   - Average Occupancy
   - Latitude & Longitude
4. **Click "Predict Price"** to get estimation
5. **View results** from both models

### Exploring Data

1. **Go to Data Explorer page**
2. **Select features** to visualize
3. **View distributions** and relationships
4. **Analyze correlations** using heatmap

### Analyzing Regions

1. **Visit Geographic Analysis page**
2. **Explore interactive map**
3. **Adjust sample size** for heatmap
4. **Review regional statistics**

---

## 📊 Deliverables

### Code Files
- ✅ `housing_price_prediction_datathon.ipynb` - Complete analysis notebook
- ✅ `streamlit_app.py` - Interactive web application
- ✅ `requirements.txt` - Package dependencies

### Generated Reports
- ✅ `housing_prediction_insights_report.txt` - Comprehensive findings
- ✅ `regional_analysis.csv` - Regional statistics
- ✅ `california_housing_heatmap.html` - Interactive map

### Trained Models
- ✅ `best_rf_model.pkl` - Random Forest model
- ✅ `lr_model.pkl` - Linear Regression model
- ✅ `feature_names.pkl` - Feature metadata

### Processed Data
- ✅ `housing_data_engineered.csv` - With engineered features
- ✅ `housing_data_regional.csv` - With regional labels

---

## 🎓 Learning Outcomes

This project demonstrates:
- End-to-end machine learning pipeline
- Feature engineering techniques
- Model evaluation and comparison
- Geographic data visualization
- Web application deployment
- Real-world data analysis

---

## 📚 References

1. Scikit-learn Documentation: https://scikit-learn.org/
2. Streamlit Documentation: https://docs.streamlit.io/
3. Folium Documentation: https://python-visualization.github.io/folium/
4. California Housing Dataset: https://scikit-learn.org/stable/datasets/real_world.html

---

## 🤝 Contributing

This is a Datathon project. For educational purposes and improvements:
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Team Information

**Datathon 2025 - Problem Statement #11**

Team Members: [Add your team name and members here]

---

## 🙏 Acknowledgments

- California Housing Dataset from scikit-learn
- Streamlit team for the amazing framework
- Folium contributors for geographic visualizations
- Datathon organizers for the opportunity

---

## 📞 Support

For questions or issues:
- Open an issue in the repository
- Contact team members
- Review the comprehensive documentation in the notebook

---

**Built with ❤️ for Datathon 2025**

Happy Predicting! 🏠📊🚀
