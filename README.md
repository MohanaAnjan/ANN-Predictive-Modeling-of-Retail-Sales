# ANN-Predictive-Modeling-of-Retail-Sales
## Sales Forecasting and Markdown Optimization.

## Overview
This project aims to develop a predictive Artificial Neural Network (ANN) model to forecast department-wide sales for each store over the next year and analyze the impact of markdowns on sales during holiday weeks. The goal is to provide actionable insights and recommendations to optimize markdown strategies and improve inventory management.

## Business Use Cases
Sales Forecasting: Accurate sales predictions to aid in inventory management and procurement.
Markdown Strategy Optimization: Identifying the best times and departments for markdowns to maximize sales.
Holiday Planning: Understanding holiday sales patterns to better prepare for peak periods.
Resource Allocation: Efficiently allocate resources and staff based on anticipated sales volumes.
Revenue Maximization: Strategies to increase overall revenue and profitability through informed decision-making.
## Approach
### Data Cleaning and Preparation:

Handle missing values.
Convert date formats.
Ensure appropriate data types.

### Exploratory Data Analysis (EDA):
Analyze sales trends.
Assess the impact of holidays on sales.
Explore feature correlations.

### Feature Engineering:

Create lag features (e.g., past sales).
Generate holiday-specific features.
Develop interaction features between markdowns and holidays.
### Modeling:

Implement Time Series Models.
Develop Deep Learning Models, including ANN using TensorFlow with various architectures.
Analyze markdown impact using statistical techniques.
### Evaluation:

Use metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE) to assess model performance.
Focus on performance during holiday weeks.
###  Insights and Recommendations:

Provide actionable insights based on model results.
Recommend strategies for sales and markdown optimization.

## Prerequisites
Python 3.x
Libraries: pandas, numpy, matplotlib, seaborn, tensorflow, scikit-learn, statsmodels
Jupyter Notebook (for interactive development and visualization)
## Usage
### Data Preparation:

Place your dataset in the data/ directory.
Modify the data_preparation.py script if necessary to match your data format.
Exploratory Data Analysis:

Run the EDA notebook eda.ipynb to explore and visualize data.
### Feature Engineering:

Adjust feature engineering in feature_engineering.py as needed.
### Model Training:

Train models using model_training.py.
Use the Jupyter Notebook model_training.ipynb for an interactive training process.
### Evaluation:

Assess model performance in model_evaluation.py.
Review results and metrics in the evaluation_results/ directory.
Insights and Recommendations:

Analyze the final insights and recommendations in insights_recommendations.ipynb.
## Results
### The project will provide:

Sales Forecasts: Predicted sales for each store and department over the next year.
Markdown Impact Analysis: Insights into how markdowns affect sales, especially during holiday periods.
Actionable Recommendations: Strategies for optimizing markdowns and improving inventory management.


## Contact
For any questions or feedback, please reach out to av.mohananjan26101997@gmail.com.
