# 🚀 Store Staffing Analyzer

A data-driven staffing optimization tool for retail clothing stores. The application reads historical sales data from a CSV, performs exploratory data analysis, forecasts future demand, computes financial metrics, and recommends optimal staffing levels to maximize gross margins.

This project demonstrates advanced data science workflows, time-series forecasting, scenario modeling, and automated professional PDF report generation.

---

# 📊 Project Overview

A branded-clothing store has been operating with a general policy of maintaining a fixed number of sales staff. Following a recent resignation, the owner needs to decide whether to hire a replacement. This tool provides a comprehensive, data-driven answer by:

* Performing exploratory data analysis (EDA) on historical sales data
* Analyzing salesperson productivity and diminishing returns
* Computing financial metrics including gross margin and break-even points
* Forecasting sales for the next 6 months using statistical models
* Comparing staffing scenarios with financial projections
* Generating a professional recommendation with full supporting evidence

---

# ✅ Key Highlights

* 🏗️ Automated End-to-End Analysis Pipeline
* 🎨 Publication-Quality Charts & Visualizations (14+ plots)
* 🔗 Dynamic Data Ingestion (Handles arbitrary sales team sizes)
* 📝 Advanced Sales Forecasting with Auto-Model Selection
* 🏷️ Multi-Scenario Staffing Comparison
* ⚡ Command-Line Interface with customizable parameters
* 📈 Professional PDF Report Generation via `fpdf2`
* 🌐 Fully dynamic computations (No hardcoded strings or variables)

---

# 🧰 Tech Stack

## Data Analysis & Modeling
* Python 3.9+
* Pandas & NumPy
* Scikit-Learn
* Statsmodels
* SciPy

## Visualization & Reporting
* Matplotlib & Seaborn
* FPDF2

---

# 🔬 Project Workflow

## 1. 🏗️ Exploratory Data Analysis & Visualization
The pipeline begins by loading and validating the historical sales CSV. It dynamically detects the active workforce and generates over 14 charts, including:
* Monthly Sales Trends with 6-Month Rolling Averages
* Salesperson Productivity Comparisons
* Sales Distributions and Correlation Heatmaps
* Employee Count vs. Gross Margin Scatter Plots

## 2. 🎨 Financial Modeling
Calculates crucial business metrics to ground the staffing recommendation:
* **Gross Margin Formula:** `(1 - COGS_rate - commission_rate) × Sales − salary × Employees`
* Historical Margin Trends and Summary Statistics
* Break-Even Sales per Employee calculations

## 3. 📝 Sales Forecasting
Evaluates three different time-series models on a 12-month hold-out validation set:
* Moving Average (6-month window)
* Linear Regression
* Exponential Smoothing (Holt-Winters)

The model with the lowest Root Mean Squared Error (RMSE) is automatically selected to generate the 6-month forward-looking forecast.

## 4. 🌐 Staffing Optimization & Scenario Comparison
Models the financial impact of four distinct hiring scenarios:
* **Scenario A:** Do Not Hire (Maintain current staffing)
* **Scenario B:** Hire Immediately (Add one employee now)
* **Scenario C:** Hire After 2 Months (Wait, then hire)
* **Scenario D:** Conditional Hire (Hire only if sales exceed a break-even threshold)

The pipeline incorporates a new-hire ramp-up curve (70% productivity moving to 100% over 6 months) and outputs an **Adaptive Staffing Rule** based on a 3-month rolling sales average.

---

# 📈 Example Response

```text
==================================================================
[OK] Analysis completed successfully in 9.4 seconds!
==================================================================

[FILES] Output files:
   - Recommendation:    output/recommendation.txt
   - Analysis Results:  output/analysis_results.csv
   - Charts:            output/charts/*.png
   - Report:            reports/Recommendation_Report.pdf

------------------------------------------------------------------
[SUMMARY] QUICK RECOMMENDATION SUMMARY
------------------------------------------------------------------
     1. EXECUTIVE SUMMARY
     Current staff count       : 7
     Recommended action        : Hire Immediately (8 from month 1)
     Avg monthly sales (Lakhs) : Rs.103.40
     Avg monthly margin (Lakhs): Rs.25.42
```

---

# 📉 Features Implemented

### Part 1 – Data Ingestion
* Dynamic CSV Parsing
* Missing value handling and active employee calculation

### Part 2 – Forecasting
* Validation-based Model Selection
* Confidence Band Generation

### Part 3 – Financial & Staffing Engine
* Break-even calculation
* Multi-scenario projection
* Diminishing returns analysis

### Part 4 – Reporting
* Plain-text Recommendation Output
* CSV Metrics Export
* Automated PDF Compilation

---

# 📁 Project Structure

```text
RevInsight_Assignment/
│
├── data/
│   ├── RevI-Test.csv              
│   └── RevI-Test-Large.csv
│
├── src/
│   ├── __init__.py                
│   ├── analyze_store.py           
│   ├── forecasting.py             
│   ├── financial_analysis.py      
│   ├── report_generator.py        
│   ├── visualization.py           
│   ├── staffing_model.py          
│   └── utils.py                   
│
├── output/
│   ├── recommendation.txt         
│   ├── analysis_results.csv       
│   └── charts/                    
│
├── reports/
│   └── Recommendation_Report.pdf  
│
├── README.md                      
├── requirements.txt               
└── run.py                         
```

---

# ▶️ Running the Project

## Prerequisites

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

## Basic Execution

Run the analysis pipeline against a dataset:

```bash
python run.py data/RevI-Test.csv
```

## Custom Parameters

You can adjust assumptions like salary and commission rate via the CLI:

```bash
python run.py data/RevI-Test.csv --salary 3.5 --commission 0.06 --output-dir my_analysis
```

---

# 🚀 Future Improvements

* **Interactive Dashboard:** Build a Streamlit/Dash web app for real-time analysis
* **Advanced Forecasting:** Incorporate ARIMA, Prophet, or LSTM models
* **External Data Integration:** Add weather, mall footfall, and holiday data
* **Employee Performance Tracking:** Individual-level ramp-up curves
* **Multi-Store Support:** Extend to analyze multiple store locations

---

# 📬 Contact

Made with ❤️ by Bishwaman Das
