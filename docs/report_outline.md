# Life Expectancy Analysis

## Project Objective
The goal of this project is to **identify, explain, and quantify the key factors that drive life expectancy across countries**.  
We combine **descriptive analysis**, **statistical inference**, and **predictive modeling** to understand how health, economic, and social variables relate to life expectancy outcomes.

The project focuses on both:
- **Explanation**: Which factors are most strongly associated with life expectancy, and how?
- **Prediction**: How accurately can life expectancy be predicted using available indicators?

---

## Data Sources
This project integrates data from two authoritative global sources:

- **World Health Organization (WHO)**  
  Health indicators such as mortality rates, immunization coverage, disease prevalence, and life expectancy.
  - https://www.who.int/data

- **World Bank**  
  Economic and social indicators such as GDP, education, population, and health expenditure.
  - https://data.worldbank.org

---

## Repository Structure

life-expectancy-analysis/
├── data/
│ ├── raw/ # Original WHO & World Bank datasets
│ ├── processed/ # Cleaned and merged datasets
│
├── notebooks/
│ ├── 01_exploration.ipynb
│ ├── 02_cleaning.ipynb
│ ├── 03_analysis.ipynb
│
├── src/
│ ├── ingestion.py
│ ├── preprocessing.py
│ ├── modeling.py
│
├── docs/
│ ├── data_dictionary.md
│ ├── methodology.md
│
├── requirements.txt
├── README.md
└── .gitignore

---

## Setup Instructions

### 1. Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## How to Run the Analysis Pipeline

The project follows a modular and reproducible analysis pipeline.

### 1. Data Ingestion
- Load raw WHO and World Bank datasets into `data/raw/`.

### 2. Data Cleaning & Preprocessing
- Handle missing values
- Normalize numerical variables
- Encode categorical variables

### 3. Dataset Merging
- Merge datasets using **country** and **year** as primary keys.

### 4. Analysis & Modeling
- Exploratory Data Analysis (EDA)
- Statistical inference and hypothesis testing
- Predictive modeling

Execution is primarily performed through the notebooks in the following order:
01_exploration → 02_cleaning → 03_analysis


---

## Outputs
- Cleaned and merged analytical dataset
- Exploratory visualizations
- Statistical inference results (correlations, regressions, hypothesis tests)
- Predictive model performance metrics
- (Optional) Interactive dashboard

---

## Project Roadmap

### Week 1 — Domain Understanding & Data Collection
- Research key determinants of life expectancy
- Identify and collect WHO and World Bank datasets

### Week 2 — Data Cleaning & Feature Engineering
- Handle missing data and inconsistencies
- Normalize and transform features
- Merge datasets into a unified analytical table

### Week 3 — Analysis & Modeling
- Descriptive and comparative analysis
- Correlation and regression analysis
- Predictive modeling

### Week 4 — Insights & Reporting
- Interpret results and derive insights
- Prepare final report and presentation
- (Optional) Dashboard development
