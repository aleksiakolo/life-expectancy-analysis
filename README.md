# Life Expectancy Analysis

## Project Objective
The goal of this project is to **identify, explain, and quantify the key factors that drive life expectancy across countries**.

We construct a longitudinal country–year dataset by combining health indicators from the World Health Organization (WHO) with socio-economic indicators from the World Bank. The project integrates:

- descriptive statistics
- exploratory data analysis (EDA)
- statistical inference
- predictive modeling

The project addresses two complementary questions:

**Explanation**
- Which factors are most strongly associated with life expectancy?
- How do health, economic, and social indicators interact?

**Prediction**
- How accurately can life expectancy be estimated from available indicators?
- Which variables contribute most to predictive performance?

---

## Data Sources

The analysis combines two international public datasets.

### 1) World Health Organization (WHO)
Contains health and demographic indicators.

Examples of variables:
- life expectancy at birth (target variable)
- adult mortality
- infant deaths
- immunization coverage
- disease indicators (e.g., HIV/AIDS)
- schooling
- BMI

File location: data/raw/world_health_organization/who.csv


Schema characteristics:
- One row per `(Country, Year)`
- Health-focused variables
- Main source of the **target variable**

---

### 2) World Bank Socio-Economic Panel
Contains economic and infrastructure indicators.

Examples of variables:
- sanitation access
- unemployment
- education expenditure
- CO₂ emissions
- undernourishment
- health expenditure
- income group and region

File location: data/raw/world_bank/wb.csv


Schema characteristics:
- One row per `(Country Name, Year)`
- Socio-economic predictors

---

### Validation
World Bank WDI export: data/raw/world_bank/wdi.csv

---

## Project Structure

```text
life-expectancy-analysis/
│
├── data/
│ ├── raw/
│ │ ├── world_health_organization/
│ │ │ └── who.csv
│ │ └── world_bank/
│ │ ├── wb.csv
│ │ ├── metadata_country.csv
│ │ ├── metadata_indicator.csv
│ │ └── wdi.csv
│ │
│ ├── interim/ # intermediate cleaned tables
│ └── processed/ # final merged analytical dataset
│
├── notebooks/
│ ├── 01_data_understanding.ipynb
│ ├── 02_cleaning_and_merge.ipynb
│ └── 03_eda_and_analysis.ipynb
│
├── src/
│ ├── clean/
│ │ ├── inspect_missingness.py
│ │ ├── filter_entities.py
│ │ ├── impute.py
│ │ ├── merge.py
│ │ └── scale_features.py
│ │
│ └── utils/
│ └── config.py
│
├── docs/
│ ├── data_dictionary.md
│ └── methodology.md
│
├── reports/
│ ├── figures/
│ └── tables/
│
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Create virtual environment
From the project root:

```bash
python3 -m venv life_expectancy_env
source life_expectancy_env/bin/activate
```

Verify the environment is active:

```bash
which python
```

It should point inside:

```
life-expectancy-analysis/life_expectancy_env/bin/python
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Workflow

The project follows a staged analysis pipeline.

### Step 1 — Data Understanding

Goals:

* understand dataset schemas
* inspect variable meanings
* check coverage (countries, years)
* identify merge keys

Outputs:

* dataset summaries
* variable descriptions
* missingness overview

Notebook:

```
notebooks/01_data_understanding.ipynb
```

### Step 2 — Cleaning and Merge

Tasks:

* standardize country names
* harmonize year format
* detect duplicates
* handle missing values
* merge WHO and World Bank datasets

Output:

```
data/processed/panel_dataset.csv
```

Notebook:

```
notebooks/02_cleaning_and_merge.ipynb
```

### Step 3 — Exploratory Data Analysis

Tasks:

* distributions of life expectancy
* correlations with predictors
* regional comparisons
* time trends

Notebook:

```
notebooks/03_eda_and_analysis.ipynb
```

### Step 4 — Modeling & Insights
Tasks:

* regression modeling
* feature importance
* interpretation in policy context
* report preparation

---

## Key Merge Definition

The merged dataset is a **panel dataset** with:

**Primary key**

```
(country, year)
```

Each row represents:

> one country in one year with health and socio-economic indicators.

---

## Outputs

The project will produce:

* cleaned merged dataset
* summary statistics tables
* visualizations
* correlation analysis
* regression results
* final written report

Outputs saved in:

```
data/processed/
reports/figures/
reports/tables/
```

---

## Project Roadmap

### Week 1 — Domain Understanding

* inspect datasets
* confirm schema
* verify merge compatibility

### Week 2 — Cleaning & Feature Engineering

* handle missing values
* standardize entities
* build merged panel dataset

### Week 3 — Analysis

* EDA
* correlations
* comparative analysis

### Week 4 — Interpretation & Reporting

* modeling
* insights
* final report and presentation
