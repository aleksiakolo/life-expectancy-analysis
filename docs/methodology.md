# Methodology

## Data Preprocessing
- Standardized country names and year formats
- Removed duplicate country–year entries
- Filtered implausible values (e.g., negative life expectancy)

## Missing Value Handling
- Low missingness: row-wise deletion
- Medium missingness: median imputation (numeric)
- High missingness: retained only if analytically essential
- No imputation for target variable (life expectancy)

## Feature Transformations
- Log transformation for skewed variables (GDP, population)
- Standard scaling for regression-based models
- One-hot encoding for categorical variables

## Statistical Analysis
- Correlation analysis (Pearson & Spearman)
- Linear regression for continuous relationships
- Hypothesis testing for group comparisons
- Variance inflation factor (VIF) for multicollinearity checks

## Predictive Modeling
- Baseline linear regression
- Regularized models (Ridge / Lasso)
- Train-test split with cross-validation
- Evaluation using RMSE and R²

## Interpretation Focus
- Emphasis on explainability over pure prediction accuracy
- Results interpreted in public-health and policy context

# Data Preprocessing Planning

This section defines the concrete preprocessing decisions that will guide all downstream analysis and modeling. The goal is to create a clean, consistent country–year panel dataset while minimizing bias introduced by missing data, scale differences, and schema mismatches.

---

### Missing Value Patterns & Handling Strategy

#### Diagnostics to Compute
In the cleaning notebook / script, compute:

- **% missing per column**
- **% missing per country**
- **% missing per year**
- (Optional) **Missingness heatmap** to visualize structure and clustering

These diagnostics will be saved as tables/figures and referenced in the report.

#### Predefined Decisions
- **Feature-level filtering**  
  - Drop features with **> 40–60% missingness** (threshold finalized after inspection).
- **Country-level filtering**  
  - Drop countries with **fewer than 5 observed years** of life expectancy.
- **Time-series gaps**
  - Interpolation allowed **only** when:
    - Gaps are short (1–2 years)
    - Indicator is slow-moving (e.g., life expectancy, education)
  - Otherwise:
    - Use grouped imputation (e.g., region/income group medians)
    - Or model-based imputation if justified
- **Target variable**
  - Rows with missing **life expectancy** are always dropped.

All imputation decisions will be explicitly documented and justified.

---

### Categorical vs Numerical Variables

#### Categorical Variables
- Country
- Region / income group (World Bank)
- Year  
  - Treated as numeric for trends
  - Treated as categorical for fixed-effects models

#### Numerical Variables
- **Target**
  - Life expectancy at birth
- **Health**
  - Immunization rates
  - Mortality rates
  - Health expenditure
- **Economic**
  - GDP per capita
  - Poverty / unemployment proxies
- **Social & Infrastructure**
  - Sanitation access
  - Clean water access
- **Education & Demographics**
  - Schooling years
  - Fertility rate
  - Population growth

This separation is fixed early to avoid leakage and inconsistent transformations.

---

### Normalization & Scaling Plan

#### Transformations
- **Log transforms** for heavy-tailed variables:
  - GDP per capita
  - Health expenditure
  - Population-related measures
- Add small constants where needed to handle zeros.

#### Standardization (for modeling)
For numerical predictors used in regression / ML models:

$$
x' = \frac{x - \mu}{\sigma}
$$

- Applied **after train/test split**
- Parameters ($\mu, \sigma$) learned on training data only

#### Interpretability Rule
- Keep **original-scale variables** alongside transformed versions
- Use original scale for:
  - Summary tables
  - Interpretation
  - Final reporting

---

### Dataset Merge Strategy

#### Target Structure
A clean **panel dataset** with:

- **Key:** (country, year)
- One row per country–year

#### Join Logic
- Merge **WHO ↔ World Bank** on (country, year)

#### Known Issues & Solutions
- **Country name mismatches**
  - Create explicit mapping file:
    ```
    data/external/country_name_map.csv
    ```
- **ISO vs name mismatch**
  - Prefer converting all datasets to **ISO3 country codes**
- **Join type**
  - Start with **inner join** to guarantee complete cases for modeling
  - Measure and report:
    - Number of rows lost
    - Countries/years dropped due to mismatches

Coverage loss statistics will be reported transparently.

---

### Target Variable & Predictor Grouping

#### Target Variable
- `life_expectancy` (WHO)

#### Predictor Groups
These groups will be used consistently across:
EDA → inferential testing → modeling → reporting

**Health System & Immunization**
- DPT, measles, polio immunization rates
- Health expenditure indicators

**Mortality & Disease Burden**
- Infant mortality
- Under-5 mortality
- Adult mortality
- HIV / TB indicators (if available)

**Economic**
- GDP per capita
- Poverty and employment proxies

**Social & Infrastructure**
- Sanitation access
- Clean water access

**Education & Demographics**
- Schooling years
- Fertility rate
- Population growth

This grouping directly informs:
- EDA section structure
- Regression blocks
- Feature importance analysis
- Final interpretation

# Exploratory Data Analysis (EDA)

This section explores the structure, distributions, and relationships in the data prior to formal statistical testing or modeling. The goal of EDA is to understand patterns, identify anomalies, and motivate subsequent inferential analysis.

---

## 1. Target Variable: Life Expectancy

### Key Questions
- How is life expectancy distributed across countries?
- How has life expectancy evolved over time?
- Do life expectancy levels differ systematically by region or income group?

### Plots
- Histogram / density plot of life expectancy
- Line plot of life expectancy over time
- Boxplot of life expectancy by region or income group

---

## 2. Health System & Immunization

### Variables
- Immunization rates (DPT, measles, polio)
- Health expenditure indicators

### EDA Focus
- Correlation between immunization coverage and life expectancy
- Relationship between health system investment and outcomes
- Missingness patterns across countries and years

### Plots
- Scatter plots: immunization rate vs life expectancy
- Time trends of immunization coverage
- Faceted plots by income group

---

## 3. Mortality & Disease Burden

### Variables
- Infant mortality
- Under-5 mortality
- Adult mortality
- HIV/TB indicators (if available)

### EDA Focus
- Inverse relationships with life expectancy
- Nonlinear effects and diminishing returns
- Identification of outliers and extreme cases

### Plots
- Log-scale scatter plots
- Correlation heatmap of mortality indicators
- Country-level life expectancy and mortality trajectories

---

## 4. Economic Indicators

### Variables
- GDP per capita
- Poverty proxy indicators

### EDA Focus
- Diminishing returns of income on life expectancy
- Cross-country and cross-income inequality

### Plots
- Log(GDP per capita) vs life expectancy
- Stratification by income group
- Economic trends over time

---

## 5. Social & Infrastructure

### Variables
- Sanitation access
- Clean water access

### EDA Focus
- Threshold effects in infrastructure coverage
- Bias introduced by missing or uneven data coverage

### Plots
- Life expectancy vs sanitation / water access
- Regional comparisons

---

## 6. Education & Demographics

### Variables
- Mean years of schooling
- Fertility rate
- Population growth

### EDA Focus
- Structural demographic relationships
- Long-term trends and transitions

### Plots
- Scatter plots with LOWESS smoothing
- Time series by region or income group

---

## 7. Multivariate Structure (Optional)

### EDA Tools
- Correlation matrix of predictors and target
- Principal Component Analysis (PCA) for structural intuition (not modeling)


