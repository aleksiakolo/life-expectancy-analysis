# Data Dictionary

This data dictionary describes the **raw datasets** used in the Life Expectancy Analysis project and the **intended merged panel schema**.

**Raw data files in this repository**
- WHO: `data/raw/world_health_organization/who.csv`
- World Bank (panel / socio-economic features): `data/raw/world_bank/wb.csv`
- World Bank (WDI export, wide by year): `data/raw/world_bank/wdi.csv`
- World Bank metadata: `data/raw/world_bank/metadata_country.csv`, `data/raw/world_bank/metadata_indicator.csv`

---

## 1) Canonical Merge Keys 

The project will build a merged country–year panel using:

- **country** (standardized country name; optionally ISO3 later)
- **year** (integer)

These keys are present in both WHO and WB panel datasets (with different column names in raw).

---

## 2) WHO Dataset (Raw) — `who.csv`

### Raw Columns 
The WHO dataset contains one row per **Country–Year** with health and socio-economic indicators.  
Key fields include:

- `Country` (string)
- `Year` (integer)
- `Status` (categorical; Developed/Developing)
- `Life expectancy ` (numerical; note trailing space in raw header)
- `Adult Mortality` (numerical)
- `infant deaths` (numerical)
- `Alcohol` (numerical)
- `percentage expenditure` (numerical)
- `Hepatitis B` (numerical)
- `Measles ` (numerical; note trailing space)
- ` BMI ` (numerical; note leading/trailing spaces)
- `under-five deaths ` (numerical; note trailing space)
- `Polio` (numerical)
- `Total expenditure` (numerical)
- `Diphtheria ` (numerical; note trailing space)
- ` HIV/AIDS` (numerical; note leading space)
- `GDP` (numerical)
- `Population` (numerical)
- ` thinness  1-19 years` (numerical; note leading space + double spaces)
- `thinness 5-9 years` (numerical)
- `Income composition of resources` (numerical)
- `Schooling` (numerical)

### Planned Standardization 
- Trim whitespace in column names (e.g., `Life expectancy ` → `life_expectancy`)
- Standardize `Country` naming to match WB (mapping file)
- Ensure `Year` is integer and unique per (Country, Year)

---

## 3) World Bank Dataset (Panel, Raw) — `wb.csv`

### Raw Columns 
This WB dataset is already in **country–year panel format** and includes socio-economic predictors.

- `Country Name` (string)
- `Country Code` (string; ISO3)
- `Region` (string)
- `IncomeGroup` (string)
- `Year` (integer)
- `Life Expectancy World Bank` (numerical)
- `Prevelance of Undernourishment` (numerical; note spelling)
- `CO2` (numerical)
- `Health Expenditure %` (numerical)
- `Education Expenditure %` (numerical)
- `Unemployment` (numerical)
- `Corruption` (numerical)
- `Sanitation` (numerical)
- `Injuries` (numerical)
- `Communicable` (numerical)
- `NonCommunicable` (numerical)

### Planned Standardization
- Standardize column names to snake_case (e.g., `Country Name` → `country`)
- Fix typos consistently (e.g., `Prevelance...` → `prevalence_undernourishment`)
- Ensure one row per (Country Name, Year)
- Use `Country Code` as optional ISO3 join helper

---

## 4) World Bank WDI Export (Raw) — `wdi.csv`

This file is in standard WDI export format:
- One row per **(Country, Indicator)**
- One column per year (1960–2024)

Key identifying columns:
- `Country Name`
- `Country Code`
- `Indicator Name`
- `Indicator Code`
- Year columns: `1960`, `1961`, ..., `2024`

### Intended Use
This dataset is **optional** and can be used to:
- validate values against official WDI indicators
- add additional predictors not present in `wb.csv`

### Planned Processing 
- Reshape wide → long with columns: (country, year, indicator_code, value)
- Filter to selected indicators
- Pivot long → wide to create a country–year panel

---

## 5) Merged Dataset Schema 

The merged dataset is created by joining:

- WHO cleaned (target + health predictors)
- WB panel cleaned (socio-economic predictors)

on:
- `country` (standardized name; optional ISO3 mapping)
- `year`

### Planned Core Columns
| Column | Source | Meaning | Type | Notes |
|---|---|---|---|---|
| `country` | WHO/WB | Standardized country name | categorical | derived from `Country` / `Country Name` |
| `year` | WHO/WB | Observation year | integer | merge key |
| `life_expectancy` | WHO | Life expectancy at birth (years) | numeric | target variable (canonical) |
| `status` | WHO | Developed/Developing | categorical | may be encoded later |
| `adult_mortality` | WHO | Adult mortality rate/proxy | numeric | raw: `Adult Mortality` |
| `infant_deaths` | WHO | Infant deaths count/rate field | numeric | raw: `infant deaths` |
| `schooling` | WHO | Years of schooling | numeric | raw: `Schooling` |
| `gdp` | WHO | GDP (WHO field) | numeric | raw: `GDP` |
| `population` | WHO | Population | numeric | raw: `Population` |
| `region` | WB | World region | categorical | from `Region` |
| `income_group` | WB | Income group | categorical | from `IncomeGroup` |
| `co2` | WB | CO2 measure | numeric | from `CO2` |
| `health_expenditure_pct` | WB | Health expenditure % | numeric | from `Health Expenditure %` |
| `education_expenditure_pct` | WB | Education expenditure % | numeric | from `Education Expenditure %` |
| `unemployment` | WB | Unemployment rate | numeric | from `Unemployment` |
| `sanitation` | WB | Sanitation indicator | numeric | from `Sanitation` |
| `undernourishment_prev` | WB | Prevalence of undernourishment | numeric | raw typo needs standardization |

**Note:** Additional WHO predictors (immunization rates, disease indicators) are included after cleaning and column standardization.

---

## 6) Cross-Dataset Compatibility Notes (Week 1)

- **Country naming mismatches are expected** between WHO and WB.
  - Plan: create mapping file `data/external/country_name_map.csv` to harmonize names.
  - Optional: use `Country Code` ISO3 for WB as a stable identifier if mapping is extensive.
- **Time coverage differs** across datasets and will be assessed during missingness diagnostics.
- **Units vary** across predictors; scaling/transformations will be applied only after train/test split during modeling.

---
