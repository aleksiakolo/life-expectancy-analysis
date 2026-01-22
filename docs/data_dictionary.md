# Data Dictionary

| Feature Name           | Source       | Meaning / Description                                  | Type        | Missingness | Notes |
|------------------------|--------------|--------------------------------------------------------|-------------|-------------|-------|
| country                | WHO / WB     | Country name                                           | Categorical | Low         | Standardized naming |
| year                   | WHO / WB     | Observation year                                       | Integer     | None        | Used as merge key |
| life_expectancy        | WHO          | Average life expectancy at birth (years)               | Numerical   | Medium      | Target variable |
| adult_mortality        | WHO          | Probability of dying between ages 15–60                | Numerical   | Medium      | Strong inverse relation |
| infant_deaths          | WHO          | Number of infant deaths per 1000 births                | Numerical   | Low         | Highly skewed |
| immunization_rate      | WHO          | Immunization coverage (%)                              | Numerical   | Medium      | Capped at [0,100] |
| gdp_per_capita         | World Bank   | GDP per capita (USD)                                   | Numerical   | Medium      | Log-transformed |
| health_expenditure     | World Bank   | Health spending per capita                             | Numerical   | High        | Imputed |
| schooling_years        | World Bank   | Average years of schooling                             | Numerical   | Medium      | Strong positive driver |
| population             | World Bank   | Total population                                       | Numerical   | Low         | Log-scaled |
| status                 | WHO          | Development status (Developed / Developing)            | Categorical | None        | Encoded |


## WHO Dataset — Schema Notes

| Item | Description |
|---|---|
| Country Identifier | Country name (string) |
| Time Field | Year (integer) |
| Uniqueness | One row per (country, year) |
| Target Variable | Life expectancy at birth |
| Units | Years |
| Missingness | Present in early years / low-income countries |
| Notes | Some indicators may change definitions over time |

## World Bank Dataset — Schema Notes

| Item | Description |
|---|---|
| Country Identifier | Country name (string) |
| Time Field | Year (integer) |
| Uniqueness | One row per (country, year, indicator) |
| Units | Mixed (% GDP, per 1,000, absolute values) |
| Missingness | High for some indicators and years |
| Notes | Requires pivoting to wide format before merge |

## Cross-Dataset Compatibility

| Aspect | Observation |
|---|---|
| Join Key | (country, year) |
| Country Name Consistency | Requires harmonization |
| Time Coverage | Uneven across datasets |
| Scale Issues | Normalization required before modeling |
