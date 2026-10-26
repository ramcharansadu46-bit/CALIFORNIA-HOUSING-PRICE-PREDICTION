# Dataset documentation - California Housing Prices

| Item | Detail |
|---|---|
| Content | 20,640 census block groups (districts) from the 1990 US Census; 9 predictors + 1 target |
| Prediction target | `median_house_value` (USD), right-censored at 500,001 |
| Original source | StatLib repository; Pace, R. K. and Barry, R. (1997), *Statistics & Probability Letters* 33(3), 291-297 |
| Kaggle listing | https://www.kaggle.com/datasets/camnugent/california-housing-prices |
| OpenML listing | https://api.openml.org/d/43705/json (dataset ID 43705) |
| File used here | https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv |
| SHA-256 of `raw/housing.csv` | `8a3727f4cf54ac1a327f69b1d5b4db54c5834ea81c6e4efc0d163300022a685e` |
| Licence | Check the licence shown on the Kaggle page before redistributing |

**How the file was obtained.** Direct Kaggle download was not possible in the environment where the project was built, so the same file was downloaded from the public GitHub repository of A. Geron (2019). The Kaggle page itself could not be opened from that environment; size and origin were cross-checked against the Kaggle description as republished by a third-party mirror (Baselight), OpenML metadata and Keras documentation. **Please compare the checksum with a file downloaded from Kaggle.** To verify against Kaggle yourself, download the file there and compare checksums (`sha256sum housing.csv` / `certutil -hashfile housing.csv SHA256` on Windows).

## Data dictionary
| Variable | Type | Unit | Description |
|---|---|---|---|
| longitude | float | degrees | Block-group longitude |
| latitude | float | degrees | Block-group latitude |
| housing_median_age | float | years | Median age of houses (capped at 52) |
| total_rooms | float | count | Total rooms in the block group |
| total_bedrooms | float | count | Total bedrooms (207 missing) |
| population | float | persons | Block-group population |
| households | float | count | Number of households |
| median_income | float | 10,000 USD | Median household income (capped at 15.0001) |
| median_house_value | float | USD | **Target** (capped at 500,001) |
| ocean_proximity | categorical | - | <1H OCEAN, INLAND, NEAR OCEAN, NEAR BAY, ISLAND |

## Known quality issues (see report Section 3.2)
207 missing `total_bedrooms`; 965 capped targets (4.68 %); 1273 capped ages (6.17 %); 49 capped incomes; no duplicates.

`processed/train.csv` and `processed/test.csv` are the 80/20 income-stratified split (seed 42) produced by the notebook.
