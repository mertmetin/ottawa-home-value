# Results (data/sample.csv)

## Cleaning

| Step | Removed | Why |
|---|---|---|
| input | 0 | 6060 rows |
| not sold | 60 | closed but not a completed sale (e.g. Deal Fell Through) |
| no price or date | 0 | ClosePrice or sale date missing |
| property type | 59 | land, farm, mobile home, other |
| duplicate key | 0 | same ListingKey twice |
| duplicate sale | 58 | same address sold again within 60 days |
| price range | 12 | price under $75,000 or over $8,000,000 (typos, parking spots, portfolios) |
| price outlier | 13 | $/sqft more than 4.0 robust SDs from its property type |
| output | 0 | 5858 rows |

## Split: recent

train 2024-12-02 to 2026-09-10, test after 2026-09-10 (last 3 months)
Train 5,461 sales, test 397.

| Model | MAE | Median abs % error | Within 5% | Within 10% | Median bias |
|---|---|---|---|---|---|
| Comps baseline | $88,382 | 10.5% | 24.4% | 47.1% | +4.9% |
| LightGBM | $36,554 | 4.4% | 54.4% | 86.4% | +1.7% |
| Neural net (MLP) | $37,609 | 4.4% | 53.7% | 85.1% | +0.8% |

LightGBM 10-90% range: contains the sale price 71.0% of the time (target 80%), median width 14.2% of the estimate.

### By property type (median abs % error)

| property_type | n | Comps baseline | LightGBM | Neural net (MLP) |
|---|---|---|---|---|
| Detached | 176 | 10.1% | 4.9% | 4.9% |
| Att/Row/Townhouse | 80 | 9.6% | 4.3% | 3.9% |
| Condo Apartment | 49 | 12.0% | 4.9% | 4.8% |
| Semi-Detached | 42 | 12.3% | 2.9% | 3.7% |
| Condo Townhouse | 34 | 11.5% | 4.0% | 4.5% |

### By price band (median abs % error)

| price_band | n | Comps baseline | LightGBM | Neural net (MLP) |
|---|---|---|---|---|
| < $400K | 69 | 12.0% | 4.2% | 5.4% |
| $400-600K | 104 | 11.3% | 4.5% | 4.2% |
| $600-800K | 103 | 10.4% | 4.4% | 4.0% |
| $800K-1.2M | 108 | 9.9% | 4.0% | 4.6% |
| > $1.2M | 13 | 10.1% | 5.4% | 4.1% |

### By neighbourhood (15 largest in the test set) (median abs % error)

| city_region | n | Comps baseline | LightGBM | Neural net (MLP) |
|---|---|---|---|---|
| 7704 - Barrhaven - Heritage Park | 26 | 12.7% | 5.2% | 3.8% |
| 4001 - Lower Town/Byward Market | 24 | 12.0% | 3.9% | 4.0% |
| 4801 - Old Ottawa South | 23 | 15.6% | 5.7% | 5.9% |
| 9008 - Kanata - Morgan's Grant/South March | 23 | 7.5% | 3.9% | 4.9% |
| 2201 - Orleans - Convent Glen | 22 | 16.5% | 4.6% | 4.5% |
| 9010 - Kanata - Emerald Meadows/Trailwest | 21 | 9.7% | 4.9% | 5.0% |
| 3701 - Alta Vista | 20 | 13.1% | 4.3% | 5.4% |
| 4401 - Glebe | 20 | 8.5% | 4.6% | 3.5% |
| 4601 - Hintonburg/Mechanicsville | 20 | 5.8% | 5.1% | 4.1% |
| 5003 - Westboro/Hampton Park | 20 | 7.2% | 4.2% | 4.4% |

## Split: shift

train first 12 months (2024-12-02 to 2025-12-02), test everything after
Train 3,007 sales, test 2,851.

| Model | MAE | Median abs % error | Within 5% | Within 10% | Median bias |
|---|---|---|---|---|---|
| Comps baseline | $91,279 | 10.8% | 23.8% | 46.9% | +0.5% |
| LightGBM | $40,792 | 4.9% | 50.8% | 83.2% | +2.2% |
| Neural net (MLP) | $51,360 | 5.8% | 43.9% | 76.1% | -2.3% |

LightGBM 10-90% range: contains the sale price 67.9% of the time (target 80%), median width 13.7% of the estimate.

### By property type (median abs % error)

| property_type | n | Comps baseline | LightGBM | Neural net (MLP) |
|---|---|---|---|---|
| Detached | 1,253 | 10.3% | 5.0% | 6.2% |
| Att/Row/Townhouse | 504 | 10.7% | 4.9% | 5.5% |
| Condo Apartment | 473 | 10.9% | 4.9% | 4.8% |
| Condo Townhouse | 229 | 11.0% | 4.5% | 5.3% |
| Semi-Detached | 220 | 12.1% | 4.5% | 5.7% |
| Duplex | 77 | 10.8% | 5.0% | 7.3% |
| Link | 35 | 14.5% | 5.5% | 5.2% |
| Common Element Condo | 30 | 11.4% | 5.7% | 5.2% |
| Triplex | 30 | 13.1% | 3.6% | 7.2% |

### By price band (median abs % error)

| price_band | n | Comps baseline | LightGBM | Neural net (MLP) |
|---|---|---|---|---|
| < $400K | 526 | 11.2% | 5.4% | 5.5% |
| $400-600K | 705 | 10.8% | 4.7% | 4.8% |
| $600-800K | 661 | 10.9% | 4.9% | 5.2% |
| $800K-1.2M | 755 | 10.5% | 4.7% | 7.2% |
| > $1.2M | 204 | 10.2% | 5.2% | 7.9% |

### By neighbourhood (15 largest in the test set) (median abs % error)

| city_region | n | Comps baseline | LightGBM | Neural net (MLP) |
|---|---|---|---|---|
| 9010 - Kanata - Emerald Meadows/Trailwest | 165 | 9.5% | 5.5% | 5.8% |
| 2201 - Orleans - Convent Glen | 151 | 11.0% | 5.4% | 6.1% |
| 4401 - Glebe | 145 | 12.2% | 5.3% | 6.0% |
| 3701 - Alta Vista | 144 | 11.1% | 4.4% | 5.9% |
| 4801 - Old Ottawa South | 133 | 10.9% | 5.0% | 6.8% |
| 1105 - Fallingbrook/Pineview | 133 | 9.0% | 4.5% | 6.8% |
| 9008 - Kanata - Morgan's Grant/South March | 132 | 10.7% | 4.4% | 6.4% |
| 4103 - Ottawa Centre | 131 | 11.0% | 5.2% | 6.2% |
| 7704 - Barrhaven - Heritage Park | 129 | 10.3% | 5.0% | 5.2% |
| 1118 - Avalon/Notting Gate/Fallingbrook | 128 | 10.3% | 4.2% | 5.5% |
| 4601 - Hintonburg/Mechanicsville | 128 | 9.5% | 5.4% | 5.2% |
| 4001 - Lower Town/Byward Market | 126 | 11.4% | 5.0% | 4.6% |
| 5003 - Westboro/Hampton Park | 125 | 9.4% | 4.6% | 7.5% |
| 7301 - Nepean - Centrepointe | 125 | 9.3% | 4.9% | 5.1% |
| 7711 - Barrhaven - Half Moon Bay | 125 | 10.6% | 4.9% | 4.8% |
