# Results (data/sample.csv)

## Cleaning

| Step | Removed | Why |
|---|---|---|
| input | 0 | 6060 rows |
| not sold | 60 | closed but not a completed sale (e.g. Deal Fell Through) |
| no price or date | 0 | ClosePrice or sale date missing |
| property type | 59 | land, farm, mobile home, other |
| duplicate key | 0 | same ListingKey twice |
| duplicate sale | 57 | same address sold again within 60 days |
| price range | 9 | price under $75,000 or over $8,000,000 (typos, parking spots, portfolios) |
| price outlier | 15 | $/sqft more than 4.0 robust SDs from its property type |
| output | 0 | 5860 rows |

## Split: recent

train 2024-12-02 to 2026-09-10, test after 2026-09-10 (last 3 months)
Train 5,468 sales, test 392.

| Model | MAE | Median abs % error | Within 5% | Within 10% | Median bias |
|---|---|---|---|---|---|
| Comps baseline | $98,576 | 12.2% | 20.2% | 39.8% | +2.1% |
| LightGBM | $41,536 | 4.9% | 51.0% | 82.9% | -0.0% |
| LightGBM, no taxes | $73,911 | 9.0% | 29.1% | 54.8% | +1.2% |
| + updates (roof, furnace...) | $39,302 | 5.0% | 50.3% | 84.4% | +0.4% |
| + description | $40,310 | 4.9% | 51.5% | 84.4% | +0.7% |
| + photos | $37,871 | 4.9% | 51.8% | 85.7% | +1.2% |
| + description + photos | $38,060 | 4.7% | 52.6% | 85.5% | +1.0% |
| Neural net (MLP) | $45,391 | 5.2% | 47.7% | 79.6% | +0.2% |

### LightGBM range (target: contains the sale price 80% of the time)

| | Contains sale price | Median width |
|---|---|---|
| Raw 10-90% quantile models | 62.8% | 12.6% |
| Calibrated (conformal) | 75.8% | 16.5% |
| ... condo (92) | 82.6% | 16.2% |
| ... freehold (300) | 73.7% | 16.6% |
| ... $400-600K (88) | 76.1% | 17.1% |
| ... $600-800K (80) | 75.0% | 16.9% |
| ... $800K-1.2M (122) | 75.4% | 16.0% |
| ... < $400K (81) | 75.3% | 16.5% |
| ... > $1.2M (21) | 81.0% | 17.0% |

### By property type (median abs % error)

| property_type | n | Comps baseline | LightGBM | LightGBM, no taxes | + updates (roof, furnace...) | + description | + photos | + description + photos | Neural net (MLP) |
|---|---|---|---|---|---|---|---|---|---|
| Detached | 183 | 11.6% | 4.8% | 9.0% | 4.9% | 4.9% | 5.2% | 4.7% | 5.3% |
| Condo Apartment | 67 | 12.5% | 4.1% | 8.2% | 4.9% | 4.5% | 4.8% | 4.6% | 4.7% |
| Att/Row/Townhouse | 64 | 12.3% | 5.1% | 9.0% | 4.5% | 5.3% | 4.9% | 4.8% | 5.1% |
| Semi-Detached | 31 | 13.4% | 6.2% | 9.1% | 5.6% | 4.8% | 4.4% | 5.6% | 6.0% |
| Condo Townhouse | 21 | 11.5% | 3.9% | 9.2% | 5.4% | 4.1% | 3.8% | 3.7% | 5.2% |

### By price band (median abs % error)

| price_band | n | Comps baseline | LightGBM | LightGBM, no taxes | + updates (roof, furnace...) | + description | + photos | + description + photos | Neural net (MLP) |
|---|---|---|---|---|---|---|---|---|---|
| < $400K | 81 | 13.4% | 5.4% | 9.1% | 5.3% | 5.2% | 4.9% | 5.0% | 4.9% |
| $400-600K | 88 | 12.8% | 5.0% | 8.8% | 5.0% | 4.9% | 4.6% | 4.4% | 6.2% |
| $600-800K | 80 | 11.8% | 4.3% | 8.8% | 4.6% | 4.8% | 4.8% | 4.8% | 5.1% |
| $800K-1.2M | 122 | 11.3% | 5.2% | 8.4% | 4.9% | 4.9% | 5.3% | 4.6% | 4.9% |
| > $1.2M | 21 | 10.2% | 4.1% | 10.3% | 4.4% | 4.7% | 3.7% | 3.8% | 5.4% |

### By neighbourhood (15 largest in the test set) (median abs % error)

| city_region | n | Comps baseline | LightGBM | LightGBM, no taxes | + updates (roof, furnace...) | + description | + photos | + description + photos | Neural net (MLP) |
|---|---|---|---|---|---|---|---|---|---|
| 7704 - Barrhaven - Heritage Park | 25 | 12.8% | 6.5% | 12.2% | 5.9% | 6.4% | 4.9% | 5.5% | 5.0% |
| 4001 - Lower Town/Byward Market | 24 | 10.6% | 4.4% | 9.7% | 3.5% | 4.4% | 2.8% | 3.1% | 4.4% |
| 4801 - Old Ottawa South | 23 | 12.4% | 5.7% | 13.1% | 4.7% | 4.9% | 5.1% | 5.8% | 4.9% |
| 9008 - Kanata - Morgan's Grant/South March | 23 | 10.9% | 6.0% | 10.5% | 5.3% | 5.3% | 5.3% | 4.8% | 6.0% |
| 9010 - Kanata - Emerald Meadows/Trailwest | 21 | 10.6% | 4.0% | 5.7% | 4.6% | 5.4% | 4.9% | 4.7% | 5.0% |
| 4401 - Glebe | 20 | 11.2% | 3.9% | 9.2% | 4.5% | 4.8% | 4.1% | 4.7% | 6.7% |
| 4601 - Hintonburg/Mechanicsville | 20 | 19.2% | 4.6% | 8.2% | 4.8% | 4.5% | 5.4% | 5.2% | 3.2% |

## Split: shift

train first 12 months (2024-12-02 to 2025-12-02), test everything after
Train 3,010 sales, test 2,850.

| Model | MAE | Median abs % error | Within 5% | Within 10% | Median bias |
|---|---|---|---|---|---|
| Comps baseline | $97,166 | 11.0% | 23.9% | 45.8% | +0.4% |
| LightGBM | $43,327 | 4.8% | 51.2% | 81.8% | +2.1% |
| LightGBM, no taxes | $70,661 | 8.1% | 32.5% | 59.5% | -0.0% |
| + updates (roof, furnace...) | $43,101 | 4.8% | 51.8% | 82.3% | +2.2% |
| + description | $42,436 | 4.7% | 52.1% | 82.3% | +1.8% |
| + photos | $42,411 | 4.8% | 51.5% | 83.0% | +2.2% |
| + description + photos | $42,335 | 4.7% | 52.3% | 82.8% | +2.2% |
| Neural net (MLP) | $56,919 | 6.3% | 40.9% | 71.4% | -1.3% |

### LightGBM range (target: contains the sale price 80% of the time)

| | Contains sale price | Median width |
|---|---|---|
| Raw 10-90% quantile models | 69.4% | 15.1% |
| Calibrated (conformal) | 77.8% | 18.1% |
| ... condo (670) | 76.4% | 21.0% |
| ... freehold (2,180) | 78.2% | 17.6% |
| ... $400-600K (659) | 71.9% | 16.1% |
| ... $600-800K (622) | 82.8% | 16.8% |
| ... $800K-1.2M (816) | 84.1% | 17.7% |
| ... < $400K (505) | 68.9% | 24.2% |
| ... > $1.2M (248) | 77.8% | 33.2% |

### By property type (median abs % error)

| property_type | n | Comps baseline | LightGBM | LightGBM, no taxes | + updates (roof, furnace...) | + description | + photos | + description + photos | Neural net (MLP) |
|---|---|---|---|---|---|---|---|---|---|
| Detached | 1,268 | 10.4% | 4.9% | 8.3% | 4.8% | 4.7% | 4.7% | 4.7% | 6.6% |
| Att/Row/Townhouse | 513 | 11.1% | 5.0% | 8.5% | 4.9% | 5.3% | 5.4% | 5.2% | 5.9% |
| Condo Apartment | 435 | 11.1% | 4.7% | 7.2% | 4.8% | 4.5% | 4.4% | 4.6% | 5.6% |
| Semi-Detached | 227 | 10.6% | 4.9% | 9.1% | 4.7% | 4.7% | 5.2% | 4.2% | 5.7% |
| Condo Townhouse | 216 | 11.4% | 4.7% | 6.3% | 4.5% | 4.5% | 4.2% | 4.5% | 6.8% |
| Duplex | 89 | 14.3% | 5.3% | 10.0% | 5.2% | 4.4% | 4.9% | 5.5% | 7.5% |
| Triplex | 45 | 11.2% | 4.4% | 9.6% | 4.6% | 4.4% | 5.1% | 4.4% | 9.9% |
| Link | 38 | 18.2% | 6.3% | 7.3% | 6.2% | 7.3% | 7.6% | 5.5% | 3.9% |

### By price band (median abs % error)

| price_band | n | Comps baseline | LightGBM | LightGBM, no taxes | + updates (roof, furnace...) | + description | + photos | + description + photos | Neural net (MLP) |
|---|---|---|---|---|---|---|---|---|---|
| < $400K | 505 | 11.9% | 5.1% | 7.3% | 5.1% | 4.9% | 4.8% | 4.8% | 6.9% |
| $400-600K | 659 | 11.1% | 5.1% | 8.6% | 5.2% | 5.3% | 5.5% | 5.2% | 5.6% |
| $600-800K | 622 | 10.6% | 4.7% | 8.7% | 4.6% | 4.4% | 4.5% | 4.4% | 4.9% |
| $800K-1.2M | 816 | 10.4% | 4.5% | 7.8% | 4.4% | 4.5% | 4.6% | 4.6% | 6.8% |
| > $1.2M | 248 | 11.9% | 5.1% | 8.5% | 5.1% | 4.9% | 5.3% | 5.1% | 10.1% |

### By neighbourhood (15 largest in the test set) (median abs % error)

| city_region | n | Comps baseline | LightGBM | LightGBM, no taxes | + updates (roof, furnace...) | + description | + photos | + description + photos | Neural net (MLP) |
|---|---|---|---|---|---|---|---|---|---|
| 9010 - Kanata - Emerald Meadows/Trailwest | 162 | 9.5% | 5.3% | 8.0% | 5.2% | 5.1% | 5.2% | 4.7% | 5.9% |
| 4401 - Glebe | 149 | 11.8% | 4.6% | 7.1% | 4.7% | 4.8% | 4.8% | 4.6% | 6.3% |
| 3701 - Alta Vista | 146 | 12.5% | 4.5% | 7.0% | 4.7% | 4.3% | 4.6% | 4.5% | 7.1% |
| 2201 - Orleans - Convent Glen | 145 | 11.7% | 4.3% | 7.9% | 4.1% | 4.4% | 4.2% | 4.3% | 5.4% |
| 1105 - Fallingbrook/Pineview | 140 | 11.7% | 4.6% | 8.6% | 4.2% | 4.3% | 4.2% | 3.8% | 7.3% |
| 4801 - Old Ottawa South | 135 | 12.3% | 5.0% | 9.0% | 4.8% | 5.3% | 5.0% | 4.6% | 8.9% |
| 9008 - Kanata - Morgan's Grant/South March | 134 | 10.7% | 5.4% | 9.5% | 4.8% | 4.6% | 4.8% | 4.9% | 6.6% |
| 1118 - Avalon/Notting Gate/Fallingbrook | 130 | 8.9% | 5.0% | 8.4% | 5.1% | 5.0% | 4.6% | 4.8% | 6.7% |
| 4103 - Ottawa Centre | 129 | 10.8% | 4.7% | 7.1% | 5.2% | 5.1% | 4.9% | 5.2% | 5.8% |
| 7711 - Barrhaven - Half Moon Bay | 127 | 12.7% | 5.3% | 9.6% | 4.6% | 4.4% | 5.1% | 5.0% | 6.6% |
| 4601 - Hintonburg/Mechanicsville | 126 | 11.9% | 5.5% | 7.6% | 5.3% | 4.9% | 5.7% | 5.3% | 5.6% |
| 5003 - Westboro/Hampton Park | 125 | 13.0% | 5.4% | 9.1% | 5.2% | 4.7% | 5.5% | 5.3% | 6.8% |
| 8211 - Stittsville (North) | 125 | 10.0% | 4.4% | 8.7% | 4.0% | 4.0% | 3.8% | 4.5% | 5.1% |
| 2605 - Blossom Park/Kemp Park/Findlay Creek | 124 | 11.5% | 5.4% | 8.2% | 5.2% | 4.7% | 4.8% | 4.4% | 6.5% |
| 4001 - Lower Town/Byward Market | 124 | 9.6% | 4.2% | 7.4% | 3.7% | 3.8% | 3.7% | 4.1% | 6.3% |
