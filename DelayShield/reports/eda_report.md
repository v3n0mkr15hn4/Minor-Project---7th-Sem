# DelayShield - dataset inspection report

## 1. Raw tables, row counts and missing values

| table                | file                                  |    rows |   columns |   missing_cells_pct |
|:---------------------|:--------------------------------------|--------:|----------:|--------------------:|
| orders               | olist_orders_dataset.csv              |   99441 |         8 |               0.617 |
| order_items          | olist_order_items_dataset.csv         |  112650 |         7 |               0     |
| customers            | olist_customers_dataset.csv           |   99441 |         5 |               0     |
| sellers              | olist_sellers_dataset.csv             |    3095 |         4 |               0     |
| products             | olist_products_dataset.csv            |   32951 |         9 |               0.825 |
| payments             | olist_order_payments_dataset.csv      |  103886 |         5 |               0     |
| reviews              | olist_order_reviews_dataset.csv       |   99224 |         7 |              21.006 |
| geolocation          | olist_geolocation_dataset.csv         | 1000163 |         5 |               0     |
| category_translation | product_category_name_translation.csv |      71 |         2 |               0     |

## 2. Schema of the tables that feed the model

### orders (99,441 rows)

| column                        | dtype          |   missing_% |   n_unique |
|:------------------------------|:---------------|------------:|-----------:|
| order_id                      | object         |        0    |      99441 |
| customer_id                   | object         |        0    |      99441 |
| order_status                  | object         |        0    |          8 |
| order_purchase_timestamp      | datetime64[ns] |        0    |      98875 |
| order_approved_at             | datetime64[ns] |        0.16 |      90733 |
| order_delivered_carrier_date  | datetime64[ns] |        1.79 |      81018 |
| order_delivered_customer_date | datetime64[ns] |        2.98 |      95664 |
| order_estimated_delivery_date | datetime64[ns] |        0    |        459 |

### order_items (112,650 rows)

| column              | dtype          |   missing_% |   n_unique |
|:--------------------|:---------------|------------:|-----------:|
| order_id            | object         |           0 |      98666 |
| order_item_id       | int64          |           0 |         21 |
| product_id          | object         |           0 |      32951 |
| seller_id           | object         |           0 |       3095 |
| shipping_limit_date | datetime64[ns] |           0 |      93318 |
| price               | float64        |           0 |       5968 |
| freight_value       | float64        |           0 |       6999 |

### products (32,951 rows)

| column                     | dtype   |   missing_% |   n_unique |
|:---------------------------|:--------|------------:|-----------:|
| product_id                 | object  |        0    |      32951 |
| product_category_name      | object  |        1.85 |         73 |
| product_name_lenght        | float64 |        1.85 |         66 |
| product_description_lenght | float64 |        1.85 |       2960 |
| product_photos_qty         | float64 |        1.85 |         19 |
| product_weight_g           | float64 |        0.01 |       2204 |
| product_length_cm          | float64 |        0.01 |         99 |
| product_height_cm          | float64 |        0.01 |        102 |
| product_width_cm           | float64 |        0.01 |         95 |

### sellers (3,095 rows)

| column                 | dtype   |   missing_% |   n_unique |
|:-----------------------|:--------|------------:|-----------:|
| seller_id              | object  |           0 |       3095 |
| seller_zip_code_prefix | int64   |           0 |       2246 |
| seller_city            | object  |           0 |        611 |
| seller_state           | object  |           0 |         23 |

### customers (99,441 rows)

| column                   | dtype   |   missing_% |   n_unique |
|:-------------------------|:--------|------------:|-----------:|
| customer_id              | object  |           0 |      99441 |
| customer_unique_id       | object  |           0 |      96096 |
| customer_zip_code_prefix | int64   |           0 |      14994 |
| customer_city            | object  |           0 |       4119 |
| customer_state           | object  |           0 |         27 |

### reviews (99,224 rows)

| column                  | dtype          |   missing_% |   n_unique |
|:------------------------|:---------------|------------:|-----------:|
| review_id               | object         |        0    |      98410 |
| order_id                | object         |        0    |      98673 |
| review_score            | int64          |        0    |          5 |
| review_comment_title    | object         |       88.34 |       4527 |
| review_comment_message  | object         |       58.7  |      36159 |
| review_creation_date    | datetime64[ns] |        0    |        636 |
| review_answer_timestamp | datetime64[ns] |        0    |      98248 |

## 3. Joins used to build the modeling table

```
orders (delivered only, all timestamps present)
  |-- order_items      : order_id  -> aggregated to one row per order
  |     |-- products   : product_id (category, weight, dimensions)
  |     '-- sellers    : seller_id of the PRIMARY seller (largest value line)
  |-- customers        : customer_id (state, city, zip prefix)
  |-- geolocation      : zip prefix centroid, twice (seller and customer)
  '-- reviews          : order_id -> score (supplier scorecard only)
```

Multi-seller orders: 1,275 of 96,470 orders (1.3%) involve more than one
seller. The order-level row keeps `seller_count` and attributes the order to
the primary seller; supplier prioritisation is handled by a separate
seller-level scorecard.

## 4. Target distribution

`delivery_delay = order_delivered_customer_date > order_estimated_delivery_date`

- Usable delivered orders: **96,470**
- Late: **7,826** (8.11%)
- On time: **88,644**
- Purchase window: 2016-09-15 to 2018-08-29

### Delay rate over time

| quarter   |   orders |   delay_rate |
|:----------|---------:|-------------:|
| 2016Q3    |        1 |       1      |
| 2016Q4    |      266 |       0.0113 |
| 2017Q1    |     4949 |       0.044  |
| 2017Q2    |     8983 |       0.0479 |
| 2017Q3    |    12215 |       0.04   |
| 2017Q4    |    17279 |       0.1008 |
| 2018Q1    |    20627 |       0.1458 |
| 2018Q2    |    19643 |       0.0509 |
| 2018Q3    |    12507 |       0.0748 |

The delay rate is far from stationary (2018 Q1 peaks well above the
2017 average), which is the reason for the chronological split.

## 5. Final modeling table

- Rows: **96,470** (one per delivered order)
- Model features: **41** (38 numeric, 3 categorical)

### Numeric features

`order_purchase_month`, `order_purchase_week`, `order_purchase_day`, `order_purchase_weekday`, `order_purchase_hour`, `order_purchase_quarter`, `is_weekend`, `estimated_lead_time_days`, `shipping_limit_days`, `lead_time_slack_days`, `lead_time_ratio`, `distance_per_promised_day`, `total_order_value`, `total_freight_value`, `freight_ratio`, `item_count`, `unique_product_count`, `seller_count`, `max_item_price`, `product_weight_g`, `product_volume_cm3`, `distance_km`, `same_state`, `customer_zip_code_prefix`, `seller_hist_orders`, `seller_hist_delay_rate`, `seller_hist_on_time_rate`, `seller_hist_avg_delivery_days`, `seller_hist_avg_freight`, `seller_hist_avg_delay_days`, `seller_orders_last_30_days`, `seller_orders_last_90_days`, `seller_workload_surge`, `category_hist_delay_rate`, `customer_state_hist_delay_rate`, `route_hist_delay_rate`, `platform_orders_last_30_days`, `platform_hist_delay_rate`

### Categorical features

`product_category`, `customer_state`, `seller_state`

### Feature summary

| feature                        |   count |      mean |       std |      min |       25% |       50% |       75% |             max |
|:-------------------------------|--------:|----------:|----------:|---------:|----------:|----------:|----------:|----------------:|
| order_purchase_month           |   96470 |     6.031 |     3.228 |    1     |     3     |     6     |     8     |     12          |
| order_purchase_week            |   96470 |    24.35  |    14.043 |    1     |    13     |    24     |    34     |     52          |
| order_purchase_day             |   96470 |    15.516 |     8.665 |    1     |     8     |    16     |    23     |     31          |
| order_purchase_weekday         |   96470 |     2.756 |     1.967 |    0     |     1     |     3     |     4     |      6          |
| order_purchase_hour            |   96470 |    14.773 |     5.328 |    0     |    11     |    15     |    19     |     23          |
| order_purchase_quarter         |   96470 |     2.355 |     1.06  |    1     |     1     |     2     |     3     |      4          |
| is_weekend                     |   96470 |     0.23  |     0.421 |    0     |     0     |     0     |     0     |      1          |
| estimated_lead_time_days       |   96470 |    23.736 |     8.761 |    2.008 |    18.33  |    23.231 |    28.408 |    155.135      |
| shipping_limit_days            |   96470 |     6.575 |     4.661 |    2.004 |     5.006 |     6.012 |     7.136 |   1052.01       |
| lead_time_slack_days           |   96470 |    11.641 |     8.579 |  -48.225 |     6.033 |    11.181 |    16.571 |    140.172      |
| lead_time_ratio                |   96470 |     2.094 |     1.02  |    0.139 |     1.479 |     1.936 |     2.492 |     25.801      |
| distance_per_promised_day      |   96470 |    23.365 |    20.265 |    0     |     9.295 |    18.52  |    31.011 |    538.756      |
| total_order_value              |   96470 |   137.04  |   209.053 |    0.85  |    45.9   |    86.5   |   149.9   |  13440          |
| total_freight_value            |   96470 |    22.786 |    21.56  |    0     |    13.85  |    17.17  |    24.02  |   1794.96       |
| freight_ratio                  |   96470 |     0.308 |     0.312 |    0     |     0.132 |     0.224 |     0.381 |     21.447      |
| item_count                     |   96470 |     1.142 |     0.539 |    1     |     1     |     1     |     1     |     21          |
| unique_product_count           |   96470 |     1.039 |     0.228 |    1     |     1     |     1     |     1     |      8          |
| seller_count                   |   96470 |     1.014 |     0.124 |    1     |     1     |     1     |     1     |      5          |
| max_item_price                 |   96470 |   125.956 |   190.43  |    0.85  |    42     |    79.05  |   139.9   |   6735          |
| product_weight_g               |   96470 |  2386.47  |  4769.81  |    0     |   300     |   750     |  2050     | 184400          |
| product_volume_cm3             |   96470 | 15440.2   | 24228.5   |  168     |  2850     |  6498     | 18750     |      1.0974e+06 |
| distance_km                    |   96470 |   599.92  |   592.291 |    0     |   189.293 |   433.863 |   796.035 |   8677.91       |
| same_state                     |   96470 |     0.36  |     0.48  |    0     |     0     |     0     |     1     |      1          |
| customer_zip_code_prefix       |   96470 | 35199.2   | 29839.8   | 1003     | 11355     | 24436     | 59056     |  99980          |
| seller_hist_orders             |   96470 |   168.684 |   281.999 |    0     |    12     |    51     |   179     |   1780          |
| seller_hist_delay_rate         |   96470 |     0.053 |     0.072 |    0     |     0     |     0.039 |     0.075 |      1          |
| seller_hist_on_time_rate       |   96470 |     0.947 |     0.072 |    0     |     0.925 |     0.961 |     1     |      1          |
| seller_hist_avg_delivery_days  |   96470 |    12.095 |     3.491 |    1.197 |    10.103 |    11.994 |    13.75  |     70.214      |
| seller_hist_avg_freight        |   96470 |    21.669 |    11.797 |    7.39  |    16.139 |    19.403 |    22.659 |    909.27       |
| seller_hist_avg_delay_days     |   96470 |   -12.102 |     5.733 | -139.397 |   -14.255 |   -11.66  |    -9.63  |     39.871      |
| seller_orders_last_30_days     |   96470 |    29.432 |    41.676 |    0     |     4     |    12     |    36     |    283          |
| seller_orders_last_90_days     |   96470 |    76.789 |   108.543 |    0     |     9     |    30     |    90     |    556          |
| seller_workload_surge          |   96470 |     1.349 |     0.746 |    0     |     0.882 |     1.169 |     1.636 |      3          |
| category_hist_delay_rate       |   96470 |     0.055 |     0.026 |    0     |     0.036 |     0.053 |     0.076 |      1          |
| customer_state_hist_delay_rate |   96470 |     0.054 |     0.037 |    0     |     0.036 |     0.044 |     0.06  |      0.273      |
| route_hist_delay_rate          |   96470 |     0.055 |     0.047 |    0     |     0.033 |     0.044 |     0.06  |      1          |
| platform_orders_last_30_days   |   96470 |  5445.62  |  1807.8   |    0     |  4086     |  5777     |  6900     |   8423          |
| platform_hist_delay_rate       |   96470 |     0.055 |     0.022 |    0.002 |     0.039 |     0.059 |     0.079 |      0.085      |

## 6. Leakage risks and how each is handled

| Risk                                           | Handling                                                                                                 |
|:-----------------------------------------------|:---------------------------------------------------------------------------------------------------------|
| Actual delivery timestamp                      | Used only to build the target; never a feature.                                                          |
| Carrier hand-over date / approval date         | Excluded - both are generated after confirmation.                                                        |
| Review score and comments                      | Post-delivery. Used only in the supplier scorecard.                                                      |
| Order status                                   | Only 'delivered' rows are modelled; status is not a feature.                                             |
| Seller history computed over the whole dataset | Replaced by lagged statistics using only orders already delivered before the current purchase timestamp. |
| Category / route / platform delay rates        | Same lagged construction as seller history.                                                              |
| Global means for cold-start fallbacks          | Running platform average available at that moment, not a full-dataset mean.                              |
| Random train/test split                        | Chronological split; validation and test come strictly after the training window.                        |

## 7. Design notes

- Orders with a delivery date before the purchase date are dropped as corrupt.
- Product weight/dimension gaps are median-imputed before aggregation.
- Distance is a great-circle proxy between zip-prefix centroids (the geolocation table has ~1M points; centroids are enough for a proxy).
- Cold-start sellers (5.5% of orders) are flagged and fall back to the running platform delay rate.
