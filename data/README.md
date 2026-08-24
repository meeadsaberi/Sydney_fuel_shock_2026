# Input data

All three files needed to reproduce Table 1 are tracked in this directory.

| File | Description | Rows |
|---|---|---|
| `raw_feb_mar_2025_2026.csv` | TfNSW permanent traffic counters, hourly, February–March 2025 and 2026. 211 stations. | 63,297 |
| `opal_all_nsw_feb_mar_2025_2026_aligned.csv` | Daily Opal tap-ons, all NSW, weekday-aligned across years. | 118 |
| `Fuel_price.csv` | Daily average unleaded retail price, Australian capital cities. | 5,828 |

## Traffic data provenance

Source: NSW Transport Traffic Volume Viewer
(https://maps.transport.nsw.gov.au/egeomaps/traffic-volumes/), backed by the
CartoDB SQL API (`ds_aadt_permanent_hourly_data`, `ds_aadt_reference`).
Snapshot downloaded 2026-05-19.

`raw_feb_mar_2025_2026.csv` is a subset of that snapshot, restricted to
February and March of 2025 and 2026 and to the columns the analysis uses:
station and region identifiers, date fields, direction, vehicle class, and the
24 hourly count columns. Station metadata not used by the analysis (address,
coordinates, LGA, postcode, holiday flags, `daily_total`) has been dropped to
keep the file small. Verified to produce output identical to the full-width
extract.

### Vehicle classification

`classification_seq` follows the TfNSW coding:

| Code | Meaning |
|---|---|
| 0 | All vehicles |
| 2 | **Light vehicles** |
| 3 | Heavy vehicles |

The analysis uses **light vehicles only** (`classification_seq == 2`). Diesel
prices rose roughly 91% over the study window against roughly 50% for petrol,
so freight faced a materially different price shock from the car travel this
study interprets. 210 of the 211 stations report the light class; the single
exception reports only `classification_seq == 0` and drops out of the panel.
