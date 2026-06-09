# Input data

These files are **not** redistributed in this repository. Download them from the
sources below and place them here (`data/`), or in the repository root, before
running the analysis.

| File | Source |
|------|--------|
| `raw_feb_mar_2025_2026.csv` | TfNSW Roads Traffic Volume Counts API — raw hourly permanent-counter records for Feb–Mar 2025 and 2026. https://opendata.transport.nsw.gov.au/data/dataset/nsw-roads-traffic-volume-counts-api |
| `opal_all_nsw_feb_mar_2025_2026_aligned.csv` | TfNSW Opal patronage data archive (All-NSW daily tap-ons). https://opendata.transport.nsw.gov.au/data/dataset/opal-patronage |
| `Fuel_price.csv` | Sydney average unleaded retail petrol price (cents/litre), weekday series. |

The analysis scripts expect these filenames. If you keep them in `data/`,
update the path constants at the top of `fuel_shock_analysis.py` accordingly.
