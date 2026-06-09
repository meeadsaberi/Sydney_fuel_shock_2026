# Sydney fuel price shock (early 2026): traffic and public transport response

Reproducible code for the paper *"Observational evidence of a limited driving
reduction without public transport substitution during a Sydney fuel price
shock"* (Blache & Saberi, rCITI, UNSW Sydney).

The analysis quantifies how road traffic and public transport patronage in New
South Wales responded to the early-2026 fuel price shock (Sydney petrol rose
from ~152 to ~249 c/L, peaking 26 March 2026), using seven complementary
statistical and time-series methods on a weekday-aligned 2025-vs-2026 sample
(9 February – 31 March).

## Headline result
Statewide weekday traffic was essentially unchanged; Greater Sydney showed a
small (~1.5–2%) reduction around the price peak; public transport patronage
showed no positive substitution response.

## Repository structure
```
fuel_shock_analysis.py      Single self-contained pipeline: read raw -> clean ->
                            M1-M7 -> autocorrelation -> Table 1. Run this to
                            reproduce every number in the paper.
fuel_shock_analysis.ipynb   Same pipeline as a Google Colab notebook (sectioned,
                            with an upload cell for the three input files).
table1_final.csv            Expected Table 1 output (for verification).
figures/
  fig_main_daily.py             Main text Fig. 1 (daily series incl. weekends;
                                weekend traffic/Opal observed, fuel interpolated).
  fig_weekday_only.py           Weekday-only "as-analysed" variant.
  fig_main_and_SI.py            Main (cleaned) + SI (raw, pre-cleaning) figures.
  fig_station_contact_sheet.py  Per-station raw-vs-cleaned QC contact sheet.
data/
  README.md                 Three input files and where to download the original data.
requirements.txt
LICENSE                     MIT
```

## Data
The three input files are included. The data in its original and raw form are public TfNSW data; see
`data/README.md`. Place them in the repository root (or in
`data/` and update the path constants at the top of `fuel_shock_analysis.py`).

## Reproduce the results
```bash
pip install -r requirements.txt
python fuel_shock_analysis.py
```
This prints Table 1 and the interrupted-time-series autocorrelation diagnostics,
and writes `clean_traffic_allnsw.csv`, `clean_traffic_sydney.csv`, and
`table1_final.csv`. The printed cells match the paper's Table 1 and Methods
verbatim. Or open `fuel_shock_analysis.ipynb` in Colab and Run all (upload the
three CSVs when prompted).

## The seven methods
M1 monthly year-on-year; M2 Welch pre/post on the daily 2026/2025 ratio;
M3 interrupted time series; M4 controlled ITS (matched 2025 control);
M5 level-baseline counterfactual; M6 Quandt–Andrews sup-F break test;
M7 fuel-price dose-response (elasticities). M3, M4, M7 use Newey–West HAC
standard errors (lag 7).

## Citation
If you use this code, please cite the paper (citation to be added on
publication).

## License
MIT — see `LICENSE`.
