# Sydney fuel price shock (early 2026): traffic and public transport response

Code and data for the paper *"Observational evidence of a limited driving reduction with no detectable public transport increase during a Sydney fuel price shock"* (Blache & Saberi, rCITI, UNSW Sydney).

The analysis examines how road traffic and public transport patronage in New
South Wales responded to the early-2026 fuel price shock (Sydney petrol rose
from ~152 to ~249 c/L, peaking 26 March 2026), on a weekday-aligned
2025-vs-2026 sample (9 February – 31 March) restricted to **light vehicles**.

Primary inference uses a single specification, the **level-baseline
counterfactual (M5)**: each 2026 weekday is compared with the matched 2025
weekday, and the mean pre-shock ratio defines a no-shock counterfactual. This
is a difference in differences in ratio form, with the matched 2025 series as
the comparison group. Six further specifications (M1–M4, M6, M7) are reported
in the Supplementary Information as sensitivity checks; they are alternative
parameterisations of one dataset, not independent sources of evidence.

## Summary results
- **Statewide light-vehicle traffic:** little cumulative change (+0.2%), with a
  peak-week reduction of −2.0%, similar to Greater Sydney.
- **Greater Sydney light-vehicle traffic:** −1.3% cumulative and −2.3% in the
  peak week (20–28 March).
- **Statewide Opal tap-ons:** no detectable increase (−0.8% cumulative, +0.2%
  peak-week). The result bounds the size of any shift to public transport
  rather than establishing its absence.

The paper reports 95% bootstrap intervals for these estimates (Table 1 and
Supplementary Table S2).

Shock onset is 28 February 2026: pre-shock 9–27 February, post-shock 2–31
March. The balanced panel comprises 91 station–direction units statewide and
14 in Greater Sydney (Transport for NSW Sydney region), over 41 aligned
weekdays. After the 80% data-quality screen, the traffic analysis retains 15
pre-shock and 21 post-shock weekdays, and the public transport analysis 13
pre-shock and 22 post-shock weekdays. Traffic has one fewer post-shock weekday
because 31 March 2026 has no matched 2025 date within the February–March 2025
extract.

## Repository structure
```
fuel_shock_analysis.py      Single self-contained pipeline: read raw -> clean ->
                            M1-M7 -> autocorrelation -> Supplementary Table S1.
                            Light vehicles only (classification_seq == 2).
fuel_shock_analysis.ipynb   Same pipeline as a Google Colab notebook (sectioned,
                            with an upload cell for the three input files).
table1_final.csv            Expected output of the seven specifications
                            (corresponds to Supplementary Table S1).
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
The three input files are included in `data/`. They are public TfNSW and Opal
data; see `data/README.md` for provenance and the vehicle-class coding. Place
them in the repository root (or in `data/` and update the path constants at
the top of `fuel_shock_analysis.py`).

## Reproduce the results
```bash
pip install -r requirements.txt
python fuel_shock_analysis.py
```
This prints the seven-specification results table and the
interrupted-time-series autocorrelation diagnostics, and writes
`clean_traffic_allnsw.csv`, `clean_traffic_sydney.csv`, and
`table1_final.csv`. The printed cells match Supplementary Table S1 and the
point estimates in the main text. Or open `fuel_shock_analysis.ipynb` in Colab
and Run all (upload the three CSVs when prompted).

## The seven specifications
M1 monthly year-on-year; M2 Welch pre/post on the daily 2026/2025 ratio;
M3 interrupted time series; M4 controlled ITS (matched 2025 control);
M5 level-baseline counterfactual (primary); M6 Quandt–Andrews sup-F break test;
M7 fuel-price dose-response (elasticities). M3, M4, M7 use Newey–West HAC
standard errors (lag 7).

## Citation
If you use this code, please cite the paper (citation to be added on
publication).

## License
MIT — see `LICENSE`.
