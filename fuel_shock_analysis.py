"""
================================================================================
 fuel_shock_analysis.py
 Behavioural response to the early-2026 NSW fuel-price shock: end-to-end pipeline
================================================================================

 WHAT THIS SCRIPT DOES
 ---------------------
 It quantifies how road traffic and public-transport use in New South Wales
 responded to the early-2026 fuel-price shock (prices rose ~50% over four weeks
 from late February 2026). It runs the complete analysis from raw inputs to the
 paper's Table 1, in six ordered sections:

   (1) reads the raw hourly permanent traffic-counter records;
   (2) reconstructs an unbiased weekday traffic series on a BALANCED PANEL of
       station-direction counters (the raw daily sum under-measures 2025 more
       than 2026, which would otherwise manufacture a spurious decline);
   (3) builds same-weekday-aligned 2025-vs-2026 panels for traffic and Opal
       patronage, applies the 9-Feb cutoff and a symmetric data-quality screen;
   (4) applies SEVEN complementary methods (M1-M7);
   (5) runs residual autocorrelation diagnostics on the ITS model;
   (6) assembles and prints Table 1 and writes table1_final.csv.

 THE THREE SERIES REPORTED
 -------------------------
   Road traffic (All-NSW)        : reconstructed, all 211 stations
   Road traffic (Greater Sydney) : reconstructed, rms_region == "Sydney"
   Public transport (All-NSW)    : Opal tap-ons (identical across traffic runs)

 THE SEVEN METHODS
 -----------------
   M1  Monthly year-on-year comparison (descriptive)
   M2  Welch pre/post test on the daily 2026/2025 ratio (descriptive)
   M3  Interrupted time series (level shift + post-shock slope, HAC SEs)
   M4  Controlled ITS (M3 plus matched 2025 volume as a control)
   M5  Level-baseline counterfactual from the pre-shock mean ratio (descriptive)
   M6  Quandt-Andrews unknown-break sup-F test
   M7  Fuel-price dose-response (elasticities, HAC SEs)

 INPUTS (same folder)
 --------------------
   raw_feb_mar_2025_2026.csv                    TfNSW hourly permanent counters
   opal_all_nsw_feb_mar_2025_2026_aligned.csv   Opal tap-ons
   Fuel_price.csv                               Sydney retail fuel prices

 OUTPUTS
 -------
   printed Table 1 and autocorrelation diagnostics;
   clean_traffic_allnsw.csv, clean_traffic_sydney.csv  (reconstructed panels);
   table1_final.csv                                    (the Table 1 cells)

 REPRODUCIBILITY
 ---------------
 The printed numbers are byte-identical to the manuscript Table 1 and Methods.
 Figures from the original notebook are intentionally omitted (they do not affect
 any number). Dependencies: pandas, numpy, scipy, statsmodels.

 Run:  python3 fuel_shock_analysis.py
================================================================================
"""

# ============================================================================
# SECTION 0  IMPORTS, CONFIGURATION, CONSTANTS
# ============================================================================
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox, acorr_breusch_godfrey

# ---- file paths
RAW_PATH  = "raw_feb_mar_2025_2026.csv"
OPAL_PATH = "opal_all_nsw_feb_mar_2025_2026_aligned.csv"
FUEL_PATH = "Fuel_price.csv"
TRAFFIC_ALLNSW_CSV = "clean_traffic_allnsw.csv"   # written by SECTION 2
TRAFFIC_SYDNEY_CSV = "clean_traffic_sydney.csv"   # written by SECTION 2

# ---- shock timing and data-quality constants (shared by all sections)
SHOCK     = pd.Timestamp("2026-02-28")   # shock onset
PEAK      = pd.Timestamp("2026-03-26")   # fuel-price peak
ANOM      = pd.Timestamp("2026-02-09")   # drop contaminated early-Feb Opal week
ANOM_FRAC = 0.80                         # symmetric data-quality screen threshold

# ---- traffic-cleaning constants
HOURS     = [f"hour_{h:02d}" for h in range(24)]
COVERAGE  = 0.90   # a unit must be matched (both years) on >=90% of aligned weekdays

# ---- Quandt-Andrews (1993) sup-F critical values
ANDREWS_CV = {"10%": 9.84, "5%": 11.79, "1%": 16.45}


# ============================================================================
# SECTION 1  READ RAW PERMANENT-COUNTER DATA
# ============================================================================
def read_raw(region_filter=None):
    """Load the raw hourly permanent-counter file; optionally restrict to an
    rms_region (e.g. 'Sydney'). Returns the raw DataFrame with parsed dates."""
    df = pd.read_csv(RAW_PATH, dtype={"station_id": str}, low_memory=False)
    if region_filter is not None:
        df = df[df["rms_region"] == region_filter].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


# ============================================================================
# SECTION 2  CLEAN / RECONSTRUCT THE BALANCED-PANEL TRAFFIC SERIES
# ----------------------------------------------------------------------------
# The raw feed is incomplete and the incompleteness is ASYMMETRIC across years
# (2025 under-reports more hours and more whole direction/class records than
# 2026). A naive daily sum therefore under-measures 2025, inflating the
# 2026/2025 ratio. We reconstruct on a balanced panel of station-direction
# units so that every unit is compared only with itself across years, which
# makes the 2026/2025 ratio immune to constant proportional undercount.
# Rules: (1) weekdays only; (2) impute missing hours from each counter's own
# hour-of-day profile; (3) unit = station x direction, classes summed;
# (4) tolerate a missing heavy class; (5) same-weekday alignment to a year-2000
# anchor; (6) keep units matched in both years on >=90% of aligned weekdays;
# (7) metric = MEAN weekday volume per unit (composition-stable).
# ============================================================================
def build_traffic_panel(region_filter, out_path, label):
    df = read_raw(region_filter)

    # Rule 2: impute missing hours by counter hour-of-day profile
    g1 = ["station_id", "cardinal_direction_seq", "classification_seq", "year"]
    for h in HOURS:
        df[h] = df[h].fillna(df.groupby(g1)[h].transform("mean"))
    g2 = ["station_id", "cardinal_direction_seq", "year"]
    for h in HOURS:
        df[h] = df[h].fillna(df.groupby(g2)[h].transform("mean"))
    df[HOURS] = df[HOURS].fillna(0)
    df["rec_total"] = df[HOURS].sum(axis=1)

    # Rule 1: weekdays only
    df = df[df["date"].dt.dayofweek < 5].copy()

    # Rules 3 & 4: unit = station x direction; sum present classes
    unit = (df.groupby(["station_id", "cardinal_direction_seq", "year", "date"])
              ["rec_total"].sum().reset_index(name="vol"))
    unit["uid"] = unit["station_id"] + "_" + unit["cardinal_direction_seq"].astype(str)

    # Rule 5: same-weekday alignment -> year-2000 anchor (x_align)
    def anchor(r):
        d = r["date"]
        d = d if r["year"] == 2026 else d - pd.Timedelta(days=1)
        return pd.Timestamp(2000, d.month, d.day)
    unit["x_align"] = unit.apply(anchor, axis=1)

    # Rule 6: balanced panel of units matched in BOTH years
    matched = (unit.pivot_table(index=["uid", "x_align"], columns="year", values="vol")
                    .dropna())
    n_anchor = unit["x_align"].nunique()
    cov = matched.reset_index().groupby("uid")["x_align"].nunique() / n_anchor
    panel = cov[cov >= COVERAGE].index
    matched = matched.reset_index()
    matched = matched[matched["uid"].isin(panel)]

    # Rule 7: per-anchor MEAN over panel units (each year)
    daily = matched.groupby("x_align")[[2025, 2026]].mean()
    daily.columns = ["v25", "v26"]
    daily = daily.reset_index()

    # emit notebook-format rows (year, date_real, x_align, volume)
    rows = []
    for _, r in daily.iterrows():
        a = r["x_align"]
        d26 = pd.Timestamp(2026, a.month, a.day)
        d25 = pd.Timestamp(2025, a.month, a.day) + pd.Timedelta(days=1)
        rows.append((2026, d26.date(), a.date(), round(r["v26"], 2)))
        rows.append((2025, d25.date(), a.date(), round(r["v25"], 2)))
    out = pd.DataFrame(rows, columns=["year", "date_real", "x_align", "volume"])
    out = out.sort_values(["x_align", "year"]).reset_index(drop=True)
    out.to_csv(out_path, index=False)
    print(f"  [clean] {label:10s}: {len(panel):3d} balanced units | "
          f"{len(daily)} aligned weekdays -> {out_path}")
    return out


# ============================================================================
# SECTION 3  LOAD OPAL + FUEL; BUILD ALIGNED, SCREENED ANALYSIS PANELS
# ----------------------------------------------------------------------------
# Produces the wide, same-weekday-aligned 2025-vs-2026 panels (trc, opc) used by
# methods M2-M7, after the 9-Feb cutoff and the symmetric 80% data-quality
# screen. fp is the weekday-only Sydney fuel-price series.
# ============================================================================
def load_fuel():
    fp = pd.read_csv(
        FUEL_PATH, skiprows=1, header=None,
        names=["date", "Sydney", "Melbourne", "Brisbane", "Adelaide",
               "Perth", "Darwin", "Hobart", "National"])
    fp = fp.dropna(subset=["date"])
    fp["date"] = pd.to_datetime(fp["date"], format="%A, %d %B %Y", errors="coerce")
    fp = fp.dropna(subset=["date"])[["date", "Sydney"]].sort_values("date")
    fp = fp[fp["date"].dt.dayofweek < 5].copy()
    fp = fp.rename(columns={"Sydney": "price"})
    return fp


def load_panel(path, val, ali):
    df = pd.read_csv(path).dropna()
    df["date_real"] = pd.to_datetime(df["date_real"])
    df["aligned"]   = pd.to_datetime(df[ali])
    df = df[df["date_real"].dt.dayofweek < 5].copy()
    w = df.pivot_table(index="aligned", columns="year", values=val).dropna()
    w.columns = ["y25", "y26"]
    w = w.reset_index().sort_values("aligned")
    w["date26"]  = w["aligned"].apply(lambda d: pd.Timestamp(2026, d.month, d.day))
    w["date25"]  = w["aligned"].apply(lambda d: pd.Timestamp(2025, d.month, d.day) + pd.Timedelta(days=1))
    w["weekday"] = w["date26"].dt.day_name()
    return w


def attach(w, fp):
    w = w.merge(fp.rename(columns={"date": "date26", "price": "P26"}), on="date26", how="left")
    w = w.merge(fp.rename(columns={"date": "date25", "price": "P25"}), on="date25", how="left")
    w["lP26"]  = np.log(w["P26"]);  w["lP25"]  = np.log(w["P25"])
    w["lprat"] = np.log(w["P26"] / w["P25"])
    w["ly26"]  = np.log(w["y26"]);  w["ly25"]  = np.log(w["y25"])
    w["lr"]    = np.log(w["y26"] / w["y25"])
    w["ratio"] = w["y26"] / w["y25"]
    w = w.sort_values("date26").reset_index(drop=True)
    w["post"] = (w["date26"] >= SHOCK).astype(int)
    return w


def screen_pairs(w, frac=ANOM_FRAC, label=""):
    """Symmetric data-quality screen: drop an aligned weekday pair if EITHER
    year's value is below `frac` of its same-weekday median."""
    w = w.copy()
    med25 = w.groupby("weekday")["y25"].transform("median")
    med26 = w.groupby("weekday")["y26"].transform("median")
    bad = (w["y25"] < frac * med25) | (w["y26"] < frac * med26)
    if bad.any():
        print(f"  [screen] {label}: dropping {int(bad.sum())} contaminated pair(s)")
    return w.loc[~bad].copy().reset_index(drop=True)


def prepare(traffic_path, fp):
    """Build the screened, aligned analysis panels for a given traffic file."""
    tr = attach(load_panel(traffic_path, "volume",      "x_align"),      fp)
    op = attach(load_panel(OPAL_PATH,    "tap_ons_sum", "date_aligned"), fp)

    trc = tr[tr["date26"] >= ANOM].copy()      # early-Feb anomaly removed
    opc = op[op["date26"] >= ANOM].copy()
    trc = screen_pairs(trc, label="Traffic")
    opc = screen_pairs(opc, label="Opal tap-ons")

    trc = trc.sort_values("date26").reset_index(drop=True); trc["t"] = np.arange(len(trc))
    opc = opc.sort_values("date26").reset_index(drop=True); opc["t"] = np.arange(len(opc))
    return trc, opc


# ============================================================================
# SECTION 4  THE SEVEN METHODS (M1-M7)
# ============================================================================

# ---------------------------------------------------------------------------
# M1  Monthly year-on-year comparison
# ---------------------------------------------------------------------------
def monthly_mean(df, val, col="date_real"):
    d = df.copy(); d["date"] = pd.to_datetime(d[col]); d["ym"] = d["date"].dt.strftime("%Y-%m")
    return d.groupby("ym")[val].mean()


def method1_monthly(traffic_path, fp):
    """M1 builds its own monthly means from the daily series, applying the
    9-Feb cutoff and an 80% low-day screen to BOTH years' raw weekday series."""
    raw_t = pd.read_csv(traffic_path).dropna()
    raw_o = pd.read_csv(OPAL_PATH).dropna()
    raw_t["date_real"] = pd.to_datetime(raw_t["date_real"])
    raw_o["date_real"] = pd.to_datetime(raw_o["date_real"])
    raw_t = raw_t[raw_t["date_real"].dt.dayofweek < 5].copy()
    raw_o = raw_o[raw_o["date_real"].dt.dayofweek < 5].copy()

    def drop_early_both(df):
        m = (((df["date_real"].dt.year == 2026) & (df["date_real"] < ANOM)) |
             ((df["date_real"].dt.year == 2025) & (df["date_real"] < pd.Timestamp("2025-02-09"))))
        return df[~m].copy()
    raw_t = drop_early_both(raw_t)
    raw_o = drop_early_both(raw_o)

    def screen_raw(df, val, frac=ANOM_FRAC):
        df = df.copy()
        df["wd"] = df["date_real"].dt.day_name(); df["yr"] = df["date_real"].dt.year
        med = df.groupby(["yr", "wd"])[val].transform("median")
        bad = df[val] < frac * med
        return df[~bad].copy()
    raw_t = screen_raw(raw_t, "volume")
    raw_o = screen_raw(raw_o, "tap_ons_sum")

    t_mo = monthly_mean(raw_t, "volume")
    o_mo = monthly_mean(raw_o, "tap_ons_sum")
    fp_w  = fp[(fp["date"] >= "2025-02-01") & (fp["date"] <= "2026-04-30")].copy()
    fp_mo = monthly_mean(fp_w, "price", col="date")

    def yoy(month):
        return pd.DataFrame({
            "month": [month],
            "fuel_2025": [fp_mo.get(f"2025-{month:02d}", np.nan)],
            "fuel_2026": [fp_mo.get(f"2026-{month:02d}", np.nan)],
            "fuel_YoY_%": [(fp_mo.get(f"2026-{month:02d}", np.nan)/fp_mo.get(f"2025-{month:02d}", np.nan)-1)*100],
            "traffic_2025": [t_mo.get(f"2025-{month:02d}", np.nan)],
            "traffic_2026": [t_mo.get(f"2026-{month:02d}", np.nan)],
            "traffic_YoY_%": [(t_mo.get(f"2026-{month:02d}", np.nan)/t_mo.get(f"2025-{month:02d}", np.nan)-1)*100],
            "opal_2025": [o_mo.get(f"2025-{month:02d}", np.nan)],
            "opal_2026": [o_mo.get(f"2026-{month:02d}", np.nan)],
            "opal_YoY_%": [(o_mo.get(f"2026-{month:02d}", np.nan)/o_mo.get(f"2025-{month:02d}", np.nan)-1)*100],
        })

    M1 = pd.concat([yoy(2), yoy(3)], ignore_index=True)
    M1["traffic_slowdown_pp"] = M1["traffic_YoY_%"] - M1.loc[0, "traffic_YoY_%"]
    M1["opal_slowdown_pp"]    = M1["opal_YoY_%"]    - M1.loc[0, "opal_YoY_%"]
    return M1


# ---------------------------------------------------------------------------
# M2  Pre/post/peak-week descriptive comparison (Welch t-test on the ratio)
# ---------------------------------------------------------------------------
def descr(w, label):
    w = w.sort_values("date26").copy()
    pre  = w.loc[w.post == 0, "ratio"]
    post = w.loc[w.post == 1, "ratio"]
    peakw = w.loc[(w.date26 >= PEAK - pd.Timedelta(days=6)) &
                  (w.date26 <= PEAK + pd.Timedelta(days=2)), "ratio"]
    _, p_post_pre = stats.ttest_ind(post, pre, equal_var=False)
    _, p_peak_pre = stats.ttest_ind(peakw, pre, equal_var=False)
    return {
        "series": label, "n_pre": len(pre), "n_post": len(post), "n_peakwk": len(peakw),
        "pre_mean": pre.mean(), "post_mean": post.mean(), "peakwk_mean": peakw.mean(),
        "post_minus_pre_pp": (post.mean() - pre.mean()) * 100,
        "p_post_vs_pre": p_post_pre,
        "peakwk_minus_pre_pp": (peakw.mean() - pre.mean()) * 100,
        "p_peakwk_vs_pre": p_peak_pre,
    }


def method2_descriptive(trc, opc):
    return pd.DataFrame([descr(trc, "Traffic"), descr(opc, "Opal tap-ons")])


# ---------------------------------------------------------------------------
# M3 / M4  Interrupted time series (single-series and controlled)
# ---------------------------------------------------------------------------
def its(w, label, controlled=False):
    w = w.sort_values("date26").copy()
    w["weekday"] = w["date26"].dt.day_name()
    w["t"] = np.arange(len(w))  # weekday observation index
    # weekday-only time since shock (0 pre-shock; increments by 1 per weekday post)
    w["tss"] = np.where(w["date26"] >= SHOCK, np.arange(len(w)) - np.argmax(w["date26"] >= SHOCK), 0)
    w["post"] = (w["date26"] >= SHOCK).astype(int)

    hac = dict(cov_type="HAC", cov_kwds={"maxlags": 7})
    formula = "ly26 ~ t + post + tss + C(weekday)" + (" + ly25" if controlled else "")
    m = smf.ols(formula, data=w).fit(**hac)

    b_post = m.params["post"]; ci_post = m.conf_int().loc["post"]; b_tss = m.params["tss"]
    # project to peak using WEEKDAY OBSERVATIONS elapsed (matches the slope unit)
    peak_tss = int(w.loc[w["date26"] <= PEAK, "tss"].max())
    return {
        "series": label,
        "level_shift_%": (np.exp(b_post) - 1) * 100,
        "level_ci_lo_%": (np.exp(ci_post[0]) - 1) * 100,
        "level_ci_hi_%": (np.exp(ci_post[1]) - 1) * 100,
        "level_p": m.pvalues["post"],
        "slope_%/wkday": (np.exp(b_tss) - 1) * 100,
        "slope_p": m.pvalues["tss"],
        "weekday_obs_to_peak": peak_tss,
        "effect_at_peak_%": (np.exp(b_post + peak_tss * b_tss) - 1) * 100,
        "n": int(m.nobs), "R2": m.rsquared, "_m": m,
    }


def method3_its(trc, opc):
    rows = [its(trc, "Traffic"), its(opc, "Opal tap-ons")]
    M3 = pd.DataFrame([{k: v for k, v in d.items() if not k.startswith("_")} for d in rows])
    return M3, rows  # rows carry the fitted models ('_m') for SECTION 5


def method4_controlled_its(trc, opc):
    rows = [its(trc, "Traffic", controlled=True), its(opc, "Opal tap-ons", controlled=True)]
    M4 = pd.DataFrame([{k: v for k, v in d.items() if not k.startswith("_")} for d in rows])
    return M4


# ---------------------------------------------------------------------------
# M5  Level-baseline counterfactual
# ---------------------------------------------------------------------------
def cf(w, label):
    w = w.sort_values("date26").copy()
    pre = w.loc[w.post == 0, "ratio"]
    mu, sd = pre.mean(), pre.std(ddof=1)
    w["cf"]  = w["y25"] * mu
    w["eff"] = w["y26"] - w["cf"]
    post = w.loc[w.post == 1].copy()
    peakw = w.loc[(w.date26 >= PEAK - pd.Timedelta(days=6)) &
                  (w.date26 <= PEAK + pd.Timedelta(days=2))].copy()
    return {
        "series": label, "pre_mean_ratio": mu, "pre_sd_ratio": sd,
        "cum_effect": post["eff"].sum(),
        "cum_%_vs_cf": post["eff"].sum() / post["cf"].sum() * 100,
        "mean_daily_effect": post["eff"].mean(),
        "peakwk_%_vs_cf": peakw["eff"].sum() / peakw["cf"].sum() * 100,
    }


def method5_counterfactual(trc, opc):
    return pd.DataFrame([cf(trc, "Traffic"), cf(opc, "Opal tap-ons")])


# ---------------------------------------------------------------------------
# M6  Quandt-Andrews unknown-break sup-F test
# ---------------------------------------------------------------------------
def supF(w, label):
    w = w.sort_values("date26").reset_index(drop=True).copy()
    w["t"] = np.arange(len(w))
    n = len(w); lo, hi = int(0.15 * n), int(0.85 * n)
    y = w["lr"].values
    X0 = sm.add_constant(w[["t"]].values)
    rss_r = sm.OLS(y, X0).fit().ssr
    Fs, ds = [], []
    for i in range(lo, hi):
        dum = np.zeros(n); dum[i:] = 1
        Xu = np.column_stack([np.ones(n), w["t"].values, dum, dum * w["t"].values])
        rss_u = sm.OLS(y, Xu).fit().ssr
        F = ((rss_r - rss_u) / 2) / (rss_u / (n - 4))
        Fs.append(F); ds.append(w.date26.iloc[i])
    Fs = np.array(Fs); k = int(np.argmax(Fs)); sf = Fs[k]
    return {
        "series": label, "break_date": ds[k].strftime("%d %b %Y"), "supF": sf,
        "days_from_28Feb": (ds[k] - SHOCK).days,
        "CV_10%": ANDREWS_CV["10%"], "CV_5%": ANDREWS_CV["5%"], "CV_1%": ANDREWS_CV["1%"],
        "reject_no_break_5%": sf > ANDREWS_CV["5%"],
    }


def method6_supF(trc, opc):
    rows = [supF(trc, "Traffic"), supF(opc, "Opal tap-ons")]
    return pd.DataFrame([{k: v for k, v in d.items() if not k.startswith("_")} for d in rows])


# ---------------------------------------------------------------------------
# M7  Fuel-price dose-response (elasticities)
# ---------------------------------------------------------------------------
def dose(w, label):
    w = w.sort_values("date26").copy()
    hac = dict(cov_type="HAC", cov_kwds={"maxlags": 7})
    # elasticity of the 2026/2025 demand ratio w.r.t. the 2026/2025 fuel-price ratio
    m_a = smf.ols("lr ~ lprat + C(weekday)", data=w).fit(**hac)
    # levels specification with matched 2025 demand/price, weekday FE and trend
    m_b = smf.ols("ly26 ~ lP26 + ly25 + lP25 + C(weekday) + t", data=w).fit(**hac)
    e_a = m_a.params["lprat"]; ci_a = m_a.conf_int().loc["lprat"]; e_b = m_b.params["lP26"]
    return {
        "series": label, "elasticity_YoYratio": e_a, "ci_lo_a": ci_a[0], "ci_hi_a": ci_a[1],
        "p_a": m_a.pvalues["lprat"], "elasticity_levels": e_b, "p_b": m_b.pvalues["lP26"],
    }


def method7_dose(trc, opc):
    return pd.DataFrame([dose(trc, "Traffic"), dose(opc, "Opal tap-ons")])


# ============================================================================
# SECTION 5  AUTOCORRELATION DIAGNOSTICS (ITS RESIDUALS)
# ----------------------------------------------------------------------------
# Computed on the residuals of the single-series ITS (M3) regression
# (ly26 ~ t + post + tss + weekday FE). The HAC standard errors used in M3/M4/M7
# accommodate any residual dependence found here.
# ============================================================================
def autocorr_report(its_model, label):
    r = its_model.resid
    dw  = durbin_watson(r)
    ac1 = pd.Series(r).autocorr(1)
    lb  = acorr_ljungbox(r, lags=[5], return_df=True)["lb_pvalue"].iloc[0]
    bg  = acorr_breusch_godfrey(its_model, nlags=5)[3]
    return {"series": label, "n": int(its_model.nobs), "DW": dw,
            "AC1": ac1, "LjungBox5_p": lb, "BreuschGodfrey5_p": bg}


# ============================================================================
# SECTION 6  RUN ALL THREE SERIES AND PRINT TABLE 1
# ============================================================================
# ---- deterministic paper-ready formatters (match the manuscript verbatim) ----
def _p(pv):
    if pv < 0.001: return "p<0.001"
    if pv < 0.10:  return f"p={pv:.3f}"
    return f"p={pv:.2f}"
def _pp(x): return f"{x:+.1f} pp"
def _pc(x): return f"{x:+.1f}%"
def _ps(pv): return "<0.001" if pv < 0.001 else f"{pv:.3f}"   # slope p (3 d.p.)

def cells(M1, M2, M3, M4, M5, M6, M7, row):
    sl = "traffic_slowdown_pp" if row == 0 else "opal_slowdown_pp"
    return {
        "M1": f"{M1.loc[1, sl]:+.1f} pp slowdown",
        "M2": f"{_pp(M2.loc[row,'post_minus_pre_pp'])} ({_p(M2.loc[row,'p_post_vs_pre'])})",
        "M3": f"{_pc(M3.loc[row,'effect_at_peak_%'])} at peak "
              f"(p_lvl={M3.loc[row,'level_p']:.2f}, p_slp={_ps(M3.loc[row,'slope_p'])})",
        "M4": f"{_pc(M4.loc[row,'effect_at_peak_%'])} at peak "
              f"(p_lvl={M4.loc[row,'level_p']:.2f}, p_slp={_ps(M4.loc[row,'slope_p'])})",
        "M5": f"{_pc(M5.loc[row,'cum_%_vs_cf'])} cum.; {_pc(M5.loc[row,'peakwk_%_vs_cf'])} peak-week",
        "M6": f"No 5% break (sup-F={M6.loc[row,'supF']:.2f}; {M6.loc[row,'break_date']})",
        "M7": f"eta={M7.loc[row,'elasticity_YoYratio']:+.3f} ({_p(M7.loc[row,'p_a'])})",
    }


def run_series(traffic_path, fp):
    """Compute M1-M7 for one traffic panel (returns the seven DataFrames plus
    the M3 fitted models for autocorrelation)."""
    trc, opc = prepare(traffic_path, fp)
    M1 = method1_monthly(traffic_path, fp)
    M2 = method2_descriptive(trc, opc)
    M3, m3_rows = method3_its(trc, opc)
    M4 = method4_controlled_its(trc, opc)
    M5 = method5_counterfactual(trc, opc)
    M6 = method6_supF(trc, opc)
    M7 = method7_dose(trc, opc)
    return dict(trc=trc, opc=opc, M1=M1, M2=M2, M3=M3, M4=M4, M5=M5, M6=M6, M7=M7,
                m3_traffic=m3_rows[0]["_m"], m3_opal=m3_rows[1]["_m"])


def main():
    print("STEP 1-2  Reading raw counters and reconstructing balanced panels ...")
    build_traffic_panel(None,     TRAFFIC_ALLNSW_CSV, "ALL NSW")
    build_traffic_panel("Sydney", TRAFFIC_SYDNEY_CSV, "SYDNEY")

    fp = load_fuel()
    print("\nSTEP 3-4  Preparing panels and running M1-M7 ...")
    print("  All-NSW run:")
    nsw = run_series(TRAFFIC_ALLNSW_CSV, fp)   # traffic All-NSW + Opal
    print("  Greater Sydney run:")
    syd = run_series(TRAFFIC_SYDNEY_CSV, fp)   # traffic Greater Sydney

    c_nsw = cells(nsw["M1"], nsw["M2"], nsw["M3"], nsw["M4"], nsw["M5"], nsw["M6"], nsw["M7"], row=0)
    c_syd = cells(syd["M1"], syd["M2"], syd["M3"], syd["M4"], syd["M5"], syd["M6"], syd["M7"], row=0)
    c_opl = cells(nsw["M1"], nsw["M2"], nsw["M3"], nsw["M4"], nsw["M5"], nsw["M6"], nsw["M7"], row=1)

    # ---- Table 1 ----
    print("\n" + "=" * 132)
    print("TABLE 1  (matches the manuscript verbatim)")
    print("=" * 132)
    print(f"{'Method':<8}{'Road traffic (All-NSW)':<42}{'Road traffic (Gr. Sydney)':<42}{'Public transport (All-NSW)'}")
    print("-" * 132)
    for m in ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]:
        print(f"{m:<8}{c_nsw[m]:<42}{c_syd[m]:<42}{c_opl[m]}")

    tab = pd.DataFrame({
        "Method": ["M1", "M2", "M3", "M4", "M5", "M6", "M7"],
        "Traffic_AllNSW":        [c_nsw[m] for m in ["M1","M2","M3","M4","M5","M6","M7"]],
        "Traffic_GreaterSydney": [c_syd[m] for m in ["M1","M2","M3","M4","M5","M6","M7"]],
        "PublicTransport_AllNSW":[c_opl[m] for m in ["M1","M2","M3","M4","M5","M6","M7"]],
    })
    tab.to_csv("table1_final.csv", index=False)

    # ---- Autocorrelation (SECTION 5) ----
    print("\n" + "=" * 132)
    print("ITS RESIDUAL AUTOCORRELATION (matches the Methods text verbatim)")
    print("=" * 132)
    ac = pd.DataFrame([
        autocorr_report(nsw["m3_traffic"], "Road traffic (All-NSW)"),
        autocorr_report(syd["m3_traffic"], "Road traffic (Greater Sydney)"),
        autocorr_report(nsw["m3_opal"],    "Public transport (All-NSW)"),
    ])
    for _, r in ac.iterrows():
        lbp = "p<0.001" if r.LjungBox5_p < 0.001 else f"p={r.LjungBox5_p:.3f}"
        print(f"{r.series:<32} n={r.n:<3} DW={r.DW:.2f}  AC(1)={r.AC1:+.2f}  "
              f"Ljung-Box(5) {lbp}  Breusch-Godfrey(5) p={r.BreuschGodfrey5_p:.3f}")

    print("\nWrote table1_final.csv")


if __name__ == "__main__":
    main()
