import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
plt.rcParams.update({"font.size":13,"axes.titlesize":13.5,"axes.titleweight":"bold",
                     "axes.labelsize":12.5,"xtick.labelsize":11.5,"ytick.labelsize":11.5,
                     "legend.fontsize":11,"font.family":"DejaVu Sans",
                     "axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none"})
HOURS=[f"hour_{h:02d}" for h in range(24)]; COVERAGE=0.90
SHOCK=pd.Timestamp("2026-02-28"); PEAK=pd.Timestamp("2026-03-26"); ANOM=pd.Timestamp("2026-02-09")
C25="#9aa4ad"; C26="#1f6feb"
X0,X1=ANOM,pd.Timestamp("2026-03-31"); FULL=pd.date_range(X0,X1,freq="D")
def to26(a): return pd.Timestamp(2026,a.month,a.day)
def axis26(d,yr):
    d=d if yr==2026 else d-pd.Timedelta(days=1); return pd.Timestamp(2026,d.month,d.day)

# ===== Traffic: cleaned EVERY day incl weekends (composition-stable basket) =====
df=pd.read_csv("raw_feb_mar_2025_2026.csv",dtype={"station_id":str},low_memory=False)
df=df[df["rms_region"]=="Sydney"].copy(); df["date"]=pd.to_datetime(df["date"])
for h in HOURS: df[h]=df[h].fillna(df.groupby(["station_id","cardinal_direction_seq","classification_seq","year"])[h].transform("mean"))
for h in HOURS: df[h]=df[h].fillna(df.groupby(["station_id","cardinal_direction_seq","year"])[h].transform("mean"))
df[HOURS]=df[HOURS].fillna(0); df["rec_total"]=df[HOURS].sum(axis=1)
u=df.groupby(["station_id","cardinal_direction_seq","year","date"],as_index=False)["rec_total"].sum()
u["uid"]=u["station_id"]+"_"+u["cardinal_direction_seq"].astype(str)
u["anchor"]=u.apply(lambda r:(r["date"] if r.year==2026 else r["date"]-pd.Timedelta(days=1)).replace(year=2000),axis=1)
pv=u.pivot_table(index=["uid","anchor"],columns="year",values="rec_total").dropna()
nd=u["anchor"].nunique(); cov=pv.reset_index().groupby("uid")["anchor"].nunique()/nd
basket=sorted(cov[cov>=COVERAGE].index)
ub=u[u.uid.isin(basket)].copy(); ub["wknd"]=ub["anchor"].map(lambda a: pd.Timestamp(a).dayofweek>=5)
typ=ub.groupby(["uid","year","wknd"])["rec_total"].mean().rename("fill").reset_index()
anchors=sorted(u["anchor"].unique())
grid=pd.MultiIndex.from_product([basket,[2025,2026],anchors],names=["uid","year","anchor"]).to_frame(index=False)
grid["wknd"]=grid["anchor"].map(lambda a: pd.Timestamp(a).dayofweek>=5)
grid=grid.merge(ub[["uid","year","anchor","rec_total"]],on=["uid","year","anchor"],how="left").merge(typ,on=["uid","year","wknd"],how="left")
grid["vol"]=grid["rec_total"].fillna(grid["fill"])
tr=grid.groupby(["year","anchor"],as_index=False)["vol"].mean(); tr["date26"]=tr.anchor.map(to26)
tr=tr[(tr.date26>=X0)&(tr.date26<=X1)]
t25=tr[tr.year==2025].set_index("date26")["vol"].reindex(FULL); t26=tr[tr.year==2026].set_index("date26")["vol"].reindex(FULL)

# ===== Opal raw daily incl weekends =====
op=pd.read_csv("opal_all_nsw_feb_mar_2025_2026_aligned.csv"); op["date_real"]=pd.to_datetime(op["date_real"])
op["date26"]=[axis26(d,y) for d,y in zip(op["date_real"],op["year"])]
o25=op[op.year==2025].set_index("date26")["tap_ons_sum"].reindex(FULL)
o26=op[op.year==2026].set_index("date26")["tap_ons_sum"].reindex(FULL)

# ===== Fuel: weekday real, weekend interpolated =====
fp=pd.read_csv("Fuel_price.csv",skiprows=1,header=None); fp[0]=pd.to_datetime(fp[0],format="%A, %d %B %Y",errors="coerce")
fp=fp.dropna(subset=[0]).rename(columns={0:"date",1:"price"}); fp["year"]=fp.date.dt.year
fp=fp[fp.year.isin([2025,2026])].copy(); fp["date26"]=[axis26(d,y) for d,y in zip(fp["date"],fp["year"])]
def fuel_series(yr):
    s=fp[fp.year==yr].set_index("date26")["price"].reindex(FULL)
    return s.interpolate(method="index",limit_area="inside"), s.notna()
f25,f25r=fuel_series(2025); f26,f26r=fuel_series(2026)

# ---- helper: split a full daily series into weekday-solid and weekend-dashed segments ----
def seg_weekday(vals):   # keep value only if BOTH endpoints of a step are weekdays -> solid within week
    v=pd.Series(vals,index=FULL).astype(float).copy()
    out=v.copy(); out[:]=np.nan
    # draw solid on consecutive weekday spans
    wkmask=FULL.dayofweek<5
    out[wkmask]=v[wkmask]
    return out.values
def seg_full(vals):      # full continuous (incl weekend) used for the thin dashed underlay
    return pd.Series(vals,index=FULL).astype(float).values

def draw(a,v25,v26,ylab,title,legloc="lower left",fuel=False,real25=None,real26=None):
    for v,c,real in [(v25,C25,real25),(v26,C26,real26)]:
        full=seg_full(v)
        a.plot(FULL,full,color=c,lw=1.6,ls=(0,(4,2)),alpha=0.9,zorder=1)         # thin dashed: weekend connectors (visible underneath)
        wk=seg_weekday(v)
        a.plot(FULL,wk,color=c,lw=2.4,zorder=2)                                  # solid weekday segments on top
        if fuel:
            a.plot(FULL[real.values],pd.Series(v,index=FULL)[real.values],color=c,lw=0,marker="o",ms=4.5,zorder=3,
                   label="2026" if c==C26 else "2025")
            we=(~real.values)&pd.Series(v,index=FULL).notna().values
            a.plot(FULL[we],pd.Series(v,index=FULL)[we],color=c,lw=0,marker="o",ms=5,mfc="white",mew=1.4,zorder=3)
        else:
            a.plot(FULL,full,color=c,lw=0,marker="o",ms=4.2,zorder=3,
                   label="2026" if c==C26 else "2025")                           # markers on every real day incl weekends
    a.set_ylabel(ylab); a.set_title(title,loc="left"); a.legend(frameon=False,ncol=2,loc=legloc)
    d=X0
    while d<=X1:
        if d.dayofweek==5: a.axvspan(d,d+pd.Timedelta(days=2),color="0.94",zorder=0,lw=0)
        d+=pd.Timedelta(days=1)
    a.axvline(SHOCK,color="crimson",ls=(0,(6,3)),lw=1.4); a.axvline(PEAK,color="green",ls=(0,(6,3)),lw=1.3)
    a.grid(axis="y",alpha=.25); a.set_xlim(X0,X1)

fig,ax=plt.subplots(3,1,figsize=(8.8,9.2),sharex=True)
draw(ax[0],t25/1e3,t26/1e3,"Traffic\n(1000 veh / counter-day)",
     f"a  Greater Sydney road traffic — cleaned daily mean per counter, incl. weekends ({len(basket)} units)")
draw(ax[1],o25/1e6,o26/1e6,"Patronage\n(million tap-ons / day)",
     "b  Public transport patronage — Opal tap-ons, All-NSW, incl. weekends")
draw(ax[2],f25/100,f26/100,"Fuel price\n(AUD / litre)",
     "c  Sydney unleaded petrol price — weekdays observed, weekends interpolated (hollow)",
     legloc="upper left",fuel=True,real25=f25r,real26=f26r)
ax[2].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax[2].set_xlabel("2026 (daily, weekday-aligned; solid = weekday, thin dashed = weekend; weekends shaded;\n"
                 "traffic & patronage weekends observed, fuel weekends interpolated (hollow); dashed vertical: 28 Feb shock, 26 Mar peak)")
plt.tight_layout()
plt.savefig("fig_main_allweek_dashed.png",dpi=200,bbox_inches="tight"); plt.savefig("fig_main_allweek_dashed.svg",bbox_inches="tight")
print(f"basket={len(basket)} units; saved")
