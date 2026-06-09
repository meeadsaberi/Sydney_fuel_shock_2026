import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
plt.rcParams.update({"font.size":10,"axes.titlesize":10,"axes.titleweight":"bold",
                     "font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})
HOURS=[f"hour_{h:02d}" for h in range(24)]; COVERAGE=0.90
SHOCK=pd.Timestamp("2026-02-28"); PEAK=pd.Timestamp("2026-03-26"); ANOM=pd.Timestamp("2026-02-09"); FRAC=0.80
C25="#9aa4ad"; C26="#1f6feb"
X0,X1=ANOM,pd.Timestamp("2026-03-31")
def to26(a): return pd.Timestamp(2026,a.month,a.day)
def axis26(d,yr):
    d=d if yr==2026 else d-pd.Timedelta(days=1); return pd.Timestamp(2026,d.month,d.day)

# reindex a (date26-indexed) series onto EVERY calendar day in the window, so weekend
# days exist as NaN -> the line breaks Fri..Mon instead of connecting across the gap
FULL=pd.date_range(X0,X1,freq="D")
def gapped(dates,vals):
    # markers on real obs only; line bridges screened-out WEEKDAYS within a week
    # but is forced to NaN on Sat/Sun so it BREAKS across every weekend.
    s=pd.Series(np.asarray(vals,dtype=float),index=pd.to_datetime(list(dates)))
    s=s[~s.index.duplicated()]
    pts=s.reindex(FULL)                                   # markers: real observations only
    line=pts.interpolate(method="index",limit_area="inside")  # fill interior gaps (incl weekends, temporarily)
    line[FULL.dayofweek>=5]=np.nan                        # re-open weekend gaps -> line breaks Fri|Mon
    return FULL,line.values,FULL,pts.values

# ---- cleaned WEEKDAY traffic average (analysis series), Greater Sydney ----
df=pd.read_csv("raw_feb_mar_2025_2026.csv",dtype={"station_id":str},low_memory=False)
df=df[df["rms_region"]=="Sydney"].copy(); df["date"]=pd.to_datetime(df["date"])
for h in HOURS: df[h]=df[h].fillna(df.groupby(["station_id","cardinal_direction_seq","classification_seq","year"])[h].transform("mean"))
for h in HOURS: df[h]=df[h].fillna(df.groupby(["station_id","cardinal_direction_seq","year"])[h].transform("mean"))
df[HOURS]=df[HOURS].fillna(0); df["rec_total"]=df[HOURS].sum(axis=1)
g=df.groupby(["station_id","cardinal_direction_seq","year","date"],as_index=False)["rec_total"].sum()
g["uid"]=g["station_id"]+"_"+g["cardinal_direction_seq"].astype(str); g=g[g["date"].dt.dayofweek<5]
g["x"]=g.apply(lambda r:(r["date"] if r.year==2026 else r["date"]-pd.Timedelta(days=1)).replace(year=2000),axis=1)
piv=g.pivot_table(index=["uid","x"],columns="year",values="rec_total").dropna()
na=g["x"].nunique(); cov=piv.reset_index().groupby("uid")["x"].nunique()/na
units=cov[cov>=COVERAGE].index
p=piv.reset_index(); p=p[p.uid.isin(units)]
tr=p.groupby("x")[[2025,2026]].mean().reset_index()
tr["date26"]=tr.x.apply(to26); tr["wd"]=tr.date26.dt.day_name()
tr=tr[tr.date26>=ANOM].copy()
m25=tr.groupby("wd")[2025].transform("median"); m26=tr.groupby("wd")[2026].transform("median")
tr=tr[~((tr[2025]<FRAC*m25)|(tr[2026]<FRAC*m26))].sort_values("date26")

# ---- Opal WEEKDAY (study series) ----
op=pd.read_csv("opal_all_nsw_feb_mar_2025_2026_aligned.csv"); op["date_real"]=pd.to_datetime(op["date_real"])
op=op[op.date_real.dt.dayofweek<5].copy()
ow=op.pivot_table(index="date_aligned",columns="year",values="tap_ons_sum").dropna(); ow.columns=["o25","o26"]
ow=ow.reset_index(); ow["date_aligned"]=pd.to_datetime(ow["date_aligned"])
ow["date26"]=ow["date_aligned"].apply(lambda a: pd.Timestamp(2026,a.month,a.day)); ow["wd"]=ow.date26.dt.day_name()
ow=ow[ow.date26>=ANOM].copy()
mm25=ow.groupby("wd").o25.transform("median"); mm26=ow.groupby("wd").o26.transform("median")
ow=ow[~((ow.o25<FRAC*mm25)|(ow.o26<FRAC*mm26))].sort_values("date26")

# ---- Fuel WEEKDAY ----
fp=pd.read_csv("Fuel_price.csv",skiprows=1,header=None); fp[0]=pd.to_datetime(fp[0],format="%A, %d %B %Y",errors="coerce")
fp=fp.dropna(subset=[0]).rename(columns={0:"date",1:"price"}); fp["year"]=fp.date.dt.year
fp=fp[fp.date.dt.dayofweek<5]; fp=fp[fp.year.isin([2025,2026])].copy()
fp["date26"]=[axis26(d,y) for d,y in zip(fp["date"],fp["year"])]
fp=fp[(fp.date26>=ANOM)&(fp.date26<=X1)]
fp25=fp[fp.year==2025].sort_values("date26"); fp26=fp[fp.year==2026].sort_values("date26")

# ---- figure ----
fig,ax=plt.subplots(3,1,figsize=(8.2,8.6),sharex=True)
def shade(a):
    d=X0
    while d<=X1:
        if d.dayofweek==5: a.axvspan(d,d+pd.Timedelta(days=2),color="0.90",zorder=0,lw=0)
        d+=pd.Timedelta(days=1)
    a.axvline(SHOCK,color="crimson",ls="--",lw=1.1); a.axvline(PEAK,color="green",ls="--",lw=1.0)
    a.grid(axis="y",alpha=.25); a.set_xlim(X0,X1)

wx,l25,fx,p25=gapped(tr.date26,tr[2025]/1e3); _,l26,_,p26=gapped(tr.date26,tr[2026]/1e3)
ax[0].plot(wx,l25,color=C25,lw=1.4); ax[0].plot(fx,p25,color=C25,lw=0,marker="o",ms=3,label="2025")
ax[0].plot(wx,l26,color=C26,lw=1.7); ax[0].plot(fx,p26,color=C26,lw=0,marker="o",ms=3,label="2026")
ax[0].set_ylabel("Traffic\n(1000 veh / counter-day)"); shade(ax[0])
ax[0].set_title(f"a  Greater Sydney road traffic — cleaned weekday mean per counter ({len(units)} units)",loc="left")
ax[0].legend(frameon=False,fontsize=8,ncol=2,loc="lower left")

wx,l25,fx,p25=gapped(ow.date26,ow.o25/1e6); _,l26,_,p26=gapped(ow.date26,ow.o26/1e6)
ax[1].plot(wx,l25,color=C25,lw=1.4); ax[1].plot(fx,p25,color=C25,lw=0,marker="o",ms=3,label="2025")
ax[1].plot(wx,l26,color=C26,lw=1.7); ax[1].plot(fx,p26,color=C26,lw=0,marker="o",ms=3,label="2026")
ax[1].set_ylabel("Patronage\n(million tap-ons / day)"); shade(ax[1])
ax[1].set_title("b  Public transport patronage — Opal tap-ons, All-NSW (weekdays)",loc="left")
ax[1].legend(frameon=False,fontsize=8,ncol=2,loc="lower left")

wx,l25,fx,p25=gapped(fp25.date26,fp25.price/100); _,l26,_,p26=gapped(fp26.date26,fp26.price/100)
ax[2].plot(wx,l25,color=C25,lw=1.4); ax[2].plot(fx,p25,color=C25,lw=0,marker="o",ms=3,label="2025")
ax[2].plot(wx,l26,color=C26,lw=1.7); ax[2].plot(fx,p26,color=C26,lw=0,marker="o",ms=3,label="2026")
ax[2].set_ylabel("Fuel price\n(AUD / litre)"); shade(ax[2])
ax[2].set_title("c  Sydney average unleaded petrol price (weekdays)",loc="left")
ax[2].legend(frameon=False,fontsize=8,ncol=2,loc="upper left")
ax[2].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax[2].set_xlabel("2026 (weekday-aligned; weekends shaded and blank; dashed: 28 Feb shock onset, 26 Mar price peak)")
plt.tight_layout()
plt.savefig("fig_study_weekday.png",dpi=200,bbox_inches="tight")
plt.savefig("fig_study_weekday.pdf",bbox_inches="tight")
print("saved | traffic n=%d opal n=%d fuel n=%d"%(len(tr),len(ow),len(fp26)))
