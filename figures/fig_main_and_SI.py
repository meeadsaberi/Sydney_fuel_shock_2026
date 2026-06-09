import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
# larger text throughout; keep fonts as editable text in SVG (no path conversion)
plt.rcParams.update({"font.size":13,"axes.titlesize":13.5,"axes.titleweight":"bold",
                     "axes.labelsize":12.5,"xtick.labelsize":11.5,"ytick.labelsize":11.5,
                     "legend.fontsize":11,"font.family":"DejaVu Sans",
                     "axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none"})
HOURS=[f"hour_{h:02d}" for h in range(24)]; COVERAGE=0.90
SHOCK=pd.Timestamp("2026-02-28"); PEAK=pd.Timestamp("2026-03-26"); ANOM=pd.Timestamp("2026-02-09"); FRAC=0.80
C25="#9aa4ad"; C26="#1f6feb"
def to26(a): return pd.Timestamp(2026,a.month,a.day)
def axis26(d,yr):
    d=d if yr==2026 else d-pd.Timedelta(days=1); return pd.Timestamp(2026,d.month,d.day)

# ============================== shared loaders ==============================
def sydney_imputed_units():
    df=pd.read_csv("raw_feb_mar_2025_2026.csv",dtype={"station_id":str},low_memory=False)
    df=df[df["rms_region"]=="Sydney"].copy(); df["date"]=pd.to_datetime(df["date"])
    for h in HOURS: df[h]=df[h].fillna(df.groupby(["station_id","cardinal_direction_seq","classification_seq","year"])[h].transform("mean"))
    for h in HOURS: df[h]=df[h].fillna(df.groupby(["station_id","cardinal_direction_seq","year"])[h].transform("mean"))
    df[HOURS]=df[HOURS].fillna(0); df["rec_total"]=df[HOURS].sum(axis=1)
    g=df.groupby(["station_id","cardinal_direction_seq","year","date"],as_index=False)["rec_total"].sum()
    g["uid"]=g["station_id"]+"_"+g["cardinal_direction_seq"].astype(str); g=g[g["date"].dt.dayofweek<5]
    return g
def fuel():
    fp=pd.read_csv("Fuel_price.csv",skiprows=1,header=None); fp[0]=pd.to_datetime(fp[0],format="%A, %d %B %Y",errors="coerce")
    fp=fp.dropna(subset=[0]).rename(columns={0:"date",1:"price"}); fp["year"]=fp.date.dt.year
    fp=fp[fp.date.dt.dayofweek<5]; fp=fp[fp.year.isin([2025,2026])].copy()
    fp["date26"]=[axis26(d,y) for d,y in zip(fp["date"],fp["year"])]; return fp

def cleaned_traffic():
    g=sydney_imputed_units()
    g["x"]=g.apply(lambda r:(r["date"] if r.year==2026 else r["date"]-pd.Timedelta(days=1)).replace(year=2000),axis=1)
    piv=g.pivot_table(index=["uid","x"],columns="year",values="rec_total").dropna()
    na=g["x"].nunique(); cov=piv.reset_index().groupby("uid")["x"].nunique()/na
    units=cov[cov>=COVERAGE].index
    p=piv.reset_index(); p=p[p.uid.isin(units)]
    tr=p.groupby("x")[[2025,2026]].mean().reset_index(); tr["date26"]=tr.x.apply(to26); tr["wd"]=tr.date26.dt.day_name()
    tr=tr[tr.date26>=ANOM].copy()
    m25=tr.groupby("wd")[2025].transform("median"); m26=tr.groupby("wd")[2026].transform("median")
    tr=tr[~((tr[2025]<FRAC*m25)|(tr[2026]<FRAC*m26))].sort_values("date26"); return tr,len(units)
def cleaned_opal():
    op=pd.read_csv("opal_all_nsw_feb_mar_2025_2026_aligned.csv"); op["date_real"]=pd.to_datetime(op["date_real"])
    op=op[op.date_real.dt.dayofweek<5].copy()
    ow=op.pivot_table(index="date_aligned",columns="year",values="tap_ons_sum").dropna(); ow.columns=["o25","o26"]
    ow=ow.reset_index(); ow["date_aligned"]=pd.to_datetime(ow["date_aligned"])
    ow["date26"]=ow["date_aligned"].apply(lambda a: pd.Timestamp(2026,a.month,a.day)); ow["wd"]=ow.date26.dt.day_name()
    ow=ow[ow.date26>=ANOM].copy()
    m25=ow.groupby("wd").o25.transform("median"); m26=ow.groupby("wd").o26.transform("median")
    ow=ow[~((ow.o25<FRAC*m25)|(ow.o26<FRAC*m26))].sort_values("date26"); return ow

def raw_traffic():
    df=pd.read_csv("raw_feb_mar_2025_2026.csv",dtype={"station_id":str},
                   usecols=["station_id","cardinal_direction_seq","rms_region","year","date","daily_total"])
    df=df[df["rms_region"]=="Sydney"].copy(); df["date"]=pd.to_datetime(df["date"]); df=df[df["date"].dt.dayofweek<5]
    unit=df.groupby(["station_id","cardinal_direction_seq","year","date"],as_index=False)["daily_total"].sum()
    avg=unit.groupby(["year","date"],as_index=False)["daily_total"].mean()
    avg["date26"]=[axis26(d,y) for d,y in zip(avg["date"],avg["year"])]
    return avg.pivot_table(index="date26",columns="year",values="daily_total").reset_index()
def raw_opal():
    op=pd.read_csv("opal_all_nsw_feb_mar_2025_2026_aligned.csv"); op["date_real"]=pd.to_datetime(op["date_real"])
    op=op[op.date_real.dt.dayofweek<5].copy(); op["date26"]=[axis26(d,y) for d,y in zip(op["date_real"],op["year"])]
    ow=op.pivot_table(index="date26",columns="year",values="tap_ons_sum").reset_index(); ow.columns=["date26","o25","o26"]; return ow

# ============================== plotting helpers ==============================
def make_gapped(X0,X1):
    FULL=pd.date_range(X0,X1,freq="D")
    def gapped(dates,vals,bridge):
        s=pd.Series(np.asarray(vals,dtype=float),index=pd.to_datetime(list(dates))); s=s[~s.index.duplicated()]
        pts=s.reindex(FULL)
        if bridge:
            line=pts.interpolate(method="index",limit_area="inside"); line[FULL.dayofweek>=5]=np.nan
        else:
            line=pts.copy(); line[FULL.dayofweek>=5]=np.nan
        return FULL,line.values,FULL,pts.values
    return FULL,gapped
def shade(a,X0,X1):
    d=X0
    while d<=X1:
        if d.dayofweek==5: a.axvspan(d,d+pd.Timedelta(days=2),color="0.90",zorder=0,lw=0)
        d+=pd.Timedelta(days=1)
    a.axvline(SHOCK,color="crimson",ls="--",lw=1.3); a.axvline(PEAK,color="green",ls="--",lw=1.2)
    a.grid(axis="y",alpha=.25); a.set_xlim(X0,X1)
def panel(a,gapped,dates,v25,v26,bridge,ylab,title,legloc="lower left"):
    wx,l25,fx,p25=gapped(dates,v25,bridge); _,l26,_,p26=gapped(dates,v26,bridge)
    a.plot(wx,l25,color=C25,lw=1.7); a.plot(fx,p25,color=C25,lw=0,marker="o",ms=3.5,label="2025")
    a.plot(wx,l26,color=C26,lw=2.0); a.plot(fx,p26,color=C26,lw=0,marker="o",ms=3.5,label="2026")
    a.set_ylabel(ylab); a.set_title(title,loc="left"); a.legend(frameon=False,ncol=2,loc=legloc)
def finish(ax,X0,X1,xlabel,fname):
    ax[2].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%d %b")); ax[2].set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(fname+".png",dpi=200,bbox_inches="tight"); plt.savefig(fname+".svg",bbox_inches="tight")
    print("saved",fname+".{png,svg}")

# ============================== FIGURE 1 (MAIN): cleaned, weekday-bridged ==============================
tr,nu=cleaned_traffic(); ow=cleaned_opal(); fp=fuel()
X0,X1=ANOM,pd.Timestamp("2026-03-31"); FULL,gapped=make_gapped(X0,X1)
fp1=fp[(fp.date26>=X0)&(fp.date26<=X1)]; f25=fp1[fp1.year==2025].sort_values("date26"); f26=fp1[fp1.year==2026].sort_values("date26")
fig,ax=plt.subplots(3,1,figsize=(8.6,9.0),sharex=True)
panel(ax[0],gapped,tr.date26,tr[2025]/1e3,tr[2026]/1e3,True,"Traffic\n(1000 veh / counter-day)",
      f"a  Greater Sydney road traffic — cleaned weekday mean per counter ({nu} units)")
panel(ax[1],gapped,ow.date26,ow.o25/1e6,ow.o26/1e6,True,"Patronage\n(million tap-ons / day)",
      "b  Public transport patronage — Opal tap-ons, All-NSW (weekdays)")
panel(ax[2],gapped,f26.date26,np.interp(mdates.date2num(f26.date26),mdates.date2num(f25.date26),f25.price)/100,f26.price/100,
      True,"Fuel price\n(AUD / litre)","c  Sydney average unleaded petrol price (weekdays)",legloc="upper left")
for a in ax: shade(a,X0,X1)
finish(ax,X0,X1,"2026 (weekday-aligned; lines connect weekdays and break at weekends; weekends shaded;\n"
       "dashed: 28 Feb shock onset, 26 Mar price peak)","fig_main_cleaned")
plt.close()

# ============================== FIGURE 2 (SI): raw, no cleaning, gaps NOT bridged ==============================
rt=raw_traffic(); ro=raw_opal()
X0,X1=pd.Timestamp("2026-02-01"),pd.Timestamp("2026-03-31"); FULL,gapped=make_gapped(X0,X1)
rt=rt[(rt.date26>=X0)&(rt.date26<=X1)].sort_values("date26"); ro=ro[(ro.date26>=X0)&(ro.date26<=X1)].sort_values("date26")
fp2=fp[(fp.date26>=X0)&(fp.date26<=X1)]; f25=fp2[fp2.year==2025].sort_values("date26"); f26=fp2[fp2.year==2026].sort_values("date26")
fig,ax=plt.subplots(3,1,figsize=(8.6,9.0),sharex=True)
panel(ax[0],gapped,rt.date26,rt[2025]/1e3,rt[2026]/1e3,False,"Traffic\n(1000 veh / counter-day)",
      "a  Greater Sydney road traffic — RAW mean per counter (all counters, no cleaning)")
panel(ax[1],gapped,ro.date26,ro.o25/1e6,ro.o26/1e6,False,"Patronage\n(million tap-ons / day)",
      "b  Public transport patronage — RAW Opal tap-ons, All-NSW (no cleaning)")
panel(ax[2],gapped,f26.date26,np.interp(mdates.date2num(f26.date26),mdates.date2num(f25.date26),f25.price)/100,f26.price/100,
      False,"Fuel price\n(AUD / litre)","c  Sydney average unleaded petrol price (weekdays)",legloc="upper left")
for a in ax: shade(a,X0,X1)
finish(ax,X0,X1,"2026 (weekday-aligned; markers = observations used; weekends shaded; RAW data, no cleaning;\n"
       "dashed: 28 Feb shock, 26 Mar peak)","fig_SI_raw")
plt.close()
print("done")

# ============================== FIGURE 3 (SI): cleaned, AS-ANALYSED, gaps NOT bridged ==============================
tr,nu=cleaned_traffic(); ow=cleaned_opal()
X0,X1=ANOM,pd.Timestamp("2026-03-31"); FULL,gapped=make_gapped(X0,X1)
fp3=fuel(); fp3=fp3[(fp3.date26>=X0)&(fp3.date26<=X1)]; f25=fp3[fp3.year==2025].sort_values("date26"); f26=fp3[fp3.year==2026].sort_values("date26")
fig,ax=plt.subplots(3,1,figsize=(8.6,9.0),sharex=True)
panel(ax[0],gapped,tr.date26,tr[2025]/1e3,tr[2026]/1e3,False,"Traffic\n(1000 veh / counter-day)",
      f"a  Greater Sydney road traffic — cleaned weekday mean per counter ({nu} units)")
panel(ax[1],gapped,ow.date26,ow.o25/1e6,ow.o26/1e6,False,"Patronage\n(million tap-ons / day)",
      "b  Public transport patronage — Opal tap-ons, All-NSW (weekdays)")
panel(ax[2],gapped,f26.date26,np.interp(mdates.date2num(f26.date26),mdates.date2num(f25.date26),f25.price)/100,f26.price/100,
      False,"Fuel price\n(AUD / litre)","c  Sydney average unleaded petrol price (weekdays)",legloc="upper left")
for a in ax: shade(a,X0,X1)
finish(ax,X0,X1,"2026 (weekday-aligned; markers = observations used in the analysis; gaps = days removed by the\n"
       "data-quality screen; weekends shaded; dashed: 28 Feb shock onset, 26 Mar price peak)","fig_SI_cleaned_asanalysed")
plt.close()
print("figure 3 done")
