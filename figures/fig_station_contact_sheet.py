import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.lines import Line2D
HOURS=[f"hour_{h:02d}" for h in range(24)]; COVERAGE=0.90
SHOCK=pd.Timestamp("2026-02-28")
DIRMAP={1:"N",3:"E",5:"S",7:"W",2:"NE",4:"SE",6:"SW",8:"NW"}
C25="#9aa4ad"; C26="#1f6feb"

# ---- load raw, build imputed copy ----
raw=pd.read_csv("raw_feb_mar_2025_2026.csv",dtype={"station_id":str},low_memory=False)
raw["date"]=pd.to_datetime(raw["date"])
imp=raw.copy()
for h in HOURS: imp[h]=imp[h].fillna(imp.groupby(["station_id","cardinal_direction_seq","classification_seq","year"])[h].transform("mean"))
for h in HOURS: imp[h]=imp[h].fillna(imp.groupby(["station_id","cardinal_direction_seq","year"])[h].transform("mean"))
imp[HOURS]=imp[HOURS].fillna(0); imp["rec_total"]=imp[HOURS].sum(axis=1)

def unit_daily(d,val,fromhours=False):
    if fromhours: pass
    g=d.groupby(["station_id","cardinal_direction_seq","year","date"],as_index=False)[val].sum()
    g["uid"]=g["station_id"]+"_"+g["cardinal_direction_seq"].astype(str)
    g=g[g["date"].dt.dayofweek<5].copy()
    g["axis"]=g.apply(lambda r:(r["date"] if r.year==2026 else r["date"]-pd.Timedelta(days=1)).replace(year=2026),axis=1)
    return g
rawd=unit_daily(raw,"daily_total")      # raw per unit (no imputation)
clnd=unit_daily(imp,"rec_total")        # cleaned per unit (hour-imputed)

# ---- balanced panel (the 50 stations / 92 units) ----
m=clnd.copy(); m["x_align"]=m["axis"].apply(lambda a: a.replace(year=2000))
piv=m.pivot_table(index=["uid","x_align"],columns="year",values="rec_total").dropna()
na=m["x_align"].nunique(); cov=piv.reset_index().groupby("uid")["x_align"].nunique()/na
units=sorted(cov[cov>=COVERAGE].index)
meta=raw[["station_id","cardinal_direction_seq","suburb","rms_region"]].drop_duplicates()
meta["uid"]=meta["station_id"]+"_"+meta["cardinal_direction_seq"].astype(str)
meta=meta.set_index("uid")
stations=sorted({u.rsplit("_",1)[0] for u in units}, key=lambda s:(0,int(s)) if s.isdigit() else (1,s))
print(f"{len(units)} units across {len(stations)} stations")

# ---- contact sheet: 25 per page (5x5), 2 pages ----
def plot_page(stlist,fname,pageno,npages):
    n=len(stlist); ncol=5; nrow=int(np.ceil(n/ncol))
    fig,axes=plt.subplots(nrow,ncol,figsize=(ncol*3.0,nrow*2.1),sharex=True)
    axes=np.atleast_2d(axes)
    for i,st in enumerate(stlist):
        ax=axes[i//ncol][i%ncol]
        sub_units=[u for u in units if u.rsplit("_",1)[0]==st]
        suburb=str(meta.loc[sub_units[0],"suburb"]) if sub_units[0] in meta.index else ""
        reg=str(meta.loc[sub_units[0],"rms_region"]) if sub_units[0] in meta.index else ""
        for u in sub_units:
            d=int(u.rsplit("_",1)[1]); dl=DIRMAP.get(d,str(d))
            for yr,c in [(2025,C25),(2026,C26)]:
                r=rawd[(rawd.uid==u)&(rawd.year==yr)].sort_values("axis")
                cl=clnd[(clnd.uid==u)&(clnd.year==yr)].sort_values("axis")
                ax.plot(r.axis,r.daily_total/1e3,color=c,lw=0.7,alpha=0.40)            # raw solid faint
                ax.plot(cl.axis,cl.rec_total/1e3,color=c,lw=1.1,ls=(0,(4,2)))          # cleaned dashed
        ax.axvline(SHOCK,color="crimson",ls=":",lw=0.8)
        ax.set_title(f"{st} · {suburb[:14]} · {reg[:8]}",fontsize=6.5)
        ax.tick_params(labelsize=6); ax.grid(alpha=.2,lw=.4)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ndir=len(sub_units)
        ax.text(0.02,0.04,f"{ndir} dir",transform=ax.transAxes,fontsize=5.5,color="0.4")
    for j in range(n,nrow*ncol): axes[j//ncol][j%ncol].axis("off")
    leg=[Line2D([0],[0],color=C25,lw=2,label="2025"),Line2D([0],[0],color=C26,lw=2,label="2026"),
         Line2D([0],[0],color="0.3",lw=1,alpha=.5,label="raw (solid)"),
         Line2D([0],[0],color="0.3",lw=1.3,ls=(0,(4,2)),label="cleaned (dashed)")]
    fig.legend(handles=leg,loc="upper center",ncol=4,fontsize=8,frameon=False,bbox_to_anchor=(0.5,1.0))
    fig.suptitle(f"Per-station daily traffic (1000 veh/day per direction), weekday-aligned to 2026 — raw vs cleaned · page {pageno}/{npages}",
                 fontsize=9,y=1.015)
    fig.supylabel("1000 vehicles / day",fontsize=8)
    plt.tight_layout(rect=[0.01,0,1,0.98]); plt.savefig(fname,dpi=170,bbox_inches="tight"); plt.close()
    print("saved",fname)

half=int(np.ceil(len(stations)/2))
plot_page(stations[:half],"stations_grid_p1.png",1,2)
plot_page(stations[half:],"stations_grid_p2.png",2,2)
