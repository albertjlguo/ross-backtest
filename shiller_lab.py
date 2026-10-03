import pandas as pd, numpy as np
pd.set_option('display.width',220); pd.set_option('display.max_columns',30); pd.set_option('display.max_rows',200)
d=pd.read_csv('results_shiller/shiller_sp500_monthly.csv',parse_dates=['Date']).set_index('Date')
d.columns=['P','D','E','CPI','R','rp','rd','re','CAPE']
full=d[d.E>0].copy()                      # 1871-01 .. 2023-06，所有字段都有真实数据
full['tr']=(full.P+full.D/12)/full.P.shift(1)-1
full['cash']=full.R.shift(1)/100/12        # 现金按长期国债收益率计（代理，偏向"等"这一边）
full['S']=(1+full.tr.fillna(0)).cumprod(); full['C']=(1+full.cash.fillna(0)).cumprod()
full['ey']=full.E.shift(3)/full.P*100      # 盈利滞后 3 个月才公布
full['gap']=full.ey-full.R
full.loc[full.CAPE<=0,'CAPE']=np.nan
f=full
print('样本',f.index[0].date(),'→',f.index[-1].date(),len(f),'个月')
for h in (12,60,120):
    f[f'fs{h}']=f.S.shift(-h)/f.S-1; f[f'fc{h}']=f.C.shift(-h)/f.C-1
    f[f'fr{h}']=(f.S.shift(-h)/f.S)/(f.CPI.shift(-h)/f.CPI)-1
ann=lambda x,h:(1+x)**(12/h)-1
print('\n== 收益率差（股票盈利收益率 − 长期国债）为负的月份占比，按年代 ==')
print((f.gap<0).groupby(f.index.year//10*10).mean().round(2).to_string())
bins=[-99,-2,-1,0,2,4,99]; lab=['<-2','-2~-1','-1~0','0~2','2~4','>4']
f['b']=pd.cut(f.gap,bins,labels=lab)
def table(sub,name):
    rows=[]
    for b,g in sub.groupby('b',observed=True):
        r=dict(bucket=b,months=len(g))
        for h in (12,60,120):
            x=g.dropna(subset=[f'fs{h}'])
            r[f's{h//12}y']=round(ann(x[f'fs{h}'],h).median()*100,1); r[f'cash{h//12}y']=round(ann(x[f'fc{h}'],h).median()*100,1)
            r[f'real{h//12}y']=round(ann(x[f'fr{h}'],h).median()*100,1)
            r[f'lose_cash{h//12}y']=round((x[f'fs{h}']<x[f'fc{h}']).mean()*100)
            r[f'loss{h//12}y']=round((x[f'fs{h}']<0).mean()*100); r[f'worst{h//12}y']=round(x[f'fs{h}'].min()*100)
        rows.append(r)
    print(f'\n== {name}：按买入时的收益率差分组（年化中位数 %，lose_cash=跑输现金的比例 %，loss=名义亏损比例 %，worst=最差总回报 %）=='); print(pd.DataFrame(rows).to_string(index=False))
table(f,'1871–2023 全样本'); table(f[f.index>='1960'],'1960 年以后')
print('\n今天：盈利收益率 3.2% − 国债 5.24% = −2.0 → 落在 <-2 / -2~-1 的交界')
# CAPE
f['cb']=pd.cut(f.CAPE,[0,15,20,25,30,99],labels=['<15','15-20','20-25','25-30','>30'])
rows=[]
for b,g in f.groupby('cb',observed=True):
    x=g.dropna(subset=['fr120']); y=g.dropna(subset=['fr60'])
    rows.append(dict(CAPE=b,months=len(g),real10y_med=round(ann(x.fr120,120).median()*100,1),real10y_min=round(ann(x.fr120,120).min()*100,1) if len(x) else None,real5y_med=round(ann(y.fr60,60).median()*100,1),loss5y=round((y.fs60<0).mean()*100),lose_cash10y=round((x.fs120<x.fc120).mean()*100) if len(x) else None))
print('\n== 按席勒市盈率（CAPE）分组：之后的实际（扣通胀）年化回报 =='); print(pd.DataFrame(rows).to_string(index=False)); print('最新 CAPE（2023-06）',f.CAPE.iloc[-1])
# 等 vs 现在买：从每个 gap<0 的月份出发
def wait_test(sub,h=120):
    out=[]
    idx=f.index; S=f.S.to_numpy(); C=f.C.to_numpy(); gap=f.gap.to_numpy(); P=f.P.to_numpy()
    for t in np.where(sub)[0]:
        if t+h>=len(f): continue
        now=S[t+h]/S[t]
        j=next((k for k in range(t+1,t+h+1) if gap[k]>=0),None)
        wait=(C[j]/C[t])*(S[t+h]/S[j]) if j else C[t+h]/C[t]
        # 分三批：先买 1/3，跌 20% 再买 1/3，跌 40% 再买 1/3，其余放现金
        w=S[t+h]/S[t]/3; cash=2/3; 
        for lvl in (0.8,0.6):
            k=next((k for k in range(t+1,t+h+1) if P[k]<=P[t]*lvl),None)
            if k: w+=(C[k]/C[t])/3*(S[t+h]/S[k]); cash-=1/3
        # 未动用的现金
        used=[next((k for k in range(t+1,t+h+1) if P[k]<=P[t]*lvl),None) for lvl in (0.8,0.6)]
        w+=sum((C[t+h]/C[t])/3 for u in used if u is None)
        out.append(dict(t=idx[t],now=now,wait=wait,stag=w,hit=j is not None,lag=(j-t) if j else np.nan,px_chg=(P[j]/P[t]-1) if j else np.nan,n_adds=sum(u is not None for u in used)))
    return pd.DataFrame(out)
for name,sub in (('收益率差<0',(f.gap<0).to_numpy()),('收益率差<-1',(f.gap<-1).to_numpy()),('CAPE>25',(f.CAPE>25).to_numpy())):
    w=wait_test(sub)
    if w.empty: continue
    a=lambda x:round(((x.median())**(1/10)-1)*100,1)
    print(f'\n== {name} 时出发，10 年后：{len(w)} 个起点（{w.t.dt.year.min()}–{w.t.dt.year.max()}）==')
    print(' 现在就买 年化中位', a(w.now),'| 等到收益率差转正再买', a(w.wait),'| 分三批', a(w.stag))
    print(' 等更好的比例', round((w.wait>w.now).mean()*100),'% | 分批更好的比例', round((w.stag>w.now).mean()*100),'% | 10 年内等到的比例', round(w.hit.mean()*100),'% | 等了多久(中位月)', w.lag.median(),'| 等到时价格比出发时', round(w.px_chg.median()*100),'%')
    print(' 最差结果：现在买',round((w.now.min()-1)*100),'%  等',round((w.wait.min()-1)*100),'%  分批',round((w.stag.min()-1)*100),'%')
    print(' 按起点年代：等更好的比例'); print((w.assign(dec=w.t.dt.year//10*10).groupby('dec').apply(lambda g: pd.Series({'n':len(g),'wait_better':round((g.wait>g.now).mean(),2),'stag_better':round((g.stag>g.now).mean(),2),'now_ann':round((g.now.median()**.1-1)*100,1),'wait_ann':round((g.wait.median()**.1-1)*100,1)}))).to_string())
# 高点买入
all_=d.copy(); lastD=full.D.iloc[-1]/full.P.iloc[-1]
all_['D2']=np.where(all_.D>0,all_.D,all_.P*lastD)            # 2023-07 之后的股息按最后一个股息率估
all_['S']=(1+((all_.P+all_.D2/12)/all_.P.shift(1)-1).fillna(0)).cumprod()
cpi=all_.CPI.where(all_.CPI>0)
print('\n== 在历史高点一次买入：多久回本（含股息），之后的回报 ==')
for pk in ('1929-09','1966-01','1972-12','2000-08','2007-10'):
    t=all_.index.get_loc(pd.Timestamp(pk+'-01')); S=all_.S.to_numpy(); P=all_.P.to_numpy()
    trough=S[t:t+240].min()/S[t]-1
    nom=next((k-t for k in range(t+1,len(S)) if S[k]>=S[t]),None)
    real_s=(all_.S/cpi).to_numpy(); realrec=next((k-t for k in range(t+1,len(S)) if real_s[k]>=real_s[t] and not np.isnan(real_s[k])),None)
    pxrec=next((k-t for k in range(t+1,len(S)) if P[k]>=P[t]),None)
    r=lambda h:(round(((S[t+h]/S[t])**(12/h)-1)*100,1) if t+h<len(S) else None)
    print(pk,'CAPE',d.CAPE.iloc[t],'| 最大跌幅',round(trough*100),'% | 回本：价格',round(pxrec/12,1) if pxrec else None,'年，含股息',round(nom/12,1) if nom else None,'年，扣通胀',round(realrec/12,1) if realrec else '未回','年 | 5/10/20 年名义年化',r(60),r(120),r(240))
