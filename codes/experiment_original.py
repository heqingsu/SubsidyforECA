"""
create by hqs   2026-10
 subsidy experiments. Run: python experiment.py
Grid solutions, not certified continuous global optima. Monetary unit: CNY/year.
"""
from pathlib import Path
from dataclasses import dataclass, replace, asdict
import json, time, platform, hashlib
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'; FIG=OUT/'figures'; TAB=OUT/'tables'
for d in [OUT,FIG,TAB]: d.mkdir(exist_ok=True,parents=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
@dataclass
class Settings:
    mu:float=.6
    lam:float=.5
    omega:float=0.
    budget:float=1e6
    pref:float=1e6
    href:float=1e7
    xref:float=1e6
    sigma:float=10000.
    eta:float=.8
    information:float=0.
    cost_scale:float=1.
    variable:float=0.
    pmax:float=60000.
    smax:float=10000.
    tau:float=1e-9

E0=pd.read_csv(ROOT/'data/synthetic_enterprises.csv')
F=pd.read_csv(ROOT/'data/synthetic_hourly_factors.csv')
phi=F.pivot(index='hour',columns='region',values='factor').to_numpy()
weights=F.pivot(index='hour',columns='region',values='reference_weight').to_numpy()
configs=[(b,G,f't{b}h_g{G}') for b in [24,6,1] for G in [1,4,8]]
checks=[]
def check(name, condition, detail=''):
    checks.append({'check':name,'passed':bool(condition),'detail':str(detail)})
def write(df,name): df.to_csv(TAB/(name+'.csv'),index=False,encoding='utf-8-sig')
def save(fig,name):
    fig.tight_layout(); fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight'); fig.savefig(FIG/(name+'.png'),dpi=200,bbox_inches='tight'); plt.close(fig)
def prepare(df,c):
    h=np.arange(24); profiles=np.array([.15+((h>=8)&(h<17)),.15+((h>=6)&(h<22)),np.ones(24)])
    load=profiles[df['shift'].to_numpy(int)]; load=load/load.sum(1,keepdims=True)
    E=df.electricity_MWh.to_numpy(); loc=df.region.to_numpy(int); q=E[:,None]*load
    baseline=E*np.sum(phi*weights)/weights.sum()
    fine=np.sum(q*phi[:,loc].T,axis=1)
    result={}; rows=[]
    for b,G,name in configs:
        coarse=np.empty_like(phi); group=8//G
        for t in range(0,24,b):
            for g in range(0,8,group):
                sl=np.s_[t:t+b,g:g+group]; coarse[sl]=np.sum(phi[sl]*weights[sl])/weights[sl].sum()
        C=(q*coarse[:,loc].T).sum(1); D=baseline-C
        B=c.information+c.eta*df.eligible_fraction.to_numpy()*df.unit_value.to_numpy()*D
        net=B-df.extra_cost.to_numpy()
        fixed=c.cost_scale*(50000+12000*(24/b)**.65+15000*G**.8)
        result[name]={'b':net,'B':B,'D':D,'C':C,'F':fixed,'fine':fine,'q':q}
        err=np.abs(C-fine)/np.maximum(fine,1e-12)
        rows.append(dict(configuration=name,hours=b,zones=G,fixed_cost=fixed,mean_signed_tonnes=D.mean(),lower_footprint_share=np.mean(D>1e-8),weighted_deviation=np.dot(E,err)/E.sum()))
    return result,pd.DataFrame(rows)

def solve(data,c,dp=250.,ds=250.,allowed=None,screen=False,cache=True,prices_fixed=None,svalues=None):
    """Enumerated grid optimum. Cache is exact at identical effective-grid prices.
    All within-tau prices are retained, not merely one optimum/configuration.
    Budget filtering occurs AFTER provider-response identification.
    """
    start=time.perf_counter(); pgrid=np.arange(0,c.pmax+dp/2,dp); pgrid=np.unique(np.r_[pgrid[pgrid<=c.pmax],c.pmax])
    sgrid=np.arange(0,c.smax+ds/2,ds); sgrid=np.unique(np.r_[sgrid[sgrid<=c.smax],c.smax])
    if svalues is not None: sgrid=np.unique(np.asarray(svalues,float))
    names=list(data) if allowed is None else allowed
    cache_arrays={}; neval=0; screened=0
    # Cache all exact effective prices needed by this finite grid.
    if cache:
        zgrid=np.unique((pgrid[None,:]-sgrid[:,None]).ravel())
        for name in names:
            d=data[name]; prob=expit((d['b'][None,:]-zgrid[:,None])/c.sigma)
            cache_arrays[name]=(zgrid,prob.sum(1),prob@d['B'])
            neval+=len(zgrid)
    records=[]
    for s in sgrid:
        candidate=[dict(configuration='no_service',subsidy=float(s),price=0.,effective_price=0.,participation=0.,profit=0.,spending=0.,benefit=0.,withdrawal=0.,withdrawal_profit=0.,provider=0.,government=0.)]
        incumbent=0.
        for name in names:
            d=data[name]; n=len(d['b']); p=pgrid if prices_fixed is None else np.array([prices_fixed[name]])
            upperprob=expit((d['b']+s)/c.sigma)
            up_profit=max(c.pmax-c.variable,0)*upperprob.sum()-d['F']
            up_obj=c.mu*upperprob.mean()+(1-c.mu)*up_profit/c.pref
            if screen and (up_profit<0 or up_obj<incumbent-c.tau): screened+=1; continue
            if cache and prices_fixed is None:
                zg,Aall,Hall=cache_arrays[name]; idx=np.searchsorted(zg,p-s); A=Aall[idx]; H=Hall[idx]
            else:
                prob=expit((d['b'][None,:]-p[:,None]+s)/c.sigma); A=prob.sum(1); H=prob@d['B']; neval+=len(p)
            profit=(p-c.variable)*A-d['F']; obj=c.mu*A/n+(1-c.mu)*profit/c.pref
            viable=profit>=0
            if not viable.any(): continue
            vmax=obj[viable].max(); incumbent=max(incumbent,float(vmax))
            for j in np.where(viable & (obj>=vmax-c.tau))[0]:
                off=expit((d['b']-p[j])/c.sigma).mean(); spending=s*A[j]
                gov=c.lam*H[j]/c.href+(1-c.lam)*off-c.omega*spending/c.xref
                candidate.append(dict(configuration=name,subsidy=float(s),price=float(p[j]),effective_price=float(p[j]-s),participation=float(A[j]/n),profit=float(profit[j]),spending=float(spending),benefit=float(H[j]),withdrawal=float(off),withdrawal_profit=float((p[j]-c.variable)*off*n-d['F']),provider=float(obj[j]),government=float(gov)))
        bestprovider=max(x['provider'] for x in candidate)
        br=[x for x in candidate if x['provider']>=bestprovider-c.tau]
        affordable=[x for x in br if x['spending']<=c.budget+1e-7]
        worst=min(br,key=lambda x:x['government'])
        if affordable:
            out=max(affordable,key=lambda x:(x['government'],-x['price']))
        else: out=max(br,key=lambda x:x['provider'])
        out=out.copy(); out['budget_feasible']=bool(affordable); out['response_count']=len(br)
        out['pessimistic_feasible']=all(x['spending']<=c.budget+1e-7 for x in br)
        out['pessimistic_government']=worst['government']; records.append(out)
    frame=pd.DataFrame(records); feasible=frame[frame.budget_feasible]
    best=feasible.sort_values(['government','subsidy'],ascending=[False,True]).iloc[0].to_dict()
    return frame,best,dict(seconds=time.perf_counter()-start,demand_evaluations=neval,screened=screened,dp=dp,ds=ds)

def main():
    c=Settings(); data,agg=prepare(E0,c); write(agg,'aggregation')
    check('energy conservation',np.allclose(data['t1h_g8']['q'].sum(1),E0.electricity_MWh))
    check('fine benchmark identity',np.allclose(data['t1h_g8']['C'],data['t1h_g8']['fine']))
    check('coarse baseline identity',np.max(np.abs(data['t24h_g1']['D']))<1e-8)
    # Analytic derivative checked independently by centered finite differences.
    d=data['t1h_g8']; z=12000.; s=2000.; dz=.01
    def objective(z):
        P=expit((d['b']-z)/c.sigma); return c.mu*P.mean()+(1-c.mu)*((z+s-c.variable)*P.sum()-d['F'])/c.pref
    P=expit((d['b']-z)/c.sigma); W=np.sum(P*(1-P))/c.sigma
    derivative=-c.mu*W/len(P)+(1-c.mu)*(P.sum()-(z+s-c.variable)*W)/c.pref
    check('analytic price derivative',np.isclose(derivative,(objective(z+dz)-objective(z-dz))/(2*dz),rtol=1e-6,atol=1e-10))
    check('effective price invariance',np.allclose(expit((d['b']-20000+3000)/c.sigma),expit((d['b']-17000)/c.sigma)))
    # Synthetic tie diagnostic: unaffordable best offer must not be replaced by inferior offer.
    toy=[(1.,2e6),(.8,0.)]; br=[x for x in toy if x[0]>=max(y[0] for y in toy)-c.tau]
    check('budget follows best response',not any(x[1]<=c.budget for x in br))
    base,best,perf=solve(data,c); write(base,'subsidy_response'); print('BASE',best,flush=True)
    refinements=[]
    for step in [500,250,125,62.5]:
        fr,be,pe=solve(data,c,step,step); refinements.append(dict(step=step,**be,**pe))
    ref=pd.DataFrame(refinements); write(ref,'grid_refinement')
    for row in refinements: check('selected feasibility step '+str(row['step']),row['profit']>=-1e-7 and row['spending']<=c.budget+1e-6)
    variants=[]
    for label,cache,screen in [('uniform',False,False),('cached',True,False),('cached_screened',True,True)]:
        fr,be,pe=solve(data,c,cache=cache,screen=screen); variants.append(dict(variant=label,**pe,government=be['government'],configuration=be['configuration']))
    # Uniform scaffold plus targeted subsidy refinement; price grid stays at 250.
    tstart=time.perf_counter(); coarse,_,pa=solve(data,c,ds=1000)
    points=set(coarse.subsidy)
    good=coarse[coarse.budget_feasible].nlargest(3,'government').subsidy.tolist()
    for j in range(1,len(coarse)):
        prev,cur=coarse.iloc[j-1],coarse.iloc[j]
        if prev.configuration!=cur.configuration or prev.budget_feasible!=cur.budget_feasible:
            good.extend([prev.subsidy,cur.subsidy])
    for center in good:
        points.update(np.arange(max(0,center-1000),min(c.smax,center+1000)+1,250))
    _,ba,pb=solve(data,c,svalues=sorted(points),screen=True)
    variants.append(dict(variant='targeted_subsidy',seconds=time.perf_counter()-tstart,demand_evaluations=pa['demand_evaluations']+pb['demand_evaluations'],screened=pb['screened'],dp=250,ds='mixed',government=ba['government'],configuration=ba['configuration']))
    comp=pd.DataFrame(variants); write(comp,'computation')
    check('screening and caching preserve grid objective',np.ptp(comp.iloc[:3].government)<1e-10)
    bench=[]
    for label,allowed,sm in [('no_subsidy',None,0),('fixed_coarse',['t24h_g1'],10000),('fixed_fine',['t1h_g8'],10000),('endogenous',None,10000)]:
        fr,be,pe=solve(data,replace(c,smax=sm),allowed=allowed); bench.append(dict(scenario=label,**be))
    write(pd.DataFrame(bench),'policy_benchmarks')
    # Conditional, fixed-price diagnostic uses the optimized zero-subsidy price for each configuration.
    fp={}
    for name in data:
        _,be,_=solve(data,replace(c,smax=0),allowed=[name]); fp[name]=be['price']
    fixed,_,_=solve(data,c,cache=False,prices_fixed=fp); write(fixed,'fixed_price_diagnostic')
    sensitivities=[]
    specs=[('cost_scale',[.5,1,1.5,2]),('eta',[.4,.6,.8,1]),('information',[0,2000,5000]),('variable',[0,500,1500]),('sigma',[5000,10000,20000]),('mu',[0,.3,.6,.9,1]),('lam',[0,.5,1]),('omega',[0,.1,.5]),('budget',[250000,500000,1000000,2000000])]
    for par,vals in specs:
        for val in vals:
            cs=replace(c,**{par:val}); dd,_=prepare(E0,cs); _,be,_=solve(dd,cs,500,500)
            sensitivities.append(dict(parameter=par,value=val,**be))
        print('sensitivity',par,flush=True)
    sens=pd.DataFrame(sensitivities); write(sens,'sensitivity')
    # Paired bootstrap = uncertainty conditional on supplied sample, not fresh generator replications.
    boots=[]
    for seed in range(20261016,20261026):
        rng=np.random.default_rng(seed); sample=E0.iloc[rng.integers(0,len(E0),len(E0))].reset_index(drop=True)
        dd,_=prepare(sample,c)
        for label,sm in [('no_subsidy',0),('endogenous',10000)]:
            _,be,_=solve(dd,replace(c,smax=sm),500,500); boots.append(dict(seed=seed,scenario=label,**be))
    boot=pd.DataFrame(boots); write(boot,'paired_bootstrap')
    check('probability bounds',base.participation.between(0,1).all())
    check('subsidy expenditure units',np.allclose(base.spending,base.subsidy*base.participation*len(E0)))
    check('signed differences retained',np.any(data['t1h_g8']['D']<0) and np.any(data['t1h_g8']['D']>0))
    check('zero subsidy cost identity',base.iloc[0].spending==0)
    # Figures matching the manuscript's seven exact filenames.
    fig,ax=plt.subplots(1,3,figsize=(12,3.4)); ax[0].plot(np.arange(24),phi); ax[0].set(xlabel='Hour of day',ylabel='tCO2e/MWh',title='(a) Synthetic factor profiles')
    ax[1].hist(E0.electricity_MWh/1000,bins=22,color='#3978a8'); ax[1].set(xlabel='Annual electricity (GWh)',ylabel='Enterprises',title='(b) Consumption')
    pd.crosstab(E0.region,E0['shift']).plot.bar(stacked=True,ax=ax[2]); ax[2].set(title='(c) Location and shift',xlabel='Synthetic zone',ylabel='Enterprises'); save(fig,'exp01_inputs')
    fig,ax=plt.subplots(1,3,figsize=(13,3.6)); ax[0].boxplot([data[n]['D'] for n in data],tick_labels=list(data),showfliers=False); ax[0].axhline(0,color='k',lw=.7); ax[0].tick_params(axis='x',rotation=70); ax[0].set(ylabel='Baseline minus refined (tCO2e)',title='(a) Signed differences')
    for a,col,title in [(ax[1],'weighted_deviation','(b) Benchmark deviation'),(ax[2],'fixed_cost','(c) Fixed cost (CNY/year)')]: a.bar(agg.configuration,agg[col]); a.tick_params(axis='x',rotation=70); a.set_title(title)
    save(fig,'exp02_aggregation')
    fig,ax=plt.subplots(1,3,figsize=(12,3.5)); price_rows=[]
    zz=np.linspace(-10000,60000,500)
    for ss in [0,5000,10000]:
        probs=expit((d['b'][None,:]-zz[:,None])/c.sigma); A=probs.sum(1); profit=(zz+ss)*A-d['F']; obj=c.mu*A/len(E0)+(1-c.mu)*profit/c.pref
        mask=(zz>=-ss)&(zz<=c.pmax-ss)
        for a,y in zip(ax,[obj,profit,A/len(E0)]): a.plot(zz[mask],y[mask],label=f's={ss:g}')
        price_rows.extend(dict(subsidy=ss,effective_price=float(zv),profit=float(pv),provider=float(ov),participation=float(av/len(E0))) for zv,pv,ov,av in zip(zz[mask],profit[mask],obj[mask],A[mask]))
    for a,title in zip(ax,['Provider objective','Profit (CNY/year)','Participation']): a.set(xlabel='Effective price (CNY)',title=title); a.legend()
    ax[1].axhline(0,color='k',lw=.8); save(fig,'exp03_pricing'); write(pd.DataFrame(price_rows),'price_response')
    fig,ax=plt.subplots(2,2,figsize=(10,7)); good=base.budget_feasible
    ax[0,0].plot(base.subsidy,base.price,label='Price'); ax[0,0].plot(base.subsidy,base.effective_price,label='Effective payment'); ax[0,0].legend(); ax[0,0].set_ylabel('CNY/enterprise')
    ax[0,1].plot(base.subsidy,base.participation,label='Endogenous price'); ax[0,1].plot(fixed.subsidy,fixed.participation,'--',label='Fixed-price diagnostic'); ax[0,1].legend(); ax[0,1].set_ylabel('Participation')
    ax[1,0].plot(base.subsidy,base.spending); ax[1,0].axhline(c.budget,color='r',ls='--',label='Budget'); ax[1,0].legend(); ax[1,0].set_ylabel('Expected expenditure (CNY)')
    ax[1,1].plot(base.subsidy,np.where(good,base.government,np.nan),label='Feasible government score'); ax[1,1].scatter(base.subsidy[~good],base.government[~good],marker='x',color='r',label='Infeasible'); ax[1,1].plot(base.subsidy,base.withdrawal,'--',label='Withdrawal participation'); ax[1,1].legend(fontsize=7)
    for a in ax.flat: a.set_xlabel('Subsidy (CNY/adopter)')
    save(fig,'exp04_subsidy')
    no=bench[0]; outcomes=[('Subsidized',best['participation'],best['profit'],best['price']),('Withdrawal',best['withdrawal'],best['withdrawal_profit'],best['price']),('Re-optimized s=0',no['participation'],no['profit'],no['price'])]
    wd=pd.DataFrame(outcomes,columns=['scenario','participation','profit','price']); write(wd,'withdrawal')
    fig,ax=plt.subplots(1,3,figsize=(12,3.6))
    for a,col in zip(ax,['participation','profit','price']): a.bar(wd.scenario,wd[col]); a.set_title(col); a.tick_params(axis='x',rotation=20)
    ax[1].axhline(0,color='k'); save(fig,'exp05_withdrawal')
    fig,ax=plt.subplots(3,3,figsize=(12,9))
    for a,(par,_) in zip(ax.flat,specs):
        sub=sens[sens.parameter==par]; a.plot(sub.value,sub.participation,'o-',label='Participation'); a.plot(sub.value,sub.withdrawal,'s--',label='Withdrawal'); a.set(xlabel=par,ylim=(0,1)); a.legend(fontsize=6)
    save(fig,'exp06_sensitivity')
    fig,ax=plt.subplots(1,3,figsize=(12,3.6)); ax[0].bar(comp.variant,comp.seconds); ax[0].set_ylabel('Seconds (single run)'); ax[1].bar(comp.variant,comp.demand_evaluations); ax[1].set_ylabel('Demand evaluations');
    for a in ax[:2]: a.tick_params(axis='x',rotation=25)
    ax[2].plot(ref.step,ref.government,'o-'); ax[2].invert_xaxis(); ax[2].set(xlabel='Grid spacing (CNY)',ylabel='Government score'); save(fig,'exp07_computation')
    # Supplement: categorical service switches and sample robustness.
    fig,ax=plt.subplots(1,2,figsize=(11,3.5)); codes={n:i for i,n in enumerate(list(data)+['no_service'])}; ax[0].step(base.subsidy,base.configuration.map(codes),where='mid'); ax[0].set(yticks=list(codes.values()),yticklabels=list(codes),xlabel='Subsidy (CNY/adopter)',title='Selected configuration')
    bc=boot[boot.scenario=='endogenous'].configuration.value_counts(); ax[1].bar(bc.index,bc.values/10); ax[1].set(ylabel='Selection frequency',title='Paired bootstrap (10 resamples)'); save(fig,'exp08_switches_bootstrap')
    # LaTeX tables and standalone figure/table inclusion fragment.
    for filename in ['aggregation','policy_benchmarks','grid_refinement','computation','withdrawal']:
        df=pd.read_csv(TAB/(filename+'.csv'))
        def esc(v): return str(v).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
        lines=[r'\begin{tabular}{'+'l'*len(df.columns)+'}',r'\toprule',' & '.join(esc(x) for x in df.columns)+r' \\',r'\midrule']
        for row in df.itertuples(index=False,name=None):
            lines.append(' & '.join(esc(('%.5g'%v) if isinstance(v,float) else v) for v in row)+r' \\')
        lines.extend([r'\bottomrule',r'\end{tabular}'])
        (TAB/(filename+'.tex')).write_text('\n'.join(lines))
    tex=['% Requires graphicx and booktabs. Copy outputs/figures into manuscript figures/.']
    for num,name in enumerate(['inputs','aggregation','pricing','subsidy','withdrawal','sensitivity','computation','switches_bootstrap'],1):
        tex.append('\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/exp%02d_%s.pdf}\n\\caption{%s. Synthetic diagnostic results.}\n\\label{fig:exp%02d}\n\\end{figure}'%(num,name,name.replace('_',' ').capitalize(),num))
    (OUT/'figures_include.tex').write_text('\n\n'.join(tex))
    write(pd.DataFrame(checks),'validation')
    summary={'settings':asdict(c),'base_grid_solution':best,'finest_grid_solution':refinements[-1], 'all_checks_passed':all(x['passed'] for x in checks),'validation_count':len(checks),'bootstrap':'paired nonparametric bootstrap; not independent generator replications','solver_scope':'finite grids; no certified continuous global optimum','versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__},'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').glob('*.csv')}}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=lambda v:v.item() if hasattr(v,'item') else str(v)))
    assert all(x['passed'] for x in checks),checks
    print('ALL CHECKS PASSED',len(checks),flush=True)
if __name__=='__main__': main()
