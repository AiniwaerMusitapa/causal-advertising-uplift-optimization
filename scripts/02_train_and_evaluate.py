"""Train four CATE estimators and response baseline; evaluate uplift without exposure."""
import hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
import duckdb,joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from econml.dml import CausalForestDML
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; FEATURES=[f'f{i}' for i in range(12)]; SAMPLE='hash(row_id,2026)%7=0'; SEED=42
def dump(path,obj): Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def classifier(): return lgb.LGBMClassifier(n_estimators=100,num_leaves=31,learning_rate=.05,min_child_samples=200,reg_lambda=5,n_jobs=2,random_state=SEED,verbosity=-1)
def regressor(): return lgb.LGBMRegressor(n_estimators=100,num_leaves=31,learning_rate=.05,min_child_samples=200,reg_lambda=5,n_jobs=2,random_state=SEED,verbosity=-1)
def split(df): return df[FEATURES].to_numpy(np.float32),df.treatment.to_numpy(np.uint8),df.visit.to_numpy(np.uint8)

def curve(y,t,score,grid=np.linspace(.01,1,100)):
    order=np.argsort(-score,kind='stable'); y=y[order]; t=t[order]; n=len(y); rows=[]; propensity=float(t.mean())
    transformed=y*(t/propensity-(1-t)/(1-propensity))
    for q in grid:
        k=max(1,int(math.ceil(q*n))); yy=y[:k]; tt=t[:k]; nt=int(tt.sum()); nc=k-nt
        uplift=float(transformed[:k].mean())
        rows.append({'fraction':float(q),'rows':k,'treated':nt,'control':nc,'uplift':uplift,'incremental_per_population':float(q*uplift)})
    x=np.r_[0,[r['fraction'] for r in rows]]; gain=np.r_[0,[r['incremental_per_population'] for r in rows]]
    auuc=float(np.trapz(gain,x)); overall=rows[-1]['uplift']; qini=float(np.trapz(gain-x*overall,x))
    at={str(int(q*100)):next(r for r in rows if abs(r['fraction']-q)<1e-9) for q in [.1,.2,.3]}
    return {'definition':'Sort descending score. With test propensity e, transformed outcome Z=Y[T/e-(1-T)/(1-e)]. uplift(q)=mean(Z|top q); gain(q)=q*uplift(q); AUUC=integral gain dq; Qini=integral [gain-q*mean(Z)] dq.',
        'auuc':auuc,'qini_coefficient':qini,'uplift_at':at,'curve':rows}

def deciles(y,t,score,boot=300):
    order=np.argsort(-score,kind='stable'); chunks=np.array_split(order,10); rng=np.random.default_rng(SEED); out=[]; propensity=float(t.mean())
    for i,idx in enumerate(chunks,1):
        yy=y[idx]; tt=t[idx]; a=yy[tt==1]; b=yy[tt==0]; z=yy*(tt/propensity-(1-tt)/(1-propensity)); uplift=float(z.mean())
        values,counts=np.unique(z,return_counts=True); draws=rng.multinomial(len(z),counts/len(z),size=boot)
        bs=(draws@values)/len(z)
        out.append({'decile':i,'rows':len(idx),'treated':len(a),'control':len(b),'treatment_rate':float(a.mean()) if len(a) else None,'control_rate':float(b.mean()) if len(b) else None,
                    'raw_difference_in_means':float(a.mean()-b.mean()) if len(a) and len(b) else None,'identifiable_by_raw_difference':bool(len(a) and len(b)),
                    'uplift':uplift,'ci_low':float(np.quantile(bs,.025)),'ci_high':float(np.quantile(bs,.975))})
    return out

def main():
    audit=json.loads((ROOT/'reports/experiment_validation.json').read_text(encoding='utf-8')); assert audit['status']=='complete'
    parquet=Path(audit['parquet']['path']); assert sha(parquet)==audit['parquet']['sha256']
    for d in ['models','figures','configs','notebooks','data/features','reports/powerbi']: (ROOT/d).mkdir(parents=True,exist_ok=True)
    con=duckdb.connect(); con.execute("SET memory_limit='2GB'"); con.execute('SET threads=2'); rel=f"read_parquet('{parquet.as_posix()}')"
    case="CASE WHEN hash(row_id,42)%100<70 THEN 'train' WHEN hash(row_id,42)%100<85 THEN 'validation' ELSE 'test' END"
    data={}
    for name in ['train','validation','test']:
        print('Load',name,flush=True)
        data[name]=con.execute(f"SELECT {','.join(FEATURES)},treatment,visit,conversion,row_id FROM {rel} WHERE {SAMPLE} AND {case}='{name}'").fetch_df()
    counts={k:len(v) for k,v in data.items()}; assert all(counts[k]>100000 for k in counts)
    xt,tt,yt=split(data['train']); xv,tv,yv=split(data['validation']); xz,tz,yz=split(data['test'])
    # Treatment assignment predictability: near-0.5 AUC supports balance; not proof by itself.
    assignment=classifier().fit(xt,tt); assignment_auc=float(roc_auc_score(tv,assignment.predict_proba(xv)[:,1]))
    models={}; predv={}; predz={}
    print('S-learner',flush=True)
    s=classifier().fit(np.c_[xt,tt],yt); models['s_learner']=s
    predv['s_learner']=s.predict_proba(np.c_[xv,np.ones(len(xv))])[:,1]-s.predict_proba(np.c_[xv,np.zeros(len(xv))])[:,1]
    predz['s_learner']=s.predict_proba(np.c_[xz,np.ones(len(xz))])[:,1]-s.predict_proba(np.c_[xz,np.zeros(len(xz))])[:,1]
    print('T-learner outcome models',flush=True)
    mu0=classifier().fit(xt[tt==0],yt[tt==0]); mu1=classifier().fit(xt[tt==1],yt[tt==1]); models.update(t_mu0=mu0,t_mu1=mu1)
    predv['t_learner']=mu1.predict_proba(xv)[:,1]-mu0.predict_proba(xv)[:,1]; predz['t_learner']=mu1.predict_proba(xz)[:,1]-mu0.predict_proba(xz)[:,1]
    print('X-learner effect models',flush=True)
    d0=mu1.predict_proba(xt[tt==0])[:,1]-yt[tt==0]; d1=yt[tt==1]-mu0.predict_proba(xt[tt==1])[:,1]
    tau0=regressor().fit(xt[tt==0],d0); tau1=regressor().fit(xt[tt==1],d1); models.update(x_tau0=tau0,x_tau1=tau1)
    propensity=float(tt.mean())
    predv['x_learner']=propensity*tau0.predict(xv)+(1-propensity)*tau1.predict(xv)
    predz['x_learner']=propensity*tau0.predict(xz)+(1-propensity)*tau1.predict(xz)
    print('Causal Forest DML on deterministic 250k train subset',flush=True)
    cf_mask=(data['train'].row_id.to_numpy(dtype=np.uint64)*np.uint64(11400714819323198485))%np.uint64(2**64-1) < np.uint64((2**64-1)*250000/len(data['train']))
    # If hash-like multiplication deviates, enforce deterministic first 250k after row-id mix order.
    cf_ids=np.flatnonzero(cf_mask); cf_ids=cf_ids[:250000] if len(cf_ids)>=250000 else np.argsort(data['train'].row_id.to_numpy() ^ 0x5DEECE66D)[:250000]
    scaler=StandardScaler().fit(xt[cf_ids]); xcf=scaler.transform(xt[cf_ids]);
    cf=CausalForestDML(model_y=lgb.LGBMRegressor(n_estimators=40,num_leaves=15,n_jobs=2,random_state=SEED,verbosity=-1),
        model_t=lgb.LGBMClassifier(n_estimators=40,num_leaves=15,n_jobs=2,random_state=SEED,verbosity=-1),
        discrete_treatment=True,n_estimators=100,max_depth=12,min_samples_leaf=100,max_samples=.45,cv=2,n_jobs=2,random_state=SEED,inference=False)
    cf.fit(yt[cf_ids],tt[cf_ids],X=xcf); models['causal_forest']=cf; models['causal_forest_scaler']=scaler
    predv['causal_forest']=cf.effect(scaler.transform(xv)); predz['causal_forest']=cf.effect(scaler.transform(xz))
    print('Response model',flush=True)
    response=classifier().fit(xt,yt); models['response_model']=response
    response_v=response.predict_proba(xv)[:,1]; response_z=response.predict_proba(xz)[:,1]
    # Validation selects one CATE policy before test summaries are computed.
    validation={name:curve(yv,tv,p) for name,p in predv.items()}; best=max(validation,key=lambda n:validation[n]['qini_coefficient'])
    frozen_at=datetime.now(timezone.utc).isoformat(); artifacts={}
    for name,model in models.items():
        path=ROOT/'models'/(name+'.joblib'); joblib.dump(model,path); artifacts[name]={'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
    # Test reporting includes all frozen predeclared methods, not test-based selection.
    test={name:curve(yz,tz,p) for name,p in predz.items()}; test['response_model']=curve(yz,tz,response_z)
    decile={name:deciles(yz,tz,p) for name,p in {**predz,'response_model':response_z}.items()}
    names=list(predz); agreement={}
    for i,a in enumerate(names):
        for b in names[i+1:]:
            k=max(1,int(.1*len(yz))); ia=set(np.argpartition(predz[a],-k)[-k:]); ib=set(np.argpartition(predz[b],-k)[-k:])
            agreement[a+'__'+b]={'spearman':float(spearmanr(predz[a],predz[b]).statistic),'top10_jaccard':len(ia&ib)/len(ia|ib)}
    bp=predz[best]; q90=np.quantile(bp,.9); segments={'high_positive':int((bp>=q90).sum()),'moderate_positive':int(((bp>.0005)&(bp<q90)).sum()),
        'near_zero_0_to_0.0005':int(((bp>=0)&(bp<=.0005)).sum()),'negative':int((bp<0).sum())}
    assert sum(segments.values())==len(bp)
    metrics={'status':'complete','completed_at_utc':datetime.now(timezone.utc).isoformat(),'frozen_at_utc':frozen_at,'outcome':'visit','estimand':'CATE/ITT by assigned treatment',
        'data_version':'corrected unbiased v2.1','full_rows':audit['rows'],'model_sample_rule':SAMPLE,'model_split_rule':audit['split_rule'],'model_rows':counts,
        'feature_list':FEATURES,'excluded':['exposure','conversion','visit','treatment as baseline covariate except S-learner intervention input'],'post_treatment_exposure_used':False,
        'treatment_ratio_train':propensity,'assignment_predictability_validation_auc':assignment_auc,
        'method_note':{'s_learner':'LightGBM outcome model with treatment toggled 0/1','t_learner':'Separate LightGBM outcome models by arm',
            'x_learner':'T-learner imputed effects; propensity-weighted effect regressors; weighting addresses unequal arm precision',
            'causal_forest':'EconML CausalForestDML; deterministic 250k training subset; LightGBM nuisance models; 100 trees',
            'response_model':'LightGBM P(visit|X), intentionally not a causal estimator'},
        'causal_forest_train_rows':len(cf_ids),'selection_metric':'validation Qini coefficient','selected_model':best,'validation':validation,'test':test,
        'test_deciles':decile,'model_agreement':agreement,'selected_policy_segments':segments,'model_artifacts':artifacts,
        'limitations':['Dataset is non-uniformly subsampled by publisher, so sample ATE is not the original campaign population lift.','No monetary values/costs; no ROI claim.','One fixed development split and one model seed.']}
    dump(ROOT/'reports/uplift_metrics.json',metrics)
    render(metrics,audit); update_manifest(metrics,audit); print(json.dumps({'selected':best,'assignment_auc':assignment_auc,'test':{k:{'qini':v['qini_coefficient'],'auuc':v['auuc'],'u10':v['uplift_at']['10']['uplift']} for k,v in test.items()}},indent=2))

def render(m,audit):
    lines=['# Uplift evaluation','',f"Outcome: visit. Model sample: {sum(m['model_rows'].values()):,} rows from {m['full_rows']:,} total rows. Train/validation/test: {m['model_rows']}.",'',
        f"Treatment assignment prediction validation AUC: {m['assignment_predictability_validation_auc']:.6f}. Treatment proportion in model train: {m['treatment_ratio_train']:.6%}.",
        '',f"Validation Qini selected `{m['selected_model']}` before test reporting.",'',
        '| Ranker | Test Qini | Test AUUC | Uplift@10% | Uplift@20% | Uplift@30% |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,x in m['test'].items(): lines.append(f"| {name} | {x['qini_coefficient']:.8f} | {x['auuc']:.8f} | {x['uplift_at']['10']['uplift']:.6%} | {x['uplift_at']['20']['uplift']:.6%} | {x['uplift_at']['30']['uplift']:.6%} |")
    lines += ['', '## Mathematical definition','',next(iter(m['test'].values()))['definition'],'',
        'The response baseline ranks by P(visit|X); causal models rank by estimated E[Y(1)-Y(0)|X]. High response propensity is therefore tested rather than assumed to equal incremental value.',
        '', '## Treatment imbalance and safeguards','',
        'The treated arm is much larger than control. T-learner has fewer control examples; X-learner combines arm-specific imputed-effect models using the observed training propensity to reflect unequal precision.',
        '`exposure` is excluded because it is post-treatment. All splits and model samples use fixed hashes. Preprocessing for Causal Forest is fit on its training subset only. Test metrics were computed after every estimator was frozen.',
        '', '## Stability and uncertainty','', 'Each test uplift decile includes a 95% stratified binary bootstrap interval (300 replicates). Full decile results, pairwise Spearman correlations, top-10% Jaccard overlap, and segment sizes are in `uplift_metrics.json`.',
        '', '## Limitations','']+[f'- {x}' for x in m['limitations']]
    (ROOT/'reports/uplift_evaluation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Power BI inputs
    pd.DataFrame([{'model':n,'selected_on_validation':n==m['selected_model'],'qini':x['qini_coefficient'],'auuc':x['auuc'],
        **{f'uplift_at_{k}':v['uplift'] for k,v in x['uplift_at'].items()}} for n,x in m['test'].items()]).to_csv(ROOT/'reports/powerbi/model_summary.csv',index=False)
    pd.DataFrame([{'model':n,**row} for n,rows in m['test_deciles'].items() for row in rows]).to_csv(ROOT/'reports/powerbi/uplift_deciles.csv',index=False)
    pd.DataFrame([{'model':n,**row} for n,x in m['test'].items() for row in x['curve'] if row['fraction'] in [.1,.2,.3,.4,.5,.6,.7,.8,.9,1.]]).to_csv(ROOT/'reports/powerbi/targeting_policy.csv',index=False)
    figures(m,audit)

def figures(m,audit):
    out=ROOT/'figures'; plt.style.use('seaborn-v0_8-whitegrid')
    bal=audit['feature_balance']; plt.figure(figsize=(7,4)); plt.barh([x['feature'] for x in bal],[x['smd'] for x in bal]); plt.axvline(0,color='black',lw=.8); plt.xlabel('Standardized mean difference'); plt.tight_layout(); plt.savefig(out/'covariate_balance.png',dpi=160); plt.close()
    g=audit['groups']; x=np.arange(2); w=.35; plt.figure(figsize=(6,4)); plt.bar(x-w/2,[g['0']['visit_rate'],g['0']['conversion_rate']],w,label='Control'); plt.bar(x+w/2,[g['1']['visit_rate'],g['1']['conversion_rate']],w,label='Treatment'); plt.xticks(x,['Visit','Conversion']); plt.ylabel('Outcome rate'); plt.legend(); plt.tight_layout(); plt.savefig(out/'outcome_comparison.png',dpi=160); plt.close()
    ate=json.loads((ROOT/'reports/ate_analysis.json').read_text(encoding='utf-8')); names=list(ate['outcomes']); vals=[ate['outcomes'][n]['ate'] for n in names]; lo=[ate['outcomes'][n]['ci95_normal'][0] for n in names]; hi=[ate['outcomes'][n]['ci95_normal'][1] for n in names]
    plt.figure(figsize=(6,4)); plt.errorbar(names,vals,yerr=[np.array(vals)-lo,np.array(hi)-vals],fmt='o',capsize=5); plt.axhline(0,color='black',lw=.8); plt.ylabel('ATE (absolute probability)'); plt.tight_layout(); plt.savefig(out/'ate_confidence_interval.png',dpi=160); plt.close()
    for kind,field,filename,ylabel in [('Qini','incremental_per_population','qini_curve.png','Cumulative incremental outcome / population'),('Uplift','uplift','uplift_curve.png','Observed uplift')]:
        plt.figure(figsize=(7,5))
        for n,z in m['test'].items(): plt.plot([r['fraction'] for r in z['curve']],[r[field] for r in z['curve']],label=n)
        plt.xlabel('Targeting fraction'); plt.ylabel(ylabel); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(out/filename,dpi=160); plt.close()
    plt.figure(figsize=(7,5))
    for n,rows in m['test_deciles'].items(): plt.plot(range(1,11),[r['uplift'] for r in rows],marker='o',label=n)
    plt.xlabel('Predicted uplift decile (1=highest)'); plt.ylabel('Observed uplift'); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(out/'uplift_by_decile.png',dpi=160); plt.close()
    plt.figure(figsize=(7,5))
    for n,z in m['test'].items(): plt.plot([r['fraction'] for r in z['curve']],[r['incremental_per_population']*m['model_rows']['test'] for r in z['curve']],label=n)
    plt.xlabel('Targeting fraction'); plt.ylabel('Estimated incremental visits in model test sample'); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(out/'targeting_policy.png',dpi=160); plt.close()
    names=list(m['test']); q=[m['test'][n]['qini_coefficient'] for n in names]; plt.figure(figsize=(8,4)); plt.bar(names,q); plt.xticks(rotation=25,ha='right'); plt.ylabel('Qini coefficient'); plt.tight_layout(); plt.savefig(out/'model_comparison.png',dpi=160); plt.close()

def update_manifest(m,audit):
    manifest={'status':'complete','completed_at_utc':m['completed_at_utc'],'dataset':audit['parquet'],'source':audit['raw'],'rows':audit['rows'],
        'split_rule':audit['split_rule'],'full_split_counts':audit['split_counts'],'model_sample_rule':m['model_sample_rule'],'model_rows':m['model_rows'],
        'features':FEATURES,'excluded_post_treatment':['exposure'],'random_seed':SEED,'selected_model':m['selected_model'],'selection_metric':m['selection_metric'],
        'model_artifacts':m['model_artifacts'],'reports':sorted(str(p.relative_to(ROOT)) for p in (ROOT/'reports').rglob('*') if p.is_file()),
        'figures':sorted(str(p.relative_to(ROOT)) for p in (ROOT/'figures').glob('*.png'))}
    dump(ROOT/'experiment_manifest.json',manifest)
if __name__=='__main__': main()
