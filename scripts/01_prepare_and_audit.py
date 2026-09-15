"""Convert official gzip CSV to Parquet and run full-data experiment/ATE audits."""
import gzip
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import duckdb
import numpy as np
from scipy.stats import norm, ks_2samp

ROOT=Path(__file__).resolve().parents[1]; FEATURES=[f'f{i}' for i in range(12)]
RAW=ROOT/'data/raw/criteo-research-uplift-v2.1.csv.gz'; PARQUET=ROOT/'data/processed/criteo_uplift_v2_1.parquet'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def dump(path,obj): Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
def lit(path): return "'"+Path(path).as_posix().replace("'","''")+"'"

def main():
    source=json.loads((ROOT/'reports/data_source.json').read_text(encoding='utf-8'))
    assert sha(RAW)==source['sha256']
    (ROOT/'data/processed').mkdir(parents=True,exist_ok=True)
    con=duckdb.connect(); con.execute("SET memory_limit='2GB'"); con.execute('SET threads=2')
    if not PARQUET.exists():
        print('Converting gzip CSV to Parquet',flush=True)
        con.execute(f"COPY (SELECT row_number() OVER ()-1 AS row_id,* FROM read_csv({lit(RAW)},header=true,auto_detect=true)) TO {lit(PARQUET)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)")
    rel=f'read_parquet({lit(PARQUET)})'
    columns=[r[0] for r in con.execute('DESCRIBE SELECT * FROM '+rel).fetchall()]
    expected=['row_id']+FEATURES+['treatment','conversion','visit','exposure']; assert columns==expected,columns
    agg=['count(*) n']+[f'sum({c}) {c}_sum' for c in ['treatment','conversion','visit','exposure']]
    totals=dict(zip(['n','treatment_sum','conversion_sum','visit_sum','exposure_sum'],con.execute('SELECT '+','.join(agg)+' FROM '+rel).fetchone()))
    n=int(totals['n']); assert n>0
    groups={}
    for t in [0,1]:
        row=con.execute(f'SELECT count(*),avg(visit),avg(conversion),avg(exposure) FROM {rel} WHERE treatment={t}').fetchone()
        groups[str(t)]={'rows':int(row[0]),'visit_rate':float(row[1]),'conversion_rate':float(row[2]),'exposure_rate':float(row[3])}
    missing=dict(con.execute('SELECT '+','.join(f'sum({c} IS NULL)::BIGINT' for c in columns)+' FROM '+rel).fetchone() and zip(columns,con.execute('SELECT '+','.join(f'sum({c} IS NULL)::BIGINT' for c in columns)+' FROM '+rel).fetchone()))
    invalid=dict(zip(['treatment','visit','conversion','exposure'],con.execute('SELECT '+','.join(f'sum({c} NOT IN (0,1))::BIGINT' for c in ['treatment','visit','conversion','exposure'])+' FROM '+rel).fetchone()))
    balance=[]
    for f in FEATURES:
        vals=con.execute(f'SELECT treatment,avg({f}),stddev_samp({f}) FROM {rel} GROUP BY treatment ORDER BY treatment').fetchall()
        (t0,m0,s0),(t1,m1,s1)=vals; pooled=math.sqrt((s0*s0+s1*s1)/2); smd=(m1-m0)/pooled if pooled else 0
        qs=con.execute(f'SELECT min({f}),quantile_cont({f},.25),median({f}),quantile_cont({f},.75),max({f}) FROM {rel}').fetchone()
        balance.append({'feature':f,'control_mean':float(m0),'treatment_mean':float(m1),'control_sd':float(s0),'treatment_sd':float(s1),'smd':float(smd),
                        'min':float(qs[0]),'q25':float(qs[1]),'median':float(qs[2]),'q75':float(qs[3]),'max':float(qs[4])})
    # Fixed hash samples support distribution tests without collecting the full table.
    samples={}
    for t in [0,1]:
        tab=con.execute(f'SELECT {",".join(FEATURES)} FROM {rel} WHERE treatment={t} AND hash(row_id,911)%100=0 LIMIT 100000').fetch_arrow_table()
        samples[t]=np.column_stack([tab.column(f).to_numpy() for f in FEATURES])
    for j,item in enumerate(balance):
        stat,p=ks_2samp(samples[0][:,j],samples[1][:,j]); item.update(ks_statistic=float(stat),ks_pvalue=float(p))
    split_counts=dict(con.execute(f"SELECT CASE WHEN hash(row_id,42)%100<70 THEN 'train' WHEN hash(row_id,42)%100<85 THEN 'validation' ELSE 'test' END s,count(*) FROM {rel} GROUP BY s").fetchall())
    audit={'status':'complete','created_at_utc':datetime.now(timezone.utc).isoformat(),'dataset_version':'corrected unbiased v2.1','rows':n,'columns':columns,
        'raw':{'path':str(RAW),'bytes':RAW.stat().st_size,'sha256':source['sha256']},'parquet':{'path':str(PARQUET),'bytes':PARQUET.stat().st_size,'sha256':sha(PARQUET)},
        'treatment_rows':groups['1']['rows'],'control_rows':groups['0']['rows'],'treatment_ratio':groups['1']['rows']/n,
        'overall':{'visit_rate':totals['visit_sum']/n,'conversion_rate':totals['conversion_sum']/n,'exposure_rate':totals['exposure_sum']/n},
        'groups':groups,'missing':missing,'invalid_binary_values':invalid,'feature_balance':balance,
        'max_abs_smd':max(abs(x['smd']) for x in balance),'mean_abs_smd':float(np.mean([abs(x['smd']) for x in balance])),
        'distribution_test_note':'Two-sample KS on fixed hash samples up to 100k/group; large-n p-values are not balance effect sizes.',
        'split_rule':'hash(row_id,42)%100: train 0-69, validation 70-84, test 85-99','split_counts':{k:int(v) for k,v in split_counts.items()},
        'features':FEATURES,'treatment':'treatment','outcomes':['visit','conversion'],
        'excluded_post_treatment':['exposure'],'post_treatment_note':'Exposure occurs after assignment and is excluded from all model features to avoid post-treatment bias.'}
    assert sum(audit['split_counts'].values())==n and not any(missing.values()) and not any(invalid.values())
    dump(ROOT/'reports/experiment_validation.json',audit)
    lines=['# Randomized experiment validation','',f"Dataset: corrected Criteo Uplift v2.1. Local rows: {n:,}; columns: {len(columns)}.",'',
        f"Treatment: {audit['treatment_rows']:,} ({audit['treatment_ratio']:.6%}); control: {audit['control_rows']:,}. Visit rate: {audit['overall']['visit_rate']:.6%}; conversion rate: {audit['overall']['conversion_rate']:.6%}.",
        '',f"Maximum absolute SMD across 12 baseline features: {audit['max_abs_smd']:.6f}; mean absolute SMD: {audit['mean_abs_smd']:.6f}.",'',
        '| Feature | Control mean | Treatment mean | SMD | KS statistic |','| --- | ---: | ---: | ---: | ---: |']
    lines += [f"| {x['feature']} | {x['control_mean']:.6f} | {x['treatment_mean']:.6f} | {x['smd']:.6f} | {x['ks_statistic']:.6f} |" for x in balance]
    lines += ['', 'Treatment assignment is imbalanced in sample size; covariate balance is assessed by SMD rather than equal group counts. Assignment predictability is added by the modeling pipeline.',
        '', '**Post-treatment safeguard:** `exposure` is observed after treatment assignment and is excluded from preprocessing, response models and every CATE estimator. Including it would condition on a treatment consequence and can bias causal effects.',
        '',f"Split: `{audit['split_rule']}`; counts: {audit['split_counts']}. Seed 42."]
    (ROOT/'reports/experiment_validation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    ate={'status':'complete','estimand':'intention-to-treat difference in means','rows':n,'bootstrap':'2,000 stratified nonparametric binary-outcome resamples; seed 42','outcomes':{}}
    rng=np.random.default_rng(42)
    for outcome in ['visit','conversion']:
        a=groups['1'][outcome+'_rate']; b=groups['0'][outcome+'_rate']; nt=groups['1']['rows']; nc=groups['0']['rows']; diff=a-b
        se=math.sqrt(a*(1-a)/nt+b*(1-b)/nc); z=diff/se; pv=2*norm.sf(abs(z))
        boot=rng.binomial(nt,a,2000)/nt-rng.binomial(nc,b,2000)/nc
        ate['outcomes'][outcome]={'treatment_mean':a,'control_mean':b,'ate':diff,'absolute_lift':diff,'relative_lift':diff/b,
            'standard_error':se,'ci95_normal':[diff-1.96*se,diff+1.96*se],'z_statistic':z,'p_value_two_sided':pv,
            'ci95_bootstrap_percentile':np.quantile(boot,[.025,.975]).tolist()}
    dump(ROOT/'reports/ate_analysis.json',ate)
    lines=['# Average treatment effect','',f"Estimand: intention-to-treat difference in means on all {n:,} rows.",'',
        '| Outcome | Treatment rate | Control rate | ATE | Relative lift | SE | 95% normal CI | 95% bootstrap CI | p-value |','| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |']
    for name,x in ate['outcomes'].items():
        lines.append(f"| {name} | {x['treatment_mean']:.6%} | {x['control_mean']:.6%} | {x['ate']:.6%} | {x['relative_lift']:.3%} | {x['standard_error']:.8f} | [{x['ci95_normal'][0]:.6%}, {x['ci95_normal'][1]:.6%}] | [{x['ci95_bootstrap_percentile'][0]:.6%}, {x['ci95_bootstrap_percentile'][1]:.6%}] | {x['p_value_two_sided']:.3g} |")
    lines += ['', 'The bootstrap resamples binary outcomes independently within treatment arms, which is the exact row-level nonparametric bootstrap distribution for group means when rows are independent and outcomes are binary.',
        'No revenue or advertising-cost fields exist, so these effects are not ROI estimates. `exposure` is not used in the estimand.']
    (ROOT/'reports/ate_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump(ROOT/'experiment_manifest.json',{'status':'data_audit_complete','dataset':audit['parquet'],'source':audit['raw'],'split_rule':audit['split_rule'],'split_counts':audit['split_counts'],'seed':42,'features':FEATURES,'excluded':['exposure'],'reports':['reports/experiment_validation.json','reports/ate_analysis.json']})
    print(json.dumps({'rows':n,'split_counts':audit['split_counts'],'max_abs_smd':audit['max_abs_smd'],'ate':ate['outcomes']},indent=2))
if __name__=='__main__': main()
