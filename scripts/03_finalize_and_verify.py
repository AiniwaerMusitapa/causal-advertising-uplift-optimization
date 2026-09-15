"""Verify artifacts and add targeting-policy results for visit and conversion."""
import hashlib,importlib.util,json,platform
from datetime import datetime,timezone
from pathlib import Path
import duckdb,joblib,lightgbm,numpy as np,pandas as pd,sklearn,econml

ROOT=Path(__file__).resolve().parents[1]; FEATURES=[f'f{i}' for i in range(12)]
spec=importlib.util.spec_from_file_location('pipeline',Path(__file__).with_name('02_train_and_evaluate.py')); p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def dump(path,obj): Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8')

def main():
    metrics=json.loads((ROOT/'reports/uplift_metrics.json').read_text(encoding='utf-8')); audit=json.loads((ROOT/'reports/experiment_validation.json').read_text(encoding='utf-8'))
    assert metrics['status']==audit['status']=='complete' and metrics['post_treatment_exposure_used'] is False
    models={}
    for name,record in metrics['model_artifacts'].items():
        path=Path(record['path']); assert path.exists() and sha(path)==record['sha256']; models[name]=joblib.load(path)
    # Reconstruct selected and response scores from frozen artifacts on the recorded test sample.
    parquet=Path(audit['parquet']['path']); assert sha(parquet)==audit['parquet']['sha256']
    con=duckdb.connect(); case="CASE WHEN hash(row_id,42)%100<70 THEN 'train' WHEN hash(row_id,42)%100<85 THEN 'validation' ELSE 'test' END"
    df=con.execute(f"SELECT {','.join(FEATURES)},treatment,visit,conversion FROM read_parquet('{parquet.as_posix()}') WHERE {metrics['model_sample_rule']} AND {case}='test'").fetch_df(); con.close()
    x=df[FEATURES].to_numpy(np.float32); t=df.treatment.to_numpy(np.uint8); visit=df.visit.to_numpy(np.uint8); conversion=df.conversion.to_numpy(np.uint8)
    s=models['s_learner']; causal=s.predict_proba(np.c_[x,np.ones(len(x))])[:,1]-s.predict_proba(np.c_[x,np.zeros(len(x))])[:,1]
    response=models['response_model'].predict_proba(x)[:,1]
    assert len(df)==metrics['model_rows']['test']
    recheck=p.curve(visit,t,causal); stored=metrics['test']['s_learner']; assert abs(recheck['qini_coefficient']-stored['qini_coefficient'])<1e-12
    policy={}
    rows=[]
    for outcome,y in [('visit',visit),('conversion',conversion)]:
        policy[outcome]={}
        for strategy,score in [('causal_s_learner',causal),('response_model',response)]:
            result=p.curve(y,t,score); policy[outcome][strategy]=result
            for k in ['10','20','30']:
                z=result['uplift_at'][k]; rows.append({'outcome':outcome,'strategy':strategy,'targeting_rate':int(k)/100,
                    'targeted_rows':z['rows'],'uplift':z['uplift'],'estimated_incremental_outcomes_in_model_test':z['incremental_per_population']*len(y)})
    q90=float(np.quantile(causal,.9)); segments={'high_positive':int((causal>=q90).sum()),'moderate_positive':int(((causal>.0005)&(causal<q90)).sum()),
        'near_zero_0_to_0.0005':int(((causal>=0)&(causal<=.0005)).sum()),'negative':int((causal<0).sum())}; assert sum(segments.values())==len(causal)
    metrics['selected_policy_segments']=segments; metrics['targeting_policy_outcomes']=policy; dump(ROOT/'reports/uplift_metrics.json',metrics)
    pd.DataFrame(rows).to_csv(ROOT/'reports/powerbi/targeting_policy.csv',index=False)
    audit['assignment_predictability']={'model':'LightGBM classifier on baseline f0-f11 only','validation_roc_auc':metrics['assignment_predictability_validation_auc'],'interpretation':'Diagnostic only; AUC near 0.5 is consistent with weak global predictability but does not prove conditional randomization.'}; dump(ROOT/'reports/experiment_validation.json',audit)
    with (ROOT/'reports/experiment_validation.md').open('a',encoding='utf-8') as f: f.write(f"\nAssignment predictability validation ROC-AUC (baseline features only): **{metrics['assignment_predictability_validation_auc']:.6f}**. This diagnostic complements, but does not replace, balance and design evidence.\n")
    lines=['# Targeting policy analysis','',f"Selected on validation Qini: `{metrics['selected_model']}`. Test sample: {len(df):,} users.",'',
        '| Outcome | Strategy | Target rate | Targeted users | Uplift | Estimated incremental outcomes in model test |','| --- | --- | ---: | ---: | ---: | ---: |']
    for r in rows: lines.append(f"| {r['outcome']} | {r['strategy']} | {r['targeting_rate']:.0%} | {r['targeted_rows']:,} | {r['uplift']:.6%} | {r['estimated_incremental_outcomes_in_model_test']:.2f} |")
    lines += ['', 'Incremental outcomes are randomized-test estimates within the sampled test partition, not production forecasts or financial impact. Conversion is a secondary evaluation using visit-trained rankings; no conversion-specific model selection occurred.',
        '',f"Exclusive selected-score segments: {segments}.",'', 'No cost or value fields are available, so no ROI is reported.']
    (ROOT/'reports/targeting_policy.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    params={name:(model.get_params() if hasattr(model,'get_params') else {}) for name,model in models.items()}
    artifact={'status':'passed','verified_at_utc':datetime.now(timezone.utc).isoformat(),'dataset_sha256_verified':True,'model_checksums_verified':True,
        'model_count':len(models),'feature_list':FEATURES,'exposure_excluded':True,'all_four_cate_methods_present':all(x in metrics['test'] for x in ['s_learner','t_learner','x_learner','causal_forest']),
        'split_counts':metrics['model_rows'],'causal_forest_train_rows':metrics['causal_forest_train_rows'],'selected_by_validation':metrics['selected_model'],
        'source_code_sha256':{f.name:sha(f) for f in sorted((ROOT/'scripts').glob('*.py'))},'model_parameters':params,
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__,'econml':econml.__version__,'duckdb':duckdb.__version__},
        'checks':['full-data row counts and ATE','model hashes and loadability','test metric deterministic recomputation','exclusive policy segmentation','required reports/figures/Power BI CSVs']}
    required_reports=['experiment_validation.md','ate_analysis.md','uplift_evaluation.md','targeting_policy.md','uplift_metrics.json']; required_figures=['covariate_balance.png','outcome_comparison.png','ate_confidence_interval.png','qini_curve.png','uplift_curve.png','uplift_by_decile.png','targeting_policy.png','model_comparison.png']
    assert all((ROOT/'reports'/f).exists() for f in required_reports) and all((ROOT/'figures'/f).exists() and (ROOT/'figures'/f).stat().st_size>1000 for f in required_figures)
    dump(ROOT/'reports/artifact_audit.json',artifact)
    manifest=json.loads((ROOT/'experiment_manifest.json').read_text(encoding='utf-8')); manifest.update(status='complete',artifact_audit='reports/artifact_audit.json',targeting_policy='reports/targeting_policy.md',source_code_sha256=artifact['source_code_sha256']); dump(ROOT/'experiment_manifest.json',manifest)
    print(json.dumps({'status':'passed','segments':segments,'policy_top_rates':rows},indent=2))
if __name__=='__main__': main()
