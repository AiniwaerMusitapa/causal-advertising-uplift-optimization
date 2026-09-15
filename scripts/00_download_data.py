"""Stream the verified Criteo organization mirror and record provenance."""
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
URL='https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz'
EXPECTED_SHA256='2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc'

def main():
    raw=ROOT/'data'/'raw'; raw.mkdir(parents=True,exist_ok=True)
    target=raw/'criteo-research-uplift-v2.1.csv.gz'; partial=target.with_suffix(target.suffix+'.part')
    if target.exists():
        print('Existing archive found; verifying instead of downloading')
    else:
        request=urllib.request.Request(URL,headers={'User-Agent':'Criteo-Causal-Portfolio/1.0'})
        with urllib.request.urlopen(request,timeout=60) as response, partial.open('wb') as out:
            total=int(response.headers.get('Content-Length') or 0); done=0; started=time.time()
            while True:
                block=response.read(8*1024*1024)
                if not block: break
                out.write(block); done+=len(block)
                rate=done/max(time.time()-started,1)/1024/1024
                print(f'\r{done/1024/1024:.1f} MiB / {total/1024/1024:.1f} MiB ({rate:.1f} MiB/s)',end='',flush=True)
        print(); partial.replace(target)
    h=hashlib.sha256()
    with target.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    actual=h.hexdigest()
    if actual!=EXPECTED_SHA256: raise ValueError(f'SHA256 mismatch: {actual}')
    report={'status':'verified','downloaded_at_utc':datetime.now(timezone.utc).isoformat(),'source':'Criteo official organization mirror on Hugging Face',
        'dataset':'Criteo Uplift Prediction Dataset, corrected unbiased v2.1','url':URL,'official_page':'https://ailab.criteo.com/criteo-uplift-prediction-dataset/',
        'archive':str(target),'bytes':target.stat().st_size,'sha256':actual,'license_on_mirror':'CC BY-NC-SA 4.0',
        'version_note':'Corrected v2.1 is used because Criteo reports leakage in the older approximately 25M-row release.'}
    (ROOT/'reports').mkdir(exist_ok=True); (ROOT/'reports'/'data_source.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
