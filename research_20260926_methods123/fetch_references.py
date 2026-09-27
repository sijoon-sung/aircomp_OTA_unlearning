"""Download pinned public source archives. Does not execute downloaded code."""
import json, urllib.request, urllib.parse, zipfile, io, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
HEADERS={'User-Agent':'research-reproducibility/1.0','Accept':'application/vnd.github+json'}
def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers=HEADERS),timeout=60) as r:
        return r.read()
records=[]
for query in ['FedOrtho','CuReNU','CureNewton']:
    url='https://api.github.com/search/repositories?q='+urllib.parse.quote(query)
    try:
        obj=json.loads(fetch(url))
        records.append({'search':query,'total_count':obj['total_count'],
                        'repositories':[r['full_name'] for r in obj['items'][:10]]})
    except Exception as e: records.append({'search':query,'error':str(e)})
for repo in ['alessiomora/FedQUIT','samaonline/Orthogonal-Convolutional-Neural-Networks','Liuhong99/Sophia']:
    name=repo.split('/')[-1]
    try:
        meta=json.loads(fetch('https://api.github.com/repos/'+repo))
        commit=json.loads(fetch('https://api.github.com/repos/'+repo+'/commits/'+meta['default_branch']))['sha']
        url='https://codeload.github.com/'+repo+'/zip/'+commit
        payload=fetch(url)
        archive=ROOT/'vendor'/(name+'-'+commit[:12]+'.zip')
        archive.write_bytes(payload)
        target=(ROOT/'vendor'/name).resolve(); target.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            for entry in z.infolist():
                parts=Path(entry.filename).parts[1:]
                if not parts: continue
                dest=target.joinpath(*parts).resolve()
                if not dest.is_relative_to(target): raise ValueError('Unsafe archive path')
                if entry.is_dir(): dest.mkdir(parents=True,exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    dest.write_bytes(z.read(entry))
        records.append({'repo':repo,'commit':commit,'url':url,'sha256':hashlib.sha256(payload).hexdigest(),
                        'license':(meta.get('license') or {}).get('spdx_id'),'status':'downloaded'})
    except Exception as e: records.append({'repo':repo,'error':str(e)})
(ROOT/'source_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(records,ensure_ascii=False,indent=2))
