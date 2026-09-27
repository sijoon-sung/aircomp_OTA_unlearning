from pathlib import Path
import urllib.request,zipfile,io,hashlib,json
root=Path(__file__).resolve().parent
url='https://proceedings.iclr.cc/paper_files/paper/2026/file/1fb4f62b8df5059a41df270a65e9b462-Supplemental-Conference.zip'
payload=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'research-reproducibility/1.0'}),timeout=60).read()
(root/'vendor/CuReNU_supplement.zip').write_bytes(payload)
base=(root/'vendor/CuReNU_supplement').resolve(); base.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    names=z.namelist()
    for entry in z.infolist():
        dest=(base/entry.filename).resolve()
        if not dest.is_relative_to(base):raise ValueError('unsafe archive path')
        if entry.is_dir():dest.mkdir(parents=True,exist_ok=True)
        else:
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(entry))
record={'url':url,'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload),'entries':names}
(root/'supplement_manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps({'bytes':len(payload),'entries':names[:60]},indent=2))
