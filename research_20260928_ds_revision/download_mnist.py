"""Download the four torchvision-listed MNIST IDX files and verify their MD5."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import urllib.request

FILES = {
    'train-images-idx3-ubyte.gz': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
    'train-labels-idx1-ubyte.gz': 'd53e105ee54ea40749a09fcbcd1e9432',
    't10k-images-idx3-ubyte.gz': '9fb629c4189551a2d022fa330f9573f3',
    't10k-labels-idx1-ubyte.gz': 'ec29112dd5afa0611ce80d1b7f02629c',
}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        p = args.out / name
        url = 'https://ossci-datasets.s3.amazonaws.com/mnist/' + name
        if not p.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                p.write_bytes(response.read())
        content = p.read_bytes()
        actual = hashlib.md5(content).hexdigest()
        if actual != expected:
            raise ValueError(f'MD5 mismatch: {name}: {actual}')
        raw = gzip.decompress(content)
        p.with_suffix('').write_bytes(raw)
        print(json.dumps({'file': name, 'MD5': actual, 'raw_bytes': len(raw), 'source': url}), flush=True)
