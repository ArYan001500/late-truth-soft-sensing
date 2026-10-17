"""Archive retention and checksum reuse without network access."""
import hashlib
import io
import json
from pathlib import Path
import tempfile

import pytest

from latetruth import data


def test_existing_archives_are_reused_or_retained(monkeypatch):
    work=Path(__file__).resolve().parents[1]/'.work'
    work.mkdir(exist_ok=True)
    directory=Path(tempfile.mkdtemp(prefix='archive-test-',dir=work))
    payload=b'synthetic test payload'; checksum=hashlib.md5(payload).hexdigest()
    archive=directory/'sample.zip';archive.write_bytes(payload)
    item=dict(dataset='sample',record=1,key='sample.zip',path='sample.zip',md5=checksum)
    record=dict(files=[dict(key='sample.zip',checksum='md5:'+checksum,size=len(payload),links={'self':'https://example.invalid/sample.zip'})])
    requests=[]
    def response(url,**kwargs):
        requests.append(url)
        return io.BytesIO(json.dumps(record).encode())
    monkeypatch.setattr(data,'manifest',lambda:[item])
    monkeypatch.setattr(data.urllib.request,'urlopen',response)
    data.download(directory,['sample'])
    assert requests==['https://zenodo.org/api/records/1']
    assert archive.read_bytes()==payload
    item['md5']='0'*32
    record['files'][0]['checksum']='md5:'+item['md5']
    with pytest.raises(RuntimeError,match='was retained'):
        data.download(directory,['sample'])
    assert archive.read_bytes()==payload
