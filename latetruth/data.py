"""Verified, resumable streaming downloads from the cited Zenodo records."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def manifest():
    return json.loads(Path(__file__).with_name('data_manifest.json').read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'md5').hexdigest()


def download(data_dir, datasets):
    """Keep archives intact; readers stream only their required CSV members."""
    records = {}
    for item in manifest():
        if item['dataset'] not in datasets:
            continue
        record = item['record']
        if record not in records:
            with urllib.request.urlopen(f'https://zenodo.org/api/records/{record}', timeout=60) as response:
                records[record] = {f['key']: f for f in json.load(response)['files']}
        info = records[record].get(item['key'])
        if info is None or info['checksum'] != 'md5:' + item['md5']:
            raise RuntimeError(f"Cited record changed or lacks verified file: {item['key']}")
        path = Path(data_dir) / item['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if digest(path) == item['md5']:
                print('Verified existing', path.name, flush=True)
                continue
            raise RuntimeError(f'Existing archive failed MD5 and was retained: {path}. Use another data directory.')
        url = info['links'].get('content', info['links']['self'])
        error = None
        for attempt in range(3):
            try:
                partial = path.with_suffix('.zip.partial')
                count = 0
                with urllib.request.urlopen(url, timeout=90) as response, partial.open('wb') as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        count += len(chunk)
                        if count % (16 * 1024 * 1024) == 0:
                            print(f"{path.name}: {count}/{info['size']} bytes", flush=True)
                if digest(partial) != item['md5']:
                    raise RuntimeError('MD5 mismatch: ' + path.name)
                if path.exists():
                    raise RuntimeError('Archive appeared during download and was retained: ' + str(path))
                partial.replace(path)
                print('Downloaded and verified', path.name, count, 'bytes', flush=True)
                error = None
                break
            except Exception as exc:
                error = exc
                print('Download retry', attempt + 1, str(exc), flush=True)
                time.sleep(attempt + 1)
        if error:
            raise RuntimeError(f'UNREACHABLE: {url}: {error}') from error
