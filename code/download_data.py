import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

BASE = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/'
ARCHIVES = {
    'GTSRB_Final_Training_Images.zip': 'd32ac4b5fa9a1cbd1994768413902e8193599d9434cf0a8eb9cfd00a6d3a290c',
    'GTSRB_Final_Test_Images.zip': 'c7e4e6327067d32654124b0fe9e82185',
    'GTSRB_Final_Test_GT.zip': 'fe31e9c9270bbcd7b84b7f21a9d9d9e5',
}


def digest(path, algorithm='md5'):
    h = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(root):
    root = Path(root).resolve()
    archives = root / 'archives'
    archives.mkdir(parents=True, exist_ok=True)
    provenance = []
    for name, expected in ARCHIVES.items():
        algorithm = 'sha256' if len(expected) == 64 else 'md5'
        target = archives / name
        if not target.exists() or digest(target, algorithm) != expected:
            partial = target.with_suffix('.zip.part')
            print(f'Downloading {name} ...', flush=True)
            urllib.request.urlretrieve(BASE + name, partial)
            if digest(partial, algorithm) != expected:
                raise ValueError(f'Checksum mismatch: {partial}')
            partial.replace(target)
        print(f'Verified {name}; extracting ...', flush=True)
        with zipfile.ZipFile(target) as archive:
            for member in archive.infolist():
                dest = (root / member.filename).resolve()
                if not dest.is_relative_to(root):
                    raise ValueError(f'Unsafe archive path: {member.filename}')
            archive.extractall(root)
        provenance.append({'archive': name, 'url': BASE + name,
                           'checksum_algorithm': algorithm, 'expected_checksum': expected,
                           'sha256': digest(target, 'sha256')})
    (root / 'download_manifest.json').write_text(
        json.dumps(provenance, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/raw')
    download(parser.parse_args().root)
