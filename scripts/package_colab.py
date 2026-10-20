"""Build a source archive; --include-data adds the private supplied dataset."""
import argparse
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--include-data', action='store_true')
parser.add_argument('--output', type=Path, default=root / 'Datathon-colab.zip')
args = parser.parse_args()
files = [root / name for name in ('case.ipynb', 'requirements-colab.txt', 'README.md')]
for directory in ('datathon', 'scripts', 'tests', 'review') + (('open',) if args.include_data else ()):
    files.extend(p for p in (root / directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(set(files)):
        archive.write(path, Path('Datathon') / path.relative_to(root))
print(f'{args.output}: {len(files)} files. Extract into your private Drive MyDrive folder.')
