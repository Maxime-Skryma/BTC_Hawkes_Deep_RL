"""Inspect files that would exist in the next commit; do not modify Git state."""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def main():
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    files=[];errors=[]
    for name in sorted(set(names)):
        if not name:continue
        p=ROOT/name
        if not p.is_file():continue  # tracked deletions will be absent from the next tree
        files.append((name,p.stat().st_size))
        generated=name.startswith(('experiences/','resultats/')) and name not in ['experiences/README.md','resultats/README.md']
        forbidden=p.suffix.lower() in ['.pt','.pth','.parquet','.bundle']
        if generated or forbidden or p.stat().st_size>5*1024**2:errors.append(name)
    print(f'Prospective tree: {len(files)} files, {sum(n for _,n in files)/1024**2:.2f} MiB')
    for name,size in sorted(files,key=lambda item:-item[1])[:5]:print(f'  {size/1024:.1f} KiB  {name}')
    if errors:
        print('FAIL: remove generated/large files from the index before committing:')
        print('\n'.join(errors));raise SystemExit(1)
    print('PASS: no generated experiments, presentation outputs or model/data binaries in the prospective tree.')
    print('Historical commits are not covered by this tree audit; inspect/rewrite history separately before pushing.')

if __name__=='__main__':main()
