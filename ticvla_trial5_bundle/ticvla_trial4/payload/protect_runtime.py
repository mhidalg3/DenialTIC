#!/usr/bin/env python3
"""Pin Isaac's installed binary stack before installing testing extras, then verify.

This does not replace the simulator's CUDA/PyTorch stack or install training deps.
Extra versions are a declared compatibility choice, not a claimed full GPU test.
"""
import argparse
import importlib.metadata as md
import json
from pathlib import Path

NATIVE=("torch", "torchvision", "torchaudio", "triton")
EXTRAS={"numpy":"1.26.4", "opencv-python":"4.11.0.86", "huggingface-hub":"0.36.0", "transformers":"4.57.1"}

def installed():
    versions={}
    for name in NATIVE:
        try: versions[name]=md.version(name)
        except md.PackageNotFoundError:
            if name in {"torch","torchvision"}:
                raise SystemExit(f"Isaac Python cannot find installed package metadata for {name}; stop rather than overwrite the binary stack.")
    return versions

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--write',type=Path);g.add_argument('--verify',type=Path)
    a=p.parse_args();v=installed()
    if a.write:
        pins={**v,**EXTRAS}
        a.write.write_text(''.join(f'{k}=={x}\n' for k,x in sorted(pins.items())))
        a.write.with_suffix('.json').write_text(json.dumps(v,indent=2)+'\n')
        print('Pinned native stack:',v)
    else:
        expected=json.loads(a.verify.with_suffix('.json').read_text())
        if expected!=v: raise SystemExit(f'Native binary-stack changed! Before={expected}; after={v}')
        for name,version in EXTRAS.items():
            if md.version(name)!=version: raise SystemExit(f'Extra pin mismatch: {name}')
        print('Native binary-stack and critical testing versions verified.')
if __name__=='__main__':main()
