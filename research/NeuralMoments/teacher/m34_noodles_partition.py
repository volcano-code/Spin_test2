"""Rebuild M32 from authenticated source parts with the original exact CGAL Boolean CLI."""
import argparse
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time

from m31_candidate_contact_screen import NOODLES, NOODLES_SHA, noodles_sources, read
from m33_noodles_inset_operand import rational_token

EPRIME_SHA = 'b68f810266a94e0c4e2489f2a1005ae0be1f7f321e9d9f092b047e374dd12133'


def packet(vertices, faces):
    def token(x):
        if isinstance(x, str):
            return x if '/' in x else x + '/1'
        return rational_token(Fraction(x))
    rows = [f'{len(vertices)} {len(faces)}']
    rows.extend(' '.join(map(token, row)) for row in vertices)
    rows.extend(' '.join(map(str, f)) for f in faces)
    return ('\n'.join(rows) + '\n').encode('ascii')


def balanced_unions(names):
    names = list(names)
    level = 0
    if len(names) < 2:
        raise ValueError('at least two fixed operands are required')
    while len(names) > 1:
        next_names = []
        for i in range(0, len(names) - 1, 2):
            name = 'F' if len(names) == 2 else f'fixed-{level:02d}-{i//2:03d}'
            yield name, names[i], names[i+1], 'union'
            next_names.append(name)
        if len(names) % 2:
            next_names.append(names[-1])
        names = next_names
        level += 1


def run_operations(ops, exe, out, expected_empty=()):
    exe = exe.resolve(strict=True)
    exe_hash = sha256(exe.read_bytes()).hexdigest()
    records = []
    for name, left, right, operation in ops:
        empty = name in expected_empty
        raw = (out/(left+'.txt')).read_bytes() + (out/(right+'.txt')).read_bytes()
        identity = dict(name=name, operation=operation, left=left, right=right,
                        input_sha256=sha256(raw).hexdigest(), executable_sha256=exe_hash)
        output = out/(name+'.json')
        receipt = out/(name+'.receipt.json')
        if receipt.exists():
            r = json.loads(receipt.read_text())
            if (any(r.get(k)!=v for k,v in identity.items()) or
                    sha256(output.read_bytes()).hexdigest()!=r['output_sha256'] or
                    sha256((out/(name+'.txt')).read_bytes()).hexdigest()!=r['operand_sha256'] or
                    (empty and (r['faces'] or r['vertices']))):
                raise ValueError('cached operand or provenance changed: ' + name)
        else:
            operand = out/'boolean-input.txt'
            operand.write_bytes(raw)
            start = time.monotonic()
            subprocess.run([str(exe), str(operand), str(output), operation], check=True)
            seconds = time.monotonic()-start
            data = output.read_bytes()
            mesh = json.loads(data)
            if not (mesh.get('succeeded') and mesh.get('operation') == operation and
                    len(mesh['faces']) == len(mesh['birth_faces']) and
                    bool(mesh['vertices']) == bool(mesh['faces']) == (not empty)):
                raise ValueError('unexpected native result or emptiness: ' + name)
            encoded = packet(mesh['vertices'],mesh['faces'])
            (out/(name+'.txt')).write_bytes(encoded)
            r = dict(**identity, seconds=seconds, vertices=len(mesh['vertices']), faces=len(mesh['faces']),
                     output_sha256=sha256(data).hexdigest(), operand_sha256=sha256(encoded).hexdigest())
            tmp = receipt.with_suffix('.tmp')
            tmp.write_text(json.dumps(r,sort_keys=True,indent=2)+'\n')
            os.replace(tmp,receipt)
        print(name, r['faces'], round(r['seconds'],2), flush=True)
        records.append(r)
    return records


def verify(exe, out):
    ops = [('U', 'Fprime', 'Dprime', 'union'),
           ('overlap', 'Fprime','Dprime','intersection'),
           ('missing','A','U','difference'),
           ('excess','U','A','difference'),
           ('fixed-missing','F','Fprime','difference')]
    source = (out/'construction.json').read_bytes()
    construction = json.loads(source)
    if construction['schema'] != 'm34-exact-cgal-partition-construction-v1':
        raise ValueError('M34 construction receipt required')
    solids = {'F','A','Dprime','Fprime'}
    outputs = [r for r in construction['operations'] if r['name'] in solids]
    if len(outputs) != 4 or {r['name'] for r in outputs} != solids:
        raise ValueError('incomplete construction receipt')
    for r in outputs:
        if sha256((out/(r['name']+'.txt')).read_bytes()).hexdigest() != r['operand_sha256']:
            raise ValueError('changed constructed operand ' + r['name'])
    checks = run_operations(ops, exe, out, expected_empty={'overlap','missing','excess','fixed-missing'})
    report = dict(schema='m34-cgal-regularized-partition-checks-v1', operations=checks,
                  construction_sha256=sha256(source).hexdigest(),
                  newton_evaluations=0)
    (out/'set-checks.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')


def build(root, exe, out):
    manifest_digest = NOODLES_SHA['frozen-parts-v7c/manifest.json']
    manifest = json.loads(read(root, NOODLES/'frozen-parts-v7c/manifest.json', manifest_digest))
    out.mkdir(parents=True, exist_ok=True)
    def save(name, v, f):
        (out/(('E' if name == 'outer-ellipsoid' else name)+'.txt')).write_bytes(packet(v, f))
    noodles_sources(root, manifest, consume=save)
    eprime = read(Path(__file__).parent, Path('M33_NOODLES_EPRIME.txt'), EPRIME_SHA)
    (out/'Eprime.txt').write_bytes(eprime)
    ops = [*balanced_unions(manifest['construction']['fixed_parts']),
           ('A','E','F','union'), ('Dprime','Eprime','F','difference'),
           ('Fprime','A','Dprime','difference')]
    records = run_operations(ops, exe, out)
    report = dict(schema='m34-exact-cgal-partition-construction-v1', source_manifest_sha256=manifest_digest,
                  inset_operand_sha256=EPRIME_SHA, operations=records, independent_set_verification=False,
                  newton_evaluations=0)
    (out/'construction.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')


if __name__ == '__main__':
    cmd = argparse.ArgumentParser(description=__doc__)
    cmd.add_argument('--main-tree',required=True,type=Path)
    cmd.add_argument('--executable',required=True,type=Path)
    cmd.add_argument('--out',required=True,type=Path)
    cmd.add_argument('--set-checks', action='store_true',
                     help='Also evaluate coverage, overlap and preservation of the fixed parts.')
    args = cmd.parse_args()
    build(args.main_tree,args.executable,args.out)
    if args.set_checks:
        verify(args.executable,args.out)
