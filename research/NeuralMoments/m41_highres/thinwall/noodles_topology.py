"""Compare Noodles D' manifold topology and a 6-adjacent, cell-centred digital complex."""
import argparse
from array import array
from hashlib import sha256
from itertools import chain
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components


def components(starts, ends, size):
    graph = sparse.coo_matrix((np.ones(len(starts), dtype=bool), (starts, ends)), shape=(size, size)).tocsr()
    return connected_components(graph, directed=False)


def source_topology(path):
    with path.open() as fd:
        nv, nf = map(int, fd.readline().split())
        for _ in range(nv):
            next(fd)
        f = np.loadtxt(fd, dtype=np.int32)
    assert f.shape == (nf, 3)
    directed = f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    e, inv, incidence = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    orientation = np.bincount(inv, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1))
    assert (incidence == 2).all() and (orientation == 0).all() and np.all(e[:, 0] != e[:, 1])
    edge_id = inv.reshape(nf, 3)
    # Each vertex link must be a single cycle. Each link node has degree 2 by the edge-incidence checks.
    link_a = 2 * edge_id + (f > np.roll(f, -1, axis=1))
    link_b = 2 * np.roll(edge_id, 1, axis=1) + (f > np.roll(f, 1, axis=1))
    nlinks = components(link_a.ravel(), link_b.ravel(), len(e)*2)[0]
    assert nlinks == nv and np.unique(f).size == nv
    nbound = components(e[:, 0], e[:, 1], nv)[0]
    assert nbound == 1
    chi = int(nv - len(e) + nf)
    return dict(vertices=nv, faces=nf, edges=len(e), edge_incidence_2=True, consistently_oriented=True,
                vertex_links_verified_single_cycle=True, boundary_components=nbound, chi_boundary=chi,
                orientable_surface_genus=(2-chi)//2, reference_solid_chi=chi//2)


def overlaps(row_a, row_b):
    i = j = 0
    while i < len(row_a) and j < len(row_b):
        a, b, id_a = row_a[i]
        c, d, id_b = row_b[j]
        lo, hi = max(a, c), min(b, d)
        if lo <= hi:
            yield lo, hi, id_a, id_b
        if b <= d:
            i += 1
        if d <= b:
            j += 1


def topology(n, ptr, starts, ends):
    # Occupied cell centres generate a cubical complex by adding each elementary edge,
    # square, and cube exactly when all its vertices are occupied.
    # Sum Euler along x first: each nonempty intersection interval contributes 1.
    src, dst = array('i'), array('i')
    faces = 0
    prev, prev_pair = {}, {}
    active_rows = np.flatnonzero(np.diff(ptr))
    starts, ends = [np.asarray(a, dtype=np.int32) for a in (starts, ends)]
    for z in range(n):
        rows, pairs = {}, {}
        for k in active_rows[np.searchsorted(active_rows, z*n):np.searchsorted(active_rows, (z+1)*n)]:
            y = k-z*n
            row = [(int(starts[i]), int(ends[i]), i) for i in range(int(ptr[k]), int(ptr[k+1]))]
            rows[y] = row
            edges = list(overlaps(rows.get(y-1, []), row))
            yz_pair = [(a, b, 0) for a, b, _, _ in edges]
            pairs[y] = yz_pair
            faces += sum(1 for _ in overlaps(prev_pair.get(y, []), yz_pair))
            for _, _, l, r in chain(edges, overlaps(prev.get(y, []), row)):
                src.append(l)
                dst.append(r)
        prev, prev_pair = rows, pairs
    n_components, labels = components(src, dst, len(starts))
    sizes = np.bincount(labels, weights=ends - starts + 1).astype(np.int64)
    sizes.sort()
    return dict(side=n, occupied_voxels=int(sizes.sum()), x_runs=len(starts),
                pairs_of_adjacent_overlapping_runs=len(src), four_row_intersections=faces,
                chi_solid_6=int(len(starts)-len(src)+faces), material_components_6=n_components,
                largest_component_voxels=int(sizes[-1]), outside_largest_voxels=int(sizes[:-1].sum()),
                largest_other_components=sorted(sizes[:-1].tolist(), reverse=True)[:12])


def selfcheck():
    gen = np.random.default_rng(20261107)
    for repetition in range(30):
        a = gen.random((7, 7, 7)) < gen.uniform(.15, .85)
        edges = np.diff(np.pad(a.astype(np.int8), ((0, 0), (0, 0), (1, 1))), axis=2)
        ptr = np.r_[0, np.cumsum((edges == 1).sum(axis=2).ravel())]
        result = topology(7, ptr, np.where(edges == 1)[2], np.where(edges == -1)[2]-1)
        assert result['material_components_6'] == ndimage.label(a)[1]
        # Count the alternating number of cubical cells directly in dense 3-D.
        chi = 0
        for shape in np.ndindex(2, 2, 2):
            corner_masks = [a[tuple(slice(v, end-s+v) for s, v, end in zip(shape, offset, a.shape))]
                            for offset in np.ndindex(*(s+1 for s in shape))]
            chi += (-1)**sum(shape) * int(np.logical_and.reduce(corner_masks).sum())
        assert chi == result['chi_solid_6']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kit', type=Path, required=True, help='extracted Spin_5090_HighRes_Kit root')
    p.add_argument('--design', type=Path, help='override kit inputs/Dprime.txt')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sides', type=int, nargs='+', default=[128, 256, 512, 1024, 2048])
    args = p.parse_args()
    selfcheck()
    sys.path.insert(0, str(args.kit.resolve() / 'geometry'))
    from noodles_resolution import parse_mesh, stencil, integrate_x_runs, DESIGN_SHA
    path = args.design or args.kit / 'inputs' / 'Dprime.txt'
    mesh, _ = parse_mesh(path)
    result = dict(schema='noodles-rle-topology-v1', subject="Noodles designable region Dprime, prior to cutting",
                  sha256=sha256(path.read_bytes()).hexdigest(), source=source_topology(path), selfcheck_dense_random_cases=30,
                  convention='cell-centre cubical complex, foreground 6-connected; reference solid chi = boundary chi/2 assuming embedded manifold bounding the source solid',
                  rows=[])
    assert result['sha256'] == DESIGN_SHA
    with TemporaryDirectory() as tmp:
        for n in args.sides:
            saved = args.kit / 'geometry' / 'Noodles_Dprime_1024_runs.npz'
            if n != 1024 or not saved.exists():
                saved = Path(tmp) / 'grid.npz'
                integrate_x_runs(stencil(mesh, n), n, saved)
            with np.load(saved) as d:
                assert int(d['side']) == n and str(d['design_sha256']) == DESIGN_SHA
                row = topology(n, d['row_ptr'], d['x_start'], d['x_end'])
            result['rows'].append(row)
            print(row, flush=True)
            args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
