# Effective density and support on a held-out analytic family

The four PINO arms use the same logistic-atom model, width, optimizer, 120 updates, batch size 8, geometry-grouped box/sphere splits, checkpoint selection and seeds 131/137/139. Vary only the node support rule (multiply interpolated effective density by fractional occupancy or use its nonempty support) and physics loss weight (0 or 0.05). The source-matched W5 voxel DeepONet is an additional architecture comparator.

The previously unused panel contains eight disjoint two-ellipsoid unions per seed, two common interior planes per geometry and one rigid Haar rotation per parent. Independently rasterize the rotated union at 8³. Sum the two closed-form cap labels and rotate the ten moments and native Jacobian; no interpolated input. Report pooled r, J and Jv relative L2 per seed and their arithmetic means. Jv uses native directions c[0], c[1] and c[5]. This analytic held-out family is excluded from all fitting, checkpoint selection and adaptation. The experiment measures geometry prediction. The two-component domain carries no claim about a connected printable top, native Boolean data or solve speed.

## Three-seed results

Each pair is the relative L2 error of the ten moments and full native Jacobian. The last column evaluates three fixed directional derivatives on the new panel.

| Training arm | Original held-out ellipsoid r / J | New double ellipsoid r / J | New Jv |
|---|---:|---:|---:|
| W5 voxel DeepONet | 0.7511 / 0.6994 | 0.7077 / 0.5027 | 0.4374 |
| Fractional support, physical loss on | 0.2060 / 0.3147 | 0.3114 / 0.4110 | 0.4079 |
| Fractional support, physical loss off | 0.2060 / 0.3147 | 0.3114 / 0.4110 | 0.4079 |
| Effective density with nonempty support, physical loss on | 0.12015 / 0.20492 | 0.1611 / 0.2866 | 0.2863 |
| Effective density with nonempty support, physical loss off | 0.12015 / 0.20492 | 0.1611 / 0.2866 | 0.2863 |

At this resolution, multiplying effective density by fractional occupancy at each quadrature node suppresses boundary weights a second time. Using the nonempty support indicator improved both error metrics on each of the three new seeds. The explicit physics penalty had negligible effect in this head. The original point-sampled ellipsoid panel also improves. The single-support option applies to inputs whose density already incorporates occupancy. Input semantics for native CAD labels and downstream convergence require their own verification. No timing or speed claim uses this test.

Run from the W6 checkout with CPU PyTorch and NumPy:
```bash
python -m research.NeuralMoments.models.pcraft_ufno.spin_pino.support_ablation --output support-holdout.json
```
