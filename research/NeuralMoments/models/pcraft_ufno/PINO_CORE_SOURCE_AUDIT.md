# PINO core source audit — W6 Spin-PINO v1

## Primary source

- Repository: https://github.com/volcano-code/test
- Pinned commit: `24eda3e9b3f87ba16251a9025761986d6cfe3079`
- Source: `structural/models/pino.py`
- Source blob: `5017f50292e43718546ad0457e0869a9eb5ddff0`
- Relevant source classes: `SpectralConv3d_Factorized`, `FNOBlock3d`, `UFNO3d`, `PINO3d`, and `PINOLoss`.
- `UFNO3d` source preserves the factorized x/y/z Fourier blocks, U-shaped encoder/bottleneck/decoder and skip connections.

## Critical distinction

The existing W6 `model.py` is explicitly an independent F-FNO-inspired model; it is not the original P-CRAFT PINO port. The new `spin_pino/` package restores the P-CRAFT UFNO3d factorized-Fourier U-shaped trunk as the primary Spin model. The old W6 M13 model remains a baseline and is not deleted.

The original P-CRAFT `PINOLoss` is for linear elasticity: equilibrium residual, constitutive residual and displacement boundary conditions. Those equations are not valid substitutes for the Spin moment task. `SpinPINOLoss` therefore preserves the PINO training paradigm (data term + physics residuals) but uses the exact identities of the smooth convex moment potential: complement identity, projective radial identity, Hessian symmetry/PSD and mass bounds.

## Reuse and license status

The upstream repository root did not expose a LICENSE file in the audited tree. This integration is recorded as internal same-organization source reuse pending explicit license/ownership clearance. Do not redistribute this source outside the project until that status is resolved. Preserve this provenance record and the upstream commit pin.

## Deliberate adaptations

- Original elasticity 9-channel displacement/stress decoder is replaced with a pooled geometry latent and Spin moment-potential head.
- The factorized spectral operator retains the original axis-wise rFFT/iFFT structure; safe mode clipping is added for small grids.
- Original coordinate channels are retained.
- Original U-shaped enc1/down1/enc2/down2/bottleneck/up2/dec2/up1/dec1 topology is retained.
- No GINO, Transolver or DeepONet module replaces the PINO trunk. Such models remain optional plug-in baselines.
- The original elasticity finite-difference `torch.roll` physics residual is not copied into Spin loss; it is unrelated to the ten-moment integral task and its periodic wrap is inappropriate for nonperiodic geometry.

## Scope

This is the first CPU-testable Spin-PINO architecture slice. It is not trained on formally admitted Cherchi labels, does not certify a CAD/Boolean Jacobian, and is not yet accepted by native Newton. The convex potential's analytic Hessian is a model-internal mathematical property, not a proof that the model weights represent the true material integral.
