"""M80 P-CRAFT-dominant operator using the M72 native RLE/query contract.

The coarse 3-D grid is an exact volume aggregation of M72 native RLE columns
when grid_size divides the native side. It is used only for P-CRAFT global
context. Every native interval is retained, sampled against the P-CRAFT
decoder feature map, and included in the mass-weighted interval reduction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import math
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .pino_backbone import PINOFeatureEncoder

SCHEMA = "m80.pcraft-native-query.v1"
SEMANTICS = "pcraft-ufno-map-sampled-native-interval-m72-bounded-odd-head-ad-jacobian.v1"
MOMENT_ORDER = ("mass", "x", "y", "z", "xx", "yy", "zz", "yz", "xz", "xy")
QUERY_ORDER = ("1", "x", "y", "z", "xx", "yy", "zz", "2yz", "2xz", "2xy")
POWERS = (0, 1, 1, 1, 2, 2, 2, 2, 2, 2)

PRESETS = {
    "smoke": dict(grid_size=8, pino_width=2, pino_modes=1, pino_layers=2,
                  interval_width=8, interval_depth=1, chunk_intervals=16),
    "large": dict(grid_size=32, pino_width=16, pino_modes=8, pino_layers=4,
                  interval_width=96, interval_depth=2, chunk_intervals=4096),
    "xl": dict(grid_size=64, pino_width=32, pino_modes=12, pino_layers=6,
              interval_width=192, interval_depth=4, chunk_intervals=2048),
    "xxl": dict(grid_size=128, pino_width=48, pino_modes=16, pino_layers=8,
               interval_width=256, interval_depth=6, chunk_intervals=1024),
}


@dataclass(frozen=True)
class NativePCRAFTConfig:
    grid_size: int = 64
    pino_width: int = 32
    pino_modes: int = 12
    pino_layers: int = 6
    interval_width: int = 192
    interval_depth: int = 4
    chunk_intervals: int = 2048
    min_native_resolution: int = 512
    checkpoint_intervals: bool = True

    def __post_init__(self):
        if self.grid_size < 8 or self.grid_size % 4:
            raise ValueError("grid_size must be >=8 and divisible by four")
        for name in ("pino_width", "pino_modes", "pino_layers", "interval_width",
                     "interval_depth", "chunk_intervals", "min_native_resolution"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.pino_width < 2 or self.interval_width < 8:
            raise ValueError("pino_width>=2 and interval_width>=8 are required")


def make_config(preset="xl", **overrides):
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}")
    return NativePCRAFTConfig(**(dict(PRESETS[preset]) | overrides))


def _geometry_hash(columns, origin, h, side, density, fixed):
    digest = sha256()
    for value in (np.asarray(columns, dtype="<i8"), np.asarray(origin, dtype="<f8"),
                  np.asarray([h, density, side], dtype="<f8"), np.asarray(fixed, dtype="<f8")):
        digest.update(value.tobytes())
    return digest.hexdigest()


def geometry_from_columns(values: Mapping[str, object]) -> dict:
    """Port of M72 NativeQuadricOracle.geometry_inputs for M69 RLE NPZ files."""
    columns = np.asarray(values["columns"])
    if columns.ndim != 2 or columns.shape[1] != 4 or columns.dtype.kind not in "iu":
        raise ValueError("columns must be integer [N,4] x,y,lo,hi RLE intervals")
    columns = np.asarray(columns, dtype=np.int64)
    origin = np.asarray(values["origin"], dtype=np.float64)
    h = float(np.asarray(values["h"]))
    side_value = np.asarray(values.get("native_resolution", values.get("side", 0)))
    if side_value.shape != () or side_value.dtype.kind not in "iu":
        raise ValueError("native_resolution/side must be an integer scalar")
    side = int(side_value)
    density = float(np.asarray(values.get("density", 1.0)))
    if origin.shape != (3,) or not np.isfinite(origin).all() or not math.isfinite(h) or h <= 0:
        raise ValueError("invalid origin or voxel width")
    if side < 1 or not math.isfinite(density) or density <= 0:
        raise ValueError("invalid native resolution or density")
    if not len(columns):
        raise ValueError("native RLE geometry has no editable intervals")
    if ((columns[:, :3] < 0).any() or (columns[:, :2] >= side).any()
            or (columns[:, 3] > side).any() or (columns[:, 3] <= columns[:, 2]).any()):
        raise ValueError("RLE interval is outside the native lattice or empty")
    order = np.lexsort((columns[:, 2], columns[:, 1], columns[:, 0]))
    columns = columns[order]
    same_xy = np.all(columns[1:, :2] == columns[:-1, :2], axis=1)
    if np.any(same_xy & (columns[1:, 2] < columns[:-1, 3])):
        raise ValueError("overlapping RLE intervals in one xy column")
    x = origin[0] + (columns[:, 0] + 0.5) * h
    y = origin[1] + (columns[:, 1] + 0.5) * h
    z0, z1 = origin[2] + columns[:, 2]*h, origin[2] + columns[:, 3]*h
    z = (z0+z1)/2
    extent = np.column_stack((np.full(len(columns), h), np.full(len(columns), h), z1-z0))
    mass = density*np.prod(extent, axis=1)
    basis = np.column_stack((np.ones(len(columns)), x, y, z,
        x*x+h*h/12, y*y+h*h/12, (z0*z0+z0*z1+z1*z1)/3, y*z, x*z, x*y))
    fixed = np.asarray(values.get("fixed_moments", np.zeros(10)), dtype=np.float64)
    if fixed.shape != (10,) or not np.isfinite(fixed).all():
        raise ValueError("fixed_moments must be finite [10]")
    design = mass @ basis
    original = np.asarray(values.get("original_moments", fixed+design), dtype=np.float64)
    if original.shape != (10,) or not np.isfinite(original).all():
        raise ValueError("original_moments must be finite [10]")
    tol = max(float(original[0]), float(mass.sum()), 1e-30)*max(side*h, 1e-12)**2*1e-10
    if not np.allclose(fixed+design, original, rtol=1e-8, atol=tol):
        raise ValueError("original moments != fixed + native editable moments")
    return dict(xyz=np.column_stack((x,y,z)), mass=mass, moment_basis=basis,
        cell_extent=extent, fixed_moments=fixed.copy(), original_moments=original.copy(),
        native_resolution=np.asarray(side, dtype=np.int64), density=density,
        geometry_sha256=_geometry_hash(columns,origin,h,side,density,fixed),
        columns=columns.copy(), origin=origin.copy(), h=h, side=side)


def load_geometry(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    if "columns" not in values:
        raise ValueError("M80 M72 adapter requires native RLE columns in geometry NPZ")
    return geometry_from_columns(values)


def conservative_coarse_grid(geometry: Mapping[str, object], grid_size: int) -> np.ndarray:
    """Exact RLE voxel-volume aggregation; never replaces the native intervals."""
    columns = np.asarray(geometry["columns"], dtype=np.int64)
    side, h = int(geometry["side"]), float(geometry["h"])
    if side % grid_size:
        raise ValueError("P-CRAFT context grid must divide the native RLE side exactly")
    factor = side//grid_size
    volume = np.zeros((grid_size,grid_size,grid_size), dtype=np.float64)
    zfirst = columns[:,2]//factor
    zlast = (columns[:,3]-1)//factor
    counts = zlast-zfirst+1
    total = int(counts.sum())
    rows = np.repeat(np.arange(len(columns),dtype=np.int64),counts)
    offsets = np.arange(total,dtype=np.int64)-np.repeat(np.cumsum(counts)-counts,counts)
    zb = zfirst[rows]+offsets
    overlap = np.minimum(columns[rows,3],(zb+1)*factor)-np.maximum(columns[rows,2],zb*factor)
    xb, yb = columns[rows,0]//factor, columns[rows,1]//factor
    np.add.at(volume,(xb,yb,zb),overlap.astype(np.float64)*h**3)
    cell = factor*h
    fraction = np.clip(volume/(cell**3),0.,1.)
    from scipy.ndimage import binary_erosion, distance_transform_edt
    occupied = fraction>0
    if not occupied.any():
        raise ValueError("coarse context grid has no editable material")
    signed = (-np.ones_like(fraction) if occupied.all()
              else distance_transform_edt(~occupied)-distance_transform_edt(occupied))
    boundary = occupied & ~binary_erosion(occupied,structure=np.ones((3,3,3),dtype=bool),border_value=0)
    return np.stack((fraction,occupied.astype(float),np.tanh(signed/2),boundary.astype(float))).astype(np.float32)


def _absolute_basis(xyz: Tensor, extent: Tensor) -> Tensor:
    lo,hi=xyz-extent/2,xyz+extent/2
    integral=(hi*hi.abs()-lo*lo.abs())/(2*extent.clamp_min(torch.finfo(extent.dtype).tiny))
    absolute=torch.where(extent>0,integral,xyz.abs())
    second=xyz.square()+extent.square()/12
    x,y,z=absolute.unbind(-1)
    return torch.stack((torch.ones_like(x),x,y,z,second[:,0],second[:,1],second[:,2],y*z,x*z,x*y),1)


def _quadric_bounds(bounds: Tensor, q: Tensor):
    lo,hi=bounds[0],bounds[1]
    sqlo=torch.where((lo<=0)&(hi>=0),torch.zeros_like(lo),torch.minimum(lo.square(),hi.square()))
    sqhi=torch.maximum(lo.square(),hi.square())
    low=[q.new_tensor(1.),*lo.unbind(),*sqlo.unbind()]
    high=[q.new_tensor(1.),*hi.unbind(),*sqhi.unbind()]
    for i,j in ((1,2),(0,2),(0,1)):
        corners=torch.stack((lo[i]*lo[j],lo[i]*hi[j],hi[i]*lo[j],hi[i]*hi[j]))
        low.append(2*corners.min()); high.append(2*corners.max())
    a,b=torch.stack(low),torch.stack(high)
    return torch.minimum(q*a,q*b).sum(-1),torch.maximum(q*a,q*b).sum(-1)


@dataclass(frozen=True)
class EncodedNativeGeometry:
    latent: Tensor
    statistics: Tensor
    fixed_moments: Tensor
    design_moments: Tensor
    absolute_moments: Tensor
    moment_scales: Tensor
    coordinate_scales: Tensor
    normalized_bounds: Tensor
    geometry_sha256: str
    native_resolution: int
    native_intervals: int
    owner: int
    versions: tuple

    def detached(self):
        names=("latent","statistics","fixed_moments","design_moments","absolute_moments",
               "moment_scales","coordinate_scales","normalized_bounds")
        return replace(self,**{name:getattr(self,name).detach() for name in names})


class IntervalResidual(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.ffn=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))
    def forward(self,x):
        return x+self.ffn(self.norm(x))


class PcraftNativeMomentOperator(nn.Module):
    """P-CRAFT spatial operator + M72 native interval response head."""
    def __init__(self,config: NativePCRAFTConfig|None=None):
        super().__init__()
        self.config=config or make_config("xl")
        c=self.config; w=c.interval_width
        self.pino_encoder=PINOFeatureEncoder(width=c.pino_width,modes=c.pino_modes,
            n_layers=c.pino_layers,grid_size=c.grid_size,use_gradient_checkpoint=True)
        self.pino_global=nn.Sequential(nn.Linear(self.pino_encoder.feature_dim,w),nn.GELU(),nn.Linear(w,w))
        self.pino_local=nn.Linear(c.pino_width,w,bias=False)
        self.statistics_context=nn.Sequential(nn.Linear(36,w),nn.SiLU(),nn.Linear(w,w))
        self.interval_embedding=nn.Sequential(nn.Linear(13,w),nn.GELU(),nn.Linear(w,w))
        self.interval_fusion=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,2*w),nn.GELU(),nn.Linear(2*w,w))
        self.interval_blocks=nn.ModuleList(IntervalResidual(w) for _ in range(c.interval_depth))
        self.query_head=nn.Sequential(nn.Linear(w+36+10,2*w),nn.SiLU(),
                                      nn.Linear(2*w,w),nn.SiLU(),nn.Linear(w,10))
        nn.init.normal_(self.query_head[-1].weight,std=.01)
        nn.init.zeros_(self.query_head[-1].bias)
        self.register_buffer("native_B",torch.tensor([1.,1.,1.,1.,1.,1.,1.,2.,2.,2.],dtype=torch.float64))
        self.counters=dict(encode_calls=0,intervals=0,queries=0,jacobians=0,exact_calls=0)

    def _versions(self):
        return tuple(p._version for p in self.parameters())

    def _chunk_pool(self,descriptor,weights,grid,pino_map,context):
        sampled=F.grid_sample(pino_map,grid,mode="bilinear",padding_mode="border",align_corners=True)
        sampled=sampled.reshape(pino_map.shape[1],-1).T
        h=self.interval_embedding(descriptor)+self.pino_local(sampled)
        h=self.interval_fusion(h+context)
        for block in self.interval_blocks:
            h=block(h)
        return (h*weights[:,None]).sum(0)

    def encode(self,geometry: Mapping[str,object],chunk: int|None=None):
        c=self.config; chunk=c.chunk_intervals if chunk is None else chunk
        if chunk<1: raise ValueError("chunk must be positive")
        if "xyz" not in geometry: geometry=geometry_from_columns(geometry)
        xyz_np=np.asarray(geometry["xyz"],dtype=np.float64)
        mass_np=np.asarray(geometry["mass"],dtype=np.float64)
        extent_np=np.asarray(geometry["cell_extent"],dtype=np.float64)
        basis_np=np.asarray(geometry["moment_basis"],dtype=np.float64)
        if xyz_np.ndim!=2 or xyz_np.shape[1]!=3 or len(xyz_np)==0: raise ValueError("invalid native interval xyz")
        if mass_np.shape!=(len(xyz_np),) or extent_np.shape!=(len(xyz_np),3) or basis_np.shape!=(len(xyz_np),10):
            raise ValueError("M72 interval arrays have inconsistent shapes")
        if not np.isfinite(xyz_np).all() or not np.isfinite(mass_np).all() or not np.isfinite(extent_np).all() or not np.isfinite(basis_np).all() or (mass_np<=0).any():
            raise ValueError("nonfinite or nonpositive native interval data")
        side=int(np.asarray(geometry["native_resolution"]))
        if side<c.min_native_resolution: raise ValueError("native geometry resolution below configured minimum")
        p=next(self.parameters()); device=p.device
        xyz=torch.as_tensor(xyz_np.copy(),dtype=torch.float64,device=device)
        mass=torch.as_tensor(mass_np.copy(),dtype=torch.float64,device=device)
        extent=torch.as_tensor(extent_np.copy(),dtype=torch.float64,device=device)
        basis=torch.as_tensor(basis_np.copy(),dtype=torch.float64,device=device)
        fixed=torch.as_tensor(np.asarray(geometry["fixed_moments"],dtype=np.float64).copy(),dtype=torch.float64,device=device)
        design=mass@basis
        original=torch.as_tensor(np.asarray(geometry["original_moments"],dtype=np.float64).copy(),dtype=torch.float64,device=device)
        if fixed.shape!=(10,) or original.shape!=(10,) or not torch.isfinite(fixed).all() or not torch.isfinite(original).all():
            raise ValueError("fixed/original moments must be finite [10]")
        lo,hi=(xyz-extent/2).amin(0),(xyz+extent/2).amax(0)
        length=(hi-lo).max().clamp_min(torch.finfo(torch.float64).eps)
        coord=length**torch.tensor(POWERS,dtype=torch.float64,device=device)
        scale=(fixed[0]+mass.sum())*coord
        if not torch.allclose(fixed+design,original,rtol=1e-8,atol=float(scale.max())*1e-10):
            raise ValueError("M72 original moments do not equal fixed + editable moments")
        absolute=mass@_absolute_basis(xyz,extent)
        if (absolute+1e-10*scale<design.abs()).any(): raise ValueError("absolute moment bound violated")
        stats=torch.cat((original/scale,fixed/scale,design/scale,(lo+hi)/(2*length),(hi-lo)/length)).to(p.dtype)[None]
        desc=torch.cat((basis/coord[None],extent/length),-1).to(p.dtype)
        weights=(mass/mass.sum()).to(p.dtype)
        grid=geometry.get("pcraft_grid")
        if grid is None: grid=conservative_coarse_grid(geometry,c.grid_size)
        grid=torch.as_tensor(np.array(grid,dtype=np.float32,copy=True),device=device).unsqueeze(0)
        if grid.shape!=(1,4,c.grid_size,c.grid_size,c.grid_size): raise ValueError("invalid P-CRAFT conservative grid")
        pino_global,pino_map=self.pino_encoder.forward_with_map(grid)
        context=(self.statistics_context(stats)+self.pino_global(pino_global)).squeeze(0)
        origin=torch.as_tensor(np.asarray(geometry["origin"],dtype=np.float64),device=device)
        span=side*float(geometry["h"])
        normalized=2*(xyz-origin)/span-1
        coords=torch.stack((normalized[:,2],normalized[:,1],normalized[:,0]),-1)
        latent=torch.zeros(w,dtype=p.dtype,device=device)
        for start in range(0,len(desc),chunk):
            stop=min(start+chunk,len(desc))
            sample_grid=coords[start:stop].to(pino_map.dtype).reshape(1,-1,1,1,3)
            args=(desc[start:stop],weights[start:stop],sample_grid,pino_map,context)
            if c.checkpoint_intervals and self.training and torch.is_grad_enabled():
                latent=latent+checkpoint(self._chunk_pool,*args,use_reentrant=False)
            else:
                latent=latent+self._chunk_pool(*args)
        geom_hash=str(geometry.get("geometry_sha256",""))
        if not geom_hash:
            d=sha256()
            for a in (xyz_np,mass_np,basis_np,extent_np): d.update(np.asarray(a,dtype="<f8").tobytes())
            geom_hash=d.hexdigest()
        self.counters["encode_calls"]+=1; self.counters["intervals"]+=len(xyz_np)
        return EncodedNativeGeometry(latent[None],stats,fixed,design,absolute,scale,coord,
            torch.stack((lo/length,hi/length)),geom_hash,side,len(xyz_np),id(self),self._versions())

    def predict(self,cache: EncodedNativeGeometry,q):
        if cache.owner!=id(self) or cache.versions!=self._versions(): raise ValueError("foreign or stale geometry cache")
        q=q.to(device=cache.latent.device,dtype=torch.float64) if isinstance(q,Tensor) else torch.tensor(np.asarray(q,dtype=np.float64),device=cache.latent.device)
        single=q.ndim==1
        if single: q=q[None]
        if q.ndim!=2 or q.shape[1]!=10 or not torch.isfinite(q).all(): raise ValueError("q must be finite [Q,10]")
        maxabs=q.abs().amax(-1,keepdim=True)
        if (maxabs<=0).any(): raise ValueError("zero quadric coefficient vector")
        scaled=(q/maxabs)*cache.coordinate_scales
        unit=scaled/torch.linalg.vector_norm(scaled,dim=-1,keepdim=True)
        latent=cache.latent.expand(len(q),-1); stats=cache.statistics.expand(len(q),-1)
        u=unit.to(latent.dtype)
        odd=(self.query_head(torch.cat((latent,stats,u),-1))-self.query_head(torch.cat((latent,stats,-u),-1))).double()/2
        result=cache.fixed_moments+.5*cache.design_moments+.5*cache.absolute_moments*torch.tanh(odd)
        lower,upper=_quadric_bounds(cache.normalized_bounds,unit)
        result=torch.where((lower>0)[:,None],cache.fixed_moments+cache.design_moments,result)
        result=torch.where((upper<0)[:,None],cache.fixed_moments,result)
        self.counters["queries"]+=len(q)
        return result[0] if single else result

    def moments_and_jacobian(self,cache,q,*,create_graph=False):
        with torch.enable_grad():
            value=q.to(device=cache.latent.device,dtype=torch.float64) if isinstance(q,Tensor) else torch.tensor(np.asarray(q,dtype=np.float64),device=cache.latent.device)
            if not value.requires_grad: value=value.detach().requires_grad_(True)
            single=value.ndim==1
            moments=self.predict(cache,value)
            rows=moments[None] if single else moments
            jac=torch.stack([torch.autograd.grad(rows[:,i].sum(),value,create_graph=create_graph,retain_graph=True)[0] for i in range(10)],dim=-2)
        self.counters["jacobians"]+=1
        return moments,jac


def load_model(path,device="cpu"):
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if payload.get("schema")!=SCHEMA or payload.get("semantics")!=SEMANTICS: raise ValueError("incompatible M80 checkpoint")
    model=PcraftNativeMomentOperator(NativePCRAFTConfig(**payload["config"]))
    model.load_state_dict(payload["model"])
    return model.to(device).eval(),payload
