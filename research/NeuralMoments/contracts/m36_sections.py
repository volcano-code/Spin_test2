"""Render two planar sections of the original and exported rational Noodles meshes."""
import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path


def contours(data, axis, coordinate, horizontal, vertical):
    # Use index edges for connectivity; project to display precision only after the section is assembled.
    vertices = [[Fraction(x) for x in p] for p in data["vertices"]]
    height = [p[axis]-coordinate for p in vertices]
    if 0 in height:
        raise ValueError("section hits a mesh vertex")
    neighbors = defaultdict(list)
    for a,b,c in data["faces"]:
        crossed = [tuple(sorted((i,j))) for i,j in ((a,b),(b,c),(c,a)) if (height[i]>0)!=(height[j]>0)]
        if crossed:
            u,v=crossed
            neighbors[u].append(v)
            neighbors[v].append(u)
    if not neighbors or any(len(v)!=2 for v in neighbors.values()):
        raise ValueError("section is empty or open")
    def project(edge):
        i,j=edge
        t=height[i]/(height[i]-height[j])
        return [float(vertices[i][k]+t*(vertices[j][k]-vertices[i][k])) for k in (horizontal,vertical)]
    remaining=set(neighbors)
    paths=[]
    while remaining:
        start=min(remaining)
        current,prev=start,None
        polygon=[]
        while current in remaining:
            remaining.remove(current)
            polygon.append(project(current))
            a,b=neighbors[current]
            current,prev=(b if a==prev else a),current
        if current!=start:raise ValueError("section loop mismatch")
        paths.append(polygon)
    return paths


def render(original, saved):
    svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 450">',
         '<rect width="960" height="450" fill="#fff"/>',
         '<g font-family="system-ui, sans-serif" fill="#172f3b">',
         '<text x="34" y="41" font-size="26" font-weight="600">Noodles · sections of the saved material</text>',
         '<rect x="36" y="60" width="17" height="12" rx="2" fill="#28596b"/>',
         '<text x="61" y="71" font-size="14">Retained material</text>',
         '<rect x="224" y="60" width="17" height="12" rx="2" fill="#e5a664"/>',
         '<text x="249" y="71" font-size="14">Removed material</text>']
    # Planes in the registered model frame, chosen away from mesh vertices.
    for index,(axis,cut,h,v,title) in enumerate(((2,Fraction(90013,100000),0,1,"Horizontal section · z = 0.90013"),
                                                  (1,Fraction(3103,100000),0,2,"Vertical section · y = 0.03103"))):
        panels=[contours(mesh,axis,cut,h,v) for mesh in (original,saved)]
        xs=[p[0] for loop in panels[0] for p in loop];ys=[p[1] for loop in panels[0] for p in loop]
        lo,hi=min(xs),max(xs); bot,top=min(ys),max(ys)
        scale=min(405/(hi-lo),255/(top-bot))
        cx=240+480*index;cy=234
        svg.append(f'<rect x="{22+480*index}" y="94" width="436" height="300" rx="7" fill="#f6f7f7"/>')
        for paths,color in zip(panels,("#e5a664","#28596b")):
            d=[]
            for loop in paths:
                xy=[f"{cx+scale*(x-(lo+hi)/2):.3f},{cy-scale*(y-(top+bot)/2):.3f}" for x,y in loop]
                d.append("M"+"L".join(xy)+"Z")
            svg.append(f'<path fill="{color}" fill-rule="evenodd" d="{" ".join(d)}"/>')
        svg.append(f'<text x="{cx}" y="421" text-anchor="middle" font-size="17">{title}</text>')
    return "\n".join(svg+["</g></svg>"])+"\n"


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("original","saved","output"):p.add_argument(name,type=Path)
    a=p.parse_args()
    a.output.write_text(render(json.loads(a.original.read_bytes()),json.loads(a.saved.read_bytes())))
