"""Regenerate the four exact polyhedral crossings and integrate their ten signed moments."""
import argparse
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from cylinder_enclosure import packet, CYLINDER, CONE
from quadric_mass_bounds import check_hash, D_PRIME


def generate(design, boolean, w1, output):
    raw=check_hash(design,D_PRIME)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as folder:
        source=Path(folder)/"input.txt"
        for name,shape in (("",CYLINDER),("cone-",CONE)):
            for side in ("inner","outer"):
                key=name+side
                poly=packet(side=="outer",shape=shape)
                (output/(key+".txt")).write_bytes(poly)
                source.write_bytes(raw+poly)
                result=output/(key+".json")
                subprocess.run([str(boolean),str(source),str(result),"intersection"],check=True)
                with (output/(key+".bounds")).open("wb") as stream:
                    subprocess.run([sys.executable,str(w1),str(result)],stdout=stream,check=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("design","boolean","w1","output"):p.add_argument(key,type=Path)
    a=p.parse_args();generate(a.design,a.boolean,a.w1,a.output)
