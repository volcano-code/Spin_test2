"""Analytic box reference for all moments, all 100 sectional derivatives and the stencil."""
import os
from fractions import Fraction as Q
from hashlib import sha256
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

from plane_report import measure

POWERS=((0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(0,2,0),(0,0,2),(0,1,1),(1,0,1),(1,1,0))
CUBE=b"""8 12
0 0 0
1 0 0
1 1 0
0 1 0
0 0 1
1 0 1
1 1 1
0 1 1
0 2 1
0 3 2
4 5 6
4 6 7
0 1 5
0 5 4
1 2 6
1 6 5
2 3 7
2 7 6
3 0 4
3 4 7
"""


class BoxPlaneReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=TemporaryDirectory()
        cls.binary=Path(cls.tmp.name)/"probe"
        subprocess.run([os.environ.get("CXX","c++"),"-O2","-std=c++17",
                        str(Path(__file__).with_name("exact_plane_probe.cpp")),
                        "-lgmpxx","-lgmp","-o",str(cls.binary)],check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_entire_box_cap_and_sectional_gram(self):
        file=Path(self.tmp.name)/"cube.txt";file.write_bytes(CUBE)
        result=measure(file,sha256(CUBE).hexdigest(),self.binary)
        height=Q(3,4)
        def encloses(ab,truth):
            self.assertLessEqual(int(ab[0]),truth*2**160)
            self.assertGreaterEqual(int(ab[1]),truth*2**160)
        for i,(x,y,z) in enumerate(POWERS):
            cap=(1-height**(z+1))/((x+1)*(y+1)*(z+1))
            encloses(result["raw_10_enclosure_numerators"][i],cap)
            for j,(a,b,c) in enumerate(POWERS):
                deriv=height**(z+c)*Q(5,4)/((x+a+1)*(y+b+1))*(2 if j>=7 else 1)
                encloses(result["J_10x10_enclosure_numerators"][i][j],deriv)
        self.assertEqual(len(result["directional_checks"]),6)

    def test_vertex_on_cut_is_rejected(self):
        file=Path(self.tmp.name)/"event.txt";file.write_bytes(CUBE.replace(b"0 0 1\n",b"0 0 3/4\n",1))
        with self.assertRaises(subprocess.CalledProcessError):
            measure(file,sha256(file.read_bytes()).hexdigest(),self.binary)
