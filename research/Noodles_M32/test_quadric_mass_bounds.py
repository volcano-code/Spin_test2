import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from fractions import Fraction as Q

from cylinder_enclosure import ring
from search_quadric_mass import evaluator


class TestQuadrics(unittest.TestCase):
    def test_bounds_on_actual_outputs(self):
        result=json.loads(Path(__file__).with_name("M36_QUADRIC_MASS.json").read_bytes())
        upper=Q(result["arbitrary_subset_saving_upper"])
        for name in ("cone","cylinder"):
            case=result[name]
            true=case["true_quadric"]; balanced=case["balanced_inner_polyhedron"]
            self.assertGreater(Q(true["inertia_retention"][0]),Q(9,10))
            self.assertLess(Q(true["dominant_axis_angle_sine_upper"]),Q(1,1000))
            self.assertLess(Q(true["fixed_axis_lateral_centroid_L1_upper"]),Q(1,1000))
            lower=Q(balanced["removed_mass_fraction"][0])
            self.assertGreater(lower,Q(265,1000));self.assertLess(lower,upper)
            self.assertLess(upper-lower,Q(3,10000))
            self.assertEqual(balanced["raw_balance_residuals"],[["0","0"]]*4)
        self.assertLess(Q(result["cone"]["true_quadric"]["height_change_model_units"][1]),0)
        self.assertGreater(Q(result["cylinder"]["true_quadric"]["height_change_model_units"][0]),0)

    def test_nested_exact_squares(self):
        inner,outer=ring(False,4),ring(True,4)
        self.assertTrue(all(a[0]*a[0]+a[1]*a[1]<b[0]*b[0]+b[1]*b[1] for a,b in zip(inner,outer)))

    def test_floating_proposal_against_unclipped_polygonal_prisms(self):
        # z in [0,1]. Containing square cross-sections, caps plus four side faces.
        v=np.array([[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0],[-2,-2,1],[2,-2,1],[2,2,1],[-2,2,1]])
        f=[[0,2,1],[0,3,2],[4,5,6],[4,6,7]]
        for i in range(4):
            j=(i+1)%4;f.extend([[i,j,j+4],[i,j+4,i+4]])
        with tempfile.TemporaryDirectory() as path:
            run=evaluator(Path(path), v[np.array(f)])
            n=32;r=.4;area=n*math.sin(2*math.pi/n)/2
            for slope in (0.,.04):
                raw=run(n,[r,0.,0.,0.,0.,slope]);base=r-.9*slope
                volume=area*(base*base+base*slope+slope*slope/3)
                zmoment=area*(base*base/2+2*base*slope/3+slope*slope/4)
                self.assertAlmostEqual(raw[0],volume,places=12)
                self.assertAlmostEqual(raw[3],zmoment,places=12)
                np.testing.assert_allclose(raw[[1,2,7,8]],0,atol=1e-12)


if __name__=="__main__":unittest.main()
