import unittest
from fractions import Fraction as Q

from cell_mass_lp import TAU, affine_interval, coefficients, solve


class CellMassTests(unittest.TestCase):
    def test_signed_interval(self):
        self.assertEqual(affine_interval([Q(-3),Q(2)],[(Q(-2),Q(1)),(Q(4),Q(5))]),(Q(5),Q(16)))

    def test_known_free_volumes_with_both_inertia_regimes(self):
        all_centers=[(-8,0,-5),(-8,0,5),(8,0,-5),(0,8,-5),(0,-8,5)]
        width=Q(1,32)
        for active in [True,False]:
            centers=all_centers if active else all_centers[1:]
            target=[sum(col)*width**3/2 for col in zip(*(coefficients(c,width) for c in centers))]
            m=[Q(0)]*10
            m[0]=Q(1)
            for j,v in zip([1,2,8,7],target[:4]):
                m[j]=v
            m[4]=target[4]/(1-TAU) * (1 if active else 2)
            sol=solve(centers,width,[(x,x) for x in m])
            self.assertTrue(sol['optimal_in_this_finite_family'])
            self.assertEqual(sol['inertia_active'],active)
            self.assertEqual(sol['basic_volume_intervals'],[[str(width**3/2)]*2]*len(centers))
            self.assertEqual(list(map(Q,sol["retained_mass_ratio"])),[1-len(centers)*width**3/2]*2)
            self.assertEqual(list(map(Q,sol["retained_inertia_ratio"])),[TAU if active else (1+TAU)/2]*2)
