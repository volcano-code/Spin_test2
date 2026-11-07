"""Closed-form and degeneracy regressions. Run with Python's unittest."""
import unittest
import numpy as np
from line_operator import B, basis, evaluate
from native_flux import flux, keys
from ray_bank import crossings, material_rays


def cube(extent, center):
    v=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])*extent+center
    f=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    return v,f


class Lines(unittest.TestCase):
    def setUp(self):
        self.bank=np.array([[0.,.1,.3]]), np.array([-1.]), np.array([1.]), 1.
        self.q=np.zeros(10)

    def test_plane_scale_and_J(self):
        q=self.q
        q[[0,1]]=[-.2,1]
        m,j,g=evaluate(self.bank,0,q)
        x=np.array([[.2,.1,.3]])
        np.testing.assert_allclose(j,basis(x).T@(basis(x)*B),atol=1e-14)
        np.testing.assert_allclose(m[[0,1,4]],[.8,.48,(1-.2**3)/3],atol=1e-14)
        np.testing.assert_allclose(j@q,0,atol=1e-14)
        m2,j2,_=evaluate(self.bank,0,3*q)
        np.testing.assert_allclose(j/3,j2)
        np.testing.assert_allclose(m,m2)
        self.assertTrue(g['derivative_supported'])

    def test_multiple_intervals_and_pair_additivity(self):
        q=self.q
        q[[0,4]]=[-.25,1.]
        m,j,_=evaluate(self.bank,0,q)
        np.testing.assert_allclose(m[[0,1,4]], [1,0,2*(1-.5**3)/3],atol=1e-14)
        d=np.arange(10)*.1+.2
        h=1.e-5
        np.testing.assert_allclose((evaluate(self.bank,0,q+h*d)[0]-evaluate(self.bank,0,q-h*d)[0])/(2*h),j@d,rtol=1e-8,atol=1e-9)
        bank=(np.repeat(self.bank[0],2,axis=0),np.array([-1.,.2]),np.array([-.2,1.]),1.)
        m2,*_=evaluate(bank,0,q)
        np.testing.assert_allclose(m,m2,atol=1e-14)

    def test_tangent_endpoint_parallel(self):
        for idx,values in [([4],[1.]),([0,1],[-1.,1.]),([0,2],[-.1,1.])]:
            q=self.q.copy()
            q[idx]=values
            m,_,diag=evaluate(self.bank,0,q)
            self.assertFalse(diag['derivative_supported'])
            if idx==[4]: self.assertAlmostEqual(m[0],2.)
        q=self.q.copy()
        q[2]=1
        m,_,g=evaluate(self.bank,0,q)
        self.assertFalse(g['derivative_supported'])
        self.assertAlmostEqual(m[0],2.)

    def test_thin_interval(self):
        q=self.q
        q[0]=1.
        m,_,_=evaluate((self.bank[0],np.array([.61]),np.array([.61001]),1.),0,q)
        self.assertAlmostEqual(m[0],.00001,places=14)

    def test_projected_native_translation(self):
        design,df=cube(.9,[0,0,1.])
        v,f=cube(.5,[0,0,1.])
        vel=np.zeros((len(v),3,10))
        vel[:,0,0]=1.
        src={'vertices_tip':v,'faces':f,'native_jacobian':vel}
        j=np.zeros((10,10))
        for axis in range(3):
            box=material_rays(v,f,axis,32)
            self.assertAlmostEqual(np.sum(box[2]-box[1])*box[3],1.)
            mat=keys(*crossings(design,df,axis,32)[:2])
            j+=flux(axis,32,mat,src)[0]
        np.testing.assert_allclose(j[:,0],[0,1,0,0,0,0,0,0,1,0],atol=5e-8)
        np.testing.assert_allclose(j[:,1:],0,atol=1e-14)


if __name__=='__main__':
    unittest.main()
