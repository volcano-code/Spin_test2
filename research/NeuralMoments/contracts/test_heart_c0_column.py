from collections import Counter
from fractions import Fraction as F

import pytest

from .heart_c0_column import check_stencil, geometry_comparison, load_capture, weights_at_zero
from .real_h8f_81_m9 import oriented_face_key


@pytest.mark.parametrize('xs', [(-3,0,5),(-4,-2,0,3,5)])
def test_exact_weights_reproduce_order_on_unequal_binary_abscissae(xs):
    xs=[F(x,16) for x in xs]
    w=weights_at_zero(xs)
    for k in range(len(xs)):
        assert sum(a*x**k for a,x in zip(w,xs)) == (k==1)


def test_central_stencils_detect_an_incorrect_component():
    x=[F(-1,128),F(0),F(1,128)]
    slopes=list(map(F,range(10)))
    points=[(v,[F(17)+s*v+3*v*v for s in slopes]) for v in x]
    report=check_stencil(points,slopes)
    assert report['passed'] and report['finite_difference']==list(map(float,slopes))
    wrong=slopes.copy(); wrong[-1]+=1
    assert not check_stencil(points,wrong)['passed']


def test_duplicate_abscissae_rejected():
    with pytest.raises(ValueError, match='duplicate coefficient'):
        weights_at_zero([F(-1),F(0),F(0)])


def test_sampled_identity_requires_vertex_ownership_source_faces_and_orientation():
    def mesh(signatures, faces, source='same'):
        return set(signatures), Counter(oriented_face_key(f,signatures) for f in faces), source
    base=mesh(('a','b','c'),[[0,1,2]])
    assert geometry_comparison(base,mesh(('b','c','a'),[[2,0,1]]))['sampled_geometry_equal']
    for trial in (mesh(('a','b','c'),[[2,1,0]]),mesh(('a','b','d'),[[0,1,2]]),
                  mesh(('a','b','c'),[[0,1,2],[0,1,2]]),mesh(('a','b','c'),[[0,1,2]],'changed')):
        assert not geometry_comparison(base,trial)['sampled_geometry_equal']


def test_bad_capture_hash_refused_before_reading_sources(tmp_path):
    (tmp_path/'capture.json').write_text('{}')
    with pytest.raises(ValueError, match='capture digest'):
        load_capture(tmp_path, 'different', 'wrong')
