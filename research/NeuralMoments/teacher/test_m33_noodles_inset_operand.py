from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path
import unittest

from m19_noodles_fda_audit import fraction_signed_ten
from m33_noodles_inset_operand import make_operand


def decode(payload):
    rows = payload.decode('ascii').splitlines()
    nv, nf = map(int, rows[0].split())
    assert len(rows) == 1+nv+nf
    points = []
    for row in rows[1:1+nv]:
        tokens = row.split()
        assert len(tokens)==3 and all('/' in t for t in tokens)
        points.append(tuple(map(Q, tokens)))
    faces = [tuple(map(int,row.split())) for row in rows[1+nv:]]
    assert all(len(f)==3 and min(f)>=0 and max(f)<nv for f in faces)
    return points, faces


class InsetOperandTests(unittest.TestCase):
    def test_fractional_translated_tetra(self):
        v = [(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
        f = [(0,2,1),(0,1,3),(0,3,2),(1,2,3)]
        pkt, report = make_operand(v, f, (1,2,3), Q(2,3))
        points, faces = decode(pkt)
        self.assertEqual(points[0], (Q(1,3), Q(2,3), Q(1)))
        self.assertEqual(points[1], (Q(1), Q(2,3), Q(1)))
        self.assertEqual(faces, f)
        moments = list(map(Q,report['inset_ten']))
        self.assertEqual(moments[:4], [Q(4,81), Q(2,81), Q(10,243), Q(14,243)])
        self.assertEqual(fraction_signed_ten(points,faces),moments)

    def test_checked_packet_bytes_and_exact_ten(self):
        here = Path(__file__).parent
        packet = (here/'M33_NOODLES_EPRIME.txt').read_bytes()
        report = json.loads((here/'M33_NOODLES_EPRIME.json').read_text())
        self.assertEqual(sha256(packet).hexdigest(),report['operand_sha256'])
        pts,faces = decode(packet)
        self.assertEqual((len(pts),len(faces)), (642,1280))
        self.assertEqual(fraction_signed_ten(pts,faces), list(map(Q,report['inset_ten'])))
        self.assertEqual(report['fixed_envelope_layer_volume_fraction'], '271/1000')
        self.assertEqual(report['fixed_envelope_layer_Iz_fraction'], '40951/100000')
        self.assertEqual(report['non_binary64_coordinates'],1698)
