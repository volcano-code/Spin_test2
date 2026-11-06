import unittest
from fractions import Fraction

from m34_noodles_partition import balanced_unions, packet


class PartitionConstructionTests(unittest.TestCase):
    def test_odd_balanced_recipe_preserves_every_leaf(self):
        leaves = [f'part-{j}' for j in range(93)]
        active = {k: [k] for k in leaves}
        count = 0
        for target, a, b, op in balanced_unions(leaves):
            self.assertEqual(op, 'union')
            self.assertNotIn(target, active)
            active[target] = active.pop(a) + active.pop(b)
            count += 1
        self.assertEqual(count, 92)
        self.assertEqual(active, {'F': leaves})

    def test_native_integer_tokens_and_exact_rational_passthrough(self):
        vertices = [(0.0,0.25,Fraction(-1,3)), ('0','-1234567890123456789/47','1/2')]
        self.assertEqual(packet(vertices, []),
                         b'2 0\n0/1 1/4 -1/3\n0/1 -1234567890123456789/47 1/2\n')

    def test_empty_native_witness_must_match_expected_and_cache(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from m34_noodles_partition import run_operations
        with TemporaryDirectory() as folder:
            d = Path(folder)
            tool = d/'test-tool'
            tool.write_text('mock fixture')
            for n in ['A','B']:
                (d/(n+'.txt')).write_text('test operand '+n)
            def execute(cmd, check):
                Path(cmd[2]).write_text('{"succeeded":true,"operation":"intersection","faces":[],"vertices":[],"birth_faces":[]}')
            op = [('out','A','B','intersection')]
            with patch('m34_noodles_partition.subprocess.run', side_effect=execute):
                with self.assertRaisesRegex(ValueError, 'emptiness'):
                    run_operations(op,tool,d)
                rec = run_operations(op,tool,d,expected_empty={'out'})[0]
                self.assertEqual((rec['faces'],rec['vertices']), (0,0))
                (d/'out.txt').write_text('corrupt')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    run_operations(op,tool,d,expected_empty={'out'})

    def test_missing_target_receipts_rejected(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from m34_noodles_partition import verify
        with TemporaryDirectory() as tmp:
            d=Path(tmp)
            (d/'construction.json').write_text('{"schema":"m34-exact-cgal-partition-construction-v1","operations":[]}')
            with self.assertRaisesRegex(ValueError,'incomplete construction'):
                verify(d/'not-invoked',d)
