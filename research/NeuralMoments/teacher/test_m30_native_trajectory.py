import io
import unittest
from unittest.mock import patch

from m29_native_first_step import PrefixComplete
from m30_native_trajectory import MAX_EVALUATIONS, Trajectory, terminal_fields
from test_m29_native_first_step import event


class TrajectoryChecks(unittest.TestCase):
    def setUp(self):
        self.s = Trajectory(None,io.StringIO(),io.StringIO())

    def test_carries_multiple_native_accepts_and_rejects(self):
        self.s.observe(event())
        self.s.observe(event(1,i=2,reason='topology_transition_relinearized'))
        self.s.observe(event(2,i=3,outcome='rejected',reason='topology_transition'))
        self.s.observe(event(2,i=4,reason='residual_decreased'))
        self.assertTrue(self.s.may_evaluate())
        self.assertEqual(self.s.accepted,2)
        self.assertEqual(self.s.indices,[2,4])

    def test_final_accepted_point_can_commit_but_never_probes_beyond_cap(self):
        self.s.calls=MAX_EVALUATIONS
        self.s.observe(event(1,i=MAX_EVALUATIONS,reason='residual_decreased'))
        self.s.observe(event(1,i=MAX_EVALUATIONS,kind='attempt_terminal',reason='converged'))
        self.assertFalse(self.s.may_evaluate())
        self.assertIsNone(self.s.status)
        with self.assertRaises(PrefixComplete):
            self.s.observe(event(2,i=MAX_EVALUATIONS,kind='newton_linear_solve'))
        self.assertEqual(self.s.status,'BUDGET_EXHAUSTED')

    def test_time_budget_and_no_spurious_terminal(self):
        with patch('m30_native_trajectory.time.monotonic',return_value=self.s.started+691):
            with self.assertRaises(PrefixComplete):
                self.s.observe(event(2,kind='newton_linear_solve'))
        self.assertEqual(self.s.status,'TIME_BUDGET_EXHAUSTED')
        self.assertIsNone(terminal_fields(None))
