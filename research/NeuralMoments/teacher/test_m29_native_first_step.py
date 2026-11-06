import io
import json
import unittest

from m29_native_first_step import CAP, PrefixComplete, Recorder


def event(iteration=0, outcome='accepted', reason='initial_evaluation_succeeded', i=1, kind='evaluation'):
    return {'type':kind, 'iteration':iteration, 'outcome':outcome,'reason':reason,'evaluation_index':i}


class PrefixChecks(unittest.TestCase):
    def setUp(self):
        self.s = Recorder(None, io.StringIO(), io.StringIO())

    def test_initial_rejection_does_not_suppress_native_restart(self):
        self.s.observe(event(outcome='rejected', reason='initial_multiplier_nonpositive'))
        self.s.observe(event(kind='attempt_terminal'))
        self.s.observe(event())
        self.assertIsNone(self.s.accepted)
        self.assertIsNone(self.s.status)

    def test_rank_event_is_not_an_accepted_optimization_step(self):
        self.s.observe(event(1, reason='regular_system_payload_captured', kind='newton_linear_solve'))
        self.s.observe(event(1,'rejected','topology_transition',i=2))
        self.assertIsNone(self.s.accepted)

    def test_accept_decision_returns_to_cpp_then_stops_before_another_trial(self):
        for reason in ('residual_decreased','topology_transition_relinearized'):
            with self.subTest(reason=reason):
                s = Recorder(None, io.StringIO(), io.StringIO())
                s.calls = 2
                s.observe(event(1, reason=reason, i=2))
                self.assertEqual(s.accepted,2)
                with self.assertRaises(PrefixComplete):
                    s.observe(event(2, kind='newton_linear_solve',i=2))
                self.assertEqual(s.status,'FIRST_TRIAL_ACCEPTED')
                self.assertEqual(s.calls,2)

    def test_no_extra_geometry_after_cap(self):
        self.s.calls = CAP
        with self.assertRaises(PrefixComplete):
            self.s.observe(event(1,'rejected','residual_not_decreased',i=CAP))
        self.assertEqual(self.s.status,'BUDGET_EXHAUSTED')
        with self.assertRaisesRegex(RuntimeError,'UNEXPECTED_NATIVE_PREFIX_CALLBACK'):
            self.s.evaluate([0]*10)

    def test_failed_source_keeps_denominator(self):
        class Geometry:
            call_counts = {}
            def __call__(self,q):
                self.call_counts['source']=1
                raise ValueError('source_failure')
        self.s.geometry = Geometry()
        with self.assertRaisesRegex(ValueError,'source_failure'):
            self.s.evaluate([0]*10)
        row=json.loads(self.s.evaluations.getvalue())
        self.assertEqual(row['index'],1)
        self.assertEqual(row['call_counts_before'],{})
        self.assertEqual(row['call_counts_after'],{'source':1})
        self.assertEqual(row['status'],'exception')
