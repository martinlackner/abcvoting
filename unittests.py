# Unit tests


import unittest


def run_test_instance(unittestinstance, profile, committeesize, tests):
    import rules_approval

    # all rules used?
    for rule in rules_approval.MWRULES:
        unittestinstance.assertTrue(rule in tests.keys())

    for rule in tests.keys():

        output = rules_approval.compute_rule(rule, profile,
                                             committeesize,
                                             resolute=False)
        unittestinstance.assertEqual(
            output, tests[rule], msg=rules_approval.MWRULES[rule] + " failed")
        output = rules_approval.compute_rule(
            rule, profile, committeesize, resolute=True)
        unittestinstance.assertEqual(
            len(output), 1,
            msg=rules_approval.MWRULES[rule] + " failed with resolute=True")
        unittestinstance.assertTrue(
            output[0] in tests[rule],
            msg=rules_approval.MWRULES[rule] + " failed with resolute=True")


class TestApprovalMultiwinner(unittest.TestCase):
    def test_createprofiles(self):
        from preferences import Profile
        from preferences import DichotomousPreferences
        num_cand = 7
        prof = Profile(num_cand)
        self.assertEqual(prof.add_preferences(
            DichotomousPreferences([0, 4, 5])),
            None)
        with self.assertRaises(Exception):
            prof.add_preferences(DichotomousPreferences([num_cand]))
        with self.assertRaises(Exception):
            prof.add_preferences(DichotomousPreferences([-1]))
        self.assertEqual(prof.add_preferences([0, 4, 5]), None)
        with self.assertRaises(Exception):
            prof.add_preferences([0, 4, 5, "1"])
        with self.assertRaises(Exception):
            prof.add_preferences(["1", 0, 4, 5])
        p1 = DichotomousPreferences([0, 4, 5])
        p2 = DichotomousPreferences([1, 2])
        self.assertEqual(prof.add_preferences([p1, p2]), None)
        self.assertTrue(prof.has_unit_weights())
        prof.add_preferences(DichotomousPreferences([0, 4, 5], 2.4))
        self.assertFalse(prof.has_unit_weights())
        self.assertEqual(prof.totalweight(), 6.4)

    def test_mwrules__toofewcandidates(self):
        from preferences import Profile
        import rules_approval
        profile = Profile(5)
        committeesize = 4
        preflist = [[0, 1, 2], [1], [1, 2], [0]]
        profile.add_preferences(preflist)

        for rule in rules_approval.MWRULES.keys():
            with self.assertRaises(Exception):
                rules_approval.compute_rule(rule, profile, committeesize)
            with self.assertRaises(Exception):
                rules_approval.compute_rule(rule, profile,
                                            committeesize, resolute=True)

    def test_mwrules_weightsconsidered(self):
        from preferences import Profile
        from preferences import DichotomousPreferences
        import rules_approval

        self.longMessage = True

        profile = Profile(3)
        profile.add_preferences(DichotomousPreferences([0]))
        profile.add_preferences(DichotomousPreferences([0]))
        profile.add_preferences(DichotomousPreferences([1], 5))
        profile.add_preferences(DichotomousPreferences([0]))
        committeesize = 1

        for rule in rules_approval.MWRULES.keys():
            if "monroe" in rule or "rule-x" in rule \
                    or rule == "phragmen-enestroem":
                # Monroe, rule x and enestroem only work with
                # unit weights:
                continue
            result = rules_approval.compute_rule(rule, profile, committeesize)
            self.assertTrue([1] in result,
                            msg=rule + " failed"+str(result))

    def test_mwrules_correct_simple(self):
        from preferences import Profile
        import rules_approval

        self.longMessage = True

        profile = Profile(4)
        profile.add_preferences([[0], [1], [2], [3]])
        committeesize = 2

        for rule in rules_approval.MWRULES.keys():
            if rule == "greedy-monroe":   # always returns one committee
                continue
            self.assertEqual(len(rules_approval.compute_rule(rule, profile,
                                                             committeesize)),
                             6, msg=rule + " failed")

        for rule in rules_approval.MWRULES.keys():
            self.assertEqual(len(rules_approval.compute_rule(rule, profile,
                                                             committeesize,
                                                             resolute=True)),
                             1, msg=rule + " failed with resolute=True")

    def test_monroe_indivisible(self):
        from preferences import Profile
        import rules_approval

        self.longMessage = True

        profile = Profile(4)
        profile.add_preferences([[0], [0], [0], [1, 2], [1, 2], [1], [3]])
        committeesize = 3

        for ilp in [True, False]:
            # max Monroe score is 6 (even for committee [0, 1, 3])
            self.assertEqual(
                rules_approval.compute_monroe(profile, committeesize,
                                              ilp=ilp, resolute=False),
                [[0, 1, 2], [0, 1, 3], [0, 2, 3]])

    # this test shows that tiebreaking is not (yet)
    # implemented for opt-Phragmen
    def test_optphrag_notiebreaking(self):
        from preferences import Profile
        from rules_approval import compute_rule

        self.longMessage = True

        profile = Profile(6)
        profile.add_preferences([[0], [0], [1, 3], [1, 3], [1, 4],
                                 [2, 4], [2, 5], [2, 5]])
        committeesize = 3

        self.assertEqual(
                len(compute_rule("optphrag", profile, committeesize,
                                 resolute=False)),
                12)

    def test_mwrules_correct_advanced_1(self):

        from preferences import Profile
        self.longMessage = True
        committeesize = 4

        profile = Profile(6)
        preflist = [[0, 4, 5], [0], [1, 4, 5], [1],
                    [2, 4, 5], [2], [3, 4, 5], [3]]
        profile.add_preferences(preflist)

        tests1 = {
            "seqpav": [[0, 1, 4, 5], [0, 2, 4, 5], [0, 3, 4, 5],
                       [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "av": [[0, 1, 4, 5], [0, 2, 4, 5], [0, 3, 4, 5],
                   [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "sav": [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5], [0, 1, 3, 4],
                    [0, 1, 3, 5], [0, 1, 4, 5], [0, 2, 3, 4], [0, 2, 3, 5],
                    [0, 2, 4, 5], [0, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3, 5],
                    [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "pav-ilp": [[0, 1, 4, 5], [0, 2, 4, 5], [0, 3, 4, 5],
                        [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "pav-noilp": [[0, 1, 4, 5], [0, 2, 4, 5], [0, 3, 4, 5],
                          [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "revseqpav": [[0, 1, 4, 5], [0, 2, 4, 5], [0, 3, 4, 5],
                          [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "minimaxav-noilp": [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5],
                                [0, 1, 3, 4], [0, 1, 3, 5], [0, 1, 4, 5],
                                [0, 2, 3, 4], [0, 2, 3, 5], [0, 2, 4, 5],
                                [0, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3, 5],
                                [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "minimaxav-ilp": [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5],
                              [0, 1, 3, 4], [0, 1, 3, 5], [0, 1, 4, 5],
                              [0, 2, 3, 4], [0, 2, 3, 5], [0, 2, 4, 5],
                              [0, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3, 5],
                              [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "phrag": [[0, 1, 4, 5], [0, 2, 4, 5], [0, 3, 4, 5],
                      [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "optphrag": [[0, 1, 2, 3]],
            "cc-ilp": [[0, 1, 2, 3]],
            "cc-noilp": [[0, 1, 2, 3]],
            "seqcc": [[0, 1, 2, 4], [0, 1, 2, 5], [0, 1, 3, 4], [0, 1, 3, 5],
                      [0, 2, 3, 4], [0, 2, 3, 5], [1, 2, 3, 4], [1, 2, 3, 5]],
            "revseqcc": [[0, 1, 2, 3]],
            "monroe-ilp": [[0, 1, 2, 3]],
            "monroe-noilp": [[0, 1, 2, 3]],
            "greedy-monroe": [[0, 2, 3, 4]],
            "slav-ilp": [[0, 1, 2, 3],
                         [0, 1, 2, 4], [0, 1, 2, 5],
                         [0, 1, 3, 4], [0, 1, 3, 5],
                         [0, 2, 3, 4], [0, 2, 3, 5],
                         [1, 2, 3, 4], [1, 2, 3, 5]],
            "slav-noilp": [[0, 1, 2, 3],
                           [0, 1, 2, 4], [0, 1, 2, 5],
                           [0, 1, 3, 4], [0, 1, 3, 5],
                           [0, 2, 3, 4], [0, 2, 3, 5],
                           [1, 2, 3, 4], [1, 2, 3, 5]],
            "seqslav": [[0, 1, 2, 4], [0, 1, 2, 5],
                        [0, 1, 3, 4], [0, 1, 3, 5],
                        [0, 2, 3, 4], [0, 2, 3, 5],
                        [1, 2, 3, 4], [1, 2, 3, 5]],
            "rule-x": [[0, 1, 4, 5], [0, 2, 4, 5],
                       [0, 3, 4, 5], [1, 2, 4, 5],
                       [1, 3, 4, 5], [2, 3, 4, 5]],
            "phragmen-enestroem": [[0, 1, 4, 5], [0, 2, 4, 5],
                                   [0, 3, 4, 5], [1, 2, 4, 5],
                                   [1, 3, 4, 5], [2, 3, 4, 5]],
        }

        run_test_instance(self, profile, committeesize, tests1)

        # and now with reversed preflist
        preflist.reverse()
        for p in preflist:
            p.reverse()
        profile = Profile(6)
        profile.add_preferences(preflist)

        run_test_instance(self, profile, committeesize, tests1)

    def test_mwrules_correct_advanced_2(self):

        from preferences import Profile
        self.longMessage = True

        # and another profile
        profile = Profile(5)
        committeesize = 3
        preflist = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2],
                    [0, 1, 2], [0, 1], [3, 4], [3, 4], [3]]
        profile.add_preferences(preflist)

        tests2 = {
            "seqpav": [[0, 1, 3]],
            "av": [[0, 1, 2]],
            "sav": [[0, 1, 3]],
            "pav-ilp": [[0, 1, 3]],
            "pav-noilp": [[0, 1, 3]],
            "revseqpav": [[0, 1, 3]],
            "minimaxav-noilp": [[0, 1, 3], [0, 2, 3], [1, 2, 3]],
            "minimaxav-ilp": [[0, 1, 3], [0, 2, 3], [1, 2, 3]],
            "phrag": [[0, 1, 3]],
            "optphrag": [[0, 1, 3], [0, 2, 3], [1, 2, 3]],
            "cc-ilp": [[0, 1, 3], [0, 2, 3], [0, 3, 4],
                       [1, 2, 3], [1, 3, 4]],
            "cc-noilp": [[0, 1, 3], [0, 2, 3], [0, 3, 4],
                         [1, 2, 3], [1, 3, 4]],
            "seqcc": [[0, 1, 3], [0, 2, 3], [0, 3, 4],
                      [1, 2, 3], [1, 3, 4]],
            "revseqcc": [[0, 1, 3], [0, 2, 3], [0, 3, 4],
                         [1, 2, 3], [1, 3, 4]],
            "monroe-ilp": [[0, 1, 3], [0, 2, 3], [1, 2, 3]],
            "monroe-noilp": [[0, 1, 3], [0, 2, 3], [1, 2, 3]],
            "greedy-monroe": [[0, 1, 3]],
            "seqslav": [[0, 1, 3]],
            "slav-ilp": [[0, 1, 3]],
            "slav-noilp": [[0, 1, 3]],
            "rule-x": [[0, 1, 3]],
            "phragmen-enestroem": [[0, 1, 3]],
        }

        run_test_instance(self, profile, committeesize, tests2)

    def test_mwrules_correct_advanced_3(self):

        from preferences import Profile
        self.longMessage = True

        # and a third profile
        profile = Profile(6)
        committeesize = 4
        preflist = [[0, 3, 4, 5], [1, 2], [0, 2, 5], [2],
                    [0, 1, 2, 3, 4], [0, 3, 4], [0, 2, 4], [0, 1]]
        profile.add_preferences(preflist)

        tests3 = {
            "seqpav": [[0, 1, 2, 4]],
            "av": [[0, 1, 2, 4], [0, 2, 3, 4]],
            "sav": [[0, 1, 2, 4]],
            "pav-ilp": [[0, 1, 2, 4]],
            "pav-noilp": [[0, 1, 2, 4]],
            "revseqpav": [[0, 1, 2, 4]],
            "minimaxav-noilp": [[0, 1, 2, 3], [0, 1, 2, 4],
                                [0, 2, 3, 4], [0, 2, 3, 5],
                                [0, 2, 4, 5]],
            "minimaxav-ilp": [[0, 1, 2, 3], [0, 1, 2, 4],
                              [0, 2, 3, 4], [0, 2, 3, 5],
                              [0, 2, 4, 5]],
            "phrag": [[0, 1, 2, 4]],
            "optphrag": [[0, 1, 2, 3], [0, 1, 2, 4],
                         [0, 1, 2, 5], [0, 2, 3, 4],
                         [0, 2, 3, 5], [0, 2, 4, 5],
                         [1, 2, 3, 4], [1, 2, 3, 5],
                         [1, 2, 4, 5]],
            "cc-ilp": [[0, 1, 2, 3], [0, 1, 2, 4],
                       [0, 1, 2, 5], [0, 2, 3, 4],
                       [0, 2, 3, 5], [0, 2, 4, 5],
                       [1, 2, 3, 4], [1, 2, 3, 5],
                       [1, 2, 4, 5]],
            "cc-noilp": [[0, 1, 2, 3], [0, 1, 2, 4],
                         [0, 1, 2, 5], [0, 2, 3, 4],
                         [0, 2, 3, 5], [0, 2, 4, 5],
                         [1, 2, 3, 4], [1, 2, 3, 5],
                         [1, 2, 4, 5]],
            "seqcc": [[0, 1, 2, 3], [0, 1, 2, 4],
                      [0, 1, 2, 5], [0, 2, 3, 4],
                      [0, 2, 3, 5], [0, 2, 4, 5]],
            "revseqcc": [[0, 1, 2, 3], [0, 1, 2, 4],
                         [0, 1, 2, 5], [0, 2, 3, 4],
                         [0, 2, 3, 5], [0, 2, 4, 5],
                         [1, 2, 3, 4], [1, 2, 3, 5],
                         [1, 2, 4, 5]],
            "monroe-ilp": [[0, 1, 2, 3], [0, 1, 2, 4],
                           [0, 1, 2, 5], [0, 2, 3, 4],
                           [0, 2, 3, 5], [0, 2, 4, 5],
                           [1, 2, 3, 4], [1, 2, 3, 5],
                           [1, 2, 4, 5]],
            "monroe-noilp": [[0, 1, 2, 3], [0, 1, 2, 4],
                             [0, 1, 2, 5], [0, 2, 3, 4],
                             [0, 2, 3, 5], [0, 2, 4, 5],
                             [1, 2, 3, 4], [1, 2, 3, 5],
                             [1, 2, 4, 5]],
            "greedy-monroe": [[0, 1, 2, 3]],
            "seqslav": [[0, 1, 2, 4]],
            "slav-ilp": [[0, 1, 2, 4]],
            "slav-noilp": [[0, 1, 2, 4]],
            "rule-x": [[0, 1, 2, 4]],
            "phragmen-enestroem": [[0, 1, 2, 4]],
        }

        run_test_instance(self, profile, committeesize, tests3)

    def test_monroescore(self):
        from preferences import Profile
        from score_functions import monroescore_flowbased, monroescore_matching
        self.longMessage = True

        # and a third profile
        profile = Profile(6)
        preflist = [[0, 1], [1], [1, 3], [4], [2], [1, 5, 3]]
        profile.add_preferences(preflist)

        self.assertEqual(monroescore_flowbased(profile, [1, 3, 2]), 5)
        self.assertEqual(monroescore_matching(profile, [1, 3, 2]), 5)
        self.assertEqual(monroescore_flowbased(profile, [2, 1, 5]), 4)
        self.assertEqual(monroescore_matching(profile, [2, 1, 5]), 4)
        self.assertEqual(monroescore_flowbased(profile, [2, 4, 5]), 3)
        self.assertEqual(monroescore_matching(profile, [2, 5, 4]), 3)


if __name__ == '__main__':
    unittest.main()
