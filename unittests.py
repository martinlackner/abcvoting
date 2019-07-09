# Unit tests

# Author: Martin Lackner


import unittest


class TestApprovalMultiwinner(unittest.TestCase):
    def test_createprofiles(self):
        from preferences import Profile
        from preferences import DichotomousPreferences
        num_cand = 7
        prof = Profile(num_cand)
        self.assertEqual(prof.add_preferences(DichotomousPreferences([0, 4, 5])), None)
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
                rules_approval.compute_rule(rule, profile, committeesize, resolute=True)

    def test_mwrules_weightsconsidered(self):
        from preferences import Profile
        from preferences import DichotomousPreferences
        import rules_approval

        self.longMessage = True

        profile = Profile(3)
        profile.add_preferences(DichotomousPreferences([0], 3))
        profile.add_preferences(DichotomousPreferences([1], 3))
        profile.add_preferences(DichotomousPreferences([0]))
        committeesize = 2

        for rule in rules_approval.MWRULES.keys():
            if rule[:6] == "monroe":
                continue  # Monroe only works with unit weights
            self.assertEqual(rules_approval.compute_rule(rule, profile, committeesize),
                             [[0, 1]],
                             msg=rule + " failed")

    def test_mwrules_correct_simple(self):
        from preferences import Profile
        import rules_approval

        self.longMessage = True

        profile = Profile(4)
        profile.add_preferences([[0], [1], [2], [3]])
        committeesize = 2

        for rule in rules_approval.MWRULES.keys():
            self.assertEqual(len(rules_approval.compute_rule(rule, profile, committeesize)),
                             6,
                             msg=rule + " failed")

        for rule in rules_approval.MWRULES.keys():
            self.assertEqual(len(rules_approval.compute_rule(rule, profile, committeesize, resolute=True)),
                             1,
                             msg=rule + " with resolute=True")

    def test_mwrules_correct_advanced(self):
        def runmwtests(tests, committeesize):
            for rule in tests.keys():
                self.assertEqual(rules_approval.compute_rule(rule, profile, committeesize, resolute=False),
                                 tests[rule],
                                 msg=rule + " failed: " + str(rules_approval.compute_rule(rule, profile, committeesize, resolute=True)))
                self.assertEqual(len(rules_approval.compute_rule(rule, profile, committeesize, resolute=True)),
                                 1,
                                 msg=rule + " failed with resolute=True: " + str(len(rules_approval.compute_rule(rule, profile, committeesize, resolute=True))) + " committees")
                self.assertTrue(rules_approval.compute_rule(rule, profile, committeesize, resolute=True)[0] in tests[rule],
                                msg=rule + " failed with resolute=True: " + str(rules_approval.compute_rule(rule, profile, committeesize, resolute=True)))

        from preferences import Profile
        import rules_approval
        self.longMessage = True

        tests1 = {
            "seqpav": [[0, 1, 4, 5], [0, 2, 4, 5], [0, 3, 4, 5], [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "av": [[0, 1, 4, 5], [0, 2, 4, 5], [0, 3, 4, 5], [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "sav": [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5], [0, 1, 3, 4], [0, 1, 3, 5], [0, 1, 4, 5], [0, 2, 3, 4], [0, 2, 3, 5], [0, 2, 4, 5], [0, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "pav-ilp": [[0, 1, 4, 5], [0, 2, 4, 5], [0, 3, 4, 5], [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "pav-noilp": [[0, 1, 4, 5], [0, 2, 4, 5], [0, 3, 4, 5], [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "revseqpav": [[0, 1, 4, 5], [0, 2, 4, 5], [0, 3, 4, 5], [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "mav-noilp": [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5], [0, 1, 3, 4], [0, 1, 3, 5], [0, 1, 4, 5], [0, 2, 3, 4], [0, 2, 3, 5], [0, 2, 4, 5], [0, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "phrag": [[0, 1, 4, 5], [0, 2, 4, 5], [0, 3, 4, 5], [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]],
            "cc-ilp": [[0, 1, 2, 3]],
            "cc-noilp": [[0, 1, 2, 3]],
            "seqcc": [[0, 1, 2, 4], [0, 1, 2, 5], [0, 1, 3, 4], [0, 1, 3, 5], [0, 2, 3, 4], [0, 2, 3, 5], [1, 2, 3, 4], [1, 2, 3, 5]],
            "revseqcc": [[0, 1, 2, 3]],
            "monroe-ilp": [[0, 1, 2, 3]],
            "monroe-noilp": [[0, 1, 2, 3]],
        }

        profile = Profile(6)
        committeesize = 4
        preflist = [[0, 4, 5], [0], [1, 4, 5], [1], [2, 4, 5], [2], [3, 4, 5], [3]]
        profile.add_preferences(preflist)

        runmwtests(tests1, committeesize)

        # and now with reversed preflist
        preflist.reverse()
        for p in preflist:
            p.reverse()
        profile = Profile(6)
        profile.add_preferences(preflist)
        runmwtests(tests1, committeesize)

        # and another profile
        profile = Profile(5)
        committeesize = 3
        preflist = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1], [3, 4], [3, 4], [3]]
        profile.add_preferences(preflist)

        tests2 = {
            "seqpav": [[0, 1, 3]],
            "av": [[0, 1, 2]],
            "sav": [[0, 1, 3]],
            "pav-ilp": [[0, 1, 3]],
            "pav-noilp": [[0, 1, 3]],
            "revseqpav": [[0, 1, 3]],
            "mav-noilp": [[0, 1, 3], [0, 2, 3], [1, 2, 3]],
            "phrag": [[0, 1, 3]],
            "cc-ilp": [[0, 1, 3], [0, 2, 3], [0, 3, 4], [1, 2, 3], [1, 3, 4]],
            "cc-noilp": [[0, 1, 3], [0, 2, 3], [0, 3, 4], [1, 2, 3], [1, 3, 4]],
            "seqcc": [[0, 1, 3], [0, 2, 3], [0, 3, 4], [1, 2, 3], [1, 3, 4]],
            "revseqcc": [[0, 1, 3], [0, 2, 3], [0, 3, 4], [1, 2, 3], [1, 3, 4]],
            "monroe-ilp": [[0, 1, 3], [0, 2, 3], [1, 2, 3]],
            "monroe-noilp": [[0, 1, 3], [0, 2, 3], [1, 2, 3]],
        }

        runmwtests(tests2, committeesize)


if __name__ == '__main__':
    unittest.main()
