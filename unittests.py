import unittest

class TestApprovalMultiwinner(unittest.TestCase):
    def test_createprofiles(self):
        from preferences import Profile
        from preferences import DichotomousPreferences
        num_cand = 7
        prof = Profile(num_cand)
        self.assertEqual(prof.add_preferences(DichotomousPreferences([0,4,5])), None)
        with self.assertRaises(Exception):
            prof.add_preferences(DichotomousPreferences([num_cand]))
        with self.assertRaises(Exception):
            prof.add_preferences(DichotomousPreferences([-1]))
        self.assertEqual(prof.add_preferences([0,4,5]), None)
        with self.assertRaises(Exception):
            prof.add_preferences([0,4,5,"1"])
        with self.assertRaises(Exception):
            prof.add_preferences(["1",0,4,5])    
        p1 = DichotomousPreferences([0,4,5] )
        p2 = DichotomousPreferences([1,2] )
        self.assertEqual(prof.add_preferences([p1,p2]), None)
        self.assertTrue(prof.has_unit_weights())
        prof.add_preferences(DichotomousPreferences([0,4,5], 2.4))
        self.assertFalse(prof.has_unit_weights())
        self.assertEqual(prof.totalweight(),6.4)

    def test_mwrules__toofewcandidates(self):
        from preferences import Profile
        import rules_approval
        profile = Profile(5)
        committeesize = 4
        preflist = [[0,1,2],[1,3],[1,2],[0]]
        profile.add_preferences(preflist)
        with self.assertRaises(Exception):
            rules_approval.compute_seqpav(profile,committeesize,tiebreaking=True)
        with self.assertRaises(Exception):
            rules_approval.compute_seqpav(profile,committeesize)
        with self.assertRaises(Exception):
            rules_approval.compute_av(profile,committeesize)
        with self.assertRaises(Exception):
            rules_approval.compute_sav(profile,committeesize)
        with self.assertRaises(Exception):
            rules_approval.compute_pav(profile,committeesize)
        with self.assertRaises(Exception):
            rules_approval.compute_pav(profile,committeesize,ilp=False)
        with self.assertRaises(Exception):
            rules_approval.compute_maxphragmen_unrefined(profile,committeesize)
        with self.assertRaises(Exception):
            rules_approval.compute_maxphragmen_refined(profile,committeesize)
        with self.assertRaises(Exception):
            rules_approval.compute_varphragmen(profile,committeesize)
        with self.assertRaises(Exception):
            rules_approval.compute_seqphragmen(profile,committeesize)
        with self.assertRaises(Exception):
            rules_approval.compute_cc(profile,committeesize)
        with self.assertRaises(Exception):
            rules_approval.compute_cc(profile,committeesize,ilp=False)
        with self.assertRaises(Exception):
            rules_approval.compute_monroe(profile,committeesize)
        with self.assertRaises(Exception):
            rules_approval.compute_monroe_bruteforce(profile,committeesize)
        

    def test_mwrules0(self):
        from preferences import Profile
        import rules_approval
        
        profile = Profile(4)
        profile.add_preferences([[0],[1],[2]])
        committeesize = 2
        
        for rule in rules_approval.mwrules.keys():
            self.assertTrue(rules_approval.method(rule,profile,committeesize))
            
        for rule in rules_approval.mwrules.keys():
            self.assertEqual(len(rules_approval.method(rule,profile,committeesize,tiebreaking=True)),1)

    def test_mwrules1(self):
        from preferences import Profile
        import rules_approval

        def tests():
            self.assertEqual(rules_approval.compute_seqpav(profile,committeesize,tiebreaking=True), [[0, 1, 4, 5 ]])
            self.assertEqual(rules_approval.compute_seqpav(profile,committeesize), [[0, 1, 4, 5],[ 0, 2, 4, 5],[ 0, 3, 4, 5 ], [ 1, 2, 4, 5 ], [1, 3, 4, 5 ], [2, 3, 4, 5]])
            self.assertEqual(rules_approval.compute_av(profile,committeesize), [[0, 1, 4, 5],[ 0, 2, 4, 5],[ 0, 3, 4, 5 ], [ 1, 2, 4, 5 ], [1, 3, 4, 5 ], [2, 3, 4, 5]])
            self.assertEqual(len(rules_approval.compute_sav(profile,committeesize)), 15)
            self.assertTrue(0 in com for com in rules_approval.compute_sav(profile,committeesize))
            self.assertEqual(rules_approval.compute_pav(profile,committeesize), [[0, 1, 4, 5],[ 0, 2, 4, 5],[ 0, 3, 4, 5 ], [ 1, 2, 4, 5 ], [1, 3, 4, 5 ], [2, 3, 4, 5]])
            self.assertEqual(rules_approval.compute_pav(profile,committeesize,ilp=False), [[0, 1, 4, 5],[ 0, 2, 4, 5],[ 0, 3, 4, 5 ], [ 1, 2, 4, 5 ], [1, 3, 4, 5 ], [2, 3, 4, 5]])
            self.assertEqual(rules_approval.compute_maxphragmen_unrefined(profile,committeesize), [[0,1,2,3]])
            self.assertEqual(rules_approval.compute_maxphragmen_refined(profile,committeesize), [[0,1,2,3]])
            self.assertEqual(rules_approval.compute_varphragmen(profile,committeesize), [[0,1,2,3]])
            self.assertEqual(rules_approval.compute_seqphragmen(profile,committeesize), [[0, 1, 4, 5],[0, 2, 4, 5],[0, 3, 4, 5], [1, 2, 4, 5],[1, 3, 4, 5],[2, 3, 4, 5]])
            self.assertEqual(rules_approval.compute_cc(profile,committeesize), [[0,1,2,3]])
            self.assertEqual(rules_approval.compute_cc(profile,committeesize,ilp=False), [[0,1,2,3]])
            self.assertEqual(rules_approval.compute_monroe(profile,committeesize), [[0,1,2,3]])
            self.assertEqual(rules_approval.compute_monroe_bruteforce(profile,committeesize), [[0,1,2,3]])

        profile = Profile(6)
        committeesize = 4
        preflist = [[0,4,5],[0],[1,4,5],[1],[2,4,5],[2],[3,4,5],[3]]
        profile.add_preferences(preflist)
        tests()
        
        # and now with reversed preflist
        preflist.reverse()
        for p in preflist:
            p.reverse()
        profile = Profile(6)
        profile.add_preferences(preflist)
        tests()
        

if __name__ == '__main__':
    unittest.main()
