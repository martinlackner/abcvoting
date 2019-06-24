# EJR example

from preferences import *
import rules_approval

num_cand = 6
prof = Profile(num_cand)
prof.add_preferences([[0,4,5],[0],[1,4,5],[1],[2,4,5],[2],[3,4,5],[3]])
com_size = 4

rules_approval.allrules(prof,com_size,includetiebreaking=True)

