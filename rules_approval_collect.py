import rules_approval

mwrules = {
    "av" : "Approval Voting",
    "sav" : "Satisfaction Approval Voting",
    "pav" : "Proportional Approval Voting (PAV) via ILP",
    "pav-noilp" : "Proportional Approval Voting (PAV) via branch-and-bound",
    "seqpav" : "Sequential Proportional Approval Voting (seq-PAV)",
    "revseqpav" : "Reverse Sequential Proportional Approval Voting (revseq-PAV)",
    "phrag" : "Phragmen's sequential rule (seq-Phragmen)",
    "monroe" : "Monroe's rule via ILP",
    "monroe-noilp" : "Monroe's rule via matching algorithm",
    "cc" : "Chamberlin-Courant (CC) via ILP",
    "cc-noilp" : "Chamberlin-Courant (CC) via branch-and-bound",
    "seqcc" : "Sequential Chamberlin-Courant (seq-CC)",
    "revseqcc" : "Reverse Sequential Chamberlin-Courant (revseq-CC)",
    "mav" : "Maximin Approval Voting",
        }

def method(name, profile, committeesize, tiebreaking=False):
    if name == "seqpav":
        return rules_approval.compute_seqpav(profile,committeesize,tiebreaking)
    elif name == "revseqpav":
        return rules_approval.compute_revseqpav(profile,committeesize,tiebreaking)
    elif name == "av":
        return rules_approval.compute_av(profile,committeesize,tiebreaking)
    elif name == "sav":
        return rules_approval.compute_sav(profile,committeesize,tiebreaking)
    elif name == "pav":
        return rules_approval.compute_pav(profile,committeesize,tiebreaking)
    elif name == "pav-noilp":
        return rules_approval.compute_pav(profile,committeesize,tiebreaking=tiebreaking,ilp=False)
    elif name == "phrag":
        return rules_approval.compute_seqphragmen(profile,committeesize,tiebreaking)
    elif name == "monroe":
        return rules_approval.compute_pav(profile,committeesize,tiebreaking)
    elif name == "monroe-noilp":
        return rules_approval.compute_pav(profile,committeesize,tiebreaking=tiebreaking,ilp=False)
    elif name == "cc":
        return rules_approval.compute_cc(profile,committeesize,tiebreaking)
    elif name == "cc-noilp":
        return rules_approval.compute_cc(profile,committeesize,tiebreaking=tiebreaking,ilp=False)
    if name == "seqcc":
        return rules_approval.compute_seqcc(profile,committeesize,tiebreaking)
    elif name == "revseqcc":
        return rules_approval.compute_revseqcc(profile,committeesize,tiebreaking)
    elif name == "mav":
        return rules_approval.compute_mav(profile,committeesize,tiebreaking)
    else:
        raise NotImplementedError("method "+str(name)+" not known")

def allrules(profile, committeesize, ilp=True, includetiebreaking=False):
    
    for rule in mwrules.keys():
        print mwrules[rule]+":"
        com = method(rule,profile,committeesize)
        rules_approval.print_committees(com) 
        
        if includetiebreaking:
            print mwrules[rule]+" (with tie-breaking):"
            com = method(rule,profile,committeesize,tiebreaking=True)
            rules_approval.print_committees(com) 
    
