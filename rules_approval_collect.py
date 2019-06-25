import rules_approval

mwrules = {
    "av" : "Approval Voting",
    "sav" : "Satisfaction Approval Voting",
    "pav-ilp" : "Proportional Approval Voting (PAV) via ILP",
    "pav-noilp" : "Proportional Approval Voting (PAV) via branch-and-bound",
    "seqpav" : "Sequential Proportional Approval Voting (seq-PAV)",
    "revseqpav" : "Reverse Sequential Proportional Approval Voting (revseq-PAV)",
    "phrag" : "Phragmen's sequential rule (seq-Phragmen)",
    "monroe-ilp" : "Monroe's rule via ILP",
    "monroe-noilp" : "Monroe's rule via matching algorithm",
    "cc-ilp" : "Chamberlin-Courant (CC) via ILP",
    "cc-noilp" : "Chamberlin-Courant (CC) via branch-and-bound",
    "seqcc" : "Sequential Chamberlin-Courant (seq-CC)",
    "revseqcc" : "Reverse Sequential Chamberlin-Courant (revseq-CC)",
    "mav" : "Maximin Approval Voting",
        }

def method(name, profile, committeesize, resolute=False):
    if name == "seqpav":
        return rules_approval.compute_seqpav(profile,committeesize,resolute=resolute)
    elif name == "revseqpav":
        return rules_approval.compute_revseqpav(profile,committeesize,resolute=resolute)
    elif name == "av":
        return rules_approval.compute_av(profile,committeesize,resolute=resolute)
    elif name == "sav":
        return rules_approval.compute_sav(profile,committeesize,resolute=resolute)
    elif name == "pav-ilp":
        return rules_approval.compute_pav(profile,committeesize,ilp=True,resolute=resolute)
    elif name == "pav-noilp":
        return rules_approval.compute_pav(profile,committeesize,ilp=False,resolute=resolute)
    elif name == "phrag":
        return rules_approval.compute_seqphragmen(profile,committeesize,resolute=resolute)
    elif name == "monroe-ilp":
        return rules_approval.compute_monroe(profile,committeesize,ilp=True,resolute=resolute)
    elif name == "monroe-noilp":
        return rules_approval.compute_monroe(profile,committeesize,ilp=False,resolute=resolute)
    elif name == "cc-ilp":
        return rules_approval.compute_cc(profile,committeesize,ilp=True,resolute=resolute)
    elif name == "cc-noilp":
        return rules_approval.compute_cc(profile,committeesize,ilp=False,resolute=resolute)
    if name == "seqcc":
        return rules_approval.compute_seqcc(profile,committeesize,resolute=resolute)
    elif name == "revseqcc":
        return rules_approval.compute_revseqcc(profile,committeesize,resolute=resolute)
    elif name == "mav":
        return rules_approval.compute_mav(profile,committeesize,resolute=resolute)
    else:
        raise NotImplementedError("method "+str(name)+" not known")

def allrules(profile, committeesize, ilp=True, include_resolute=False):
    
    for rule in mwrules.keys():
        if not ilp and "-ilp" in rule:
            continue
        print mwrules[rule]+":"
        com = method(rule,profile,committeesize)
        rules_approval.print_committees(com) 
        
        if include_resolute:
            print mwrules[rule]+" (with tie-breaking):"
            com = method(rule,profile,committeesize,resolute=True)
            rules_approval.print_committees(com) 
    
