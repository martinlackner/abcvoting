# Implementations of many approval-based multwinner voting rules

# Author: Martin Lackner

import sys
import itertools
from gmpy2 import mpq
import rules_approval_ilp
import functools
from bipartite_matching import matching 
from rules_approval_collect import mwrules, method, allrules

# verifies if a sufficient number of approved candidates exists
def __enough_approved_candiates(profile, committeesize):
    appr = set()
    for pref in profile.preferences:
        appr.update(pref.approved)
    if len(appr) < committeesize:
        raise Exception("committeesize = "+str(committeesize)+" is larger than number of approved candidates")

# returns score function given its name 
def __get_scorefct(scorefct_str, committeesize):
    if scorefct_str == 'pav':
        return __pav_score_fct
    elif scorefct_str == 'cc':
        return __cc_score_fct
    elif scorefct_str == 'av':
        return __av_score_fct
    elif scorefct_str[:4] == 'geom':
        base = mpq(scorefct_str[4:])
        return functools.partial(__geom_score_fct, base=base)
    elif scorefct_str.startswith('generalizedcc'):
        param = mpq(scorefct_str[13:])
        return functools.partial(__generalizedcc_score_fct, l=param, committeesize=committeesize)
    elif scorefct_str.startswith('lp-av'):
        param = mpq(scorefct_str[5:])
        return functools.partial(__lp_av_score_fct, l=param)    
    else:
        raise Exception("Scoring function", scorefct_str, "does not exist.")


# sorts a list of committees, converts them to lists, and removes duplicates
def __sort_committees(committees):
    return [sorted(list(c)) for c in sorted(set(map(tuple, committees)))]


# required for reverse sequential Thiele methods
def __getcandidatesthatcontributeleast(profile, comm,utilityfct):
    marg_util_cand = [0] * profile.num_cand
        # marginal utility gained by adding candidate to the committee
    for pref in profile.preferences:
        for c in pref.approved:
            marg_util_cand[c] += pref.weight * utilityfct(len(pref.approved.intersection(comm))) 
    for c in range(profile.num_cand):
        if not c in comm:
            marg_util_cand[c] = max(marg_util_cand) + 1 # do not choose candidates that already have been removed 
    # find smallest elements in marg_util_cand and return indices
    return [cand for cand in range(profile.num_cand) if marg_util_cand[cand] == min(marg_util_cand)], min(marg_util_cand)


# computes the Thiele score of a committee subject to a given score function (scorefct_str)
def thiele_score(profile,committee,scorefct_str="pav"):
    scorefct = __get_scorefct(scorefct_str,len(committee))
    score = 0
    for pref in profile.preferences:
        cand_in_com = 0
        for _ in set(committee) & pref.approved:
            cand_in_com += 1
            score += pref.weight*scorefct(cand_in_com)
    return score

# computes arbitrary Thiele methods via branch-and-bound
def compute_thiele_methods_branchandbound(profile,committeesize,scorefct_str,resolute=False):
    __enough_approved_candiates(profile, committeesize)
    scorefct = __get_scorefct(scorefct_str,committeesize)

    best_committees = []
    init_com = compute_seq_thiele_methods_with_resolute(profile,committeesize,scorefct_str)
    best_score = thiele_score(profile, init_com[0],scorefct_str)
    part_coms = [[]]
    while part_coms:
        part_com = part_coms.pop(0)
        if len(part_com) == committeesize:  #potential committee, check if at least as good as previous best committee
            score = thiele_score(profile,part_com,scorefct_str)
            if score == best_score:
                best_committees.append(part_com) 
            elif score > best_score:
                best_committees = [part_com]
                best_score = score
        else:
            if len(part_com)>0:
                largest_cand = part_com[-1]
            else:
                largest_cand = -1
            missing_candidates = committeesize - len(part_com)
            marg_util_cand = __additional_thiele_scores(profile,part_com,scorefct)
            upper_bound = sum(sorted(marg_util_cand[largest_cand+1:])[-missing_candidates:]) + thiele_score(profile,part_com,scorefct_str)
            if upper_bound >= best_score:
                for c in range(largest_cand+1,profile.num_cand-missing_candidates+1):
                    part_coms.insert(0,part_com+[c])
    
    committees = __sort_committees(best_committees)
    if resolute:
        return [committees[0]]
    else:
        return committees


# Sequential PAV
def compute_seqpav(profile,committeesize,resolute=False):
    if resolute:
        return compute_seq_thiele_methods_with_resolute(profile,committeesize,'pav')
    else:
        return compute_seq_thiele_methods(profile,committeesize,'pav')
    
# Reverse Sequential PAV
def compute_revseqpav(profile,committeesize,resolute=False):
    if resolute:
        return compute_revseq_thiele_methods_with_resolute(profile,committeesize,'pav')
    else:
        return compute_revseq_thiele_methods(profile,committeesize,'pav')


# Sequential Chamberlin-Courant
def compute_seqcc(profile,committeesize,resolute=False):
    if resolute:
        return compute_seq_thiele_methods_with_resolute(profile,committeesize,'cc')
    else:
        return compute_seq_thiele_methods(profile,committeesize,'cc')
    

# Reverse Sequential Chamberlin-Courant
def compute_revseqcc(profile,committeesize,resolute=False):
    if resolute:
        return compute_revseq_thiele_methods_with_resolute(profile,committeesize,'cc')
    else:
        return compute_revseq_thiele_methods(profile,committeesize,'cc')


def __generalizedcc_score_fct(i,l,committeesize):
    # corresponds to (1,1,1,..,1,0,..0) of length *committeesize* with *l* zeros
    # l=committeesize-1 ... Chamberlin-Courant
    # l=0 ... Approval Voting
    if i > committeesize - l:
        return 0
    if i > 0:
        return 1
    else:
        return 0
        
def __lp_av_score_fct(i,l):
    # l-th root of i
    # l=1 ... Approval Voting
    # l=\infty ... Chamberlin-Courant
    if i == 1:
        return 1
    else: 
        return i ** mpq(1,l) - (i-1) ** mpq(1,l)
    
def __geom_score_fct(i,base):
    if i == 0:
        return 0
    else:
        return mpq(1,base**i)

def __pav_score_fct(i):
    if i == 0:
        return 0
    else:
        return mpq(1,i)

def __av_score_fct(i):
    if i >= 1:
        return 1
    else:
        return 0

def __cc_score_fct(i):
    if i == 1:
        return 1
    else:
        return 0

def __cumulative_score_fct(scorefct, cand_in_com):
    return sum(scorefct(i+1) for i in range(cand_in_com))

# returns a list of length num_cand 
# the i-th entry contains the marginal score increase gained by adding candidate i  
def __additional_thiele_scores(profile, committee, scorefct):
    marg = [0] * profile.num_cand
    for pref in profile.preferences:
        for c in pref.approved:
            if pref.approved & set(committee):
                marg[c] += pref.weight * scorefct(len(pref.approved & set(committee))+1)
            else:
                marg[c] += pref.weight * scorefct(1)
    for c in committee:
        marg[c] = -1
    return marg

# Satisfaction Approval Voting (SAV)
def compute_sav(profile,committeesize, resolute=False):
    return compute_av(profile, committeesize, resolute, sav=True)

# Approval Voting (AV)
def compute_av(profile, committeesize, resolute=False, sav=False):
    __enough_approved_candiates(profile, committeesize)

    appr_scores = [0] * profile.num_cand
    for pref in profile.preferences:
        for cand in pref.approved:
            if sav:
                # Satisfaction Approval Voting
                appr_scores[cand] += mpq(pref.weight,len(pref.approved))
            else:
                # (Classic) Approval Voting
                appr_scores[cand] += pref.weight
    cutoff = sorted(appr_scores)[-committeesize]  # smallest score to be in the committee
    certain_cand = [c for c in range(profile.num_cand) if appr_scores[c] > cutoff]
    possible_cand =  [c for c in range(profile.num_cand) if appr_scores[c]==cutoff]
    if resolute:
        return __sort_committees([(certain_cand + possible_cand[:committeesize-len(certain_cand)])])
    else:
        return __sort_committees([(certain_cand + list(selection)) for selection in itertools.combinations(possible_cand, committeesize-len(certain_cand))])


# Sequential Thiele methods without resolute
def compute_seq_thiele_methods(profile,committeesize,scorefct_str):
    __enough_approved_candiates(profile, committeesize)
    scorefct = __get_scorefct(scorefct_str, committeesize)

    comm_scores = {():0}
    
    for _ in range(0,committeesize):  # size of partial committees currently under consideration
        comm_scores_next = {}
        for committee, score in comm_scores.iteritems():
            additional_score_cand = __additional_thiele_scores(profile,committee,scorefct)  
                # marginal utility gained by adding candidate to the committee
            for c in range(profile.num_cand):
                if additional_score_cand[c] >= max(additional_score_cand):
                    comm_scores_next[tuple(sorted(committee + (c,)))] = comm_scores[committee]+additional_score_cand[c]
        # remove suboptimal committees
        comm_scores = {}
        cutoff = max(comm_scores_next.values())
        for com, score in comm_scores_next.iteritems():
            if score >= cutoff:
                comm_scores[com] = score
    return __sort_committees(comm_scores.keys())

# Sequential Thiele methods with resolute
def compute_seq_thiele_methods_with_resolute(profile,committeesize,scorefct_str):
    __enough_approved_candiates(profile, committeesize)
    scorefct = __get_scorefct(scorefct_str, committeesize)

    committee = []

    for _ in range(0,committeesize):  # size of partial committees currently under consideration
        additional_score_cand = __additional_thiele_scores(profile,committee,scorefct)  
        committee.append(additional_score_cand.index(max(additional_score_cand)))
    return [sorted(committee)]
    
 
# Reverse Sequential Thiele methods without resolute
def compute_revseq_thiele_methods(profile,committeesize,scorefct_str):
    __enough_approved_candiates(profile, committeesize)
    scorefct = __get_scorefct(scorefct_str, committeesize)

    allcandcomm = tuple(range(profile.num_cand))
    comm_scores = {allcandcomm:thiele_score(profile,allcandcomm,scorefct_str)}
    
    for _ in range(0,profile.num_cand-committeesize):
        comm_scores_next = {}
        for committee, score in comm_scores.iteritems():
            cands_to_remove, score_reduction = __getcandidatesthatcontributeleast(profile,committee,scorefct)
            for c in cands_to_remove:
                comm_scores_next[tuple(set(committee) - set([c]))] = score - score_reduction
        # remove suboptimal committees
        comm_scores = {}
        cutoff = max(comm_scores_next.values())
        for com, score in comm_scores_next.iteritems():
            if score >= cutoff:
                comm_scores[com] = score
    return __sort_committees(comm_scores.keys())


# Reverse Sequential Thiele methods with resolute
def compute_revseq_thiele_methods_with_resolute(profile,committeesize,scorefct_str):
    __enough_approved_candiates(profile, committeesize)
    scorefct = __get_scorefct(scorefct_str, committeesize)

    committee = set(range(profile.num_cand))

    for _ in range(0,profile.num_cand-committeesize):  
        cands_to_remove, score_reduction = __getcandidatesthatcontributeleast(profile,committee,scorefct)
        committee.remove(cands_to_remove[0])
    return [sorted(list(committee))]


# Phragmen's Sequential Rule
def compute_seqphragmen(profile, committeesize, resolute=False):
    __enough_approved_candiates(profile, committeesize)

    load = {v:0 for v in profile.preferences}
    com_loads = {():load}
    
    approvers_weight = {}
    for c in range(profile.num_cand):
        approvers_weight[c] = sum(v.weight for v in profile.preferences if c in v.approved)

    for _ in range(0,committeesize):  # size of partial committees currently under consideration
        com_loads_next = {}
        for committee, load in com_loads.iteritems():
            approvers_load = {}
            for c in range(profile.num_cand):
                approvers_load[c] = sum(v.weight * load[v] for v in profile.preferences if c in v.approved)
            new_maxload = [mpq(approvers_load[c] + 1, approvers_weight[c]) if approvers_weight[c] > 0 else committeesize+1 for c in range(profile.num_cand)]
            for c in range(profile.num_cand):
                if c in committee:
                    new_maxload[c] = sys.maxint
            for c in range(profile.num_cand):
                if new_maxload[c] <= min(new_maxload):
                    new_load = {}
                    for v in profile.preferences:
                        if c in v.approved:
                            new_load[v] = new_maxload[c]
                        else:
                            new_load[v] = load[v]
                    com_loads_next[tuple(sorted(committee + (c,)))] = new_load
        # remove suboptimal committees
        com_loads = {}
        cutoff = min([max(load) for load in com_loads_next.values()])
        for com, load in com_loads_next.iteritems():
            if max(load) <= cutoff:
                com_loads[com] = load
    
    committees = __sort_committees(com_loads.keys())
    if resolute:
        return [committees[0]]
    else:
        return committees


# Maximin Approval Voting
def compute_mav(profile, committeesize, ilp=False, resolute = False):

    if ilp:
        raise NotImplementedError("MAV is not implemented as an ILP.")

    def hamming(a,b,elements):
        diffs = 0
        for x in elements:
            if (x in a and not x in b) or (x in b and not x in a):
                diffs += 1
        return diffs

    def mavscore(committee, profile):
        score = 0
        for vote in profile.preferences:
            hamdistance = hamming(vote.approved,committee,range(profile.num_cand))
            if hamdistance > score:
                score = hamdistance
        return score

    __enough_approved_candiates(profile, committeesize)
    
    opt_committees = []
    opt_mavscore = profile.num_cand + 1
    for comm in itertools.combinations(range(profile.num_cand),committeesize):
        score = mavscore(comm, profile)
        if score < opt_mavscore:
            opt_committees = [comm]
            opt_mavscore = score
        elif mavscore(comm, profile) == opt_mavscore:
            opt_committees.append(comm)
    
    opt_committees = __sort_committees(opt_committees)
    if resolute:
        return [opt_committees[0]]
    else:
        return __sort_committees(opt_committees)


# Proportional Approval Voting
def compute_pav(profile, committeesize, ilp=True, resolute=False):
    if ilp:
        return rules_approval_ilp.compute_thiele_methods_ilp(profile,committeesize,'pav', resolute)
    else:
        return compute_thiele_methods_branchandbound(profile,committeesize,'pav', resolute)


# Chamberlin-Courant
def compute_cc(profile, committeesize, ilp=True, resolute=False):
    if ilp:
        return rules_approval_ilp.compute_thiele_methods_ilp(profile,committeesize,'cc', resolute)
    else:
        return compute_thiele_methods_branchandbound(profile,committeesize,'cc', resolute)


# Monroe's rule
def compute_monroe(profile, committeesize, ilp=True, resolute=False):
    if ilp:
        return rules_approval_ilp.compute_monroe_ilp(profile,committeesize, resolute)
    else:
        return compute_monroe_bruteforce(profile, committeesize, resolute)


def __monroescore(committee, profile):
    graph = {}
    sizeofdistricts = len(profile.preferences) / len(committee)
    for cand in committee:
        interestedvoters = []
        for i in range(len(profile.preferences)):
            if cand in profile.preferences[i].approved:
                interestedvoters.append(i)
        for j in range(sizeofdistricts):
            graph[str(cand)+"/"+str(j)] = interestedvoters
    m, _, _ = matching.bipartiteMatch(graph)
    return len(m)
    

# Monroe's rule, computed via (brute-force) matching
def compute_monroe_bruteforce(profile, committeesize, resolute=False):
    __enough_approved_candiates(profile, committeesize)
    
    if not profile.has_unit_weights():
        raise Exception("Monroe is only defined for unit weights (weight=1)")
    if len(profile.preferences) % committeesize != 0:
        raise NotImplementedError("compute_monroe_bruteforce() currently works only if the number of voters is divisible by the committee size")
    opt_committees = []
    opt_monroescore = -1
    for comm in itertools.combinations(range(profile.num_cand),committeesize):
        score = __monroescore(comm, profile)
        if score > opt_monroescore:
            opt_committees = [comm]
            opt_monroescore = score
        elif __monroescore(comm, profile) == opt_monroescore:
            opt_committees.append(comm)
    
    opt_committees = __sort_committees(opt_committees)
    if resolute:
        return [opt_committees[0]]
    else:
        return opt_committees



# nicely print a list of committees 
def print_committees(committees, print_max=10):
    if committees is None:
        print "Error: no committees returned"
        return
    if len(committees) == 1:
        print " 1 committee"
    else:
        if len(committees)>print_max:
            print " ", len(committees), "committees, printing ", print_max, "of them"
        else:
            print " ", len(committees), "committees"
    for com in sorted(map(tuple, committees[:print_max])):
        print "    {", ", ".join(map(str, com)), "}"
    print 

