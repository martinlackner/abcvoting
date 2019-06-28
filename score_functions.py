# Score functions for Thiele methods

# Author: Martin Lackner


from gmpy2 import mpq


# returns score function given its name 
def get_scorefct(scorefct_str, committeesize):
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


# computes the Thiele score of a committee subject to a given score function (scorefct_str)
def thiele_score(profile,committee,scorefct_str="pav"):
    scorefct = get_scorefct(scorefct_str,len(committee))
    score = 0
    for pref in profile.preferences:
        cand_in_com = 0
        for _ in set(committee) & pref.approved:
            cand_in_com += 1
            score += pref.weight*scorefct(cand_in_com)
    return score


def generalizedcc_score_fct(i,l,committeesize):
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

def cumulative_score_fct(scorefct, cand_in_com):
    return sum(scorefct(i+1) for i in range(cand_in_com))

# returns a list of length num_cand 
# the i-th entry contains the marginal score increase gained by adding candidate i  
def additional_thiele_scores(profile, committee, scorefct):
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
