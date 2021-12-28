
## Pseudocode:
# Bottom-up approach
"""

initialise empty Anchor
A = []
while True:
    candidates = [A + a_i for a_i in features]
    best = get_best_candidate(candidates)
    if prec(best) >= tau:
        break
    else:
        while lb(best) < tau < up(best):
            sample instance
            predict instance
            update candidates' precision and bounds
        if lb(A) > tau:
            break
return best
"""

# get_best_candidate:

"""
A =  best mean candidate
A' =  best ub candidate
while A.lb > A'.ub (with tolerance eps):
    sample instance that satisfies A / A' from neighborhood
    predict instance with model
    update candidates' precision and bounds
return best mean candidate
...

"""

# beam search:
"""
A* = []
A_0 = []
while True:
    candidates = [A + a_i for a_i in features if cov(A + a_i) > cov(A)]
    A_t = get_b_best_candidates(candidates)
    if not A_t:
        break
    for A in [A for A in A_t if lb(A) > tau]:
        if cov(A) > cov(A*):
            A* = A
return A*

"""

# get_b_best_candidates:
"""

"""

"""
question:
ucb mit delta (pseudocode)
update bounds https://github.com/marcotcr/anchor/blob/d55f1b480f2326819ae43b1f5c4e7b8892912f30/anchor/anchor_base.py#L76
dup_bernoulli bekommt als argument beta/n_samples
beta -> compute beta: https://github.com/marcotcr/anchor/blob/d55f1b480f2326819ae43b1f5c4e7b8892912f30/anchor/anchor_base.py#L53

b-best-candidates?
pairwise comparison?
"""
