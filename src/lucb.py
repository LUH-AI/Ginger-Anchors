import numpy as np


# Require: eps >= 0 (tolerance level), U, L (confidence bounds)
# t = 1 (number of stage of the algorithm), B(1) = inf (stopping index)
# for a=1...K do
    # Sample arm a, compute confidence bounds Ua(1), La(1)
# end for
# while B(t) > eps do
    # Draw arm ut and lt. t = t + 1.
    # Update confidence bounds, set J(t) and arms ut, lt
    # B(t) = Uut (t) - Llt (t)
# end while
# return J(t).

# hyperparameters (see https://proceedings.mlr.press/v30/Kaufmann13.pdf, Section 5)
ALPHA = 1.1
HYPER_K = 405.5
EPS = 0.1
DELTA = 0.05

def get_best_candidate(anchors):
    # LUCB?
    # init bounds by sampling each anchor once
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
    pass


def compute_beta(t):
    pass

def compute_ub(n_correct, n_samples):
    pass

def compute_lb():
    pass

def d(x,y):
    # Kullback-Leibler Divergenz
    # 
    return x * np.log(x / y) + (1 - x) * np.log( (1-x) / (1-y))