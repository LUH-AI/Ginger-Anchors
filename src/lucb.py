import numpy as np

# hyperparameters (see https://proceedings.mlr.press/v30/Kaufmann13.pdf, Section 5)
ALPHA = 1.1
HYPER_K = 405.5
EPS = 0.20
DELTA = 0.1
INIT_SAMPLES = 10

def get_b_best_candidates(anchors, instance, model, B, delta, eps):
    """
    Determine the best b anchors by precision. Poses anchor selection as a multi-armed bandit problem.
    Implements the LUCB algorithm (see https://proceedings.mlr.press/v30/Kaufmann13.pdf) in an explore-m setting.

    :param anchors: List of candidates anchors
    :type anchors: list
    :param instance: Instance to be explained
    :type instance: np.ndarray
    :param model: Model to be explained
    :type model: callable
    :param tau: precision threshold
    :type tau: float
    :return: Best anchor among candidates
    :rtype: TabularAnchor
    """    
    t = 1
    y = model.predict(instance)
    beta = compute_beta(t, len(anchors), delta)
    for a in anchors:
        for _ in range(INIT_SAMPLES):
            a_x = a.sample_instance()
            a_y = model.predict(a_x)
            a.n_samples += 1
            if a_y == y:
                a.correct += 1

            a.compute_ub(beta)
            a.compute_lb(beta)

    best_ub_anchor = sorted(anchors, key=lambda a : a.ub, reverse=True)[0]
    best_mean_anchor = sorted(anchors, key=lambda a : a.mean, reverse=True)[0]
    
    while abs(best_ub_anchor.ub) - abs(best_mean_anchor.lb) > eps:
        t += 1
        beta = compute_beta(t, len(anchors), delta)
        for a in (best_mean_anchor, best_ub_anchor):
            a_x = a.sample_instance()
            a_y = model.predict(a_x)
            a.n_samples += 1
            if a_y == y:
                a.correct += 1

            a.compute_ub(beta)
            a.compute_lb(beta)
        best_ub_anchor = sorted(anchors, key=lambda a : a.ub, reverse=True)[0]
        best_mean_anchor = sorted(anchors, key=lambda a : a.mean, reverse=True)[0]

    return sorted(anchors, key=lambda a : a.mean, reverse=True)[:B]


def get_best_candidate(anchors, instance, model, delta, eps):
    """Determine the best anchor by precision. Poses anchor selection as a multi-armed bandit problem.
    Implements the LUCB algorithm (see https://proceedings.mlr.press/v30/Kaufmann13.pdf)

    :param anchors: List of candidates anchors
    :type anchors: list
    :param instance: Instance to be explained
    :type instance: np.ndarray
    :param model: Model to be explained
    :type model: callable
    :param tau: precision threshold
    :type tau: float
    :return: Best anchor among candidates
    :rtype: TabularAnchor
    """    
    # init bounds by sampling each anchor
    t = 1
    y = model.predict(instance)
    beta = compute_beta(t, len(anchors), delta)
    for a in anchors:
        for _ in range(INIT_SAMPLES):
            a_x = a.sample_instance()
            a_y = model.predict(a_x)
            a.n_samples += 1
            if a_y == y:
                a.correct += 1
            a.compute_ub(beta)
            a.compute_lb(beta)


    best_ub_anchor = sorted(anchors, key=lambda a : a.ub, reverse=True)[0]
    best_mean_anchor = sorted(anchors, key=lambda a : a.mean, reverse=True)[0]

    while abs(best_ub_anchor.ub) - abs(best_mean_anchor.lb) > eps:
        t += 1
        beta = compute_beta(t, len(anchors), delta)
        for a in (best_mean_anchor, best_ub_anchor):
            n = 100
            for _ in range(n):
                a_x = a.sample_instance()
                a_y = model.predict(a_x)
                a.n_samples += 1
                if a_y == y:
                    a.correct += 1

            a.compute_ub(beta)
            a.compute_lb(beta)
        best_ub_anchor = sorted(anchors, key=lambda a : a.ub, reverse=True)[0]
        best_mean_anchor = sorted(anchors, key=lambda a : a.mean, reverse=True)[0]


    return best_mean_anchor



def compute_beta(t, K, delta):
    term = np.log(HYPER_K * K * (t ** ALPHA) / delta)
    return term + np.log(term)


def kullback_leibler(x,y):
    """Kullback Leibler Divergenz.
    Used to compute upper and lower bounds

    :param x: x
    :type x: float
    :param y: y
    :type y: float
    :return: KL distance
    :rtype: float
    """    
    p = min(1 - 1e-5, max(1e-6, x))
    q = min(1 - 1e-5, max(1e-6, y))
    return p * np.log(p / q) + (1 - p) * np.log( (1-p) / (1-q))