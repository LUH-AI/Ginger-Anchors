import numpy as np

# hyperparameters (see https://proceedings.mlr.press/v30/Kaufmann13.pdf, Section 5)
ALPHA = 1.1
HYPER_K = 405.5
EPS = 0.1
DELTA = 0.05
INIT_SAMPLES = 10

def get_best_candidate(anchors, instance, model, tau):
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
    beta = compute_beta(t, len(anchors))
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

    while best_mean_anchor.lb - best_ub_anchor.ub > EPS:
        t += 1
        beta = compute_beta(t, len(anchors))
        for a in (best_mean_anchor, best_ub_anchor):
            a_x = a.sample_instance()
            a_y = model.predict(a_x)
            a.n_samples += 1
            if a_y == y:
                a.correct += 1

            a.compute_ub(beta)
            a.compute_lb(beta)
            # print("Mean = ", a.mean)
        best_ub_anchor = sorted(anchors, key=lambda a : a.ub, reverse=True)[0]
        best_mean_anchor = sorted(anchors, key=lambda a : a.mean, reverse=True)[0]

        
        while best_mean_anchor.lb <= tau and tau <= best_mean_anchor.ub:
            print("Refine")
            t += 1
            beta = compute_beta(t, len(anchors))
            a_x = a.sample_instance()
            a_y = model.predict(a_x)
            a.n_samples += 1
            if a_y == y:
                a.correct += 1

            a.compute_ub(beta)
            a.compute_lb(beta)

    return best_mean_anchor



def compute_beta(t, K):
    term = np.log((HYPER_K * K * t**ALPHA) / DELTA)
    return term * np.log(term)


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
    x = min(1 - 1e-5, max(1e-6, x))
    y = min(1 - 1e-5, max(1e-6, y))
    return x * np.log(x / y) + (1 - x) * np.log( (1-x) / (1-y))