import numpy as np

# hyperparameters (see https://proceedings.mlr.press/v30/Kaufmann13.pdf, Section 5)
ALPHA = 1.1
HYPER_K = 405.5
EPS = 0.1
DELTA = 0.05
INIT_SAMPLES = 10

def get_best_candidate(anchors, instance, model):
    # LUCB?
    # init bounds by sampling each anchor once
    t = 1
    y = model.predict(instance)
    beta = compute_beta(t, len(anchors))
    for a in anchors:
        for _ in range(INIT_SAMPLES):
            a_x = a.sample_instance()
            a_y = model.predict(a_x)
            a.n_samples += 1
            # print(a_y, y)
            if a_y == y:
                a.correct += 1

            a.compute_ub(beta)
            a.compute_lb(beta)
            # print("Mean = ", a.mean)

    
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

    return best_mean_anchor



def compute_beta(t, K):
    term = np.log((HYPER_K * K * t**ALPHA) / DELTA)
    return term * np.log(term)


def kullback_leibler(x,y):
    # Kullback-Leibler Divergenz
    x = min(1 - 1e-5, max(1e-6, x))
    y = min(1 - 1e-5, max(1e-6, y))
    return x * np.log(x / y) + (1 - x) * np.log( (1-x) / (1-y))