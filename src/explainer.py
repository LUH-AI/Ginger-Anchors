import pandas as pd

from custom_anchor import TabularAnchor

class Explainer:

    def __init__(self, X : pd.DataFrame) -> None:
        self.X = X
        self.features = X.columns
        # TODO np.quantile for each feature
        self.quantiles = {}

    def explain_bottom_up(self, instance, model, tau=0.95):
        # initialise empty Anchor
        anchor = TabularAnchor()
        while True:
            # get quantile of explain instance, add to current anchor
            quantiles = None
            candidates = generate_candidates(anchor, self.features, quantiles)
            # all this LUCB?
            anchor = get_best_candidate(candidates)
            if anchor.precision >= tau:
                break
            else:
                while anchor.lb <= tau and tau <= anchor.up:
                    pass
                    # sample instance
                    # predict instance
                    # update candidates' precision and bounds
                if anchor.lb > tau:
                    break
        return anchor

    def explain_beam_search(self):
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
        raise NotImplementedError()

def generate_candidates(current_anchors, all_features, bounds):
    new_anchors = current_anchors


    return new_anchors


def get_best_candidate():
    # LUCB?
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
