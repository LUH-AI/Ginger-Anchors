
class TabularAnchor:

    def __init__(self) -> None:
        self.ub = None # upper bound
        self.lb = None # lower bound
        self.mean = None # expected precision
        self.coverage = None
        self.n_samples = 0

        self.gt = [] # greater than rules such as ("height", 150)
        self.lt = [] # less than rules
        self.eq = [] # equal to rules

    def sample_instance(self):
        sample = None
        return sample

    def is_satisfied(self, instance):
        g = all(instance[condition] >= value for condition, value in self.gt)
        l = all(instance[condition] <= value for condition, value in self.lt)
        e = all(instance[condition] == value for condition, value in self.eq)
        return g and l and e


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
