import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

class TabularAnchor:

    def __init__(self, cs : CS.ConfigurationSpace) -> None:
        self.ub = None # upper bound
        self.lb = None # lower bound
        self.mean = None # expected precision
        self.coverage = None
        self.n_samples = 0

        self.gt = [] # greater than rules such as ("height", 150)
        self.lt = [] # less than rules
        self.eq = [] # equal to rules
        self.rules = [] # raw rules

        self.cs = cs

    def get_current_features(self):
        return [feat for feat,_ in self.gt + self.lt + self.eq]

    def sample_instance(self):
        return self.cs.sample_configuration()

    def is_satisfied(self, instance):
        if self.gt == []:
            g = True
        else:
            g = all(instance[condition] >= value for condition, value in self.gt)

        if self.lt == []:
            l = True
        else:
            l = all(instance[condition] <= value for condition, value in self.lt)

        if self.eq == []:
            e = True
        else:
            e = all(instance[condition] == value for condition, value in self.eq)

        return g and l and e

    def add_rule(self, rule):
        # add a new rule and adjust the configspace
        new_cs = CS.ConfigurationSpace()
        self.rules.append(rule)
        if len(rule) == 5:
            f, o1, v1, o2, v2 = rule
            if "<=" == o1 and ">=" == o2:
                self.lt.append((f, v1))
                self.gt.append((f, v2))
                old_hp = self.cs.get_hyperparameter(f)
                hp_class = old_hp.__class__
                new_hp = hp_class(f, lower=v2, upper=v1, log=False)
                new_cs.add_hyperparameter(new_hp)

            elif ">=" == o1 and "<=" == o2:
                self.gt.append((f, v1))
                self.lt.append((f, v2))
                old_hp = self.cs.get_hyperparameter(f)
                hp_class = old_hp.__class__
                new_hp = hp_class(f, lower=v1, upper=v2, log=False)
                new_cs.add_hyperparameter(new_hp)
            else:
                raise Exception("Unvalid rule", rule)
        elif len(rule) == 3:
            f, o, v = rule
            if "<=" in o:
                self.lt.append((f, v))
                old_hp = self.cs.get_hyperparameter(f)
                hp_class = old_hp.__class__
                new_hp = hp_class(f, lower=old_hp.lower, upper=v, log=False)
                new_cs.add_hyperparameter(new_hp)
            elif ">=" in o:
                self.gt.append((f, v))
                old_hp = self.cs.get_hyperparameter(f)
                hp_class = old_hp.__class__
                new_hp = hp_class(f, lower=v, upper=old_hp.upper, log=False)
                new_cs.add_hyperparameter(new_hp)
            elif "==" == o:
                self.eq.append((f, v))
                new_hp = CSH.Constant(f, v)
                new_cs.add_hyperparameter(new_hp)
        else:
            raise Exception("Unvalid rule", rule)

        for f in self.cs.get_hyperparameter_names():
            if not f in new_cs.get_hyperparameter_names():
                print("Take old", f)
                new_cs.add_hyperparameter(self.cs.get_hyperparameter(f))
            else:
                print("Take new", f)
        self.cs = new_cs
        print(self.cs)

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
