import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from lucb import kullback_leibler

class TabularAnchor:

    def __init__(self, cs : CS.ConfigurationSpace) -> None:
        self.ub = None # upper bound
        self.lb = None # lower bound
        self.coverage = None
        self.n_samples = 0
        self.correct = 0

        self.gt = [] # greater than rules such as ("height", 150)
        self.lt = [] # less than rules
        self.eq = [] # equal to rules
        self.rules = [] # raw rules

        self.cs = cs

    @property
    def mean(self):
        if self.n_samples == 0:
            return 0
        return self.correct / self.n_samples


    def compute_ub(self, beta):
        # LUCB paper equation 4
        p_mean = self.correct / self.n_samples
        ub = p_mean + np.sqrt(beta / 2* self.n_samples)
        q = min(ub, 1)
        if self.n_samples * kullback_leibler(p_mean, q) > beta:
            ub = (ub + p_mean) / 2
        self.ub = ub


    def compute_lb(self, beta):
        p_mean = self.correct / self.n_samples
        lb = p_mean - np.sqrt(beta / 2* self.n_samples)
        q = max(min(lb, 1), 0)
        if self.n_samples * kullback_leibler(p_mean, q) > beta:
            lb = (lb + p_mean) / 2
        self.lb = lb

    def get_current_features(self):
        return [feat for feat,_ in self.gt + self.lt + self.eq]

    def sample_instance(self):
        return self.cs.sample_configuration().get_array().reshape(1, -1)

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
                new_cs.add_hyperparameter(self.cs.get_hyperparameter(f))
        self.cs = new_cs

