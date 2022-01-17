import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from lucb import kullback_leibler

class TabularAnchor:

    def __init__(self, cs : CS.ConfigurationSpace, all_features) -> None:
        """Anchor for explaining an instance of tabular Dataset. 

        :param cs: ConfigurationSpace that is used for sampling
        :type cs: CS.ConfigurationSpace
        :param all_features: all features of the Dataset
        :type all_features: list
        """        
        self.ub = None # upper bound
        self.lb = None # lower bound
        self.n_samples = 0
        self.coverage = 0
        self.correct = 0

        self.gt = [] # greater than rules such as ("height", 150)
        self.lt = [] # less than rules
        self.eq = [] # equal to rules
        self.rules = [] # raw rules
        
        self.all_features = all_features # all features in correct order
        self.cs = cs

    @property
    def mean(self):
        if self.n_samples == 0:
            return 0
        return self.correct / self.n_samples

    def reset(self):
        self.ub = None # upper bound
        self.lb = None # lower bound
        self.n_samples = 0
        self.coverage = 0
        self.correct = 0

    def compute_coverage(self, X):
        """Compute the coverage of the current rules with respect to the given dataset.
        Note: Coverage is not defined by multiplying feature range 
        but counting the occurences in the dataset satisfying the current rules.

        :param X: Dataset
        :type X: pd.Dataframe
        :return: coverage
        :rtype: float
        """        
        cov_array = np.array([((X[f] >= self.cs.get_hyperparameter(f).lower) & (X[f] <= self.cs.get_hyperparameter(f).upper)) for f in self.all_features])
        return (np.all(cov_array, axis=0).sum())


    def compute_ub(self, beta):
        """Computes upper bound for given beta

        :param beta: beta parameter
        :type beta: float
        """        
        # LUCB paper equation 4
        p_mean = self.correct / self.n_samples
        ub = p_mean + np.sqrt(beta / 2* self.n_samples)
        q = min(ub, 1)
        if self.n_samples * kullback_leibler(p_mean, q) > beta:
            ub = (ub + p_mean) / 2
        self.ub = ub


    def compute_lb(self, beta):
        """Computes lower bound for given beta

        :param beta: beta parameter
        :type beta: float
        """        
        p_mean = self.correct / self.n_samples
        lb = p_mean - np.sqrt(beta / 2* self.n_samples)
        q = max(min(lb, 1), 0)
        if self.n_samples * kullback_leibler(p_mean, q) > beta:
            lb = (lb + p_mean) / 2
        self.lb = lb

    def get_current_features(self):
        """Features that are currently used in a rule.

        :return: features
        :rtype: list
        """        
        return [feat for feat,_ in self.gt + self.lt + self.eq]

    def sample_instance(self):
        """Sample one instance w.r.t the current rules.

        :return: instance
        :rtype: np.ndarray
        """        
        c = self.cs.sample_configuration()
        sample = []
        # make sure features are in correct order
        for f in self.all_features:
            sample.append(c.get(f))
        return np.array(sample).reshape(1, -1)

    def is_satisfied(self, instance):
        """Returns true if the instance is valid according to the current rules.

        :param instance: instance to check
        :type instance: np.ndarray
        :return: True if instance satisfies all rules
        :rtype: boolean
        """        
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
        """Parses rule tuple and updates current rules and configspace.

        :param rule: Rule regarding one feature
        :type rule: tuple
        """        
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

