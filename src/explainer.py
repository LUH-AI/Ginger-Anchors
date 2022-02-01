from copy import deepcopy
import enum
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import pandas as pd
import numpy as np
import random


from custom_anchor import TabularAnchor
from lucb import get_best_candidate, get_b_best_candidates
from utils import new_logger

# TODO: Categorical hyperparameters in quantiles, rule generation (check for hp type, == if categorical),

class Explainer:

    def __init__(self, X : pd.DataFrame) -> None:
        """An explainer object from which explanations
        (aka anchors) for single instances can be computed.

        :param X: Dataset used to train the model
        :type X: pd.DataFrame
        """
        self.logger = new_logger(self.__class__.__name__)  
        self.X = X
        self.features = list(X.columns)
        self.quantiles = {}
        # { feature : [bound1, bound2, bound3] }
        for f in self.features:
            # More quantiles might lead to less predicates per anchor but 
            # leads to tighter rules (worse coverage)
            self.quantiles[f] = np.quantile(X[f], [0.25, 0.5, 0.75])# np.arange(0,1, 0.05))#
        
        self.feature2index = {f : self.features.index(f) for f in self.features}
        self.cs = get_configspace_for_dataset(X)
        

    def explain_bottom_up(self, instance, model, tau=0.95):
        """Bottom-up Construction of Anchors introduced in M. Ribeiro "Anchors: High-Precision Model-Agnostic Explanations".
        Rules of the Anchor are greedily generated.

        :param instance: instance to be explained
        :type instance: np.ndarray
        :param model: the model that is estimated by the anchor
        :type model: model
        :param tau: desired level of precision, defaults to 0.95
        :type tau: float, optional
        :return: anchor
        :rtype: TabularAnchor
        """        
        # initialise empty Anchor
        self.logger.debug(f"Start bottom-up search for {instance}.")
        anchor = TabularAnchor(self.cs, self.features)
        # get quantiles of instance
        rules = generate_rules_for_instance(self.quantiles, instance, self.feature2index)
        self.logger.debug(f"Generated rules: {rules}")
        random.shuffle(rules)
        while True:
            # add unused rules to current anchor
            candidates = self.generate_candidates(anchor, rules)
            if candidates == []:
                exit("No anchors found, ¯\\_(ツ)_/¯")
            # treat anchors as Mulit-Armed Bandidates
            anchor = get_best_candidate(candidates, instance, model, tau)
            self.logger.info(f"Current best: P={anchor.mean} (based on {anchor.n_samples} samples), Rules: {anchor.rules}")
            if anchor.mean >= tau:
                break
        
        anchor.compute_coverage(self.X)
        self.logger.info(f"Found anchor: P={anchor.mean}, C={anchor.coverage}, Rules:{anchor.rules}")
        return anchor

    def explain_beam_search(self, instance, model, tau=0.95, B=1):
        """
        Finds in anchor that explains the given instance w.r.t the model by using beam search.
        In each iteration, beam search keeps a set of good candidates.

        :param instance: instance to be explained
        :type instance: np.ndarray
        :param model: the model that is estimated by the anchor
        :type model: model
        :param tau: desired level of precision, defaults to 0.95
        :type tau: float, optional
        :return: anchor
        :rtype: TabularAnchor
        """        
        self.logger.debug(f"Start bottom-up search for {instance}.")
        # Init B anchors
        current_anchors = [TabularAnchor(self.cs, self.features)]
        best_anchor = current_anchors[0]
        rules = generate_rules_for_instance(self.quantiles, instance, self.feature2index)
        self.logger.debug(f"Generated rules: {rules}")
        random.shuffle(rules)
        i = 0
        while True:
            # generate candidates for multiple anchors
            candidates = []
            for a in current_anchors:
                candidates.extend(self.generate_candidates(a, rules, min_cov=best_anchor.coverage))
            
            if len(candidates) == 0:
                break
            
            current_anchors = get_b_best_candidates(candidates, instance, model, tau, B)

            for a in current_anchors:
                self.logger.info(f"Current best: P={a.mean} (based on {a.n_samples} samples), Rules: {a.rules}")

            sufficiently_precise_anchors = [a for a in current_anchors if a.mean > tau]
            for a in sufficiently_precise_anchors:
                cov = a.compute_coverage(self.X)
                if cov > best_anchor.coverage:
                    best_anchor = a
                    self.logger.info(f"Current best: P={best_anchor.mean} (based on {best_anchor.n_samples} samples), Rules: {best_anchor.rules}")

        self.logger.info(f"Found anchor: P={best_anchor.mean}, C={best_anchor.coverage}, Rules:{best_anchor.rules}")
        return best_anchor

    def generate_candidates(self, anchor, rules, min_cov=0):
        """Generates new anchor candidates by adding different rules to copies of the same anchor.

        for rules x, y, z, anchor a
        a1 := a + x
        a2 := a + y
        a3 := a + z
        :param anchor: Anchor to be expanded
        :type anchor: TabularAnchor
        :param rules: New rules not yet in anchor
        :type rules: list
        :return: New anchor candidates
        :rtype: list
        """    
        anchors_copy = [deepcopy(anchor) for _ in range(len(rules))]
        new_anchors = []
        for i, rule in enumerate(rules):
            # do not enforce different rules on the same feature
            if rule[0] in anchor.get_current_features():
                continue
            # reset bounds and add new rule
            new = anchors_copy[i]
            new.reset_bounds()

            new.add_rule(rule)
            new.compute_coverage(self.X)
            if new.coverage > min_cov:
                new_anchors.append(new)

        return new_anchors

def get_configspace_for_dataset(X : pd.DataFrame):
    """Creates a ConfigSpace for the given dataset.
    The hyperparameters bounds are taken from
    the respective minimum and maximum values of the features.

    :param X: Dataset for which to create the configspace
    :type X: pd.DataFrame
    :return: ConfigSpace
    :rtype: CS.ConfigurationSpace
    """    
    cs = CS.ConfigurationSpace()
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#basics-dtypes
    for f in X.columns:
        if X[f].dtype in ("category", "string", "object", "boolean"):
            hp = CSH.CategoricalHyperparameter(f, choices=X[f].unique())
        elif "Int" in str(X[f].dtype):
            hp = CSH.UniformIntegerHyperparameter(f, lower=X[f].min(), upper=X[f].max(), log=False)
        elif "float" in str(X[f].dtype):
            hp = CSH.UniformFloatHyperparameter(f, lower=X[f].min(), upper=X[f].max(), log=False)

        cs.add_hyperparameter(hp)
    return cs

def generate_rules_for_instance(quantiles, instance, feature2index):
    """Generate rules that are satisified by the given instance.
    Rules bounds are determined by given quantiles.
    Each quantile generates a 
    - greater than lower bound
    - smaller than upper bound
    - between upper and lower bound

    :param quantiles: Quantile bounds per feature
    :type quantiles: dict
    :param instance: The instance to be explained
    :type instance: np.ndarray
    :param feature2index: Mapping from feature name to index in instance
    :type feature2index: dict
    :return: List of Rule tuples
    :rtype: list
    """    
    if len(instance.shape) >= 2:
        instance = instance.squeeze(0)
    rules = []
    for f, f_quantile in quantiles.items():
        f_idx = feature2index[f]
        for i, bound in enumerate(f_quantile):
            if i == 0:
                if instance[f_idx] <= bound:
                    rules.append((f, "<=", bound))
                    break

            if i == len(f_quantile) - 1:
                if instance[f_idx] >= bound:
                    rules.append((f, ">=", bound))
                    break
            
            if instance[f_idx] > bound and instance[f_idx] < f_quantile[i+1]:
                rules.append((f, ">=", bound, "<=", f_quantile[i+1]))
                rules.append((f, ">=", bound))
                rules.append((f, "<=", f_quantile[i+1]))
                break

    return rules




