from copy import deepcopy
import os
from shutil import rmtree
import time

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import pandas as pd
import numpy as np
import random

from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
from functools import partial

from custom_anchor import TabularAnchor
from lucb import get_best_candidate, get_b_best_candidates
from utils import new_logger
from bo_search import evaluate_rules_from_cs


class Explainer:

    def __init__(self, X : pd.DataFrame, seed=42, logger=None) -> None:
        """An explainer object from which explanations
        (aka anchors) for single instances can be computed.

        :param X: Dataset used to train the model
        :type X: pd.DataFrame
        :param seed: seed for configspace
        :type seed: int
        """
        if logger is None:
            self.logger = new_logger(self.__class__.__name__)
        else:
            self.logger = logger

        self.X = X
        self.features = list(X.columns)
        self.quantiles = {}
        # { feature : [bound1, bound2, bound3] }
        for f in self.features:
            # More quantiles might lead to less predicates per anchor but 
            # leads to tighter rules (worse coverage)
            self.quantiles[f] = np.quantile(X[f], [0.25, 0.5, 0.75])
        
        self.feature2index = {f : self.features.index(f) for f in self.features}
        self.cs = get_configspace_for_dataset(X, seed)
        self.seed = seed
        

    def explain_bottom_up(self, instance, model, tau=0.95, delta=0.1, epsilon=0.2):
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
        prediction = model.predict(instance)[0]
        anchor = TabularAnchor(self.cs, self.features, self.seed, prediction)
        # get quantiles of instance
        rules = generate_rules_for_instance(self.quantiles, instance, self.feature2index)
        self.logger.debug(f"Generated rules: {rules}")
        random.shuffle(rules)
        while True:
            # add unused rules to current anchor
            candidates = self.generate_candidates(anchor, rules)
            # no candidates found
            if candidates == []:
                return None
            # treat anchors as Mulit-Armed Bandidates
            anchor = get_best_candidate(candidates, instance, model, delta, epsilon)
            self.logger.debug(f"Current best: P={anchor.mean} (based on {anchor.n_samples} samples), Rules: {anchor.rules}")
            if anchor.mean >= tau:
                break
        
        anchor.compute_coverage(self.X)
        self.logger.debug(f"Found anchor: P={anchor.mean}, C={anchor.coverage}, Rules:{anchor.rules}")
        return anchor

    def explain_beam_search(self, instance, model, tau=0.95, B=1, delta=0.1, epsilon=0.2, timeout=60, seed=42):
        """
        Finds in anchor that explains the given instance w.r.t the model by using beam search.
        In each iteration, beam search keeps a set of good candidates.

        :param instance: instance to be explained
        :type instance: np.ndarray
        :param model: the model that is estimated by the anchor
        :type model: model
        :param tau: desired level of precision, defaults to 0.95
        :type tau: float, optional
        :param B: B candidates to hold
        :type B: int, optional
        :param delta: Hyperparameter, higher leads to less exploration, in [0,1], defaults to 0.1
        :type delta: float, optional
        :param epsilon: Break condition, desired difference of ub and lb, defaults to 0.2
        :type epsilon: float, optional
        :param timeout: Maximum compute time in seconds, defaults to 60
        :type timeout: int, optional
        :param seed: Random seed, defaults to 42
        :type seed: int, optional
        :return: anchor
        :rtype: TabularAnchor
        """
        random.seed(seed)
        np.random.seed(seed)      
        self.logger.debug(f"Start bottom-up search for {instance}.")
        # Init B anchors
        prediction = model.predict(instance)[0]
        current_anchors = [TabularAnchor(self.cs, self.features, cls=prediction)]
        best_anchor = current_anchors[0]
        rules = generate_rules_for_instance(self.quantiles, instance, self.feature2index)
        self.logger.debug(f"Generated rules: {rules}")
        random.shuffle(rules)

        start = time.time()
        trajectory = []
        while time.time() - start < timeout:
            # generate candidates for multiple anchors
            candidates = []
            for a in current_anchors:
                candidates.extend(self.generate_candidates(a, rules, min_cov=best_anchor.coverage))
            
            if len(candidates) == 0:
                break
            
            current_anchors = get_b_best_candidates(candidates, instance, model, B, delta, epsilon)

            level_traj = []
            for a in current_anchors:
                a.compute_coverage(self.X)
                level_traj.append((a.mean, a.n_samples, a.coverage))
                self.logger.debug(f"Current best: P={a.mean} (based on {a.n_samples} samples), Rules: {a.rules}")
            trajectory.append(level_traj)

            sufficiently_precise_anchors = [a for a in current_anchors if a.mean > tau]
            for a in sufficiently_precise_anchors:
                cov = a.compute_coverage(self.X)
                if cov > best_anchor.coverage:
                    best_anchor = a
                    self.logger.debug(f"Current best: P={best_anchor.mean} (based on {best_anchor.n_samples} samples), Rules: {best_anchor.rules}")

        if time.time() - start > timeout:
            return None
            
        self.logger.debug(f"Found anchor: P={best_anchor.mean}, C={best_anchor.coverage}, Rules:{best_anchor.rules}")
        best_anchor.trajectory = trajectory
        return best_anchor

    def explain_bayesian_optimiziation(self, instance, model, tau=0.95, evaluations=100, samples_per_iteration=100, seed=42, keep_trajectory=False):
        """Find rules via Bayesion Optimization.
        Each feature in the dataset gets a lower and upper bound hyperparameter.
        Then, SMAC samples bounds aka rules that are evaluated w.r.t precision and coverage.

        :param instance: instance to be explained
        :type instance: np.ndarray
        :param model: the model that is estimated by the anchor
        :type model: model
        :param tau: desired level of precision, defaults to 0.95
        :type tau: float, optional
        :param evaluations: number of sample evaluations for smac, defaults to 100
        :type evaluations: int, optional
        :param samples_per_iteration: number of samples for anchor evaluation, defaults to 100
        :type samples_per_iteration: int, optional
        :param seed: random seed for smac, defaults to 42
        :type seed: int, optional
        :param keep_trajectory: Whether to delete the smac log or not, defaults to False
        :type keep_trajectory: boolean, optional
        :return: anchor
        :rtype: TabularAnchor
        """        
        if len(instance.shape) == 2:
            inst = instance.squeeze(0)
        smac_cs = CS.ConfigurationSpace()

        for f in self.features:
            hp = self.cs.get_hyperparameter(f)

            low_hp =  hp.__class__(f + "_lower", lower=hp.lower, upper=inst[self.features.index(f)], log=False)
            up_hp = hp.__class__(f + "_upper", lower=inst[self.features.index(f)], upper=hp.upper, log=False)
            smac_cs.add_hyperparameters([low_hp, up_hp])

            lower_mask = CSH.CategoricalHyperparameter(f + "_lower_mask", choices=[0, 1])
            upper_mask = CSH.CategoricalHyperparameter(f + "_upper_mask", choices=[0, 1])
            smac_cs.add_hyperparameters([lower_mask, upper_mask])

            cat_lower = CS.EqualsCondition(low_hp, lower_mask, 0)
            cat_upper = CS.EqualsCondition(up_hp, upper_mask, 0)
            smac_cs.add_condition(cat_lower)
            smac_cs.add_condition(cat_upper)

        self.logger.debug(f"Configspace for SMAC: {smac_cs}")

        output_dir = "./smac_run"
        scenario = Scenario({
            "run_obj" : "quality",
            "runcount-limit" : evaluations,
            "cs" : smac_cs,
            "output_dir" : output_dir,
        })
        self.logger.info(f"Running smac for {evaluations} evaluations, write to {output_dir}")

        # pass all arguments except configuration which is done by smac
        tae_func = partial(
            evaluate_rules_from_cs,
            model=model,
            X=self.X,
            features=self.features,
            explain=instance,
            iterations=samples_per_iteration
            )

        optimizer = SMAC4HPO(
            scenario=scenario,
            rng=np.random.RandomState(seed),
            tae_runner=tae_func,
        )
        _ = optimizer.optimize()

        # gather all configurations that were good enough
        run_dirs = sorted(os.listdir(output_dir), key=lambda x : len(x))
        newest_run_dir = run_dirs[0]
        runs = pd.read_json(f"{output_dir}/{newest_run_dir}/traj.json", lines=True)

        runs = runs[runs["cost"] < 1 - tau]
        self.logger.debug(f"Found {len(runs)} incumbents matching {tau=}.")
        if len(runs) == 0:
            return None

        anchor_candidates = []
        for _, run in runs.iterrows():
            a = generate_anchor_from_configuration(self.cs, run["incumbent"], self.features)
            anchor_candidates.append(a)

        # sample for precision estimate
        y = model.predict(instance)
        for anchor in anchor_candidates:
            for _ in range(300):
                a_x = anchor.sample_instance()
                a_y = model.predict(a_x)
                anchor.n_samples += 1
                if a_y == y:
                    anchor.correct += 1
            anchor.compute_coverage(self.X)
        
        if not keep_trajectory:
            self.logger.debug("Removing smac folder content")
            rmtree(f"{output_dir}/{newest_run_dir}")
        
        # return best coverage
        anchor_candidates = sorted(anchor_candidates, key=lambda a : a.coverage)
        best_cov_anchor = anchor_candidates[-1]
        best_cov_anchor.cls = model.predict(instance)[0]
        return best_cov_anchor

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
        :param min_cov: Minimum coverage for new candidates
        :type min_cov: float
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

def get_configspace_for_dataset(X : pd.DataFrame, seed=42):
    """Creates a ConfigSpace for the given dataset.
    The hyperparameters bounds are taken from
    the respective minimum and maximum values of the features.

    :param X: Dataset for which to create the configspace
    :type X: pd.DataFrame
    :param seed: random seed for the configspace, defaults to 42
    :type seed: int, optional
    :return: ConfigSpace
    :rtype: CS.ConfigurationSpace
    """    
    cs = CS.ConfigurationSpace(seed)
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

def generate_anchor_from_configuration(cs, configuration, features):
    """Construct an anchor object according to the bounds of the configuration.

    :param cs: ConfigSpace containing the range of the whole dataset
    :type cs: CS.ConfigurationSpace
    :param configuration: Configuration that contains upper and lower bounds for each features
    :type configuration: cs.configuration_space.Configuration
    :param features: names of all features
    :type features: list
    :return: anchor
    :rtype: TabularAnchor
    """    
    anchor = TabularAnchor(cs, features)
    for f in features:
        lower_bound = configuration.get(f + "_lower")
        upper_bound = configuration.get(f + "_upper")
        if not upper_bound and not lower_bound:
            continue
        elif not lower_bound and upper_bound:
            rule = f, "<=", upper_bound
        elif lower_bound and not upper_bound:
            rule = f, ">=", lower_bound
        else:
            rule = f, "<=", upper_bound, ">=", lower_bound
        anchor.add_rule(rule)
    return anchor