import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np

from custom_anchor import TabularAnchor

def evaluate_rules_from_cs(configuration, model, X, features, explain, iterations):
    y = model.predict(explain)
    # TODO: construct configspace from configuration
    cs = CS.ConfigurationSpace()
    anchor = TabularAnchor(cs, features)
    anchor.compute_coverage(X)

    for i in iterations:
        a_x = anchor.sample_instance()
        a_y = model.predict(a_x)
        anchor.n_samples += 1
        if a_y == y:
            anchor.correct += 1
    return 1 - (anchor.mean + anchor.coverage)