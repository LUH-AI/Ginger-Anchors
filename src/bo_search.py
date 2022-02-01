import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np

from custom_anchor import TabularAnchor

def evaluate_rules_from_cs(configuration, **kwargs):
    print(kwargs)
    features = 0
    model = None
    instance = None
    y = model(instance)
    iterations = 100
    # TODO: need: dataset for coverage, features, model, original instance / class
    # TODO: construct configspace from configuration
    cs = ...
    anchor = TabularAnchor(cs, features)
    anchor.compute_coverage()

    for i in iterations:
        a_x = anchor.sample_instance()
        a_y = model.predict(a_x)
        anchor.n_samples += 1
        if a_y == y:
            anchor.correct += 1
    return 1 - (anchor.mean + anchor.coverage)