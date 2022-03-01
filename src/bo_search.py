import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np

from custom_anchor import TabularAnchor

def evaluate_rules_from_cs(configuration, model, X, features, explain, iterations):
    """Creates an anchor from the rules generated from the configuration
    and estimates the precision.


    :param configuration: Configuration that contains upper and lower bounds for each features
    :type configuration: CS.configuration_space.Configuration
    :param model: The model to be estimated by the anchor
    :type model: model
    :param X: Dataset
    :type X: pd.Dataframe
    :param features: all features of the dataset
    :type features: list
    :param explain: Instance to be explained
    :type explain: np.ndarray
    :param iterations: number of samples to calculate precision
    :type iterations: int
    :return: 1 - precision
    :rtype: float
    """    
    y = model.predict(explain)[0]

    cs = create_configspace_from_configuration(configuration, features, X)
 
    anchor = TabularAnchor(cs, features, y)
    anchor.compute_coverage(X)

    for _ in range(iterations):
        a_x = anchor.sample_instance()
        a_y = model.predict(a_x)
        anchor.n_samples += 1
        if a_y == y:
            anchor.correct += 1
    # maximize precision
    return 1 - anchor.mean

def create_configspace_from_configuration(configuration, features, X):
    """Creates a configspace from the given bounds of the configuration.

    :param configuration: Configuration that contains upper and lower bounds for each features
    :type configuration: CS.configuration_space.Configuration
    :param features: list of all features in the dataset
    :type features: list
    :param X: Dataset
    :type X: pd.DataFrame
    :return: ConfigSpace with new bounds
    :rtype: CS.ConfigurationSpace
    """    
    cs = CS.ConfigurationSpace()
    for f in features:
        lm = configuration.get(f + "_lower_mask")
        um = configuration.get(f + "_upper_mask")
        if um == 1:
            upper_bound = X[f].max()
        else:
            upper_bound = configuration.get(f + "_upper")
        if lm == 1:
            lower_bound = X[f].min()
        else:
            lower_bound = configuration.get(f + "_lower")
            
        f_hp = CSH.UniformFloatHyperparameter(f, lower=lower_bound, upper=upper_bound, log=False)
        cs.add_hyperparameter(f_hp)
    return cs
