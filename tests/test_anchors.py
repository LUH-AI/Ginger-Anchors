import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import sys
import os
cwd = os.getcwd()
sys.path.insert(0, cwd + "/src")

from custom_anchor import TabularAnchor
from explainer import Explainer, generate_rules_for_instance



@pytest.fixture(scope="session")
def prepared_data():
    print("Preparing data...")
    data = pd.read_csv("data/wheat_seeds.csv")
    X_df = data.drop(columns=["Type"])
    X = data.drop(columns=["Type"]).to_numpy()
    y = data["Type"].to_numpy()
    instance = X[3].reshape(1, -1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # prepare model
    print("Preparing classifier...")
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(sum(preds == y_test) / len(y_test))

    prepared = {}
    prepared["model"] = model
    prepared["X"] = X_df
    prepared["instance"] = instance
    return prepared

def test_empty_anchor_coverage(prepared_data):
    X = prepared_data["X"]
    exp = Explainer(X)
    anchor = TabularAnchor(exp.cs, exp.features)
    assert anchor.compute_coverage(prepared_data["X"]) == 1

def test_generate_candidates(prepared_data):
    # test whether rules are added correctly
    exp = Explainer(prepared_data["X"])
    anchor = TabularAnchor(exp.cs, exp.features)
    rules = [(exp.features[0], ">=", 0), (exp.features[1], ">=", 1)]
    candidates = exp.generate_candidates(anchor, rules)
    assert len(candidates) == len(rules)

    # tests whether rules for features that already exist are skipped
    anchor = TabularAnchor(exp.cs, exp.features)
    anchor.add_rule((exp.features[0], ">=", 0))
    rules = [(exp.features[0], "<=", 1), (exp.features[1], ">=", 1)]
    candidates = exp.generate_candidates(anchor, rules)
    assert len(candidates) == len(rules) - 1

def test_sampling_instance(prepared_data):
    # test for correct order of features by checking feature range
    X = prepared_data["X"]
    exp = Explainer(X)
    anchor = TabularAnchor(exp.cs, exp.features)
    x = anchor.sample_instance()
    assert x.shape == (1, len(exp.features))
    for f,i in exp.feature2index.items():
        assert x[0][i] >= X[f].min()
        assert x[0][i] <= X[f].max()
    

def test_rule_generation(prepared_data):
    # test rule generation for given instance and quantiles
    exp = Explainer(prepared_data["X"])
    instance = prepared_data["instance"]
    # [[13.84   13.94    0.8955  5.324   3.379   2.259   4.805 ]]
    # Quantiles:
    # Area [12.33  14.43  17.455]
    # Perimeter [13.47  14.37  15.805]
    # Compactness [0.8571 0.8734 0.8868]
    # Kernel.Length [5.267 5.541 6.002]
    # Kernel.Width [2.9545 3.245  3.5645]
    # Asymmetry.Coeff [2.57  3.631 4.799]
    # Kernel.Groove [5.046 5.228 5.879]
    want_rules = [
        ("Area", ">=", 12.33),
        ("Area", "<=", 14.43),
        ("Area", ">=", 12.33,"<=", 14.43),
        ("Perimeter", ">=", 13.47),
        ("Perimeter", "<=", 14.37),
        ("Perimeter", ">=", 13.47,"<=", 14.37),
        ("Compactness", ">=", 0.8868),
        ("Kernel.Length", ">=", 5.267),
        ("Kernel.Length", "<=", 5.541),
        ("Kernel.Length", ">=", 5.267,"<=", 5.541),
        ("Kernel.Width", ">=", 3.245),
        ("Kernel.Width", "<=", 3.5645),
        ("Kernel.Width", ">=", 3.245,"<=", 3.5645),
        ("Asymmetry.Coeff", "<=", 2.57),
        ("Kernel.Groove", "<=", 5.046)
    ]
    got_rules = generate_rules_for_instance(exp.quantiles, instance, exp.feature2index)
    assert len(got_rules) == len(want_rules)
    for r in got_rules:
        l = list(r)
        l[2] = round(l[2], 4)
        got_rules[got_rules.index(r)] = tuple(l)
    for r in want_rules:
        assert r in got_rules


def test_add_rule(prepared_data):
    # test adding rules to anchor
    exp = Explainer(prepared_data["X"])
    anchor = TabularAnchor(exp.cs, exp.features)
    anchor.add_rule((exp.features[1], ">=", 2))
    hp = anchor.cs.get_hyperparameter(exp.features[1])
    assert len(anchor.gt) == 1
    assert hp.lower == 2
    assert isinstance(hp, CSH.UniformFloatHyperparameter)

    anchor.add_rule((exp.features[0], ">=", 2, "<=", 3))
    hp = anchor.cs.get_hyperparameter(exp.features[0])
    assert len(anchor.gt) == 2 and len(anchor.lt) == 1
    assert hp.lower == 2 and hp.upper == 3
    assert isinstance(hp, CSH.UniformFloatHyperparameter)

    hp = anchor.cs.get_hyperparameter(exp.features[2])
    assert hp.lower == prepared_data["X"][exp.features[2]].min()

def test_instance_satisfies_anchor_bottomup(prepared_data):
    exp = Explainer(prepared_data["X"])
    anchor = exp.explain_bottom_up(prepared_data["instance"], prepared_data["model"], tau=0.95)
    assert anchor.is_satisfied(prepared_data["instance"])
    assert anchor.mean > 0.95
    assert len(anchor.rules) > 1
    
def test_instance_satisfies_anchor_beam(prepared_data):
    exp = Explainer(prepared_data["X"])
    anchor = exp.explain_beam_search(prepared_data["instance"], prepared_data["model"], B=3, tau=0.95)
    assert anchor.is_satisfied(prepared_data["instance"])
    assert anchor.mean > 0.95
    assert len(anchor.rules) > 1


def test_instance_satisfies_bayesian_optimization(prepared_data):
    exp = Explainer(prepared_data["X"])
    anchor = exp.explain_bayesian_optimiziation(prepared_data["instance"], prepared_data["model"], evaluations=30, tau=0.7)
    assert anchor.is_satisfied(prepared_data["instance"])
    assert anchor.mean > 0.7
    assert len(anchor.rules) > 1
