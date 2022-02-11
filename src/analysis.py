import itertools
import time
from datetime import datetime
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from explainer import Explainer

# prepare data
def setup():
    print("Preparing data...")
    data = pd.read_csv("data/wheat_seeds.csv")
    # y = pd.read_csv("data/german_labels.csv")
    # X_df = data
    # X = data
    X_df = data.drop(columns=["Type"])
    X = data.drop(columns=["Type"]).to_numpy()
    y = data["Type"].to_numpy()
    instance = X[155].reshape(1, -1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # prepare model
    print("Preparing classifier...")
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Classifier accuracy:", sum(preds == y_test) / len(y_test))
    return model, X, X_df

def log_run(anchor, elapsed_time, b, d, e, logfile):
    if anchor is None:
        prec = 0
        coverage = 0
        rules = []
        traj = []
        samples = 0
    else:
        prec = anchor.mean
        coverage = anchor.coverage
        rules = anchor.rules
        traj = anchor.trajectory
        samples = anchor.n_samples

    run_dict = {
        "B" : b,
        "delta" : d,
        "epsilon" : e,
        "precision" : prec,
        "coverage" : coverage,
        "rules" : rules,
        "clock_time" : elapsed_time,
        "n_samples" : samples,
        "trajectory_pnc" : traj
    }

    run_str = json.dumps(run_dict)
    with open(logfile, "a") as f:
        f.write(run_str + "\n")
    return


# for different 𝐵, 𝛿 and 𝜖 collect:
#   - runtime + n_samples
#   - precision
#   - coverage   
#   - bounds
# -> handle timeouts
#
# nice to have -> samples, precision and bounds per arm for one run
# parallel coordinates trajectory

# analysing BO

# number of evaluations -> coverage, precision
# quantization factor

def run_analysis(B, delta, epsilon, exp, model, instance, timeout, logfile, seed):
    for b, d, e in itertools.product(B, delta, epsilon):
        start_time = time.time()
        anchor = exp.explain_beam_search(instance, model, tau=0.95, B=b, delta=d, epsilon=e, timeout=timeout, seed=seed)
        end_time = time.time()
        if anchor is None:
            elapsed_time = -1
        else:
            elapsed_time = end_time - start_time
        log_run(anchor, elapsed_time, b, d, e, logfile)
    return

if "__main__" == __name__:
    now = datetime.now()
    seeds = [42, 55, 87, 1337]
    instances = [3, 111, 155]
    model, X, X_df = setup()
    timeout = 200
    # grid search
    for instance_idx in instances:
        instance = X[instance_idx].reshape(1, -1)
        for seed in seeds:
            exp = Explainer(X_df, seed)
            logfile = f"final_analysis_{now.strftime('%d.%m.%y_%H_%M_%S')}_{seed}_{instance_idx}.jsonl"
            B = [1, 2, 3, 4, 5, 6, 7]
            delta = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
            # delta bigger -> beta smaller -> bounds less far from mean -> more confident in our sampled precision
            epsilon = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
            run_analysis(B, delta, epsilon, exp, model, instance, timeout, logfile, seed)