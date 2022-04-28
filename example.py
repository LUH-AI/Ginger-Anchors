import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ginger_anchors.explainer import Explainer

if "__main__" == __name__:
    # Prepare data
    print("Preparing data ...")
    data = pd.read_csv("data/wheat_seeds.csv")
    i_idx = 111
    X_df = data.drop(columns=["Type"])
    X = data.drop(columns=["Type"]).to_numpy()
    y = data["Type"].to_numpy()
    instance = X[i_idx].reshape(1, -1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Prepare model
    print("Preparing classifier ...")
    model = RandomForestClassifier(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Classifier accuracy:", sum(preds == y_test) / len(y_test))

    exp = Explainer(X_df)

    # Get beam search explanation
    anchor = exp.explain_beam_search(instance, model, tau=0.95, B=3)
    print("\nBeam search explanation:")
    print(anchor.get_explanation())

    # Get smac explanation
    anchor = exp.explain_bayesian_optimiziation(
        instance, model, evaluations=64, samples_per_iteration=500, tau=0.8
    )
    print("\nSMAC explanation")
    print(anchor.get_explanation())
