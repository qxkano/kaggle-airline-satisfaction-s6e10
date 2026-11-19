"""Fit and evaluate a single decision tree on the cleaned course database."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

from stage_analysis import write_depth_curve, write_stage_analysis
from course_pipeline import CATEGORIES, DB, OUT, ROOT

SEED = 42


def draw(model, y, scores, classes):
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.6), layout="constrained")
    fpr, tpr, _ = roc_curve(y, scores)
    a.plot(fpr, tpr, color="#176b87", label=f"AUC = {roc_auc_score(y, scores):.4f}")
    a.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    a.set(xlabel="False positive rate", ylabel="True positive rate", title="A. Holdout ROC")
    a.legend(loc="lower right")
    ConfusionMatrixDisplay.from_predictions(y, classes, ax=b, display_labels=["Other", "Satisfied"],
                                            cmap="Blues", colorbar=False, values_format=",d")
    b.set_title("B. Holdout classification")
    fig.savefig(OUT / "tree_evaluation.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 5), layout="constrained")
    transform, tree = model.steps[0][1], model.steps[1][1]
    plot_tree(tree, feature_names=list(transform.get_feature_names_out()), max_depth=2,
              class_names=["Other", "Satisfied"], filled=True, rounded=True, impurity=False,
              proportion=True, fontsize=8, ax=ax)
    fig.savefig(OUT / "tree_rules.png", dpi=200)
    plt.close(fig)


def main():
    if not DB.is_file():
        raise FileNotFoundError("Build the database with course_pipeline.py first")
    with closing(sqlite3.connect(DB)) as conn:
        def read(split):
            return pd.read_sql_query("SELECT * FROM responses WHERE split=? ORDER BY id",
                                     conn, params=(split,), index_col="id").drop(columns="split")
        train, test = read("train"), read("test")
    y = train.pop("satisfaction").astype(int)
    test = test.drop(columns="satisfaction")
    fit_ids, val_ids = train_test_split(train.index, test_size=.2, stratify=y, random_state=SEED)
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True),
         [c for c in train if c not in CATEGORIES]),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(CATEGORIES))
    ], verbose_feature_names_out=False)
    model = make_pipeline(preprocess, DecisionTreeClassifier(criterion="entropy", max_depth=6,
                          min_samples_leaf=200, random_state=SEED))
    model.fit(train.loc[fit_ids], y.loc[fit_ids])
    yv = y.loc[val_ids]
    prob = model.predict_proba(train.loc[val_ids])[:, 1]
    pred = (prob >= .5).astype(int)
    metrics = dict(seed=SEED, fit=len(fit_ids), holdout=len(val_ids), features=train.shape[1],
                   auc=roc_auc_score(yv, prob), accuracy=accuracy_score(yv, pred),
                   precision=precision_score(yv,pred), recall=recall_score(yv,pred),
                   f1=f1_score(yv,pred), matrix=confusion_matrix(yv,pred).tolist())
    OUT.mkdir(parents=True, exist_ok=True)
    draw(model, yv, prob, pred)
    # Use the evaluated tree for both the saved model and the delivered predictions.
    path = ROOT / "models/course_tree.joblib"
    path.parent.mkdir(exist_ok=True)
    joblib.dump(model, path)
    scores = model.predict_proba(test)[:,1]
    assert np.array_equal(joblib.load(path).predict_proba(test.iloc[:100])[:,1], scores[:100])
    write_stage_analysis(OUT, y.loc[fit_ids], model.predict_proba(train.loc[fit_ids])[:, 1], yv, prob, test, scores)
    write_depth_curve(OUT, model, train.loc[fit_ids], y.loc[fit_ids], train.loc[val_ids], yv)
    submission = pd.DataFrame({"id":test.index, "satisfaction":scores})
    assert submission.id.is_unique and submission.satisfaction.between(0,1).all()
    submission.to_csv(OUT / "tree_predictions.csv", index=False)
    metrics["test_rows"] = len(test)
    metrics["depth"] = int(model.steps[-1][1].get_depth())
    metrics["leaves"] = int(model.steps[-1][1].get_n_leaves())
    (OUT / "tree_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
