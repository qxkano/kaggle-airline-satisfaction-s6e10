"""Compare fitted and held-out scores; summarize predictions without test labels."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score


def write_stage_analysis(out, fit_y, fit_p, val_y, val_p, test, test_p):
    groups = []
    for group, y, p in [("fit", fit_y, fit_p), ("holdout", val_y, val_p)]:
        groups.append(dict(split=group, n=len(y), auc=roc_auc_score(y, p),
                           accuracy=accuracy_score(y, p >= .5), f1=f1_score(y, p >= .5),
                           recall=recall_score(y, p >= .5)))
    metrics = pd.DataFrame(groups)
    metrics.to_csv(out / "fit_holdout.csv", index=False)
    scored = pd.DataFrame({"cabin":test["Class"], "p":test_p, "positive":test_p >= .5})
    by = scored.groupby("cabin").agg(n=("p","size"), mean_score=("p","mean"),
                                     classified_positive=("positive","sum"), positive_share=("positive","mean"))
    by.to_csv(out / "prediction_cabin.csv")
    edges = np.linspace(0, 1, 11)
    # Each interval includes its left boundary; the final interval also includes 1.
    counts, _ = np.histogram(test_p, bins=edges)
    assert counts.sum() == len(scored) and by.n.sum() == len(scored)
    pd.DataFrame({"lower":edges[:-1], "upper":edges[1:], "n":counts,
                  "share":counts/len(scored)}).to_csv(out / "prediction_bins.csv", index=False)

    fonts = {f.name for f in font_manager.fontManager.ttflist}
    zh = next((f for f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC") if f in fonts), None)
    def label(cn, en):
        return cn if zh else en
    with plt.rc_context({"font.family": zh or "DejaVu Sans", "axes.unicode_minus":False}):
        fig, ax = plt.subplots(figsize=(7, 2.9), layout="constrained")
        cols = ["auc","accuracy","f1","recall"]
        x = np.arange(len(cols))
        for j, (_, row) in enumerate(metrics.iterrows()):
            y = row[cols].astype(float)
            bars = ax.bar(x+(j-.5)*.38, y, .38, color=["#19697d","#d38b3a"][j],
                          label=[label("拟合集","Fit"),label("留出集","Holdout")][j])
            ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
        ax.set(ylim=(0,1.09), xticks=x, xticklabels=["ROC-AUC",label("准确率","Accuracy"),"F1",label("召回率","Recall")])
        ax.legend(loc="lower right", ncol=2); fig.savefig(out/"fit_holdout.png", dpi=200); plt.close(fig)
        fig, (a,b) = plt.subplots(1,2,figsize=(8.3,3.2),layout="constrained")
        a.bar(edges[:-1]+.05, 100*counts/len(scored), width=.086, color="#19697d")
        a.axvline(.5,color="#8c4a12",linestyle="--",lw=1.2)
        a.set(xlim=(0,1),xlabel=label("模型输出的满意概率","Satisfaction score"),ylabel=label("预测集记录占比（%）","Share of records (%)"))
        bars=b.bar([label("商务舱","Business"),label("经济舱","Eco"),label("超级经济舱","Eco Plus")],
                   100*by.loc[["Business","Eco","Eco Plus"],"positive_share"],color="#438775")
        b.bar_label(bars, fmt="%.1f%%",padding=2)
        b.set(ylim=(0,100),ylabel=label("判为满意的比例（%）","Classified satisfied (%)"))
        fig.savefig(out/"prediction_profile.png",dpi=200);plt.close(fig)


def write_depth_curve(out, model, fit_x, fit_y, val_x, val_y):
    """Score the fitted tree when traversal stops at successive depths."""
    transform, clf = model.steps[0][1], model.steps[-1][1]
    tree = clf.tree_
    parent = np.arange(tree.node_count)
    depths = np.zeros(tree.node_count, dtype=int)
    for node, children in enumerate(zip(tree.children_left, tree.children_right)):
        for child in children:
            if child >= 0:
                parent[child] = node
                depths[child] = depths[node] + 1
    values = tree.value[:, 0, :]
    positive = values[:, list(clf.classes_).index(1)] / values.sum(axis=1)
    rows = []
    for split, x, y in [("fit", fit_x, fit_y), ("holdout", val_x, val_y)]:
        nodes = clf.apply(transform.transform(x))
        for limit in range(clf.get_depth(), -1, -1):
            scores = positive[nodes]
            rows.append((limit, split, roc_auc_score(y, scores)))
            nodes = np.where(depths[nodes] == limit, parent[nodes], nodes)
    table = pd.DataFrame(rows, columns=["depth", "split", "auc"]).pivot(
        index="depth", columns="split", values="auc").sort_index()
    assert np.allclose(table.loc[0], .5)
    reference = pd.read_csv(out / "fit_holdout.csv").set_index("split")
    assert np.allclose(table.iloc[-1], reference.loc[table.columns, "auc"])
    table.to_csv(out / "tree_depth_auc.csv")
    chinese = next((f.name for f in font_manager.fontManager.ttflist
                    if f.name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC")), None)
    with plt.rc_context({"font.family": chinese or "DejaVu Sans"}):
        fig, ax = plt.subplots(figsize=(7, 3.15), layout="constrained")
        ax.plot(table.index, table.fit, color="#22576b", marker="o", markersize=7,
                label="拟合集" if chinese else "Fit")
        ax.plot(table.index, table.holdout, color="#cb7424", linestyle="--", marker="x",
                label="留出集" if chinese else "Holdout")
        ax.set(xlabel="预测时允许经过的最大深度" if chinese else "Maximum traversal depth",
               ylabel="ROC-AUC", xticks=table.index, ylim=(.47, 1))
        ax.legend(); ax.grid(axis="y", alpha=.2)
        fig.savefig(out / "tree_depth_auc.png", dpi=200)
        plt.close(fig)
