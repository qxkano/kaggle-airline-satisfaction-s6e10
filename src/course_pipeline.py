"""Build the course SQLite database, quality audit, grouped tables and figures."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "results/course"
DB = ROOT / "data/processed/airline.sqlite"
RATINGS = ["Inflight wifi service", "Departure/Arrival time convenient",
           "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
           "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
           "Baggage handling", "Checkin service", "Cleanliness"]
CATEGORIES = {"Gender": {"Female", "Male"}, "Customer Type": {"Loyal Customer", "disloyal Customer"},
              "Type of Travel": {"Business travel", "Personal Travel"}, "Class": {"Business", "Eco", "Eco Plus"}}
ARR = "Arrival Delay in Minutes"
DEP = "Departure Delay in Minutes"
FIELDS = ["id", "Age", "Flight Distance", *RATINGS, DEP, ARR, *CATEGORIES]


def check(ok, message):
    if not ok:
        raise ValueError(message)


def prepare(frame, split):
    expected = FIELDS + (["satisfaction"] if split == "train" else [])
    check(set(frame.columns) == set(expected), f"{split}: unexpected CSV columns")
    for col, choices in CATEGORIES.items():
        frame[col] = frame[col].str.strip()
        check(frame[col].isin(choices).all(), f"{split}: unknown {col}")
    for col in ["id", "Age", "Flight Distance", *RATINGS, DEP, ARR]:
        values = pd.to_numeric(frame[col], errors="raise")
        observed = values.dropna()
        check(((observed >= 0) & (observed % 1 == 0)).all(), f"{split}: invalid {col}")
        check(col == ARR or values.notna().all(), f"{split}: missing {col}")
        if col in RATINGS:
            check(observed.le(5).all(), f"{split}: invalid rating {col}")
        frame[col] = values
    if split == "train":
        label = frame["satisfaction"].astype("string").str.lower().map({"true": 1, "false": 0})
        check(label.notna().all(), "unknown or missing training label")
        frame["satisfaction"] = label
    else:
        frame["satisfaction"] = None
    frame["split"] = split
    return frame


def build(conn):
    types = [f'"{c}" ' + ("TEXT" if c in CATEGORIES else "INTEGER") +
             (" PRIMARY KEY" if c == "id" else "" if c == ARR else " NOT NULL") for c in FIELDS]
    conn.execute("CREATE TABLE responses (" + ", ".join(types) +
                 ", split TEXT NOT NULL, satisfaction INTEGER, CHECK ((split='train' AND satisfaction IS NOT NULL AND satisfaction IN (0,1)) OR (split='test' AND satisfaction IS NULL)))")
    audit = []
    for split in ["train", "test"]:
        source = RAW / f"{split}.csv"
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        rows = missing = 0
        for frame in pd.read_csv(source, chunksize=50000):
            frame = prepare(frame, split)
            rows += len(frame)
            missing += int(frame[ARR].isna().sum())
            frame.to_sql("responses", conn, if_exists="append", index=False)
        stored, nulls, labels = conn.execute(
            f'SELECT COUNT(*), COUNT(*)-COUNT("{ARR}"), COUNT(satisfaction) FROM responses WHERE split=?',
            (split,)).fetchone()
        check(rows > 0 and (rows, missing) == (stored, nulls), "source/database mismatch")
        check(labels == (rows if split == "train" else 0), "label count mismatch")
        audit.append(dict(source=source.name, sha256=digest, input_rows=rows, stored_rows=stored,
                          missing_arrival=nulls, known_labels=labels))
    conn.execute('CREATE INDEX ix_split_class ON responses(split,"Class","Type of Travel")')
    check(conn.execute("PRAGMA integrity_check").fetchone() == ("ok",), "database integrity check failed")
    return audit


def summarize(conn, audit):
    groups = {"cabin": '"Class" AS cabin',
              "cabin_travel": '"Class" AS cabin, "Type of Travel" AS travel',
              "arrival": """CASE WHEN "Arrival Delay in Minutes" IS NULL THEN 'missing'
              WHEN "Arrival Delay in Minutes"=0 THEN '0' WHEN "Arrival Delay in Minutes"<=15 THEN '1-15'
              WHEN "Arrival Delay in Minutes"<=60 THEN '16-60' ELSE '61+' END AS band"""}
    tables = {}
    for name, dimensions in groups.items():
        keys = "1,2" if name == "cabin_travel" else "1"
        sql = f"""SELECT {dimensions}, COUNT(*) AS n, SUM(satisfaction) AS satisfied,
                  AVG(satisfaction) AS rate FROM responses WHERE split='train'
                  GROUP BY {keys} ORDER BY {keys}"""
        table = pd.read_sql_query(sql, conn)
        check(table.n.sum() == audit[0]["stored_rows"], "incomplete group coverage")
        tables[name] = table
    return tables


def plot(tables):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    wide = tables["cabin_travel"].pivot(index="cabin", columns="travel", values="rate")
    (100 * wide).plot.bar(ax=axes[0], rot=0, color=["#176b87", "#cd8131"])
    axes[0].legend(title="Travel purpose", fontsize=8)
    axes[0].set(xlabel="Cabin class", title="A. Cabin and travel purpose")
    delay = tables["arrival"].set_index("band").loc[["0", "1-15", "16-60", "61+", "missing"]]
    axes[1].bar(delay.index, 100 * delay.rate, color=["#176b87"] * 4 + ["#858585"])
    axes[1].set(xlabel="Arrival delay (minutes)", title="B. Observed and missing delays")
    for ax in axes:
        ax.set(ylim=(0, 100), ylabel="Satisfied passengers (%)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "satisfaction_groups.png", dpi=180)
    plt.close(fig)


def main():
    DB.parent.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=DB.parent) as temp:
        candidate = Path(temp) / DB.name
        conn = sqlite3.connect(candidate)
        try:
            audit = build(conn)
            tables = summarize(conn, audit)
            conn.commit()
        finally:
            conn.close()
        os.replace(candidate, DB)
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / "quality.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    plot(tables)
    print(f"Stored {sum(item['stored_rows'] for item in audit):,} rows in {DB}")
    print(f"Course results: {OUT}")


if __name__ == "__main__":
    main()
