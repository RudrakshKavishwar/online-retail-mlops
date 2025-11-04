# train_and_export.py
"""
Train churn model from Online Retail dataset and export pipeline.

Robust header normalization (maps common column name variants),
Excel/CSV support, optional sampling for fast runs, MLflow local logging,
and saves trained pipeline to exported_model/rf_churn_pipe.joblib.

Usage:
  python train_and_export.py --data data/online_retail_II.xlsx --mlflow
  python train_and_export.py --sample 0.05    # trains on 5% of rows (fast)
"""

import argparse
import os
import re
import sys
import pathlib
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score
import mlflow
import mlflow.sklearn

# ---------- Helpers: column normalization & loading ----------

def normalize_col_name(c):
    if pd.isna(c):
        return ""
    s = str(c).strip().lower()
    s = re.sub(r'[\s\-_]+', '', s)
    s = re.sub(r'[^\w]', '', s)
    return s

CANON_TOKENS = {
    "InvoiceDate": ["invoicedate", "invoicedatetime", "date"],
    "InvoiceNo": ["invoiceno", "invoicenumber", "invoice", "invoiceid"],
    "CustomerID": ["customerid", "custid", "customer", "customeridnumber"],
    "UnitPrice": ["unitprice", "price", "unit_price"],
    "Quantity": ["quantity", "qty"]
}

CANON_ORDER = ["InvoiceDate", "InvoiceNo", "CustomerID", "UnitPrice", "Quantity"]

def map_columns(cols):
    norm = [normalize_col_name(c) for c in cols]
    mapping = {}
    used_canons = set()
    used_cols = set()

    for canon in CANON_ORDER:
        tokens = CANON_TOKENS.get(canon, [])
        best_idx = None
        best_score = -1
        for i, n in enumerate(norm):
            if i in used_cols:
                continue
            if n == canon.lower():
                best_idx = i
                best_score = 2
                break
            for tok in tokens:
                if tok == n:
                    best_idx = i
                    best_score = 2
                    break
                if tok in n and best_score < 1:
                    best_idx = i
                    best_score = 1
        if best_idx is not None:
            mapping[cols[best_idx]] = canon
            used_canons.add(canon)
            used_cols.add(best_idx)
    return mapping

def load_table(path):
    path_l = path.lower()
    if path_l.endswith((".xls", ".xlsx")):
        try:
            xls = pd.ExcelFile(path, engine="openpyxl")
        except Exception as e:
            raise RuntimeError(f"Failed to open Excel file {path}: {e}")
        for sheet in xls.sheet_names:
            try:
                df_try = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
            except Exception:
                continue
            mapped = map_columns(list(df_try.columns))
            if len(set(mapped.values()) & set(["InvoiceNo","InvoiceDate","CustomerID","UnitPrice","Quantity"])) >= 3:
                return df_try, mapped, sheet
        raise RuntimeError("Could not find valid sheet.")
    else:
        df_try = pd.read_csv(path, encoding="latin1")
        mapped = map_columns(list(df_try.columns))
        return df_try, mapped, None

# ---------- Feature preparation & training ----------

def prepare_features(df, label_window_days=90):
    required_canonical = {"InvoiceNo", "InvoiceDate", "CustomerID", "UnitPrice", "Quantity"}
    if not required_canonical.issubset(set(df.columns)):
        raise RuntimeError(f"Dataframe missing required canonical columns: {required_canonical - set(df.columns)}")

    df = df.dropna(subset=["CustomerID"]).copy()
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C", na=False)].copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"]).copy()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce").fillna(0.0)
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)

    snapshot_date = df["InvoiceDate"].max()
    cutoff_date = snapshot_date - pd.Timedelta(days=label_window_days)
    label_window_end = cutoff_date + pd.Timedelta(days=label_window_days)

    df_cut = df[df["InvoiceDate"] <= cutoff_date].copy()
    df_future = df[(df["InvoiceDate"] > cutoff_date) & (df["InvoiceDate"] <= label_window_end)].copy()

    agg_recent = df_cut.groupby("CustomerID")["InvoiceDate"].agg(lambda x: (cutoff_date - x.max()).days).rename("Recency")
    agg_freq = df_cut.groupby("CustomerID")["InvoiceNo"].nunique().rename("Frequency")
    agg_avg_price = df_cut.groupby("CustomerID")["UnitPrice"].mean().rename("AvgUnitPrice")
    agg_qty = df_cut.groupby("CustomerID")["Quantity"].sum().rename("TotalQuantity")
    agg_monetary = (df_cut.assign(spend=df_cut["UnitPrice"] * df_cut["Quantity"])
                    .groupby("CustomerID")["spend"].sum().rename("Monetary"))

    rfm = pd.concat([agg_recent, agg_freq, agg_avg_price, agg_qty, agg_monetary], axis=1).reset_index()
    rfm.fillna(0, inplace=True)

    future_counts = df_future.groupby("CustomerID").size().rename("future_count").reset_index()
    labels = rfm[["CustomerID"]].merge(future_counts, on="CustomerID", how="left")
    labels["future_count"] = labels["future_count"].fillna(0).astype(int)
    labels["churn"] = (labels["future_count"] == 0).astype(int)

    data = rfm.merge(labels[["CustomerID", "churn"]], on="CustomerID", how="left")
    data["churn"] = data["churn"].fillna(1).astype(int)
    return data, cutoff_date

def train_and_save(X_train, y_train, model_dir, model_name="rf_churn_pipe.joblib"):
    preproc = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    pipe = make_pipeline(preproc, clf)
    pipe.fit(X_train, y_train)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    joblib.dump(pipe, model_path)
    return pipe, model_path

# ---------- Main ----------

def main(argv=None):
    p = argparse.ArgumentParser(description="Train churn model from Online Retail data")
    p.add_argument("--data", type=str, default=os.path.join("data", "online_retail_II.xlsx"), help="Path to dataset")
    p.add_argument("--sample", type=float, default=0.0, help="Sample fraction")
    p.add_argument("--model-dir", type=str, default="exported_model", help="Save dir")
    p.add_argument("--mlflow", action="store_true", help="Enable MLflow")
    args = p.parse_args(argv)

    if not os.path.exists(args.data):
        print(f"ERROR: data file not found at {args.data}", file=sys.stderr)
        sys.exit(2)

    print("Loading data from:", args.data)
    df_raw, mapped_cols, sheet_used = load_table(args.data)
    print("Detected sheet:", sheet_used)
    if mapped_cols:
        print("Auto column mapping:")
        for orig, canon in mapped_cols.items():
            print(f"  {orig} -> {canon}")
        df_raw = df_raw.rename(columns=mapped_cols)

    data, cutoff_date = prepare_features(df_raw)
    print("Prepared customer-level data:", len(data))

    feature_cols = ["Recency", "Frequency", "Monetary", "AvgUnitPrice", "TotalQuantity"]
    X, y = data[feature_cols], data["churn"]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pipe, model_path = train_and_save(X_train, y_train, args.model_dir)
    print("Saved model to:", model_path)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    print(f"AUC: {auc:.4f}   AP: {ap:.4f}")
    print(classification_report(y_test, y_pred))

    # ✅ Safe MLflow logging for Windows (uses file:///C:/... URI)
    if args.mlflow:
        mlruns_dir = pathlib.Path(os.getcwd(), "mlruns").absolute()
        mlruns_uri = mlruns_dir.as_uri()  # Builds file:///C:/... path automatically
        os.makedirs(mlruns_dir, exist_ok=True)

        mlflow.set_tracking_uri(mlruns_uri)
        mlflow.set_experiment("online-retail-churn")

        with mlflow.start_run(run_name="run"):
            mlflow.log_param("model", "RandomForestPipeline")
            mlflow.log_metric("auc", float(auc))
            mlflow.log_metric("average_precision", float(ap))
            mlflow.sklearn.log_model(pipe, artifact_path="model")
        print("✅ MLflow run saved successfully in ./mlruns")

if __name__ == "__main__":
    main()
