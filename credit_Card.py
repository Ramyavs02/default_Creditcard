# ...existing code...
import os
import sys
import json
import logging
from pathlib import Path
from urllib.parse import quote_plus
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import mysql.connector
import joblib
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings("ignore")
# load environment variables from a .env file (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # python-dotenv not installed; environment variables must be set externally
    pass

# optional imblearn
try:
    from imblearn.over_sampling import SMOTE
    _HAVE_IMBLEARN = True
except Exception:
    SMOTE = None
    _HAVE_IMBLEARN = False
    logging.getLogger().warning("imblearn not installed; SMOTE disabled. Install with: python -m pip install imbalanced-learn")

# CONFIG (edit if needed) — prefer absolute path or place CSV next to script
DEFAULT_CSV_NAME = "default of credit card clients.csv"
CSV_PATH = os.getenv("CSV_PATH", r"D:\ProgramData\ML-DL-GenAI\ML-DL-GenAI-Jul25-Batch\4_Machine_Learning\2_ML_Files\4_Classification\default of credit card clients.csv")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD")   # require env var
if not DB_PASSWORD:
    logging.getLogger().error("DB_PASSWORD environment variable not set. Export DB_PASSWORD before running.")
    raise SystemExit("Missing DB_PASSWORD environment variable")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_NAME = os.getenv("DB_NAME", "creditCard_db")
TABLE_NAME = "default_of_credit_card_clients"
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
SEED = 1

# expected schema for validation (update if your dataset differs)
EXPECTED_COLS = [
    'ID','LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE',
    'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
    'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
    'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6',
    'default payment next month'
]

os.makedirs(OUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def resolve_csv_path():
    # Candidates: explicit CSV_PATH, script dir, cwd, parent notebook export
    script_dir = Path(__file__).parent
    candidates = []
    if CSV_PATH:
        candidates.append(Path(CSV_PATH))
    candidates += [
        script_dir / DEFAULT_CSV_NAME,
        Path.cwd() / DEFAULT_CSV_NAME,
    ]
    for p in candidates:
        if p and p.exists():
            logging.info("Using CSV: %s", p.resolve())
            return str(p.resolve())
    searched = ", ".join(str(p) for p in candidates if p)
    raise FileNotFoundError(f"CSV not found. Searched: {searched}. Put '{DEFAULT_CSV_NAME}' in one of these locations or set CSV_PATH env var.")

def ensure_database(user, password, host, port, db_name):
    conn = mysql.connector.connect(user=user, password=password, host=host, port=port)
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
    conn.commit()
    cur.close()
    conn.close()
    logging.info("Database ensured: %s", db_name)

def get_engine(user, password, host, port, db_name):
    pwd = quote_plus(password)
    uri = f"mysql+mysqlconnector://{user}:{pwd}@{host}:{port}/{db_name}"
    return create_engine(uri, pool_pre_ping=True)

def validate_schema(df, expected_cols):
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    logging.info("Schema validated. columns: %d", len(df.columns))
    return True

def upload_csv(df, engine, table):
    try:
        with engine.begin() as conn:          # transaction scope (atomic)
            df.to_sql(table, conn, if_exists="replace", index=False, chunksize=1000, method="multi")
        logging.info("Uploaded %d rows to %s.%s", len(df), engine.url.database, table)
    except Exception as e:
        logging.exception("Failed to upload CSV to DB: %s", e)
        raise

def basic_eda(df):
    df = df.rename(columns={'default payment next month':'defaulter'}, errors='ignore')
    df.describe().T.to_csv(os.path.join(OUT_DIR, "describe.csv"))
    df.dtypes.to_csv(os.path.join(OUT_DIR, "dtypes.csv"))
    df.isnull().sum().to_csv(os.path.join(OUT_DIR, "nulls.csv"))
    # five point summary
    numcols = df.select_dtypes(include=[np.number]).columns
    quint = pd.DataFrame({
        'min': df[numcols].min(),
        'Q1': df[numcols].quantile(0.25),
        'median': df[numcols].median(),
        'Q3': df[numcols].quantile(0.75),
        'max': df[numcols].max(),
    })
    quint.to_csv(os.path.join(OUT_DIR, "five_point_summary.csv"))
    # simple plots
    if 'defaulter' in df.columns:
        df['defaulter'].value_counts().to_csv(os.path.join(OUT_DIR,"target_counts.csv"))
        plt.figure(figsize=(4,3))
        df['defaulter'].value_counts().plot(kind='bar', color=['tab:green','tab:red'])
        plt.title('Defaulter counts'); plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"target_counts.png")); plt.close()
    df[numcols].hist(figsize=(12,10))
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"histograms.png")); plt.close()
    # correlation
    plt.figure(figsize=(10,8)); sns.heatmap(df[numcols].corr(), cmap='coolwarm'); plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"corr.png")); plt.close()
    return df

def clean_data(df):
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    df = df.rename(columns={'default payment next month':'defaulter'}, errors='ignore')
    cat_cols = [c for c in ['SEX','EDUCATION','MARRIAGE'] if c in df.columns]
    for c in cat_cols:
        df[c] = df[c].fillna(0).astype(int).astype('category')
    numcols = [c for c in df.select_dtypes(include=[np.number]).columns if c!='defaulter']
    for c in numcols:
        if df[c].isnull().any():
            df[c].fillna(df[c].median(), inplace=True)
    return df

def remove_outliers_zscore(df, thresh=4.0):
    numcols = df.select_dtypes(include=[np.number]).columns
    if len(numcols) == 0:
        return df
    z = np.abs(stats.zscore(df[numcols], nan_policy='omit'))
    mask = (z < thresh).all(axis=1)
    logging.info("Outlier filter kept %d/%d rows", mask.sum(), len(df))
    return df.loc[mask].copy()

def prepare_features(df, top_k=8, use_smote=False):
    df_enc = pd.get_dummies(df, drop_first=True)
    if 'defaulter' not in df_enc.columns:
        raise RuntimeError("target 'defaulter' missing")
    X = df_enc.drop(columns=['defaulter'])
    # scikit-learn requires feature names to be strings (no mixed types).
    # Ensure all column names are strings to avoid TypeError during validation.
    try:
        X.columns = X.columns.astype(str)
    except Exception:
        # fallback: map to str
        X.columns = [str(c) for c in X.columns]
    y = df_enc['defaulter']
    sel = SelectKBest(mutual_info_classif, k=min(top_k, X.shape[1]))
    # prepare data for sklearn: fill NaNs and ensure all feature names are plain Python strings
    X_for_fit = X.fillna(0).copy()
    X_for_fit.columns = [str(c) for c in X_for_fit.columns]
    # ensure target is numeric
    try:
        y_numeric = y.astype(int)
    except Exception:
        y_numeric = y
    sel.fit(X_for_fit, y_numeric)
    cols = X.columns[sel.get_support()].tolist()
    Xs = X[cols].fillna(0)
    if use_smote:
        if not _HAVE_IMBLEARN:
            raise RuntimeError("SMOTE requested but imbalanced-learn (imblearn) is not installed. Install with: python -m pip install imbalanced-learn")
        sm = SMOTE(random_state=SEED)
        Xs_res, y_res = sm.fit_resample(Xs, y)
        logging.info("After SMOTE distribution: %s", y_res.value_counts().to_dict())
        return Xs_res, y_res, cols
    return Xs, y, cols

def save_model_version(obj, name_base, out_dir):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = f"{name_base}_{ts}.joblib"
    path = os.path.join(out_dir, fname)
    joblib.dump(obj, path)
    logging.info("Saved %s", path)
    return path

def write_manifest(out_dir):
    files = [p.name for p in Path(out_dir).glob("*")]
    manifest = {"generated_at": datetime.utcnow().isoformat(), "files": files}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    logging.info("Wrote manifest with %d files", len(files))

def train_baseline(df, use_smote=False):
    X, y, features = prepare_features(df, top_k=8, use_smote=use_smote)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED)
    gs = GridSearchCV(clf, {'C':[0.01,0.1,1,10]}, cv=5, scoring='roc_auc', n_jobs=-1)
    gs.fit(Xtr, y_train)
    best = gs.best_estimator_
    y_pred = best.predict(Xte)
    y_proba = best.predict_proba(Xte)[:,1]
    report = classification_report(y_test, y_pred, output_dict=True)
    with open(os.path.join(OUT_DIR,"model_report.json"),"w") as f:
        json.dump(report, f, indent=2)
    logging.info("Best params: %s", gs.best_params_)
    logging.info("ROC AUC: %.4f", roc_auc_score(y_test, y_proba))

    # save versioned model and scaler
    model_path = save_model_version(best, "logistic_model", OUT_DIR)
    scaler_path = save_model_version(scaler, "scaler", OUT_DIR)
    pd.Series(features).to_csv(os.path.join(OUT_DIR,"top_features.csv"), index=False)
    return best, scaler, features

def write_daily_report(df, engine):
    if 'defaulter' in df.columns:
        report = df[df['defaulter']==1].copy()
        try:
            with engine.begin() as conn:
                report.to_sql("daily_defaulters_report", conn, if_exists='replace', index=False)
        except Exception:
            logging.exception("Failed to write daily_defaulters_report table")
            raise
        report.to_csv(os.path.join(OUT_DIR,"daily_defaulters.csv"), index=False)
        logging.info("Daily report rows: %d", len(report))
    else:
        logging.info("No defaulter column; skipping daily report")

def main():
    csv_file = resolve_csv_path()
    ensure_database(DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME)
    engine = get_engine(DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME)
    try:
        with engine.connect() as conn:
            logging.info("Connected to DB: %s", conn.engine.url.database)
    except Exception as e:
        logging.error("Connection failed: %s", e)
        sys.exit(1)
    df = pd.read_csv(csv_file)

    # validate schema before upload (adjust EXPECTED_COLS if your dataset differs)
    try:
        validate_schema(df, EXPECTED_COLS)
    except Exception as e:
        logging.getLogger().warning("Schema validation failed: %s. Proceeding without strict validation.", e)

    upload_csv(df, engine, TABLE_NAME)
    df_sql = pd.read_sql_table(TABLE_NAME, engine)
    df_sql = basic_eda(df_sql)
    df_clean = clean_data(df_sql)
    df_clean = remove_outliers_zscore(df_clean)
    train_baseline(df_clean, use_smote=False)
    write_daily_report(df_sql, engine)

    # write manifest of outputs
    write_manifest(OUT_DIR)
    logging.info("Pipeline complete. Outputs in %s", OUT_DIR)

if __name__ == "__main__":
    main()
# ...existing code...