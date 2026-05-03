import os, json
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

app = Flask(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
MODEL = None
SCALER = None
TOP_FEATURES = None
RESULTS = {}
DF_ALL = None

DATA_DIR = os.path.dirname(__file__)

def load_and_train():
    global MODEL, SCALER, TOP_FEATURES, RESULTS, DF_ALL

    def load_all():
        frames = []
        for deg in [27, 30, 35, 40, 45]:
            acc = pd.read_csv(os.path.join(DATA_DIR, f"accel_features_windows_{deg}deg_new.csv"))
            mic = pd.read_csv(os.path.join(DATA_DIR, f"mic_feature_extraction_{deg}deg.csv"))
            cur = pd.read_csv(os.path.join(DATA_DIR, f"Current_feature_extraction_{deg}deg.csv"))
            acc['temp'] = deg; mic['temp'] = deg; cur['temp'] = deg
            acc = acc.add_prefix("acc_")
            mic = mic.add_prefix("mic_")
            cur = cur.add_prefix("cur_")
            frames.append(pd.concat([acc, mic, cur], axis=1))
        return pd.concat(frames, axis=0).reset_index(drop=True)

    df = load_all()
    DF_ALL = df
    label_map = {27: 0, 30: 1, 35: 2, 40: 3, 45: 4}
    df['label'] = df['acc_temp'].map(label_map)
    X = df.drop(columns=['label', 'acc_temp', 'mic_temp', 'cur_temp'])
    y = df['label']

    # Feature importance
    xgb_fi = XGBClassifier(n_estimators=300, max_depth=5, random_state=42, eval_metric='mlogloss')
    xgb_fi.fit(X, y)
    importance = pd.Series(xgb_fi.feature_importances_, index=X.columns).sort_values(ascending=False)
    TOP_FEATURES = importance.head(30).index.tolist()
    X_sel = X[TOP_FEATURES]

    # Train/test split
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(X_sel, y, test_size=0.2, stratify=y, random_state=42)
    SCALER = StandardScaler()
    X_tr = SCALER.fit_transform(X_tr_raw)
    X_te = SCALER.transform(X_te_raw)

    classes = np.unique(y_tr)
    weights = compute_class_weight('balanced', classes=classes, y=y_tr)
    sw = np.array([dict(zip(classes, weights))[i] for i in y_tr])

    MODEL = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                          subsample=0.9, colsample_bytree=0.9, gamma=0.1,
                          random_state=42, eval_metric='mlogloss')
    MODEL.fit(X_tr, y_tr, sample_weight=sw)

    y_pred = MODEL.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    cm = confusion_matrix(y_te, y_pred).tolist()
    report = classification_report(y_te, y_pred,
                                   target_names=['27°','30°','35°','40°','45°'],
                                   output_dict=True)

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for tr_idx, te_idx in skf.split(X_sel, y):
        Xtr, Xte = X_sel.iloc[tr_idx], X_sel.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        cw = dict(zip(np.unique(ytr), compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)))
        sw2 = np.array([cw[i] for i in ytr])
        m = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                          subsample=0.9, colsample_bytree=0.9, gamma=0.1,
                          random_state=42, eval_metric='mlogloss')
        m.fit(Xtr_s, ytr, sample_weight=sw2)
        cv_scores.append(float(accuracy_score(yte, m.predict(Xte_s))))

    top10 = importance.head(10)
    feat_names = [f.replace('acc_','ACC: ').replace('mic_','MIC: ').replace('cur_','CUR: ')
                  for f in top10.index.tolist()]
    feat_vals = [round(float(v), 4) for v in top10.values.tolist()]

    top5 = importance.head(5).index.tolist()
    class_feature_means = {}
    abs_min, abs_max = {}, {}
    for f in top5:
        vals = [df[df['label']==i][f].mean() for i in range(5)]
        abs_min[f] = min(vals); abs_max[f] = max(vals)
    for deg, lbl in label_map.items():
        sub = df[df['label']==lbl][top5]
        class_feature_means[str(deg)] = {
            f: round(float(sub[f].mean()), 4) for f in top5
        }

    RESULTS.update({
        "accuracy": round(float(acc), 4),
        "cv_scores": [round(s, 4) for s in cv_scores],
        "cv_mean": round(float(np.mean(cv_scores)), 4),
        "cv_std": round(float(np.std(cv_scores)), 4),
        "confusion_matrix": cm,
        "report": report,
        "feat_names": feat_names,
        "feat_vals": feat_vals,
        "class_counts": [int((y == i).sum()) for i in range(5)],
        "total_samples": int(len(df)),
        "total_features": int(X.shape[1]),
        "class_feature_means": class_feature_means,
        "top5_features": [f.replace('acc_','ACC: ').replace('mic_','MIC: ').replace('cur_','CUR: ')
                          for f in top5]
    })

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/results")
def api_results():
    return jsonify(RESULTS)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        # Expect {"features": {feature_name: value, ...}}  OR  {"row": [values...]}
        if "features" in data:
            row = pd.DataFrame([data["features"]])[TOP_FEATURES]
        else:
            row = pd.DataFrame([data["row"]], columns=TOP_FEATURES)
        scaled = SCALER.transform(row)
        pred = int(MODEL.predict(scaled)[0])
        proba = MODEL.predict_proba(scaled)[0].tolist()
        class_names = ['27°','30°','35°','40°','45°']
        return jsonify({"predicted_class": class_names[pred], "probabilities": dict(zip(class_names, [round(p,4) for p in proba]))})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})

if __name__ == "__main__":
    load_and_train()
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
