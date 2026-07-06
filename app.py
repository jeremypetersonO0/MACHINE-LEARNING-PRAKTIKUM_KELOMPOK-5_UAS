"""
Flask app untuk deploy model SAINT (deteksi Parkinson).

Struktur input:
- feature_columns.pkl berisi 22 nama kolom (urutan sama seperti saat training)
- Kolom pertama (index 0) SECARA ARSITEKTUR tidak pernah benar-benar dipakai model
  (di training, itu jadi slot dummy kategori yang selalu di-nolkan), jadi form
  hanya minta 21 fitur (feature_columns[1:]) dan kolom pertama diisi placeholder
  0 sebelum discaling (aman, karena MinMaxScaler men-scale tiap kolom independen).
"""

import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify

from saint_arch import SAINT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# ================= LOAD ARTIFACTS =================
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))   # 22 nama kolom
model_config = joblib.load(os.path.join(BASE_DIR, "model_config.pkl"))

# ================= LOAD METRICS (untuk tab "Akurasi") =================
with open(os.path.join(BASE_DIR, "metrics.json")) as f:
    _raw_metrics = json.load(f)


def build_metrics_context(m):
    """Susun dict metrics supaya gampang dipakai di template (persen, dst)."""
    items = [
        {"label": "Accuracy", "value": f"{m['accuracy']*100:.1f}%", "pct": m['accuracy']*100},
        {"label": "Precision", "value": f"{m['precision']*100:.1f}%", "pct": m['precision']*100},
        {"label": "Recall (Sensitivity)", "value": f"{m['recall']*100:.1f}%", "pct": m['recall']*100},
        {"label": "Specificity", "value": f"{m['specificity']*100:.1f}%", "pct": m['specificity']*100},
        {"label": "F1-Score", "value": f"{m['f1']*100:.1f}%", "pct": m['f1']*100},
        {"label": "AUC-ROC", "value": f"{m['auc']*100:.1f}%", "pct": m['auc']*100},
    ]
    return {
        "items_list": items,
        "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
        "is_placeholder": m.get("is_placeholder", False),
    }


METRICS_CTX = build_metrics_context(_raw_metrics)

# Fitur yang benar-benar ditampilkan di form (buang kolom pertama, sesuai arsitektur training)
INPUT_FEATURES = feature_columns[1:]   # 21 fitur

# ================= PENJELASAN TIAP FITUR (untuk tab "Cek Sekarang") =================
FEATURE_INFO = {
    "MDVP:Fhi(Hz)": {
        "desc": "Frekuensi getaran suara tertinggi (nada paling tinggi) yang terekam selama pasien mengucapkan vokal berkelanjutan.",
        "range": "sekitar 100 – 600 Hz",
    },
    "MDVP:Flo(Hz)": {
        "desc": "Frekuensi getaran suara terendah (nada paling rendah) yang terekam pada rekaman yang sama.",
        "range": "sekitar 65 – 240 Hz",
    },
    "MDVP:Jitter(%)": {
        "desc": "Seberapa tidak stabil jarak antar-getaran pita suara, dalam persen. Makin tinggi, suara makin \"bergetar\".",
        "range": "sekitar 0.001 – 0.033",
    },
    "MDVP:Jitter(Abs)": {
        "desc": "Versi lain dari ukuran di atas, dinyatakan dalam satuan waktu (detik) bukan persen.",
        "range": "sekitar 0.000007 – 0.00026",
    },
    "MDVP:RAP": {
        "desc": "Ukuran ketidakstabilan nada dalam jangka sangat pendek, dihitung dari 3 getaran berurutan.",
        "range": "sekitar 0.0007 – 0.02",
    },
    "MDVP:PPQ": {
        "desc": "Mirip RAP, tapi dihitung dari 5 getaran berurutan untuk menangkap pola yang sedikit lebih panjang.",
        "range": "sekitar 0.0008 – 0.02",
    },
    "Jitter:DDP": {
        "desc": "Rata-rata selisih ketidakstabilan nada antar-getaran berurutan; secara matematis berkaitan langsung dengan RAP.",
        "range": "sekitar 0.002 – 0.06",
    },
    "MDVP:Shimmer": {
        "desc": "Seberapa tidak stabil volume/amplitudo suara antar getaran. Makin tinggi, suara makin \"bergetar\" dari sisi kekerasannya.",
        "range": "sekitar 0.009 – 0.12",
    },
    "MDVP:Shimmer(dB)": {
        "desc": "Ukuran shimmer yang sama, tapi dinyatakan dalam satuan desibel (dB).",
        "range": "sekitar 0.085 – 1.3",
    },
    "Shimmer:APQ3": {
        "desc": "Ketidakstabilan volume suara dalam jangka pendek, dihitung dari 3 getaran berurutan.",
        "range": "sekitar 0.004 – 0.06",
    },
    "Shimmer:APQ5": {
        "desc": "Mirip APQ3, tapi dihitung dari 5 getaran berurutan.",
        "range": "sekitar 0.005 – 0.08",
    },
    "MDVP:APQ": {
        "desc": "Ukuran ketidakstabilan volume suara standar, dihitung dari rentang getaran yang lebih panjang (11 siklus).",
        "range": "sekitar 0.007 – 0.14",
    },
    "Shimmer:DDA": {
        "desc": "Rata-rata selisih ketidakstabilan volume antar-getaran berurutan; berkaitan langsung dengan APQ3.",
        "range": "sekitar 0.013 – 0.17",
    },
    "NHR": {
        "desc": "Rasio antara komponen suara \"berisik\" (noise) dibanding komponen suara yang jernih/beraturan (harmonik).",
        "range": "sekitar 0.0006 – 0.31",
    },
    "HNR": {
        "desc": "Kebalikan dari NHR — mengukur seberapa \"bersih\" suara dibanding derau, dalam desibel. Makin tinggi biasanya makin sehat.",
        "range": "sekitar 8 – 33 dB",
    },
    "RPDE": {
        "desc": "Ukuran seberapa acak atau tidak beraturan pola getaran suara dari waktu ke waktu (analisis non-linear).",
        "range": "sekitar 0.25 – 0.68",
    },
    "DFA": {
        "desc": "Mengukur pola skala/fraktal pada sinyal suara — bagaimana fluktuasi kecil terhubung dengan pola yang lebih besar.",
        "range": "sekitar 0.57 – 0.83",
    },
    "spread1": {
        "desc": "Ukuran non-linear pertama dari variasi frekuensi dasar suara. Biasanya bernilai negatif.",
        "range": "sekitar -7.9 – -2.4",
    },
    "spread2": {
        "desc": "Ukuran non-linear kedua dari variasi frekuensi dasar suara, melengkapi spread1.",
        "range": "sekitar 0.006 – 0.45",
    },
    "D2": {
        "desc": "Mengukur seberapa kompleks/rumit dinamika sinyal suara (dimensi korelasi).",
        "range": "sekitar 1.4 – 3.6",
    },
    "PPE": {
        "desc": "Ukuran ketidakstabilan nada yang dirancang tahan terhadap gangguan kualitas rekaman.",
        "range": "sekitar 0.04 – 0.53",
    },
}


def build_field_info():
    """Susun list fitur + penjelasan, urutan sama seperti INPUT_FEATURES."""
    fields = []
    for feat in INPUT_FEATURES:
        info = FEATURE_INFO.get(feat, {"desc": "", "range": ""})
        fields.append({"name": feat, "desc": info["desc"], "range": info["range"]})
    return fields


FIELD_INFO = build_field_info()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= BUILD MODEL =================
model = SAINT(
    categories=model_config["categories"],
    num_continuous=model_config["num_continuous"],
    dim=model_config["dim"],
    dim_out=model_config["dim_out"],
    depth=model_config["depth"],
    heads=model_config["heads"],
    attn_dropout=model_config["attn_dropout"],
    ff_dropout=model_config["ff_dropout"],
    mlp_act=nn.GELU(),   # sama seperti saat training di SAINT.py
).to(device)

state_dict = torch.load(os.path.join(BASE_DIR, "saint_model.pt"), map_location=device)
model.load_state_dict(state_dict)
model.eval()


def predict_one(raw_values_21):
    """
    raw_values_21: list/array berisi 21 nilai fitur mentah (belum discaling),
    urutan HARUS sama dengan INPUT_FEATURES.
    """
    # Susun ulang jadi 22 kolom (index 0 = placeholder 0, sisanya nilai asli)
    full_row = np.zeros((1, len(feature_columns)), dtype=np.float32)
    full_row[0, 1:] = raw_values_21

    # Scaling pakai scaler yang sama seperti saat training
    scaled_row = scaler.transform(full_row)

    # x_cont = semua kolom KECUALI kolom pertama (index 0), sesuai arsitektur training
    x_cont = torch.tensor(scaled_row[:, 1:], dtype=torch.float32).to(device)
    x_categ = torch.zeros((1, 1), dtype=torch.long).to(device)

    with torch.no_grad():
        _, outputs = model(x_categ, x_cont)
        proba = torch.softmax(outputs, dim=1)[0]

    prob_parkinson = float(proba[1].cpu().item())
    pred_label = int(prob_parkinson >= 0.5)

    return pred_label, prob_parkinson


@app.route("/", methods=["GET"])
def landing():
    return render_template("landing.html")


@app.route("/app", methods=["GET"])
def app_page():
    return render_template(
        "main.html", features=INPUT_FEATURES, field_info=FIELD_INFO,
        metrics=METRICS_CTX, active_tab="tentang"
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil nilai dari form HTML, urutan sesuai INPUT_FEATURES
        raw_values = []
        for feat in INPUT_FEATURES:
            val = request.form.get(feat)
            if val is None or val.strip() == "":
                return render_template(
                    "main.html", features=INPUT_FEATURES, field_info=FIELD_INFO,
                    metrics=METRICS_CTX, active_tab="cek", error=f"Kolom '{feat}' belum diisi."
                )
            raw_values.append(float(val))

        pred_label, prob_parkinson = predict_one(raw_values)

        result = {
            "label": "Parkinson" if pred_label == 1 else "Sehat (Healthy)",
            "prob_parkinson": round(prob_parkinson * 100, 2),
            "prob_healthy": round((1 - prob_parkinson) * 100, 2),
        }

        return render_template(
            "main.html", features=INPUT_FEATURES, field_info=FIELD_INFO,
            metrics=METRICS_CTX, active_tab="cek", result=result
        )

    except Exception as e:
        return render_template(
            "main.html", features=INPUT_FEATURES, field_info=FIELD_INFO,
            metrics=METRICS_CTX, active_tab="cek", error=str(e)
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Endpoint JSON untuk dipakai programatik (bukan lewat form HTML).
    Body JSON contoh:
    {
        "features": {
            "MDVP:Fhi(Hz)": 157.3,
            "MDVP:Flo(Hz)": 116.6,
            ...  (21 fitur sesuai INPUT_FEATURES, urutan bebas asal nama kolom cocok)
        }
    }
    """
    try:
        data = request.get_json(force=True)
        features_dict = data.get("features", {})

        missing = [f for f in INPUT_FEATURES if f not in features_dict]
        if missing:
            return jsonify({"error": f"Fitur berikut belum diisi: {missing}"}), 400

        raw_values = [float(features_dict[f]) for f in INPUT_FEATURES]
        pred_label, prob_parkinson = predict_one(raw_values)

        return jsonify({
            "prediction": "Parkinson" if pred_label == 1 else "Healthy",
            "prob_parkinson": round(prob_parkinson, 4),
            "prob_healthy": round(1 - prob_parkinson, 4),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)