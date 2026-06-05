import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import mlflow
import mlflow.sklearn

# Menyembunyikan peringatan Git Python di MLflow
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

def train_with_tuning():
    # 1. Hubungkan ke MLflow tracking server lokal
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Putri_Emotion_Tuning")

    # 2. Menggunakan Absolute Path agar tidak membingungkan Python
    train_path = r"C:\PTR\SMSML_Putri_Yulianti\namadataset_preprocessing\train_clean.csv"
    test_path = r"C:\PTR\SMSML_Putri_Yulianti\namadataset_preprocessing\test_clean.csv"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("[Error] Data bersih tidak ditemukan! Periksa kembali foldernya.")
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df['Utterance'].fillna("").astype(str)
    y_train = train_df['Emotion_Encoded']
    X_test = test_df['Utterance'].fillna("").astype(str)
    y_test = test_df['Emotion_Encoded']

    # 3. Ekstraksi Fitur Teks dengan TF-IDF
    print("[*] Melakukan ekstraksi fitur TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Simpan vectorizer lokal untuk dijadikan artefak pelengkap nanti
    os.makedirs("temp_artifacts", exist_ok=True)
    joblib.dump(vectorizer, "temp_artifacts/tfidf_vectorizer.pkl")

    # 4. Daftar Hyperparameter untuk Tuning
    candidate_params = [
        {"C": 0.1, "max_iter": 300},
        {"C": 1.0, "max_iter": 300},
        {"C": 5.0, "max_iter": 300}
    ]

    print("[*] Memulai pelatihan dan Manual Logging ke MLflow...")

    for i, params in enumerate(candidate_params):
        with mlflow.start_run(run_name=f"LogisticRegression_Run_{i+1}"):
            model = LogisticRegression(C=params['C'], max_iter=params['max_iter'], random_state=42)
            model.fit(X_train_vec, y_train)

            y_pred = model.predict(X_test_vec)
            acc = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

            # Manual Logging
            mlflow.log_param("C", params['C'])
            mlflow.log_param("max_iter", params['max_iter'])
            mlflow.log_param("model_architecture", "Logistic Regression")

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(model, artifact_path="model")
            mlflow.log_artifact("temp_artifacts/tfidf_vectorizer.pkl", artifact_path="metadata")

            print(f"[+] Run {i+1} Berhasil -> C: {params['C']} | Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

    print("\n[*] Seluruh pencatatan eksperimen selesai. Silakan cek MLflow UI Anda!")

if __name__ == "__main__":
    train_with_tuning()
