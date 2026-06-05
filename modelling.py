import argparse
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=300)
    args = parser.parse_args()

    # Hubungkan ke MLflow (secara lokal di server runner GitHub Actions)
    mlflow.set_experiment("CI_Retraining_Automation")

    # Memuat data bersih yang berada di dalam folder project
    train_df = pd.read_csv("namadataset_preprocessing/train_clean.csv")
    test_df = pd.read_csv("namadataset_preprocessing/test_clean.csv")

    X_train = train_df['Utterance'].fillna("").astype(str)
    y_train = train_df['Emotion_Encoded']
    X_test = test_df['Utterance'].fillna("").astype(str)
    y_test = test_df['Emotion_Encoded']

    # Vektorisasi
    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    with mlflow.start_run():
        model = LogisticRegression(C=args.C, max_iter=args.max_iter, random_state=42)
        model.fit(X_train_vec, y_train)
        
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)

        # Logging ke MLflow
        mlflow.log_param("C", args.C)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        print(f"[CI Sukses] Model dilatih otomatis dengan Akurasi: {acc:.4f}")

if __name__ == "__main__":
    # Memanggil fungsi main() yang benar agar argparse di terminal aktif
    main()
