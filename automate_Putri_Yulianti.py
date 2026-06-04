# automate_Putri_Yulianti.py
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def run_preprocessing(input_csv_path, output_dir="namadataset_preprocessing"):
    print(f"[*] Memulai otomatisasi preprocessing untuk: {input_csv_path}")
    
    # 1. Validasi berkas input
    if not os.path.exists(input_csv_path):
        print(f"[Error] Berkas mentah tidak ditemukan di {input_csv_path}")
        sys.exit(1)
        
    # 2. Memuat Data Mentah
    df_raw = pd.read_csv(input_csv_path)
    
    # 3. Pembersihan Data (Handling Missing Values & Duplicates)
    # Menghapus baris jika kolom teks (Utterance) atau target (Emotion) kosong
    df_cleaned = df_raw.dropna(subset=['Utterance', 'Emotion']).copy()
    df_cleaned = df_cleaned.drop_duplicates().copy()
    
    # 4. Text Cleansing Sederhana
    df_cleaned['Utterance'] = df_cleaned['Utterance'].astype(str).str.lower().str.strip()
    
    # 5. Encoding Label Kategorikal (Emotion) menjadi Numerik
    label_encoder = LabelEncoder()
    df_cleaned['Emotion_Encoded'] = label_encoder.fit_transform(df_cleaned['Emotion'])
    
    # 6. Split Data Latih dan Uji (80:20)
    X = df_cleaned['Utterance']
    y = df_cleaned['Emotion_Encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 7. Menyimpan Hasil Preprocessing ke Folder Output
    os.makedirs(output_dir, exist_ok=True)
    
    train_df = pd.DataFrame({'Utterance': X_train, 'Emotion_Encoded': y_train})
    test_df = pd.DataFrame({'Utterance': X_test, 'Emotion_Encoded': y_test})
    
    train_path = os.path.join(output_dir, "train_clean.csv")
    test_path = os.path.join(output_dir, "test_clean.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"[+] Preprocessing Sukses!")
    print(f"[+] Data latih bersih disimpan di: {train_path} ({len(train_df)} baris)")
    print(f"[+] Data uji bersih disimpan di: {test_path} ({len(test_df)} baris)")

if __name__ == "__main__":
    # Skrip ini menerima argumen jalur file mentah dari terminal
    # Contoh jalankan: python automate_Putri_Yulianti.py /path/to/train_sent_emo.csv
    if len(sys.argv) < 2:
        print("[!] Penggunaan: python automate_Putri_Yulianti.py <jalur_file_csv_mentah>")
    else:
        raw_data_path = sys.argv[1]
        run_preprocessing(raw_data_path)
