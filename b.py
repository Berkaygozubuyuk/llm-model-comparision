import os  # İşletim sistemi seviyesinde dosya ve klasör işlemleri için
import pandas as pd  # Veri yükleme ve tablo işlemleri için
import numpy as np  # Sayısal hesaplamalar ve vektör işlemleri için
import torch  # GPU kullanımını kontrol etmek ve tensor işlemleri için
from sentence_transformers import SentenceTransformer  # Metinleri embedding vektörlerine dönüştürmek için ön-eğitimli model
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test setlerine ayırmak için
from sklearn.ensemble import RandomForestClassifier  # Rastgele orman sınıflandırıcısı modeli
from sklearn.metrics import accuracy_score, classification_report  # Model performansını ölçmek için metrikler
import matplotlib.pyplot as plt  # Grafik çizimleri için

# Önbellek klasörü ve batch boyutu tanımlamaları
CACHE_DIR = 'cache'
BATCH_SIZE = 64

def load_data(path):
    """
    Excel dosyasını yükler, sütunları yeniden adlandırır ve DataFrame döner.
    path: Excel dosyasının yolu
    """
    df = pd.read_excel(path, engine='openpyxl')  # openpyxl motoru ile Excel okuma
    df = df.rename(columns={
        'Soru': 'question',
        'gpt4o cevabı': 'gpt4o',
        'deepseek cevabı': 'deepseek',
        'Hangisi iyi? (1: gpt4o daha iyi, 2: deepseek daha iyi, '
           '3: ikisi de yeterince iyi, 4: ikisi de kötü)': 'label'
    })
    return df


def encode_texts(df, model_name, model_key):
    """
    Metinleri embedding vektörlerine dönüştürür ve önbellekler.
    df: DataFrame, model_name: HuggingFace model adı, model_key: önbellek dosya anahtarı
    """
    os.makedirs(CACHE_DIR, exist_ok=True)  # Önbellek klasörünü oluştur
    cache_q = os.path.join(CACHE_DIR, f"{model_key}_q.npy")
    cache_g = os.path.join(CACHE_DIR, f"{model_key}_g.npy")
    cache_d = os.path.join(CACHE_DIR, f"{model_key}_d.npy")

    # Eğer zaten kaydedilmişse önbellekten yükle
    if os.path.exists(cache_q) and os.path.exists(cache_g) and os.path.exists(cache_d):
        emb_q = np.load(cache_q)
        emb_g = np.load(cache_g)
        emb_d = np.load(cache_d)
    else:
        # GPU varsa cuda kullan, yoksa CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # SentenceTransformer modeli yükle (HuggingFace üzerinden)
        model = SentenceTransformer(model_name, device=device)
        questions = df['question'].astype(str).tolist()
        gpt4o     = df['gpt4o'].astype(str).tolist()
        deepseek  = df['deepseek'].astype(str).tolist()

        # Sorular ve cevaplar için gömme (embedding) hesapla
        emb_q = model.encode(
            questions, batch_size=BATCH_SIZE,
            show_progress_bar=True, convert_to_tensor=False
        )
        emb_g = model.encode(
            gpt4o, batch_size=BATCH_SIZE,
            show_progress_bar=True, convert_to_tensor=False
        )
        emb_d = model.encode(
            deepseek, batch_size=BATCH_SIZE,
            show_progress_bar=True, convert_to_tensor=False
        )

        # Hesaplanan embeddingleri .npy dosyalarına kaydet
        np.save(cache_q, emb_q)
        np.save(cache_g, emb_g)
        np.save(cache_d, emb_d)

    return np.array(emb_q), np.array(emb_g), np.array(emb_d)


def build_features(s, g, d):
    """
    Üç embedding arasından farklı özellikler oluşturur:
    - Ham vektörler (s, g, d)
    - Fark vektörleri (s-g, s-d, g-d)
    - Mutlak farklar (|s-g|, |s-d|)
    - Mutlak farkların farkı (|s-g| - |s-d|)
    """
    feats = {
      's':      s,
      'g':      g,
      'd':      d,
      's-g':    s - g,
      's-d':    s - d,
      'g-d':    g - d,
      '|s-g|':  np.abs(s - g),
      '|s-d|':  np.abs(s - d),
    }
    # Mutlak farkların farkını hesapla
    feats['|s-g| - |s-d|'] = feats['|s-g|'] - feats['|s-d|']
    return feats


def evaluate_model(X, y, test_size=0.2, random_state=42):
    """
    Veriyi eğitim/test olarak böler, RandomForest eğitir ve performansı döner.
    X: özellik matrisi, y: etiketler
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=100, random_state=random_state, n_jobs=-1
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)  # Doğruluk skoru
    report = classification_report(y_te, y_pred)  # Sınıflandırma raporu
    return acc, report


def main():
    # Veri dosyasını oku
    df = load_data(os.path.join('data', 'ogrenci_sorular_2025.xlsx'))
    y = df['label'].values  # Etiketleri al

    # Kullanılacak embedding modelleri
    models = {
      'e5':       'intfloat/multilingual-e5-large-instruct',
      'cosmosE5': 'ytu-ce-cosmos/turkish-e5-large',
      'jina':     'jinaai/jina-embeddings-v3'
    }

    summary = []  # Sonuç özetini tutacak liste
    os.makedirs('results/figures', exist_ok=True)  # Grafik klasörünü oluştur

    # Her model için embedding, özellik ve değerlendirme
    for mname, mpath in models.items():
        print(f"\n--- Model: {mname} ---")
        s_emb, g_emb, d_emb = encode_texts(df, mpath, mname)
        feats = build_features(s_emb, g_emb, d_emb)

        # Farklı özellik kombinasyonları
        combos = {
          's,g,d':       np.hstack([feats['s'], feats['g'], feats['d']]),
          'differences': np.hstack([feats['s-g'], feats['s-d'], feats['g-d']]),
          'absolutes':   np.hstack([feats['|s-g|'], feats['|s-d|'], feats['|s-g| - |s-d|']]),
          'all':         np.hstack(list(feats.values()))
        }

        # Her özellik setini değerlendir
        for cname, X in combos.items():
            acc, report = evaluate_model(X, y)
            print(f"{cname:12s} — Acc: {acc:.3f}")
            print(report)
            summary.append({'model': mname, 'features': cname, 'accuracy': acc})

    # Özet sonuçlarını CSV'ye kaydet
    res_df = pd.DataFrame(summary)
    res_df.to_csv('results/b_summary.csv', index=False)

    # Performans grafiğini oluştur ve kaydet
    features = ['s,g,d', 'differences', 'absolutes', 'all']
    x_base = np.arange(len(features))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (mname, _) in enumerate(models.items()):
        subset = res_df[res_df.model == mname]
        x = x_base + idx * width
        ax.bar(x, subset.accuracy, width=width, label=mname)
    ax.set_xticks(x_base + width * (len(models) - 1) / 2)
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Bölüm B: Model Performansları')
    ax.legend()
    plt.tight_layout()
    fig.savefig('results/figures/b_performance.png')

if __name__ == '__main__':
    main()
