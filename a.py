import os  # işletim sistemi ile etkileşim
import pandas as pd  # tabloları dataframe formatında okumak ve işlemek için
import numpy as np  # sayısal diziler ve matris işlemleri için
from sentence_transformers import SentenceTransformer  # cümleleri vektör embedding haline getiren model
from sklearn.metrics.pairwise import cosine_similarity  # matematiksel işlemden dolayı
from scipy.stats import spearmanr  # sıralı veriler arasındaki spearman korelasyonunu hesaplamak için
import matplotlib.pyplot as plt  # grafik işleri için


def load_and_sample(path, n=1000, seed=42):
    df = pd.read_excel(path)  # Excel dosyasını yüklüyoruz
    df = df.rename(columns={  # türkçe sütun adlarını ingilizceye çevirirsek pythonda daha iyi verim alırız
        'Soru': 'question',
        'gpt4o cevabı': 'gpt4o',
        'deepseek cevabı': 'deepseek',
        'Hangisi iyi? (1: gpt4o daha iyi, 2: deepseek daha iyi, '
            '3: ikisi de yeterince iyi, 4: ikisi de kötü)': 'label'
    })
    sampled = df.sample(n=n, random_state=seed)  # her seferinde aynı rastgele örneklemeyi almak için seed
    return sampled.reset_index(drop=True)  # indeksleri sıfırlayıp örneklenmiş datafframei döndürüyorum


def compute_embeddings(df, model_name='intfloat/multilingual-e5-large-instruct'):
    model = SentenceTransformer(model_name)  # belirtilen modeli yüklüyorum
    emb_q = model.encode(df['question'].tolist(), convert_to_tensor=True, show_progress_bar=True)  # soruları vektöre çeviriyoruz
    emb_g4 = model.encode(df['gpt4o'].tolist(),    convert_to_tensor=True, show_progress_bar=True)  # gpt cevaplarını vektöre çevirme işlemi
    emb_ds = model.encode(df['deepseek'].tolist(), convert_to_tensor=True, show_progress_bar=True)  # Depseek cevaplarını vektöre çevirme işlemi
    return emb_q, emb_g4, emb_ds  # soruya ve her iki cevaba ait embeddingleri döndür


def evaluate_topk(df, emb_q, emb_resp, prefix, k=5):
    sims_matrix = cosine_similarity(emb_q.cpu(), emb_resp.cpu())  # tüm soru-cevap benzerlik matrisini hesaplama işlemi
    top1, top5 = [], []
    for i, sims in enumerate(sims_matrix):
        ranked = np.argsort(sims)[::-1]  # benzerlik skorlarını azalan sırada sıralama ve indeks alıyoruz
        top1.append(ranked[0] == i)  # en yüksek skorun hedef cevap olup olmadığını kontrol etmeliyiz
        top5.append(i in ranked[:k])  # gerçek cevabın ilk k içinde olup olmadığını kontrol etmek lazım
    df[f'{prefix}_top1'] = top1  # sütunları ekleme
    df[f'{prefix}_top5'] = top5
    return df  # güncellenmiş dataframei geri döndürüyorum


def compute_correlations(df, prefix):
    corr1, p1 = spearmanr(df['label'], df[f'{prefix}_top1'])  # label ile top1 arasındaki Spearman korelasyonu ve p değerini hesaplama işlemi
    corr5, p5 = spearmanr(df['label'], df[f'{prefix}_top5'])  # label ile top5 arasındaki Spearman korelasyonu ve p değerini hesaplama
    return {
        f'{prefix}_top1_spearman': corr1,  # top1 korelasyon katsayısı
        f'{prefix}_top1_pval':       p1,    # top1 p değeri
        f'{prefix}_top5_spearman': corr5,  # top5 korelasyon katsayısı
        f'{prefix}_top5_pval':       p5,    # top5 p değeri
    }


def main():
    data_path = os.path.join('data', 'ogrenci_sorular_2025.xlsx')  # veri dosyasının yolu
    df = load_and_sample(data_path, n=1000)  # veriyi yükleme ve örnek seçme bin tanee

    emb_q, emb_g4, emb_ds = compute_embeddings(df)  # embedding hesaplamalarını yapıyorum

    df = evaluate_topk(df, emb_q, emb_g4,  prefix='gpt4o',    k=5)  # gpt için top1/top5 değerlendirmesi
    df = evaluate_topk(df, emb_q, emb_ds, prefix='deepseek', k=5)  # Deepseek için top1/top5 değerlendirmesi

    metrics = {}  # korelasyon sonuçlarını saklayacağım sözlüğü başlatıyorum
    metrics.update(compute_correlations(df, 'gpt4o'))      # gpt korelasyonlarını ekliyorum
    metrics.update(compute_correlations(df, 'deepseek'))   # Deepsek korelasyonlarını ekliyorum

    os.makedirs('results/figures', exist_ok=True)
    df.to_csv(os.path.join('results', 'metrics_detailed.csv'), index=False)  # detaylı metrikleri csvyyee kaydediyorum
    summary_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['value'])  # özet metrikleri dataframe dönüştürüyoruz
    summary_df.to_csv(os.path.join('results', 'summary_metrics.csv'))  # özet metrikleri CSV olarak kaydediyorum

    print("=== Spearman Correlations ===")
    for name, val in metrics.items():
        print(f"{name}: {val:.3f}")  # her metrik ismi ve değerini yazdırıyorum

    top1_rates = {  # her modelin top1 başarı oranını hesaplıyorum
        'gpt4o': df['gpt4o_top1'].mean(),
        'deepseek': df['deepseek_top1'].mean()
    }
    top5_rates = {  # her modelin top5 başarı oranını hesaplıyorum
        'gpt4o': df['gpt4o_top5'].mean(),
        'deepseek': df['deepseek_top5'].mean()
    }

    plt.figure()
    plt.bar(list(top1_rates.keys()), list(top1_rates.values()))  # Top1 bar grafiği
    plt.ylabel('Top1 Başarı Oranı')
    plt.title('Top1 Başarı Oranları')
    plt.savefig(os.path.join('results', 'figures', 'top1_rates.png'))  # grafiği dosyaya kaydediyorum

    plt.figure()
    plt.bar(list(top5_rates.keys()), list(top5_rates.values()))
    plt.ylabel('Top5 Başarı Oranı')
    plt.title('Top5 Başarı Oranları')
    plt.savefig(os.path.join('results', 'figures', 'top5_rates.png'))  # grafiği dosyaya kaydetmemiz lazım

    corr_df = summary_df[summary_df.index.str.contains('_spearman')].copy()  # sadece korelasyon değerlerini seçiyorum
    corr_df.index = corr_df.index.str.replace('_spearman', '')
    plt.figure()
    plt.imshow(corr_df.values, aspect='auto')  # korelasyon matrisini görselleştirme işlemi
    plt.xticks(np.arange(corr_df.shape[1]), corr_df.columns)
    plt.yticks(np.arange(corr_df.shape[0]), corr_df.index)
    plt.colorbar()
    plt.title('Spearman Korelasyonları')
    plt.savefig(os.path.join('results', 'figures', 'correlations_heatmap.png'))  # grafiği kaydetmem lazım

    fail_gpt4o = df[df['gpt4o_top1'] == False].head(5)  # Top1 başarısız gpt örneklerini seçmemiz lazım
    print("\n=== Örnek Başarısız GPT4O Soruları ===")
    print(fail_gpt4o[['question','gpt4o','deepseek','label']])  # başarısız örnekler


if __name__ == '__main__':
    main()
