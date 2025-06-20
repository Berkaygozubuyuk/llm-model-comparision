# llm-model-comparision
Yapay Zeka ile Soru-Cevap Değerlendirmesi ve Model Karşılaştırması
Detaylı grafikler ve sonuçlar raporda bulunabilir
🔹 A) Sorudan Cevaba
Amaç:
Öğrenci sorularına verilen GPT-4o ve DeepSeek yanıtlarının, sorularla olan semantik benzerliği ve insan etiketleriyle olan uyumu değerlendirilmiştir.

Adımlar:

Excel'den 1000 örnek alınır.

Sorular ve yanıtlar için çok-dilli E5-large-instruct embedding’leri çıkarılır.

Gömlemler arasında kozinüs benzerliği ile Top-1 ve Top-5 isabet oranları hesaplanır.

Bu otomatik skorlarla insan etiketleri arasında Spearman korelasyonu ölçülür.

Sonuçlar CSV ve görseller olarak kaydedilir, örnek başarısızlıklar gösterilir.

Sonuçlar:

GPT-4o, DeepSeek’ten biraz daha başarılı.

Ancak Top-1 başarı oranı %80’in altında.

İnsan etiketleri ile benzerlik skorları zayıf ilişki gösteriyor → bu yaklaşım güvenilir değil.

🔹 B) Hangisi İyi?
Amaç:
Farklı embedding modelleri (E5, Cosmos-E5, Jina) ve vektör kombinasyonları kullanılarak, hangi yapıların insan etiketini daha iyi tahmin ettiğini belirlemek.

Yaklaşım:

Her embedding modeli için soru ve yanıt embedding’leri çıkarıldı.

Özellik mühendisliği ile fark ve mutlak fark vektörleri üretildi.

RandomForestClassifier ile sınıflandırma yapıldı.

Başarı ölçütleri (accuracy, precision, recall, F1) karşılaştırıldı.

Sonuçlar:

Jina modeli ham vektörlerde iyi performans gösterdi.

E5 modeli fark vektörlerinde en başarılı oldu.

Cosmos-E5 ve Jina, tüm özellikler birleştirildiğinde en yüksek doğruluğa ulaştı (yaklaşık %86–89).

