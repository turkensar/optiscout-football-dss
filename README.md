# ⚽ OptiScout: AI-Driven Football Scouting & Squad Optimization DSS

Avrupa'nın 5 büyük ligindeki (Premier League, La Liga, Serie A, Bundesliga, Ligue 1) **2026-2027 sezonu FBref** verileri üzerine inşa edilmiş; **Veri Mühendisliği**, **Makine Öğrenmesi (Cosine Similarity)** ve **Yöneylem Araştırması (Mixed-Integer Linear Programming - MILP)** yöntemlerini birleştiren interaktif bir **Karar Destek Sistemi (Decision Support System - DSS)**.

---

## 🎯 Projenin Amacı ve Çözülen Problemler

Modern futbol analitiğinde oyuncu izleme (scouting) ve kadro planlama süreçleri üç temel analitik probleme dayanır:
1. **Adil Performans Kıyaslaması:** Farklı süreler alan ve farklı mevkilerde (`GK`, `DF`, `MF`, `FW`) görev yapan oyuncuların 90 dakika başına (`Per 90`) normalize edilerek kendi mevkileri içinde yüzdelik dilimlerle (`Percentile 0-100`) puanlanması.
2. **İkame & Fırsat Oyuncu Keşfi (Player Similarity):** Takımdan ayrılan veya yüksek maliyetli bir yıldız oyuncunun oyun profiline istatistiksel olarak en çok benzeyen genç yeteneklerin (U23 Wonderkid) çok boyutlu uzayda bulunması.
3. **Kısıtlı Kadro Optimizasyonu (Squad Building):** Taktik diziliş, yaş ortalaması sınırı ve kulüp başına maksimum oyuncu kısıtları altında matematiksel olarak en yüksek verimi üreten İlk 11'in kurulması.

---

## 🏗️ Sistem Mimarisi ve Metodoloji

| Modül | Dosya | Kullanılan Yöntem & Algoritma | Açıklama |
| :--- | :--- | :--- | :--- |
| **1. Veri Mühendisliği & Skorlama** | `data_processor.py` | Per-90 Normalizasyon & Mevki Bazlı Yüzdelik Dilim (`Percentile Rank`) | Ham FBref verilerini temizler, mevkileri ayrıştırır ve Hücum, Yaratıcılık, Savunma, Disiplin alt kırılımlarıyla `0-100` arası **OptiScout_Score** üretir. |
| **2. İkame Oyuncu Motoru** | `similarity_engine.py` | `StandardScaler` (Z-Score) + `Cosine Similarity` (`scikit-learn`) | Oyuncuların mevki bazlı Per-90 vektörlerini standartlaştırır ve seçilen referans oyuncuya en yüksek kosinüs benzerliği gösteren alternatifleri listeler. |
| **3. Optimum İlk 11 Çözücü** | `squad_optimizer.py` | Karışık Tamsayılı Doğrusal Programlama (`scipy.optimize.milp` - HiGHS Solver) | Seçilen formasyon (`4-3-3`, `4-2-3-1`, `3-5-2` vb.), maksimum yaş ortalaması ve kulüp sınırı kısıtları altında toplam kadro skorunu maksimize eder. |
| **4. İnteraktif KDS Arayüzü** | `app.py` | `Streamlit` & `Plotly` | Dört kadranlı aykırı değer (outlier) haritası, yüzdelik dilim radar karşılaştırması ve yeşil saha üzerinde taktik diziliş görselleştirmesi sunar. |

---

## 📸 Uygulama Ekran Görüntüleri

### 1. Oyuncu Sıralaması ve Performans Skorlaması
![Overview](images/overview.png)

### 2. Dört Kadran Performans Haritası (Outlier Tespiti)
![Quadrant Analysis](images/quadrant.png)

### 3. Makine Öğrenmesi ile İkame Oyuncu & Wonderkid Keşfi (Cosine Similarity)
![Similarity Engine](images/similarity.png)

### 4. Doğrusal Programlama (MILP) ile Kısıtlı Optimum İlk 11 ve Taktik Tahtası
![Optimal XI](images/optimal_xi.png)

---

## 🧮 Matematiksel Optimizasyon Modeli (MILP)

Kadro optimizasyon modülü, $N$ oyuncudan oluşan havuzda her oyuncu için ikili karar değişkeni $x_i \in \{0, 1\}$ tanımlayarak aşağıdaki modeli çözer:

$$\max \sum_{i=1}^{N} \left( \text{Score}_i \cdot w_i \right) x_i$$

**Kısıtlar (Constraints):**
* **İlk 11 Kısıtı:** $\sum_{i=1}^{N} x_i = 11$
* **Formasyon Mevki Kısıtı:** Her mevki $p \in \{\text{GK, DF, MF, FW}\}$ için $\sum_{i \in P_p} x_i = R_p$
* **Kulüp Kotası Kısıtı:** Her kulüp $c$ için $\sum_{i \in C_c} x_i \le K_{\max}$
* **Maksimum Yaş Ortalaması Kısıtı:** $\sum_{i=1}^{N} \text{Age}_i \cdot x_i \le 11 \cdot \text{Age}_{\max}$

---

## 🚀 Kurulum ve Çalıştırma

```bash
# 1. Repoyu klonlayın
git clone https://github.com/turkensar/optiscout-football-dss.git
cd optiscout-football-dss

# 2. Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt

# 3. Streamlit karar destek arayüzünü başlatın
python -m streamlit run app.py
```
