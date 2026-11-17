import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


class PlayerSimilarityEngine:
    """
    Seçilen hedef oyuncunun oyun stiline (Ağırlıklı Kosinüs Benzerliği) ve
    kalite/üretim seviyesine (Ağırlıklı Standart Öklid / RBF Yakınlığı) en çok
    benzeyen alternatif / fırsat oyuncuları Hibrit Benzerlik ile bulur.
    Adaylar hedefin rol grubundan (ör. kanat, bek) seçilir; FBref'in "MF"
    dediği bir kanat da kanatlarla kıyaslanır.

    Oyuncular isimle değil `Row_Key` (isim | doğum yılı | kulüp) ile seçilir;
    çünkü veri setinde aynı isimli farklı oyuncular ve sezon içinde transfer
    olup iki satırda görünen oyuncular bulunur.
    """

    POSITION_FEATURES = {
        "FW": [
            "Gls_adj", "Ast_adj", "GA_non_pk_adj", "Sh_adj", "SoT_adj",
            "G_Sh_adj", "Crs_adj", "Fld_adj", "Def_Actions_adj"
        ],
        "MF": [
            "Gls_adj", "Ast_adj", "GA_non_pk_adj", "Sh_adj", "Crs_adj",
            "Fld_adj", "TklW_adj", "Int_adj", "Def_Actions_adj"
        ],
        "DF": [
            "TklW_adj", "Int_adj", "Def_Actions_adj", "Crs_adj",
            "Ast_adj", "Gls_adj", "Fld_adj", "Fls_adj"
        ],
        "GK": [
            "SavePct_adj", "CSPct_adj", "GA90_adj", "Saves_adj"
        ],
    }

    # Çoklu doğrusal bağlantıyı (multicollinearity) dengelemek için metrik ağırlıkları.
    # Birbirini tekrar eden metrikler yarım ağırlık alır, böylece aynı bilgi profili
    # birden çok kez çekmez:
    # - Bitiricilik/şut: Gls, GA_non_pk (gol+asist), Sh, SoT birbirine çok bağlı.
    # - Savunma: Def_Actions = TklW + Int; üçü tam ağırlıkta savunmayı 3 kez sayardı.
    POSITION_WEIGHTS = {
        "FW": {"Gls_adj": 0.50, "GA_non_pk_adj": 0.50, "Sh_adj": 0.50, "SoT_adj": 0.50},
        "MF": {"Gls_adj": 0.50, "GA_non_pk_adj": 0.50, "Sh_adj": 0.50, "TklW_adj": 0.50, "Int_adj": 0.50},
        "DF": {"TklW_adj": 0.50, "Int_adj": 0.50},
        "GK": {},
    }

    # Rol grubu -> kullanilacak ozellik seti ve "rolu bilinmeyen" oyuncularin
    # aday havuzuna katilacagi FBref genis mevkileri
    ROLE_FEATURES = {
        "GK": "GK", "CB": "DF", "FB": "DF", "DM": "MF", "CM": "MF",
        "AM": "MF", "W": "FW", "ST": "FW",
    }
    ROLE_BROAD_POOL = {
        "GK": ["GK"], "CB": ["DF"], "FB": ["DF", "MF"], "DM": ["MF"], "CM": ["MF"],
        "AM": ["MF", "FW"], "W": ["MF", "FW"], "ST": ["FW"],
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _get_feature_weights(self, pos_key: str, features: list) -> np.ndarray:
        """Seçili pozisyon grubu için metrik ağırlık vektörünü döndürür."""
        weight_map = self.POSITION_WEIGHTS.get(pos_key, {})
        return np.array([weight_map.get(f, 1.0) for f in features], dtype=float)

    def find_similar_players(
        self,
        target_key: str,
        top_n: int = 10,
        max_age: float = 40.0,
        leagues: list = None,
        exclude_same_squad: bool = True,
        style_weight: float = 0.65,
        level_weight: float = 0.35,
        gamma: float = 0.25,
    ) -> pd.DataFrame:
        """
        Hibrit `Similarity_Score` (-100 ile 100 arası):
        - `Style_Score`: ağırlıklı kosinüs benzerliği x 100 (oyun stili / profilin yönü).
        - `Level_Score`: ağırlıklı ortalama kare z-mesafesi üzerinden RBF çekirdeği
          x 100 (kalite / üretim hacminin yakınlığı).
        - Benzerlik = kosinüs x (style_weight + level_weight x seviye). Seviye, stili
          toplamaya değil ÇARPAN olarak girer: profili ilgisiz (kosinüs ~0) bir oyuncu
          sırf seviyesi yakın diye yüksek puan alamaz ve skor 0 civarında sıçramaz.
          Her alanda +1σ olan bir oyuncu ile +3σ olan bir yıldız aynı yönde olsalar bile
          %100 yerine ~%78 benzer çıkar. Zıt profillerde (kosinüs < 0) skor kosinüstür.
        """
        if style_weight < 0 or level_weight < 0 or style_weight + level_weight <= 0:
            raise ValueError("style_weight ve level_weight negatif olamaz, toplamlari pozitif olmali")
        total_w = style_weight + level_weight
        style_weight, level_weight = style_weight / total_w, level_weight / total_w
        if target_key not in self.df["Row_Key"].values:
            raise ValueError(f"Oyuncu bulunamadı: {target_key}")

        target_row = self.df[self.df["Row_Key"] == target_key].iloc[0]
        group = target_row.get("Score_Group", target_row["Primary_Pos"])

        if group in self.ROLE_FEATURES:
            pos_key = self.ROLE_FEATURES[group]
            # Rolü bilinen hedef: aynı rol grubundakiler + rolü bilinmeyen uygun oyuncular
            features = self.POSITION_FEATURES[pos_key]
            unknown_role = ~self.df["Score_Group"].isin(self.ROLE_FEATURES.keys())
            pool_mask = (self.df["Score_Group"] == group) | (
                unknown_role & self.df["Primary_Pos"].isin(self.ROLE_BROAD_POOL[group])
            )
        else:
            pos_key = target_row["Primary_Pos"]
            # Rolü bilinmeyen hedef: FBref ana mevkisindeki tüm oyuncular
            features = self.POSITION_FEATURES[pos_key]
            pool_mask = self.df["Primary_Pos"] == pos_key

        features = [f for f in features if f in self.df.columns]
        pos_df = self.df[pool_mask].copy().reset_index(drop=True)

        # 1. Z-Score (StandardScaler) ile tüm metrikleri aynı ölçeğe getir
        scaler = StandardScaler()
        z_matrix = scaler.fit_transform(pos_df[features].fillna(0.0))

        # 2. Çoklu doğrusal bağlantı (multicollinearity) ağırlıklarını uygula
        weights = self._get_feature_weights(pos_key, features)
        sqrt_w = np.sqrt(weights)
        weighted_z_matrix = z_matrix * sqrt_w

        target_idx = pos_df.index[pos_df["Row_Key"] == target_key][0]
        target_vec_weighted = weighted_z_matrix[target_idx].reshape(1, -1)
        target_vec_z = z_matrix[target_idx].reshape(1, -1)

        # 3. Stil Benzerliği: Ağırlıklı Kosinüs Benzerliği
        cos_sim = cosine_similarity(target_vec_weighted, weighted_z_matrix).flatten()

        # 4. Seviye (Kalite) Benzerliği: Ağırlıklı Ortalama Karekök Z-Mesafesi (WRMSD) + RBF
        diff_sq = (z_matrix - target_vec_z) ** 2
        wrmsd_sq = np.sum(diff_sq * weights, axis=1) / np.sum(weights)
        level_sim = np.exp(-gamma * wrmsd_sq)

        # 5. Hibrit Benzerlik Skoru: seviye, pozitif kosinüsü ölçekleyen bir çarpandır.
        # Kosinüs 0'a yaklaştıkça skor da sürekli biçimde 0'a iner (eşik sıçraması yok).
        hybrid_sim = np.where(
            cos_sim > 0.0,
            cos_sim * (style_weight + level_weight * level_sim),
            cos_sim,
        )

        pos_df["Style_Score"] = (cos_sim * 100.0).round(1)
        pos_df["Level_Score"] = (level_sim * 100.0).round(1)
        pos_df["Similarity_Score"] = (hybrid_sim * 100.0).round(1)

        # Hedef oyuncunun kendisini (diğer kulüpteki satırı dahil) çıkar, filtreleri uygula
        results = pos_df[pos_df["Player_ID"] != target_row["Player_ID"]].copy()
        results = results[results["Age"] <= max_age]

        # None -> lig filtresi yok; boş liste -> hiç lig seçilmemiş, sonuç da boş
        if leagues is not None:
            results = results[results["League"].isin(leagues)]

        if exclude_same_squad:
            results = results[results["Squad"] != target_row["Squad"]]

        return results.sort_values(
            by=["Similarity_Score", "OptiScout_Score"], ascending=[False, False]
        ).head(top_n)
