import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


class PlayerSimilarityEngine:
    """
    Seçilen hedef oyuncunun oyun stiline (küçük örneklem düzeltmeli Per 90
    metriklerine) en çok benzeyen alternatif / fırsat oyuncuları
    Kosinüs Benzerliği (Cosine Similarity) ile bulur. Adaylar hedefin rol
    grubundan (ör. kanat, bek) seçilir; FBref'in "MF" dediği bir kanat da
    kanatlarla kıyaslanır.

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

    def find_similar_players(
        self,
        target_key: str,
        top_n: int = 10,
        max_age: float = 40.0,
        leagues: list = None,
        exclude_same_squad: bool = True,
    ) -> pd.DataFrame:
        """
        `Similarity_Score`: ham kosinüs benzerliği x 100 (-100 ile +100 arası).
        0 = ilişkisiz profil, 100 = aynı yönde profil. Eski (cos+1)/2 dönüşümü
        ilişkisiz oyuncuları "%50 benzer" gösterdiği için kaldırıldı.
        """
        if target_key not in self.df["Row_Key"].values:
            raise ValueError(f"Oyuncu bulunamadı: {target_key}")

        target_row = self.df[self.df["Row_Key"] == target_key].iloc[0]
        group = target_row.get("Score_Group", target_row["Primary_Pos"])

        if group in self.ROLE_FEATURES:
            # Rolü bilinen hedef: aynı rol grubundakiler + rolü bilinmeyen uygun oyuncular
            features = self.POSITION_FEATURES[self.ROLE_FEATURES[group]]
            unknown_role = ~self.df["Score_Group"].isin(self.ROLE_FEATURES.keys())
            pool_mask = (self.df["Score_Group"] == group) | (
                unknown_role & self.df["Primary_Pos"].isin(self.ROLE_BROAD_POOL[group])
            )
        else:
            # Rolü bilinmeyen hedef: FBref ana mevkisindeki tüm oyuncular
            features = self.POSITION_FEATURES[target_row["Primary_Pos"]]
            pool_mask = self.df["Primary_Pos"] == target_row["Primary_Pos"]

        features = [f for f in features if f in self.df.columns]
        pos_df = self.df[pool_mask].copy().reset_index(drop=True)

        # Z-Score (StandardScaler) ile tüm metrikleri aynı ölçeğe getir
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(pos_df[features].fillna(0.0))

        target_idx = pos_df.index[pos_df["Row_Key"] == target_key][0]
        sim_scores = cosine_similarity(
            feature_matrix[target_idx].reshape(1, -1), feature_matrix
        ).flatten()

        pos_df["Similarity_Score"] = (sim_scores * 100.0).round(1)

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
