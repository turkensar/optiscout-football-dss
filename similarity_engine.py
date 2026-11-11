import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


class PlayerSimilarityEngine:
    """
    Seçilen hedef oyuncunun oyun stiline (Per 90 metriklerine) en çok benzeyen
    alternatif / fırsat oyuncuları Kosinüs Benzerliği (Cosine Similarity) ile bulur.
    """

    POSITION_FEATURES = {
        "FW": [
            "Gls_p90", "Ast_p90", "GA_non_pk_p90", "Sh/90", "SoT/90",
            "G/Sh", "Crs_p90", "Fld_p90", "Def_Actions_p90"
        ],
        "MF": [
            "Gls_p90", "Ast_p90", "GA_non_pk_p90", "Sh/90", "Crs_p90",
            "Fld_p90", "TklW_p90", "Int_p90", "Def_Actions_p90"
        ],
        "DF": [
            "TklW_p90", "Int_p90", "Def_Actions_p90", "Crs_p90",
            "Ast_p90", "Gls_p90", "Fld_p90", "Fls_p90"
        ],
        "GK": [
            "Save%", "CS%", "GA90", "Saves_p90"
        ],
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def find_similar_players(
        self,
        target_player: str,
        top_n: int = 10,
        max_age: float = 40.0,
        leagues: list = None,
        exclude_same_squad: bool = True,
    ) -> pd.DataFrame:
        if target_player not in self.df["Player"].values:
            raise ValueError(f"Oyuncu bulunamadı: {target_player}")

        target_row = self.df[self.df["Player"] == target_player].iloc[0]
        pos = target_row["Primary_Pos"]
        features = [f for f in self.POSITION_FEATURES[pos] if f in self.df.columns]

        # Aynı ana mevkideki oyuncuları al
        pos_df = self.df[self.df["Primary_Pos"] == pos].copy().reset_index(drop=True)

        # Z-Score (StandardScaler) ile tüm metrikleri aynı ölçeğe getir
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(pos_df[features].fillna(0.0))

        # Kosinüs Benzerliği Matrisini hesapla (-1 ile +1 arasını 0-100% ölçeğine çevir)
        target_idx = pos_df[pos_df["Player"] == target_player].index[0]
        sim_scores = cosine_similarity(
            feature_matrix[target_idx].reshape(1, -1), feature_matrix
        ).flatten()

        pos_df["Similarity_%"] = ((sim_scores + 1.0) / 2.0 * 100.0).round(1)

        # Kendisini çıkar ve filtreleri uygula
        results = pos_df[pos_df["Player"] != target_player].copy()
        results = results[results["Age"] <= max_age]

        if leagues:
            results = results[results["League"].isin(leagues)]

        if exclude_same_squad:
            results = results[results["Squad"] != target_row["Squad"]]

        return results.sort_values(
            by=["Similarity_%", "OptiScout_Score"], ascending=[False, False]
        ).head(top_n)