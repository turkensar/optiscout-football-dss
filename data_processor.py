import os
import numpy as np
import pandas as pd


class FootballDataProcessor:
    """
    FBref 2026-2027 veri setini temizler, 90 dakika başına (Per 90) metrikleri üretir
    ve her oyuncuya kendi mevkisi (GK, DF, MF, FW) içinde 0-100 arası
    yüzdelik dilim (Percentile) ile ağırlıklı 'OptiScoutScore' hesaplar.
    """

    def __init__(
        self,
        light_path: str = "data/players_data_light-2026_2027.csv",
        full_path: str = "data/players_data-2026_2027.csv",
    ):
        self.light_path = light_path
        self.full_path = full_path

    def load_raw_data(self) -> pd.DataFrame:
        # Tam veri seti varsa ve okunabiliyorsa oradan ekstra metrikleri de çek, yoksa light kullan
        if os.path.exists(self.full_path):
            df = pd.read_csv(self.full_path)
        elif os.path.exists(self.light_path):
            df = pd.read_csv(self.light_path)
        else:
            raise FileNotFoundError(
                f"Veri dosyası bulunamadı: {self.light_path}"
            )
        return df

    def clean_and_engineer(self, min_minutes: int = 180) -> pd.DataFrame:
        df = self.load_raw_data().copy()

        # 1. Metin Sütunlarını Temizle (Nation: 'us USA' -> 'USA', Comp: 'eng Premier League' -> 'Premier League')
        df["Nation_Clean"] = (
            df["Nation"]
            .astype(str)
            .apply(lambda x: x.split()[-1] if " " in x else x)
        )
        df["League"] = (
            df["Comp"]
            .astype(str)
            .apply(lambda x: " ".join(x.split()[1:]) if " " in x else x)
        )

        # 2. Pozisyonları Ayrıştır (Ör: 'FW,MF' -> Primary_Pos: 'FW', Secondary_Pos: 'MF')
        pos_split = df["Pos"].astype(str).str.split(",", expand=True)
        df["Primary_Pos"] = pos_split[0].str.strip()
        df["Secondary_Pos"] = (
            pos_split[1].str.strip() if pos_split.shape[1] > 1 else None
        )

        # Sadece 4 ana mevkii filtrele
        valid_positions = ["GK", "DF", "MF", "FW"]
        df = df[df["Primary_Pos"].isin(valid_positions)].copy()

        # 3. Sayısal Sütunları Güvenli Şekilde Dönüştür
        numeric_cols = [
            "Age", "MP", "Starts", "Min", "90s", "Gls", "Ast", "G+A",
            "G-PK", "PK", "PKatt", "CrdY", "CrdR", "G+A-PK", "Sh", "SoT",
            "SoT%", "Sh/90", "SoT/90", "G/Sh", "G/SoT", "Crs", "TklW",
            "Int", "Fld", "Fls", "GA", "GA90", "SoTA", "Saves", "Save%",
            "W", "D", "L", "CS", "CS%", "PKsv",
        ]
        # Tam veri setinde bulunabilecek ekstra gelişmiş metrikler
        optional_advanced_cols = ["xG", "npxG", "xAG", "PrgC", "PrgP", "KP", "SCA", "GCA"]
        for col in numeric_cols + optional_advanced_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # 4. Minimum Süre Filtresi (İstatistiksel sapmaları önlemek için)
        df = df[df["Min"] >= min_minutes].copy()
        df["90s_safe"] = df["90s"].replace(0, np.nan)
        df["90s_safe"] = df["90s_safe"].fillna(df["Min"] / 90.0).clip(lower=0.5)

        # 5. "Per 90" (90 Dakika Başına) Metrikleri Türet
        df["Gls_p90"] = (df["Gls"] / df["90s_safe"]).round(2)
        df["Ast_p90"] = (df["Ast"] / df["90s_safe"]).round(2)
        df["GA_non_pk_p90"] = (df["G+A-PK"] / df["90s_safe"]).round(2)
        df["Crs_p90"] = (df["Crs"] / df["90s_safe"]).round(2)
        df["TklW_p90"] = (df["TklW"] / df["90s_safe"]).round(2)
        df["Int_p90"] = (df["Int"] / df["90s_safe"]).round(2)
        df["Def_Actions_p90"] = ((df["TklW"] + df["Int"]) / df["90s_safe"]).round(2)
        df["Fld_p90"] = (df["Fld"] / df["90s_safe"]).round(2)
        df["Fls_p90"] = (df["Fls"] / df["90s_safe"]).round(2)
        df["Saves_p90"] = (df["Saves"] / df["90s_safe"]).round(2)

        # Disiplin Ceza Puanı (Daha az faul ve kart = daha yüksek disiplin)
        df["Discipline_Raw"] = -1.0 * (
            (df["CrdY"] + 3 * df["CrdR"] + 0.2 * df["Fls"]) / df["90s_safe"]
        )

        # 6. Mevki Bazlı Yüzdelik Dilim (Percentile) ve OptiScout Bileşik Skoru (0-100)
        df = self._compute_position_scores(df)

        return df.reset_index(drop=True)

    def _compute_position_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Her mevkii kendi içinde değerlendirerek 0-100 arası alt skorlar
        ve genel 'OptiScout_Score' üretir.
        """
        df["Attacking_Score"] = 0.0
        df["Creativity_Score"] = 0.0
        df["Defensive_Score"] = 0.0
        df["Goalkeeping_Score"] = 0.0
        df["Discipline_Score"] = 0.0
        df["OptiScout_Score"] = 0.0

        for pos in ["FW", "MF", "DF", "GK"]:
            mask = df["Primary_Pos"] == pos
            sub = df.loc[mask]
            if sub.empty:
                continue

            # Yüzdelik dilim (0-100) yardımcı fonksiyonu
            def pct(col_name: str, ascending: bool = True) -> pd.Series:
                if col_name not in sub.columns:
                    return pd.Series(50.0, index=sub.index)
                r = sub[col_name].rank(pct=True, ascending=ascending) * 100.0
                return r.fillna(50.0)

            att_score = (
                0.40 * pct("GA_non_pk_p90")
                + 0.25 * pct("Gls_p90")
                + 0.20 * pct("SoT/90")
                + 0.15 * pct("G/Sh")
            )
            cre_score = (
                0.45 * pct("Ast_p90")
                + 0.30 * pct("Crs_p90")
                + 0.25 * pct("Fld_p90")
            )
            def_score = (
                0.50 * pct("Def_Actions_p90")
                + 0.25 * pct("TklW_p90")
                + 0.25 * pct("Int_p90")
            )
            gk_score = (
                0.40 * pct("Save%")
                + 0.30 * pct("CS%")
                + 0.20 * pct("GA90", ascending=False)  # Daha az gol yeme daha iyidir
                + 0.10 * pct("Saves_p90")
            )
            disc_score = pct("Discipline_Raw")

            df.loc[mask, "Attacking_Score"] = att_score.round(1)
            df.loc[mask, "Creativity_Score"] = cre_score.round(1)
            df.loc[mask, "Defensive_Score"] = def_score.round(1)
            df.loc[mask, "Goalkeeping_Score"] = gk_score.round(1)
            df.loc[mask, "Discipline_Score"] = disc_score.round(1)

            # Mevkilere özel ağırlık matrisi
            if pos == "FW":
                overall = (
                    0.60 * att_score
                    + 0.25 * cre_score
                    + 0.05 * def_score
                    + 0.10 * disc_score
                )
            elif pos == "MF":
                overall = (
                    0.30 * att_score
                    + 0.35 * cre_score
                    + 0.25 * def_score
                    + 0.10 * disc_score
                )
            elif pos == "DF":
                overall = (
                    0.10 * att_score
                    + 0.15 * cre_score
                    + 0.60 * def_score
                    + 0.15 * disc_score
                )
            else:  # GK
                overall = 0.90 * gk_score + 0.10 * disc_score

            df.loc[mask, "OptiScout_Score"] = overall.round(1)

        return df


if __name__ == "__main__":
    processor = FootballDataProcessor()
    clean_df = processor.clean_and_engineer(min_minutes=270)
    print(f"Filtre Sonrası Aktif Oyuncu Sayısı (Min >= 270 dk): {len(clean_df)}")

    for pos in ["FW", "MF", "DF", "GK"]:
        print(f"\n=== EN YÜKSEK SKORLU TOP 5 {pos} (2026-2027) ===")
        top5 = (
            clean_df[clean_df["Primary_Pos"] == pos]
            .sort_values("OptiScout_Score", ascending=False)
            .head(5)
        )
        cols = [
            "Player", "Squad", "League", "Age", "Min",
            "Gls", "Ast", "Def_Actions_p90", "OptiScout_Score"
        ]
        print(top5[cols].to_string(index=False))