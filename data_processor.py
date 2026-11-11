import os
import numpy as np
import pandas as pd


class FootballDataProcessor:
    """
    FBref 2026-2027 veri setini temizler, 90 dakika basina (Per 90) metrikleri uretir
    ve her oyuncuya kendi mevkisi (GK, DF, MF, FW) icinde 0-100 arasi
    yuzdelik dilim (Percentile) ile agirlikli 'OptiScoutScore' hesaplar.
    """

    def __init__(
        self,
        light_path: str = "data/players_data_light-2026_2027.csv",
        full_path: str = "data/players_data-2026_2027.csv",
    ):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.light_path = os.path.join(base_dir, light_path) if not os.path.isabs(light_path) else light_path
        self.full_path = os.path.join(base_dir, full_path) if not os.path.isabs(full_path) else full_path

    def load_raw_data(self) -> pd.DataFrame:
        if os.path.exists(self.full_path):
            df = pd.read_csv(self.full_path)
        elif os.path.exists(self.light_path):
            df = pd.read_csv(self.light_path)
        else:
            raise FileNotFoundError(f"Veri dosyasi bulunamadi: {self.light_path}")
        return df

    def clean_and_engineer(self, min_minutes: int = 180) -> pd.DataFrame:
        df = self.load_raw_data().copy()

        # 1. Metin Sutunlarini Tip-Guvenli Temizle (NaN float hatalarini engeller)
        df["Nation"] = df["Nation"].fillna("").astype(str)
        df["Comp"] = df["Comp"].fillna("").astype(str)
        df["Pos"] = df["Pos"].fillna("MF").astype(str)

        df["Nation_Clean"] = df["Nation"].apply(
            lambda x: str(x).split()[-1] if len(str(x).split()) > 0 else str(x)
        )
        df["League"] = df["Comp"].apply(
            lambda x: " ".join(str(x).split()[1:]) if len(str(x).split()) > 1 else str(x)
        )

        # 2. Pozisyonlari Ayristir (Or: 'FW,MF' -> Primary_Pos: 'FW')
        pos_split = df["Pos"].str.split(",", expand=True)
        df["Primary_Pos"] = pos_split[0].str.strip()
        df["Secondary_Pos"] = (
            pos_split[1].str.strip() if pos_split.shape[1] > 1 else None
        )

        # Sadece 4 ana mevkii filtrele
        valid_positions = ["GK", "DF", "MF", "FW"]
        df = df[df["Primary_Pos"].isin(valid_positions)].copy()

        # 3. Sayisal Sutunlari Guvenli Sekilde Donustur
        numeric_cols = [
            "Age", "MP", "Starts", "Min", "90s", "Gls", "Ast", "G+A",
            "G-PK", "PK", "PKatt", "CrdY", "CrdR", "G+A-PK", "Sh", "SoT",
            "SoT%", "Sh/90", "SoT/90", "G/Sh", "G/SoT", "Crs", "TklW",
            "Int", "Fld", "Fls", "GA", "GA90", "SoTA", "Saves", "Save%",
            "W", "D", "L", "CS", "CS%", "PKsv",
        ]
        optional_advanced_cols = ["xG", "npxG", "xAG", "PrgC", "PrgP", "KP", "SCA", "GCA"]
        for col in numeric_cols + optional_advanced_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # 4. Minimum Sure Filtresi
        df = df[df["Min"] >= min_minutes].copy()
        df["90s_safe"] = df["90s"].replace(0, np.nan)
        df["90s_safe"] = df["90s_safe"].fillna(df["Min"] / 90.0).clip(lower=0.5)

        # 5. "Per 90" Metrikleri Turet
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

        # Disiplin Puani
        df["Discipline_Raw"] = -1.0 * (
            (df["CrdY"] + 3 * df["CrdR"] + 0.2 * df["Fls"]) / df["90s_safe"]
        )

        # 6. Yuzdelik Dilim ve OptiScout Skoru
        df = self._compute_position_scores(df)

        return df.reset_index(drop=True)

    def _compute_position_scores(self, df: pd.DataFrame) -> pd.DataFrame:
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
                + 0.20 * pct("GA90", ascending=False)
                + 0.10 * pct("Saves_p90")
            )
            disc_score = pct("Discipline_Raw")

            df.loc[mask, "Attacking_Score"] = att_score.round(1)
            df.loc[mask, "Creativity_Score"] = cre_score.round(1)
            df.loc[mask, "Defensive_Score"] = def_score.round(1)
            df.loc[mask, "Goalkeeping_Score"] = gk_score.round(1)
            df.loc[mask, "Discipline_Score"] = disc_score.round(1)

            if pos == "FW":
                overall = 0.60 * att_score + 0.25 * cre_score + 0.05 * def_score + 0.10 * disc_score
            elif pos == "MF":
                overall = 0.30 * att_score + 0.35 * cre_score + 0.25 * def_score + 0.10 * disc_score
            elif pos == "DF":
                overall = 0.10 * att_score + 0.15 * cre_score + 0.60 * def_score + 0.15 * disc_score
            else:
                overall = 0.90 * gk_score + 0.10 * disc_score

            df.loc[mask, "OptiScout_Score"] = overall.round(1)

        return df
