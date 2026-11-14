import os
import numpy as np
import pandas as pd


SEASON_START_YEAR = 2026

# Bayesian shrinkage (kucuk orneklem duzeltmesi) icin "sanal" on bilgi miktarlari.
# Ornek: 450 dk -> az oynamis oyuncunun per-90 degeri, mevki ortalamasina dogru
# 5 maclik sanal veriyle cekilir. Cok oynayan oyuncu neredeyse hic etkilenmez.
SHRINK_MINUTES = 450.0
SHRINK_SHOTS = 10.0        # G/Sh icin
SHRINK_SHOTS_FACED = 20.0  # Save% icin
SHRINK_MATCHES = 5.0       # CS% icin

VALID_POSITIONS = ["GK", "DF", "MF", "FW"]

# Sözleşme durumu -> tahmini transfer maliyeti. Hesap anı bir sonraki yaz transfer dönemi.
# Sözleşmesi o tarihte biten oyuncu bedelsiz gelir (yalnızca imza ücreti/menajer payı gibi
# masraflar: piyasa değerinin %15'i); son yılına giren oyuncuyu kulüp bedelsiz kaybetmemek
# için indirimli satar (%60); diğerleri tam piyasa değeri. Oranlar varsayımdır.
TRANSFER_WINDOW = pd.Timestamp("2027-07-01")
TM_SNAPSHOT_DATE = pd.Timestamp("2026-07-06")  # Transfermarkt verisinin tarihi
FEE_FACTOR_FREE = 0.15
FEE_FACTOR_LAST_YEAR = 0.60
CONTRACT_STATUS_LABELS = {
    "free": "Bedelsiz 2027",
    "last_year": "Son yılı",
    "long": "Uzun",
    "unknown": "—",
}

# Transfermarkt alt mevkisi -> skorlama rol grubu
SUB_POSITION_TO_ROLE_GROUP = {
    "Goalkeeper": "GK",
    "Centre-Back": "CB",
    "Left-Back": "FB", "Right-Back": "FB",
    "Defensive Midfield": "DM",
    "Central Midfield": "CM",
    "Attacking Midfield": "AM", "Second Striker": "AM",
    "Left Winger": "W", "Right Winger": "W", "Left Midfield": "W", "Right Midfield": "W",
    "Centre-Forward": "ST",
}
ROLE_GROUP_LABELS = {
    "GK": "Kaleci", "CB": "Stoper", "FB": "Bek", "DM": "Ön Libero", "CM": "Merkez Orta Saha",
    "AM": "10 Numara", "W": "Kanat", "ST": "Santrfor",
    "DF": "Defans (rol bilinmiyor)", "MF": "Orta Saha (rol bilinmiyor)", "FW": "Forvet (rol bilinmiyor)",
}
# Bu sayidan az oyunculu rol grubunda yuzdelik dilim anlamsizlasir -> genis gruba dusulur
MIN_ROLE_GROUP = 20

# Genel skor agirliklari: att=Hucum, cre=Yaraticilik, def=Savunma, gk=Kalecilik, disc=Disiplin
ROLE_WEIGHTS = {
    "GK": {"gk": 0.90, "disc": 0.10},
    "CB": {"def": 0.65, "cre": 0.10, "att": 0.10, "disc": 0.15},
    "FB": {"def": 0.45, "cre": 0.35, "att": 0.05, "disc": 0.15},
    "DM": {"def": 0.50, "cre": 0.25, "att": 0.10, "disc": 0.15},
    "CM": {"def": 0.30, "cre": 0.35, "att": 0.25, "disc": 0.10},
    "AM": {"att": 0.45, "cre": 0.45, "def": 0.05, "disc": 0.05},
    "W":  {"att": 0.55, "cre": 0.35, "def": 0.05, "disc": 0.05},
    "ST": {"att": 0.75, "cre": 0.15, "def": 0.05, "disc": 0.05},
}
# Yaraticilik alt skoru: asist / orta / faul yaptirma agirliklari. Ortalar yalnizca
# bekler icin anlamli bir yaraticilik gostergesi; digerlerinde agirligi dusuk tutulur
# (aksi halde cok orta yapan bir kanat, icine kat eden bir driplingciyi geride birakir).
CREATIVITY_WEIGHTS_DEFAULT = {"Ast_adj": 0.60, "Crs_adj": 0.15, "Fld_adj": 0.25}
CREATIVITY_WEIGHTS_WIDE_DEF = {"Ast_adj": 0.45, "Crs_adj": 0.30, "Fld_adj": 0.25}
CREATIVITY_WEIGHTS = {"FB": CREATIVITY_WEIGHTS_WIDE_DEF, "DF": CREATIVITY_WEIGHTS_WIDE_DEF}

# Metrik puani = (1 - Z_BLEND) x yuzdelik dilim + Z_BLEND x olceklenmis z-skor.
# Yuzdelik dilim tek basina buyuklugu ezer (15 gol ile 4 gol ayni "%99" olur);
# z-skor ise farkin ne kadar buyuk oldugunu korur. z, +-Z_CAP ile sinirlanir.
Z_BLEND = 0.5
Z_CAP = 3.0

# Alt skorlar rol ICINDE hesaplanir; bu yuzden yalnizca o role anlamli olanlar tutulur.
# (Orn. kalecilerde neredeyse herkesin hucum degeri 0 oldugu icin tek bir asist z-skoru
# tavana vurdurur ve kaleciyi "en iyi hucumcu" gibi gosterirdi.)
OUTFIELD_SUB_SCORES = ["Attacking_Score", "Creativity_Score", "Defensive_Score"]
GK_SUB_SCORES = ["Goalkeeping_Score"]


def applicable_sub_scores(group: str) -> list:
    """Bir skor grubu (rol veya FBref mevkisi) icin anlamli alt skor sutunlari."""
    if group == "GK":
        return GK_SUB_SCORES + ["Discipline_Score"]
    if group in (None, "", "ALL"):
        return []  # Roller arasi: alt skorlar karsilastirilamaz
    return OUTFIELD_SUB_SCORES + ["Discipline_Score"]


# Rolu bilinmeyen oyuncular icin (onceki FBref mevki agirliklari)
BROAD_WEIGHTS = {
    "FW": {"att": 0.65, "cre": 0.25, "def": 0.05, "disc": 0.05},
    "MF": {"att": 0.30, "cre": 0.35, "def": 0.25, "disc": 0.10},
    "DF": {"att": 0.10, "cre": 0.15, "def": 0.60, "disc": 0.15},
    "GK": {"gk": 0.90, "disc": 0.10},
}


class FootballDataProcessor:
    """
    FBref 2026-2027 veri setini temizler, 90 dakika basina (Per 90) metrikleri uretir
    ve her oyuncuya kendi ROL grubu (kaleci, stoper, bek, ön libero, kanat, santrfor...)
    icinde, yuzdelik dilim ile z-skor karisimina dayali 0-100 arasi agirlikli
    'OptiScout_Score' hesaplar. Rol bilgisi Transfermarkt alt mevkisinden gelir; bilinmiyorsa
    FBref ana mevkisi (GK, DF, MF, FW) kullanilir.

    Ham per-90 degerleri (`*_p90`) tablolarda gosterilir; skorlama ve benzerlik
    ise kucuk orneklem etkisi azaltilmis (`*_adj`) degerler uzerinden yapilir.
    """

    def __init__(
        self,
        light_path: str = "data/players_data_light-2026_2027.csv",
        full_path: str = "data/players_data-2026_2027.csv",
        market_path: str = "data/market_values.csv",
    ):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.light_path = os.path.join(base_dir, light_path) if not os.path.isabs(light_path) else light_path
        self.full_path = os.path.join(base_dir, full_path) if not os.path.isabs(full_path) else full_path
        self.market_path = os.path.join(base_dir, market_path) if not os.path.isabs(market_path) else market_path
        self.source_file = None
        self.has_market_values = False

    def load_raw_data(self) -> pd.DataFrame:
        if os.path.exists(self.full_path):
            path = self.full_path
        elif os.path.exists(self.light_path):
            path = self.light_path
        else:
            raise FileNotFoundError(f"Veri dosyasi bulunamadi: {self.light_path}")
        self.source_file = os.path.basename(path)
        return pd.read_csv(path)

    def clean_and_engineer(self, min_minutes: int = 180) -> pd.DataFrame:
        df = self.load_raw_data().copy()

        # 1. Metin Sutunlarini Tip-Guvenli Temizle (NaN float hatalarini engeller)
        df["Player"] = df["Player"].fillna("").astype(str).str.strip()
        df["Squad"] = df["Squad"].fillna("").astype(str).str.strip()
        df["Nation"] = df["Nation"].fillna("").astype(str)
        df["Comp"] = df["Comp"].fillna("").astype(str)
        df["Pos"] = df["Pos"].fillna("MF").astype(str)

        df["Nation_Clean"] = df["Nation"].apply(
            lambda x: str(x).split()[-1] if len(str(x).split()) > 0 else str(x)
        )
        df["League"] = df["Comp"].apply(
            lambda x: " ".join(str(x).split()[1:]) if len(str(x).split()) > 1 else str(x)
        )

        # 2. Pozisyonlari Ayristir (Or: 'FW,MF' -> Primary_Pos: 'FW', Secondary_Pos: 'MF')
        pos_split = df["Pos"].str.split(",", expand=True)
        df["Primary_Pos"] = pos_split[0].str.strip()
        if pos_split.shape[1] > 1:
            df["Secondary_Pos"] = pos_split[1].fillna("").str.strip()
        else:
            df["Secondary_Pos"] = ""
        df.loc[~df["Secondary_Pos"].isin(VALID_POSITIONS), "Secondary_Pos"] = ""

        df = df[df["Primary_Pos"].isin(VALID_POSITIONS)].copy()

        # 3. Yas: bos yas 0 ile doldurulmaz (0 yasindaki oyuncu yas filtrelerini ve
        # MILP yas ortalamasi kisitini bozar). Once dogum yilindan turet, o da yoksa at.
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df["Born"] = pd.to_numeric(df["Born"], errors="coerce")
        df["Age"] = df["Age"].fillna(SEASON_START_YEAR - df["Born"])
        df = df[df["Age"].notna() & (df["Age"] > 0)].copy()

        # 4. Benzersiz kimlik: isim tek basina benzersiz degil (ayni isimli farkli
        # oyuncular ve sezon icinde transfer olup iki satirda gorunen oyuncular var).
        df["Player_ID"] = df["Player"] + "|" + df["Born"].fillna(0).astype(int).astype(str)
        df["Row_Key"] = df["Player_ID"] + "|" + df["Squad"]
        df["Label"] = df["Player"] + " (" + df["Squad"] + ")"

        # Transfermarkt piyasa degerleri (scripts/build_market_values.py ile uretilir)
        df = self._attach_market_values(df)

        # 5. Sayisal Sutunlari Guvenli Sekilde Donustur
        numeric_cols = [
            "MP", "Starts", "Min", "90s", "Gls", "Ast", "G+A",
            "G-PK", "PK", "PKatt", "CrdY", "CrdR", "G+A-PK", "Sh", "SoT",
            "SoT%", "Sh/90", "SoT/90", "G/Sh", "G/SoT", "Crs", "TklW",
            "Int", "Fld", "Fls", "GA", "GA90", "SoTA", "Saves", "Save%",
            "W", "D", "L", "CS", "CS%", "PKsv",
        ]
        optional_advanced_cols = ["xG", "npxG", "xAG", "PrgC", "PrgP", "KP", "SCA", "GCA"]
        for col in numeric_cols + optional_advanced_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        df["90s_safe"] = df["90s"].replace(0, np.nan)
        df["90s_safe"] = df["90s_safe"].fillna(df["Min"] / 90.0).clip(lower=0.5)

        # 6. Ham "Per 90" Metrikleri (tablolarda gosterim icin)
        # DIKKAT: FBref'teki "G+A-PK" sutunu toplam degil, zaten 90 dakika basina
        # orandir. Tekrar 90'a bolunmemesi icin penaltisiz gol+asist SAYISI ayrica uretilir.
        df["NPGA"] = df["G-PK"] + df["Ast"]
        df["Def_Actions"] = df["TklW"] + df["Int"]
        df["Disc_Points"] = df["CrdY"] + 3 * df["CrdR"] + 0.2 * df["Fls"]

        df["Gls_p90"] = (df["Gls"] / df["90s_safe"]).round(2)
        df["Ast_p90"] = (df["Ast"] / df["90s_safe"]).round(2)
        df["GA_non_pk_p90"] = (df["NPGA"] / df["90s_safe"]).round(2)
        df["Crs_p90"] = (df["Crs"] / df["90s_safe"]).round(2)
        df["TklW_p90"] = (df["TklW"] / df["90s_safe"]).round(2)
        df["Int_p90"] = (df["Int"] / df["90s_safe"]).round(2)
        df["Def_Actions_p90"] = (df["Def_Actions"] / df["90s_safe"]).round(2)
        df["Fld_p90"] = (df["Fld"] / df["90s_safe"]).round(2)
        df["Fls_p90"] = (df["Fls"] / df["90s_safe"]).round(2)
        df["Saves_p90"] = (df["Saves"] / df["90s_safe"]).round(2)

        # 7. Kucuk orneklem duzeltmeli (shrinkage) metrikler. On bilgi (prior),
        # dakika filtresinden ONCE, oyuncunun ROL grubundan hesaplanir: bir kanat,
        # FBref onu "MF" ya da "FW" dese de kanat ortalamasina dogru cekilir.
        # Alt mevkisi bilinmeyen oyuncular icin FBref ana mevkisi kullanilir.
        df["Prior_Group"] = (
            df["TM_Sub_Position"].map(SUB_POSITION_TO_ROLE_GROUP).fillna(df["Primary_Pos"])
        )
        k90 = SHRINK_MINUTES / 90.0
        per90_map = {
            "Gls_adj": "Gls",
            "Ast_adj": "Ast",
            "GA_non_pk_adj": "NPGA",
            "Sh_adj": "Sh",
            "SoT_adj": "SoT",
            "Crs_adj": "Crs",
            "TklW_adj": "TklW",
            "Int_adj": "Int",
            "Def_Actions_adj": "Def_Actions",
            "Fld_adj": "Fld",
            "Fls_adj": "Fls",
            "Saves_adj": "Saves",
            "GA90_adj": "GA",
        }
        for new_col, num in per90_map.items():
            df[new_col] = self._shrink_rate(df, num, "90s_safe", k90)

        df["Discipline_adj"] = -1.0 * self._shrink_rate(df, "Disc_Points", "90s_safe", k90)
        df["G_Sh_adj"] = self._shrink_rate(df, "G-PK", "Sh", SHRINK_SHOTS)
        df["SavePct_adj"] = 100.0 * self._shrink_rate(df, "Saves", "SoTA", SHRINK_SHOTS_FACED)
        df["CSPct_adj"] = 100.0 * self._shrink_rate(df, "CS", "MP", SHRINK_MATCHES)

        # 8. Minimum Sure Filtresi
        df = df[df["Min"] >= min_minutes].copy()

        # 9. Yuzdelik Dilim ve OptiScout Skoru
        df = self._compute_position_scores(df)

        return df.reset_index(drop=True)

    def _attach_market_values(self, df: pd.DataFrame) -> pd.DataFrame:
        if os.path.exists(self.market_path):
            mv = pd.read_csv(self.market_path)
            wanted = ["Row_Key", "Market_Value_EUR", "TM_Sub_Position",
                      "TM_Contract_Expiry", "TM_Foot", "TM_Height_cm"]
            for col in wanted:
                if col not in mv.columns:  # eski market_values.csv ile geriye uyumluluk
                    mv[col] = np.nan
            mv = mv[wanted].drop_duplicates("Row_Key")
            df = df.merge(mv, on="Row_Key", how="left")
            self.has_market_values = bool(mv["Market_Value_EUR"].notna().any())
        else:
            df["Market_Value_EUR"] = np.nan
            df["TM_Sub_Position"] = None
            df["TM_Contract_Expiry"] = np.nan
            df["TM_Foot"] = None
            df["TM_Height_cm"] = np.nan
        df["Market_Value_M"] = (df["Market_Value_EUR"] / 1e6).round(1)
        return self._attach_contract_costs(df)

    @staticmethod
    def _attach_contract_costs(df: pd.DataFrame) -> pd.DataFrame:
        """Sözleşme bitişinden tahmini transfer maliyeti (Est_Fee_EUR) üretir."""
        expiry = pd.to_datetime(df["TM_Contract_Expiry"], errors="coerce")
        # Veri tarihinden önce bitmiş görünen sözleşmeler eskimiş kayıttır (yenilenmiş olabilir)
        expiry = expiry.where(expiry >= TM_SNAPSHOT_DATE)
        years_left = (expiry - TRANSFER_WINDOW).dt.days / 365.25
        status = np.select(
            [expiry.isna(), years_left <= 0.05, years_left <= 1.05],
            ["unknown", "free", "last_year"],
            default="long",
        )
        factor = pd.Series(status, index=df.index).map(
            {"free": FEE_FACTOR_FREE, "last_year": FEE_FACTOR_LAST_YEAR, "long": 1.0, "unknown": 1.0}
        )
        df["Contract_Expiry"] = expiry.dt.strftime("%Y-%m")
        df["Contract_Years_Left"] = years_left.clip(lower=0).round(1)
        df["Contract_Status"] = status
        df["Contract_Label"] = pd.Series(status, index=df.index).map(CONTRACT_STATUS_LABELS)
        df["Est_Fee_EUR"] = df["Market_Value_EUR"] * factor
        df["Est_Fee_M"] = (df["Est_Fee_EUR"] / 1e6).round(1)
        return df

    @staticmethod
    def _shrink_rate(df: pd.DataFrame, num: str, denom: str, k: float) -> pd.Series:
        """(sayi + grup_ortalamasi * k) / (payda + k). Payda kucukse deger
        grup (rol) ortalamasina yaklasir; payda buyudukce ham orana yakinsar."""
        grp = df.groupby("Prior_Group")
        prior = grp[num].transform("sum") / grp[denom].transform("sum").replace(0, np.nan)
        prior = prior.fillna(0.0)
        return (df[num] + prior * k) / (df[denom] + k)

    def _compute_position_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Yuzdelik dilimler mumkun oldugunca ROL grubu icinde hesaplanir (bek bekle,
        kanat kanatla kiyaslanir) ve her rolun kendi agirliklari kullanilir.
        Transfermarkt alt mevkisi bilinmeyen oyuncular ile cok kucuk rol gruplari
        (< MIN_ROLE_GROUP oyuncu) FBref'in genis mevki grubu (DF/MF/FW) icinde puanlanir.
        `Score_Group` sutunu her oyuncunun hangi grupta puanlandigini gosterir.
        """
        for col in ["Attacking_Score", "Creativity_Score", "Defensive_Score",
                    "Goalkeeping_Score", "Discipline_Score", "OptiScout_Score"]:
            df[col] = 0.0

        sub_pos = df["TM_Sub_Position"] if "TM_Sub_Position" in df.columns else pd.Series(None, index=df.index)
        role = sub_pos.map(SUB_POSITION_TO_ROLE_GROUP)
        role_sizes = role.value_counts()
        small_roles = set(role_sizes[role_sizes < MIN_ROLE_GROUP].index)
        use_role = role.notna() & ~role.isin(small_roles)
        df["Score_Group"] = np.where(use_role, role, df["Primary_Pos"])

        # 1) Rol gruplari
        for grp in ROLE_WEIGHTS:
            mask = use_role & (role == grp)
            if mask.any():
                self._score_group(df, mask, mask, ROLE_WEIGHTS[grp], grp)

        # 2) Genis FBref grubu: yuzdelik, mevkideki TUM oyuncular arasinda hesaplanir,
        # ama yalnizca rol skoru olmayan oyunculara yazilir.
        for pos, weights in BROAD_WEIGHTS.items():
            ref_mask = df["Primary_Pos"] == pos
            target_mask = ref_mask & ~use_role
            if target_mask.any():
                self._score_group(df, ref_mask, target_mask, weights, pos)

        return df

    @staticmethod
    def _score_group(df: pd.DataFrame, ref_mask: pd.Series, target_mask: pd.Series,
                     weights: dict, group: str) -> None:
        """`ref_mask` grubuna gore metrik puanlarini hesaplar, `target_mask` satirlarina yazar."""
        sub = df.loc[ref_mask]

        def pct(col_name: str, ascending: bool = True) -> pd.Series:
            """0-100 metrik puani: yuzdelik dilim ile olceklenmis z-skorun karisimi."""
            if col_name not in sub.columns:
                return pd.Series(50.0, index=sub.index)
            col = sub[col_name]
            rank = (col.rank(pct=True, ascending=ascending) * 100.0).fillna(50.0)
            std = col.std(ddof=0)
            z = (col - col.mean()) / std if std and std > 0 else col * 0.0
            if not ascending:
                z = -z
            z_scaled = (50.0 + (50.0 / Z_CAP) * z.clip(-Z_CAP, Z_CAP)).fillna(50.0)
            return (1.0 - Z_BLEND) * rank + Z_BLEND * z_scaled

        cre_w = CREATIVITY_WEIGHTS.get(group, CREATIVITY_WEIGHTS_DEFAULT)

        parts = {
            "att": 0.40 * pct("GA_non_pk_adj") + 0.25 * pct("Gls_adj")
                   + 0.20 * pct("SoT_adj") + 0.15 * pct("G_Sh_adj"),
            "cre": sum(w * pct(col) for col, w in cre_w.items()),
            "def": 0.50 * pct("Def_Actions_adj") + 0.25 * pct("TklW_adj") + 0.25 * pct("Int_adj"),
            "gk": 0.40 * pct("SavePct_adj") + 0.30 * pct("CSPct_adj")
                  + 0.20 * pct("GA90_adj", ascending=False) + 0.10 * pct("Saves_adj"),
            "disc": pct("Discipline_adj"),
        }
        overall = sum(weights.get(k, 0.0) * v for k, v in parts.items())

        idx = target_mask[target_mask].index
        df.loc[idx, "Attacking_Score"] = parts["att"].loc[idx].round(1)
        df.loc[idx, "Creativity_Score"] = parts["cre"].loc[idx].round(1)
        df.loc[idx, "Defensive_Score"] = parts["def"].loc[idx].round(1)
        df.loc[idx, "Goalkeeping_Score"] = parts["gk"].loc[idx].round(1)
        df.loc[idx, "Discipline_Score"] = parts["disc"].loc[idx].round(1)
        df.loc[idx, "OptiScout_Score"] = overall.loc[idx].round(1)

        # Role uygulanamayan alt skorlar bos (NaN) birakilir
        not_applicable = OUTFIELD_SUB_SCORES if group == "GK" else GK_SUB_SCORES
        df.loc[idx, not_applicable] = np.nan
