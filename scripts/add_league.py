"""
FBref'ten elle indirilen 5 lig tablosunu (standart, kaleci, sut, oynama suresi,
cesitli) tek bir CSV'de birlestirir ve ana veri dosyasiyla AYNI sutun adlarina
cevirir. Cikti `data/leagues/` klasorune yazilir; uygulama bu klasordeki her
dosyayi Big-5 verisine otomatik ekler.

Girdi: FBref'te her tablonun "Share & Export -> Get table as CSV" metni,
su adlarla bir klasorde:
    stats.csv  keepers.csv  shooting.csv  playingtime.csv  misc.csv

Kullanim (repo kok dizininde):
    python scripts/add_league.py --src data/super_lig --comp "tr Süper Lig" --out data/leagues/super_lig-2026_2027.csv

Baska bir lig icin (ör. Eredivisie) ayni 5 tabloyu indirip --comp "nl Eredivisie" verin.
"""

import argparse
import csv
import os
import re
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ID_COL = "FBref_ID"

# Ana veri dosyasindaki (players_data_light) sutun sirasi
OUTPUT_COLUMNS = [
    "Rk", "Player", "Nation", "Pos", "Squad", "Comp", "Age", "Born", "MP", "Starts", "Min", "90s",
    "Gls", "Ast", "G+A", "G-PK", "PK", "PKatt", "CrdY", "CrdR", "G+A-PK",
    "Sh", "SoT", "SoT%", "Sh/90", "SoT/90", "G/Sh", "G/SoT", "PK_stats_shooting", "PKatt_stats_shooting",
    "Crs", "TklW", "Int", "Fld", "CrdY_stats_misc", "CrdR_stats_misc", "2CrdY", "Fls", "OG",
    "GA", "GA90", "SoTA", "Saves", "Save%", "W", "D", "L", "CS", "CS%",
    "PKatt_stats_keeper", "PKA", "PKsv", "PKm",
    "Mn/MP", "Min%", "PPM", "onG", "onGA", "+/-", "+/-90", "On-Off",
]

# tablo -> {FBref sutunu: cikti sutunu}. Yalnizca listelenen sutunlar alinir.
TABLES = {
    "stats": {c: c for c in ["Rk", "Player", "Nation", "Pos", "Squad", "Age", "Born", "MP", "Starts",
                             "Min", "90s", "Gls", "Ast", "G+A", "G-PK", "PK", "PKatt", "CrdY", "CrdR",
                             "G+A-PK"]},
    "shooting": {"Sh": "Sh", "SoT": "SoT", "SoT%": "SoT%", "Sh/90": "Sh/90", "SoT/90": "SoT/90",
                 "G/Sh": "G/Sh", "G/SoT": "G/SoT", "PK": "PK_stats_shooting",
                 "PKatt": "PKatt_stats_shooting"},
    "misc": {"Crs": "Crs", "TklW": "TklW", "Int": "Int", "Fld": "Fld", "CrdY": "CrdY_stats_misc",
             "CrdR": "CrdR_stats_misc", "2CrdY": "2CrdY", "Fls": "Fls", "OG": "OG"},
    "keepers": {"GA": "GA", "GA90": "GA90", "SoTA": "SoTA", "Saves": "Saves", "Save%": "Save%",
                "W": "W", "D": "D", "L": "L", "CS": "CS", "CS%": "CS%", "PKatt": "PKatt_stats_keeper",
                "PKA": "PKA", "PKsv": "PKsv", "PKm": "PKm"},
    "playingtime": {"Mn/MP": "Mn/MP", "Min%": "Min%", "PPM": "PPM", "onG": "onG", "onGA": "onGA",
                    "+/-": "+/-", "+/-90": "+/-90", "On-Off": "On-Off"},
}


def read_fbref_table(path: str) -> pd.DataFrame:
    """FBref 'Get table as CSV' metnini okur.

    - Ustteki grup basligi satirini ve '--' ile baslayan aciklama satirlarini atlar.
    - Ayni adla tekrar eden sutunlarda (ör. toplam Gls ve 90 dk basina Gls) ILKINI tutar.
    - Son sutun (baslikta '-9999') FBref oyuncu kimligidir.
    - Tablo icinde tekrarlanan baslik satirlarini ('Rk' degeri 'Rk' olanlar) atar.
    """
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.reader(fh) if r and not r[0].startswith("--")]
    header_idx = next((i for i, r in enumerate(rows[:5]) if r and r[0] == "Rk"), None)
    if header_idx is None:
        raise ValueError(f"{path}: 'Rk' ile baslayan baslik satiri bulunamadi")
    header = rows[header_idx]
    body = [r for r in rows[header_idx + 1:] if r and r[0] != "Rk"]

    keep_idx, names, seen = [], [], set()
    for i, name in enumerate(header):
        if i == len(header) - 1:
            keep_idx.append(i)
            names.append(ID_COL)
        elif name and name not in seen:
            seen.add(name)
            keep_idx.append(i)
            names.append(name)
    bad = [r for r in body if len(r) != len(header)]
    if bad:
        raise ValueError(f"{path}: {len(bad)} satirin sutun sayisi basliktan farkli (ilk: {bad[0][:3]})")
    data = [[r[i] for i in keep_idx] for r in body]
    return pd.DataFrame(data, columns=names).replace("", pd.NA)


def split_positions(pos) -> str:
    """'FWMF' -> 'FW,MF' (kopyalanan metinde virgul kaybolabiliyor)."""
    if pd.isna(pos):
        return pos
    text = str(pos).replace(",", "").strip()
    parts = re.findall(r"GK|DF|MF|FW", text)
    return ",".join(parts) if parts else text


def build_league(src_dir: str, comp: str) -> pd.DataFrame:
    tables = {}
    for name in TABLES:
        path = os.path.join(src_dir, f"{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Eksik tablo: {path}")
        tables[name] = read_fbref_table(path)

    base = tables["stats"]
    missing = [c for c in TABLES["stats"] if c not in base.columns]
    if missing:
        raise ValueError(f"stats.csv icinde eksik sutunlar: {missing}")
    df = base[list(TABLES["stats"]) + [ID_COL]].copy()

    # Diger tablolar (oyuncu kimligi, kulup) ile eklenir: sezon icinde kulup degistiren
    # oyuncu her kulup icin ayri satirdir.
    for name in ["shooting", "misc", "keepers", "playingtime"]:
        tbl = tables[name]
        mapping = {k: v for k, v in TABLES[name].items() if k in tbl.columns}
        part = tbl[[ID_COL, "Squad"] + list(mapping)].rename(columns=mapping)
        part = part.drop_duplicates([ID_COL, "Squad"])
        df = df.merge(part, on=[ID_COL, "Squad"], how="left")

    df["Comp"] = comp
    df["Pos"] = df["Pos"].map(split_positions)
    # FBref yasi "yil-gun" bicimindedir (ör. 22-149) -> tam yil
    df["Age"] = pd.to_numeric(df["Age"].astype("string").str.split("-").str[0], errors="coerce")

    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    numeric = [c for c in OUTPUT_COLUMNS if c not in ("Player", "Nation", "Pos", "Squad", "Comp")]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[OUTPUT_COLUMNS + [ID_COL]]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default=os.path.join(BASE_DIR, "data", "super_lig"))
    parser.add_argument("--comp", default="tr Süper Lig",
                        help="Ana verideki 'Comp' bicimi: '<ulke kodu> <lig adi>'")
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "data", "leagues", "super_lig-2026_2027.csv"))
    args = parser.parse_args(argv)

    df = build_league(args.src, args.comp)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    gk = df["Pos"].astype(str).str.startswith("GK")
    print(f"{args.comp}: {len(df)} oyuncu, {df['Squad'].nunique()} kulup")
    print(f"  kaleci verisi: {df.loc[gk, 'Saves'].notna().sum()}/{gk.sum()} | "
          f"oynama suresi verisi: {df['Min%'].notna().sum()}/{len(df)}")
    print(f"Yazildi: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
