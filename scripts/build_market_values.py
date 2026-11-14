"""
FBref oyuncularini Transfermarkt piyasa degerleriyle eslestirir ve
`data/market_values.csv` dosyasini uretir.

Kaynak: transfermarkt-datasets (CC0) players tablosu
    https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/players.csv.gz

Kullanim (repo kok dizininde):
    python scripts/build_market_values.py
    python scripts/build_market_values.py --tm data/players.csv.gz --fbref data/players_data_light-2026_2027.csv

Eslestirme adimlari (her FBref satiri icin, ayni dogum yilindaki adaylar arasinda):
    1. exact   : normalize edilmis tam isim ayni
    2. tokens  : bir ismin tum kelimeleri digerinin icinde (ör. "Vinicius Junior" <-> "Vinicius Jose Paixao de Oliveira Junior")
    3. surname : soyad ayni, ilk ismin ilk 3 harfi ayni (ör. "Alex Grimaldo" <-> "Alejandro Grimaldo")
    4. fuzzy   : difflib benzerligi >= 0.85
Birden fazla aday kalirsa kulup adi benzerligi ile secilir; yine belirsizse eslestirilmez.
"""

import argparse
import difflib
import os
import re
import sys
import unicodedata

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# NFKD'nin ayristirmadigi harfler
SPECIAL_CHARS = str.maketrans({
    "ø": "o", "Ø": "o", "ß": "ss", "æ": "ae", "Æ": "ae", "ł": "l", "Ł": "l",
    "đ": "d", "Đ": "d", "ı": "i", "œ": "oe", "Œ": "oe", "þ": "th", "ð": "d",
})

CLUB_NOISE = {"fc", "cf", "ac", "as", "sc", "ss", "ssc", "afc", "rc", "ud", "cd", "sv", "vfb",
              "vfl", "tsg", "fk", "1", "de", "club", "calcio", "football", "futbol", "hotspur",
              "united", "city", "the", "and", "1899", "1846", "1904", "05", "04", "09", "olympique"}


def normalize(text: str) -> str:
    text = str(text).translate(SPECIAL_CHARS)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def club_tokens(name: str) -> set:
    return {t for t in normalize(name).split() if t not in CLUB_NOISE and len(t) > 1}


def club_similarity(a: str, b: str) -> float:
    ta, tb = club_tokens(a), club_tokens(b)
    if ta and tb and (ta & tb):
        return 1.0
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def name_score(fb: str, tm: str) -> tuple:
    """(yontem, skor) dondurur; eslesme yoksa (None, 0)."""
    if fb == tm:
        return "exact", 1.0
    fb_t, tm_t = set(fb.split()), set(tm.split())
    if fb_t and tm_t and (fb_t <= tm_t or tm_t <= fb_t):
        return "tokens", 0.95
    fb_l, tm_l = fb.split(), tm.split()
    if (len(fb_l) > 1 and len(tm_l) > 1 and fb_l[-1] == tm_l[-1]
            and fb_l[0][:3] == tm_l[0][:3]):
        return "surname", 0.9  # "alex grimaldo" <-> "alejandro grimaldo"
    sm = difflib.SequenceMatcher(None, fb, tm)
    if sm.real_quick_ratio() < 0.85 or sm.quick_ratio() < 0.85:
        return None, 0.0
    ratio = sm.ratio()
    if ratio >= 0.85:
        return "fuzzy", ratio
    return None, 0.0


def load_transfermarkt(path: str) -> pd.DataFrame:
    tm = pd.read_csv(path)
    tm = tm[tm["market_value_in_eur"].notna()].copy()
    if "last_season" in tm.columns:
        tm = tm[tm["last_season"] >= 2022]
    tm["birth_year"] = pd.to_datetime(tm["date_of_birth"], errors="coerce").dt.year
    tm = tm[tm["birth_year"].notna()].copy()
    tm["birth_year"] = tm["birth_year"].astype(int)
    tm["norm_name"] = tm["name"].map(normalize)
    return tm


def load_fbref(path: str) -> pd.DataFrame:
    fb = pd.read_csv(path, usecols=["Player", "Squad", "Born", "Age"])
    fb["Player"] = fb["Player"].fillna("").astype(str).str.strip()
    fb["Squad"] = fb["Squad"].fillna("").astype(str).str.strip()
    fb["Born"] = pd.to_numeric(fb["Born"], errors="coerce")
    fb = fb[fb["Born"].notna()].copy()
    fb["Born"] = fb["Born"].astype(int)
    fb["norm_name"] = fb["Player"].map(normalize)
    # data_processor.py ile ayni anahtar
    fb["Row_Key"] = fb["Player"] + "|" + fb["Born"].astype(str) + "|" + fb["Squad"]
    return fb.drop_duplicates("Row_Key")


def match(fb: pd.DataFrame, tm: pd.DataFrame) -> pd.DataFrame:
    by_year = {year: grp for year, grp in tm.groupby("birth_year")}
    out = []
    for row in fb.itertuples(index=False):
        cands = by_year.get(row.Born)
        if cands is None:
            continue
        scored = []
        for c in cands.itertuples(index=False):
            method, score = name_score(row.norm_name, c.norm_name)
            if method:
                scored.append((score, club_similarity(row.Squad, c.current_club_name), method, c))
        if not scored:
            continue
        best_name = max(s[0] for s in scored)
        top = [s for s in scored if s[0] == best_name]
        if len(top) > 1:
            top.sort(key=lambda s: s[1], reverse=True)
            # Kulup de ayirt etmiyorsa belirsiz -> eslestirme
            if top[0][1] - top[1][1] < 0.2:
                continue
        score, club_sim, method, c = top[0]
        if method in ("fuzzy", "surname") and club_sim < 0.5:
            continue  # zayif isim benzerligi + farkli kulup: guvenilmez
        out.append({
            "Row_Key": row.Row_Key,
            "Player": row.Player,
            "Squad": row.Squad,
            "TM_Player_ID": c.player_id,
            "TM_Name": c.name,
            "TM_Club": c.current_club_name,
            "TM_Sub_Position": getattr(c, "sub_position", None),
            "TM_Contract_Expiry": getattr(c, "contract_expiration_date", None),
            "TM_Foot": getattr(c, "foot", None),
            "TM_Height_cm": getattr(c, "height_in_cm", None),
            "Market_Value_EUR": int(c.market_value_in_eur),
            "Match_Method": method,
        })
    return pd.DataFrame(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tm", default=os.path.join(BASE_DIR, "data", "players.csv.gz"))
    parser.add_argument("--fbref", default=os.path.join(BASE_DIR, "data", "players_data_light-2026_2027.csv"))
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "data", "market_values.csv"))
    args = parser.parse_args(argv)

    if not os.path.exists(args.tm):
        print(f"Transfermarkt dosyasi bulunamadi: {args.tm}", file=sys.stderr)
        return 1

    fb = load_fbref(args.fbref)
    tm = load_transfermarkt(args.tm)
    matched = match(fb, tm)
    matched.to_csv(args.out, index=False)

    rate = 100.0 * len(matched) / max(len(fb), 1)
    print(f"FBref satiri: {len(fb)} | eslesen: {len(matched)} ({rate:.1f}%)")
    print(matched["Match_Method"].value_counts().to_string())
    print(f"Yazildi: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
