"""
Big-5 istatistiklerini ve Transfermarkt verisini Kaggle'dan indirip repodaki veri
dosyalarini yeniler. GitHub Actions'ta haftalik calisir (.github/workflows/update-data.yml),
istenirse yerelde de calistirilabilir.

Kaggle API anahtari gerekir: KAGGLE_USERNAME ve KAGGLE_KEY ortam degiskenleri
ya da ~/.kaggle/kaggle.json dosyasi.

Kullanim (repo kok dizininde):
    python scripts/update_data.py                 # Big-5 + Transfermarkt
    python scripts/update_data.py --skip-big5     # yalnizca Transfermarkt
    python scripts/update_data.py --big5-file indirilen/players_data-2026_2027.csv --tm-file players.csv
                                                  # Kaggle'a baglanmadan, elle indirilen dosyalarla

Guvenlik: Yeni dosya bariz bicimde bozuksa (eksik sutun, oyuncu sayisinda buyuk dusus)
hicbir sey yazilmaz ve betik hata koduyla cikar.
"""

import argparse
import datetime as dt
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.insert(0, BASE_DIR)

SEASON = "2026_2027"
BIG5_DATASET = "hubertsidorowicz/football-players-stats-2026-2027"
BIG5_FULL_NAME = f"players_data-{SEASON}.csv"
BIG5_LIGHT_NAME = f"players_data_light-{SEASON}.csv"
# Transfermarkt verisinin haftalik guncellenen Kaggle kopyasi (transfermarkt-datasets, CC0)
TM_DATASET = "davidcariboo/player-scores"
TM_FILE_NAME = "players.csv"

VERSIONS_PATH = os.path.join(DATA_DIR, "data_versions.json")

# Yeni dosya eskisinin bu oranindan az oyuncu iceriyorsa bozuk kabul edilir
MIN_ROW_RATIO = 0.8
MIN_TM_PLAYERS = 20000


class UpdateError(RuntimeError):
    pass


def kaggle_download(dataset: str, dest: str, file_name: str) -> None:
    """Kaggle CLI ile veri setinden TEK bir dosyayi indirir (tum seti degil) ve zip ise acar."""
    if shutil.which("kaggle") is None:
        raise UpdateError("kaggle komutu bulunamadi. Kurulum: pip install kaggle")
    os.makedirs(dest, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", dataset, "-f", file_name, "-p", dest, "--force"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise UpdateError(f"Kaggle indirmesi basarisiz ({dataset}/{file_name}):\n{result.stderr or result.stdout}")
    for archive in glob.glob(os.path.join(dest, "*.zip")):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                # Yalnizca beklenen dosya, klasor yolu olmadan cikarilir
                if os.path.basename(member) == file_name:
                    with zf.open(member) as src, open(os.path.join(dest, file_name), "wb") as dst:
                        shutil.copyfileobj(src, dst)


def find_file(root: str, name: str) -> str:
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    if not hits:
        raise UpdateError(f"Indirilen dosyalar arasinda {name} yok: {os.listdir(root)}")
    return hits[0]


def build_light(full_path: str, light_path: str) -> pd.DataFrame:
    """Tam Big-5 dosyasindan, repodaki light dosyayla AYNI sutunlara sahip yeni bir light uretir.

    Kaggle'in kendi light dosyasi oynama suresi sutunlarini (Min%, onG, +/- ...) icermez;
    bu yuzden sutun listesi mevcut light dosyadan alinir.
    """
    old = pd.read_csv(light_path)
    full = pd.read_csv(full_path)
    missing = [c for c in old.columns if c not in full.columns]
    if missing:
        raise UpdateError(f"Yeni Big-5 dosyasinda eksik sutunlar: {missing}")
    if len(full) < MIN_ROW_RATIO * len(old):
        raise UpdateError(f"Yeni Big-5 dosyasi supheli: {len(full)} satir (onceki {len(old)})")
    new = full[list(old.columns)]
    for col in ("Player", "Squad", "Comp", "Min"):
        if new[col].isna().mean() > 0.05:
            raise UpdateError(f"Yeni Big-5 dosyasinda '{col}' sutunu buyuk olcude bos")
    return new


def update_big5(full_src: str) -> dict:
    light_path = os.path.join(DATA_DIR, BIG5_LIGHT_NAME)
    new = build_light(full_src, light_path)
    old_minutes = pd.read_csv(light_path, usecols=["Min"])["Min"].sum()
    new.to_csv(light_path, index=False)
    # Tam dosya repoda tutulmaz (.gitignore) ama yerelde varsa o da yenilenir
    full_dst = os.path.join(DATA_DIR, BIG5_FULL_NAME)
    if os.path.abspath(full_src) != os.path.abspath(full_dst) and (
            os.path.exists(full_dst) or not os.environ.get("GITHUB_ACTIONS")):
        shutil.copyfile(full_src, full_dst)
    return {"rows": int(len(new)), "minutes_before": float(old_minutes),
            "minutes_after": float(new["Min"].sum())}


def update_transfermarkt(tm_src: str) -> dict:
    tm = pd.read_csv(tm_src, usecols=["player_id", "name", "market_value_in_eur", "last_season"])
    if len(tm) < MIN_TM_PLAYERS:
        raise UpdateError(f"Transfermarkt dosyasi supheli: yalnizca {len(tm)} oyuncu")
    from scripts.build_market_values import main as build_market_values
    rc = build_market_values(["--tm", tm_src])
    if rc != 0:
        raise UpdateError("Piyasa degeri eslestirmesi basarisiz")
    return {"players": int(len(tm)), "latest_season": int(tm["last_season"].max())}


def write_versions(**updates) -> dict:
    versions = {}
    if os.path.exists(VERSIONS_PATH):
        with open(VERSIONS_PATH, encoding="utf-8") as fh:
            versions = json.load(fh)
    versions.update({k: v for k, v in updates.items() if v is not None})
    with open(VERSIONS_PATH, "w", encoding="utf-8") as fh:
        json.dump(versions, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    return versions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-big5", action="store_true")
    parser.add_argument("--skip-tm", action="store_true")
    parser.add_argument("--big5-file", help="Elle indirilmis tam Big-5 dosyasi (Kaggle'a baglanmaz)")
    parser.add_argument("--tm-file", help="Elle indirilmis Transfermarkt players.csv (Kaggle'a baglanmaz)")
    args = parser.parse_args(argv)

    today = dt.date.today().isoformat()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            big5_info = tm_info = None
            if not args.skip_big5:
                src = args.big5_file
                if not src:
                    kaggle_download(BIG5_DATASET, os.path.join(tmp, "big5"), BIG5_FULL_NAME)
                    src = find_file(os.path.join(tmp, "big5"), BIG5_FULL_NAME)
                big5_info = update_big5(src)
                print(f"Big-5: {big5_info['rows']} oyuncu, toplam dakika "
                      f"{big5_info['minutes_before']:,.0f} -> {big5_info['minutes_after']:,.0f}")
            if not args.skip_tm:
                src = args.tm_file
                if not src:
                    kaggle_download(TM_DATASET, os.path.join(tmp, "tm"), TM_FILE_NAME)
                    src = find_file(os.path.join(tmp, "tm"), TM_FILE_NAME)
                tm_info = update_transfermarkt(src)
                print(f"Transfermarkt: {tm_info['players']} oyuncu")
            def file_date(path):
                # Elle verilen dosyada tarih = dosyanin degistirilme tarihi; Kaggle'dan inende bugun
                return dt.date.fromtimestamp(os.path.getmtime(path)).isoformat() if path else today
            write_versions(
                big5_updated=file_date(args.big5_file) if big5_info else None,
                transfermarkt_snapshot=file_date(args.tm_file) if tm_info else None,
            )
    except UpdateError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
