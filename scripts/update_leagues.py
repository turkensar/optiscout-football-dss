"""
Ek liglerin (Süper Lig, Primeira Liga, Eredivisie, Belçika) FBref tablolarini kendi
bilgisayarinda acilan bir tarayici penceresiyle ceker ve data/leagues/ dosyalarini yeniler.

FBref otomatik istekleri Cloudflare ile engelledigi icin bu betik GitHub'da degil,
SENIN bilgisayarinda calisir. Gercek bir Chrome penceresi acilir; Cloudflare dogrulamasi
cikarsa onu sen tamamlarsin, betik tabloyu bekleyip devam eder. Sayfalar arasinda
FBref'in hiz sinirina uymak icin birkac saniye beklenir (20 sayfa ~ 2-3 dakika).

Ilk kurulum (bir kez):
    pip install -r requirements-update.txt
    python -m playwright install chromium

Kullanim (repo kok dizininde):
    python scripts/update_leagues.py                     # 4 lig
    python scripts/update_leagues.py --league super_lig  # tek lig
    python scripts/update_leagues.py --skip-market-values  # piyasa degeri eslestirmesini atla

Calisma sirasi: tablolari cek -> data/<lig>/*.csv -> add_league.py -> build_market_values.py.
Bir tablo bariz bicimde eksikse (oyuncu sayisi oncekinin %80'inin altinda) o lig guncellenmez.
"""

import argparse
import csv
import datetime as dt
import io
import json
import os
import random
import re
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.insert(0, BASE_DIR)

SEASON = "2026_2027"
PROFILE_DIR = os.path.join(os.path.expanduser("~"), ".optiscout-browser")
MIN_ROW_RATIO = 0.8
DELAY_RANGE = (4.0, 6.5)        # FBref: dakikada en fazla ~10 istek
CHALLENGE_TIMEOUT_S = 180       # Cloudflare dogrulamasi icin beklenecek sure

# klasor adi -> (FBref lig no, URL adi, Comp degeri)
LEAGUES = {
    "super_lig": (26, "Super-Lig", "tr Süper Lig"),
    "primeira_liga": (32, "Primeira-Liga", "pt Primeira Liga"),
    "eredivisie": (23, "Eredivisie", "nl Eredivisie"),
    "belgian_pro_league": (37, "Belgian-Pro-League", "be Belgian Pro League"),
}

# dosya adi -> (FBref sayfa adi, oyuncu tablosunun id'si)
PAGES = {
    "stats": ("stats", "stats_standard"),
    "keepers": ("keepers", "stats_keeper"),
    "shooting": ("shooting", "stats_shooting"),
    "playingtime": ("playingtime", "stats_playing_time"),
    "misc": ("misc", "stats_misc"),
}


class UpdateError(RuntimeError):
    pass


def page_url(comp_id: int, slug: str, page: str) -> str:
    return f"https://fbref.com/en/comps/{comp_id}/{page}/{slug}-Stats"


def _cell_text(cell) -> str:
    text = cell.get_text(" ", strip=True)
    # Sayilardaki binlik ayiricilari kaldir ("1,234" -> "1234"); FBref'in CSV ciktisi da boyledir
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", text):
        text = text.replace(",", "")
    return text


def find_table(html: str, table_id: str):
    """Tabloyu bulur; FBref bazi tablolari HTML yorumu icinde gonderir, onlara da bakar."""
    from bs4 import BeautifulSoup, Comment

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id=table_id)
    if table is not None:
        return table
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        if f'id="{table_id}"' in comment:
            table = BeautifulSoup(str(comment), "html.parser").find("table", id=table_id)
            if table is not None:
                return table
    return None


def table_html_to_csv(html: str, table_id: str) -> str:
    """FBref oyuncu tablosunu, sitenin 'Get table as CSV' ciktisiyla ayni bicimde metne cevirir.

    Cikti: grup basligi satiri, sutun basligi satiri (son sutun '-9999'), oyuncu satirlari
    (son hucre FBref oyuncu kimligi). add_league.read_fbref_table bu bicimi okur.
    """
    table = find_table(html, table_id)
    if table is None:
        raise UpdateError(f"Sayfada '{table_id}' tablosu yok")
    head_rows = table.find("thead").find_all("tr")
    header = [_cell_text(c) for c in head_rows[-1].find_all(["th", "td"])]
    group = []
    if len(head_rows) > 1:
        for c in head_rows[0].find_all(["th", "td"]):
            span = int(c.get("colspan", 1))
            label = c.get_text(" ", strip=True)
            group.extend([label] * span)
    group = (group + [""] * len(header))[:len(header)]

    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(group + ["-additional"])
    writer.writerow(header + ["-9999"])
    n = 0
    for tr in table.find("tbody").find_all("tr"):
        classes = " ".join(tr.get("class", []))
        if "thead" in classes or "spacer" in classes or "partial_table" in classes:
            continue
        cells = tr.find_all(["th", "td"])
        if not cells or cells[0].get_text(strip=True) == "Rk":
            continue
        player_cell = tr.find(attrs={"data-stat": "player"})
        player_id = player_cell.get("data-append-csv", "") if player_cell is not None else ""
        if not player_id:
            continue
        writer.writerow([_cell_text(c) for c in cells] + [player_id])
        n += 1
    if n == 0:
        raise UpdateError(f"'{table_id}' tablosunda oyuncu satiri yok")
    return out.getvalue()


def count_rows(csv_text_or_path: str, is_path: bool = False) -> int:
    if is_path:
        if not os.path.exists(csv_text_or_path):
            return 0
        with open(csv_text_or_path, encoding="utf-8-sig") as fh:
            csv_text_or_path = fh.read()
    rows = list(csv.reader(io.StringIO(csv_text_or_path)))
    return sum(1 for r in rows if r and r[0] not in ("Rk", "") and not r[0].startswith("--")
               and r[-1] not in ("-9999", "-additional"))


class FBrefBrowser:
    """Kalici profille gercek bir tarayici penceresi. Cloudflare cerezleri profilde kalir,
    bu yuzden dogrulama genelde yalnizca ilk calistirmada istenir."""

    def __init__(self, headless: bool = False):
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        kwargs = dict(user_data_dir=PROFILE_DIR, headless=headless, viewport={"width": 1280, "height": 900})
        try:
            self.ctx = self._pw.chromium.launch_persistent_context(channel="chrome", **kwargs)
        except Exception:
            # Chrome kurulu degilse Playwright'in kendi Chromium'u
            self.ctx = self._pw.chromium.launch_persistent_context(**kwargs)
        self.page = self.ctx.pages[0] if self.ctx.pages else self.ctx.new_page()

    def fetch(self, url: str, table_id: str) -> str:
        self.page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        deadline = time.time() + CHALLENGE_TIMEOUT_S
        warned = False
        while time.time() < deadline:
            html = self.page.content()
            if f'id="{table_id}"' in html:
                return html
            title = (self.page.title() or "").lower()
            if not warned and ("just a moment" in title or "bir dakika" in title or "challenge" in html[:5000].lower()):
                print("   Tarayicida dogrulama cikti; lutfen pencerede tamamla. Bekliyorum...")
                warned = True
            if "429" in title or "too many requests" in html[:3000].lower():
                raise UpdateError("FBref hiz sinirina takildi (429). Bir saat sonra tekrar dene.")
            time.sleep(2)
        raise UpdateError(f"Tablo yuklenmedi: {url}")

    def close(self):
        try:
            self.ctx.close()
        finally:
            self._pw.stop()


def update_league(browser, folder: str, first: bool) -> str:
    comp_id, slug, comp = LEAGUES[folder]
    src_dir = os.path.join(DATA_DIR, folder)
    os.makedirs(src_dir, exist_ok=True)
    fetched = {}
    for name, (page, table_id) in PAGES.items():
        if not first:
            time.sleep(random.uniform(*DELAY_RANGE))
        first = False
        url = page_url(comp_id, slug, page)
        print(f"   {name:<12} {url}")
        text = table_html_to_csv(browser.fetch(url, table_id), table_id)
        old_n = count_rows(os.path.join(src_dir, f"{name}.csv"), is_path=True)
        new_n = count_rows(text)
        if old_n and new_n < MIN_ROW_RATIO * old_n:
            raise UpdateError(f"{folder}/{name}: {new_n} oyuncu (onceki {old_n}); guncellenmedi")
        fetched[name] = text
    # Bes tablo da saglamsa yaz (yarim guncelleme olmasin)
    for name, text in fetched.items():
        with open(os.path.join(src_dir, f"{name}.csv"), "w", encoding="utf-8", newline="") as fh:
            fh.write(text)

    from scripts.add_league import main as add_league
    out = os.path.join(DATA_DIR, "leagues", f"{folder}-{SEASON}.csv")
    if add_league(["--src", src_dir, "--comp", comp, "--out", out]) != 0:
        raise UpdateError(f"{folder}: add_league basarisiz")
    return out


def mark_updated():
    path = os.path.join(DATA_DIR, "data_versions.json")
    versions = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            versions = json.load(fh)
    versions["leagues_updated"] = dt.date.today().isoformat()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(versions, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--league", choices=sorted(LEAGUES), action="append",
                        help="Yalnizca bu lig(ler). Birden fazla kez verilebilir.")
    parser.add_argument("--skip-market-values", action="store_true")
    parser.add_argument("--tm", help="Transfermarkt players.csv yolu (varsayilan: build_market_values'in varsayilani)")
    args = parser.parse_args(argv)

    try:
        import playwright  # noqa: F401
        import bs4  # noqa: F401
    except ImportError:
        print("Eksik paket. Once: pip install -r requirements-update.txt && python -m playwright install chromium",
              file=sys.stderr)
        return 1

    leagues = args.league or list(LEAGUES)
    browser = FBrefBrowser()
    done, failed = [], []
    try:
        for i, folder in enumerate(leagues):
            print(f"[{i + 1}/{len(leagues)}] {LEAGUES[folder][2]}")
            try:
                update_league(browser, folder, first=(i == 0))
                done.append(folder)
            except UpdateError as exc:
                print(f"   ATLANDI: {exc}", file=sys.stderr)
                failed.append(folder)
                if "429" in str(exc):
                    break
    finally:
        browser.close()

    if done:
        mark_updated()
        if not args.skip_market_values:
            from scripts.build_market_values import main as build_market_values
            tm_args = ["--tm", args.tm] if args.tm else []
            tm_default = os.path.join(DATA_DIR, "players.csv.gz")
            if args.tm or os.path.exists(tm_default):
                print("Piyasa degerleri yeniden eslestiriliyor (~3 dk)...")
                build_market_values(tm_args)
            else:
                print("Not: data/players.csv.gz yok; piyasa degeri eslestirmesi haftalik GitHub isinde yapilacak.")
    print(f"Guncellenen: {', '.join(done) or '-'}" + (f" | Basarisiz: {', '.join(failed)}" if failed else ""))
    if done:
        print("Simdi kontrol et: python -m pytest -q   ardindan git add data && git commit && git push")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
