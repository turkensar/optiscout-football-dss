"""Ek lig (Süper Lig) entegrasyonu: birlestirme betigi, lig katsayisi ve Big-5 degismezligi."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_processor import FootballDataProcessor, add_decision_score  # noqa: E402
from scripts.add_league import (ID_COL, OUTPUT_COLUMNS, build_league,  # noqa: E402
                                read_fbref_table, split_positions)

SUPER_LIG_SRC = os.path.join(ROOT, "data", "super_lig")
LEAGUES_DIR = os.path.join(ROOT, "data", "leagues")
NO_LEAGUES = os.path.join(ROOT, "data", "__no_extra_leagues__")


# --- Birlestirme betigi -------------------------------------------------------

def test_read_fbref_table_keeps_first_duplicate_and_id(tmp_path):
    text = (
        "Playing Time,Playing Time,,,,,Per 90,Per 90,,-additional\n"
        "Rk,Player,Squad,Gls,Ast,Gls,Ast,G+A-PK,Matches,-9999\n"
        "1,A Player,Club,4,2,0.40,0.20,0.55,Matches,abc123\n"
        "Rk,Player,Squad,Gls,Ast,Gls,Ast,G+A-PK,Matches,-9999\n"  # tabloda tekrar eden baslik
        "2,B Player,Club,1,0,0.10,0.00,0.10,Matches,def456\n"
    )
    path = tmp_path / "t.csv"
    path.write_text(text, encoding="utf-8")
    t = read_fbref_table(str(path))
    assert list(t["Player"]) == ["A Player", "B Player"]
    assert t.loc[0, "Gls"] == "4"          # toplam, 90 dk basina deger degil
    assert t.loc[0, "G+A-PK"] == "0.55"
    assert t.loc[1, ID_COL] == "def456"


def test_read_fbref_table_rejects_shifted_rows(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Rk,Player,Squad,-9999\n1,A,B,x1\n2,C,D\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_fbref_table(str(path))


@pytest.mark.parametrize("raw,expected", [
    ("FWMF", "FW,MF"), ("DF", "DF"), ("MF,FW", "MF,FW"), ("DFMF", "DF,MF"),
])
def test_split_positions(raw, expected):
    assert split_positions(raw) == expected


@pytest.mark.skipif(not os.path.isdir(SUPER_LIG_SRC), reason="ham Süper Lig tablolari yok")
def test_build_super_lig_matches_main_format():
    out = build_league(SUPER_LIG_SRC, "tr Süper Lig")
    assert list(out.columns) == OUTPUT_COLUMNS + [ID_COL]
    assert len(out) > 300
    assert out[ID_COL].notna().all()
    assert (out["Comp"] == "tr Süper Lig").all()
    # Yas "22-149" bicimi tam yila cevrilmis olmali
    assert out["Age"].between(15, 45).all()
    # Pozisyonlar ana veriyle ayni bicimde (virgullu)
    pos = out["Pos"].dropna()  # FBref bazi genc oyuncularda mevki vermiyor
    assert len(pos) >= len(out) - 5
    assert pos.str.fullmatch(r"(GK|DF|MF|FW)(,(GK|DF|MF|FW))?").all()
    # Kalecilerin kaleci verisi, herkesin oynama suresi verisi var
    gk = out["Pos"].str.startswith("GK")
    assert out.loc[gk, "Saves"].notna().all()
    assert out["Min%"].notna().all()


# --- Lig katsayisi ve Big-5 degismezligi --------------------------------------

@pytest.fixture(scope="module")
def big5_only():
    return FootballDataProcessor(leagues_dir=NO_LEAGUES).clean_and_engineer(min_minutes=270)


@pytest.fixture(scope="module")
def with_leagues():
    if not os.path.isdir(LEAGUES_DIR):
        pytest.skip("ek lig dosyasi yok")
    return FootballDataProcessor().clean_and_engineer(min_minutes=270)


def test_adding_a_league_does_not_change_big5_scores(big5_only, with_leagues):
    big = with_leagues[with_leagues["Is_Reference"]].set_index("Row_Key")
    base = big5_only.set_index("Row_Key")
    assert set(big.index) == set(base.index)
    for col in ["OptiScout_Score", "Attacking_Score", "Defensive_Score", "Impact_Score",
                "Goalkeeping_Score", "Gls_adj", "Def_Actions_adj"]:
        diff = (big[col] - base[col].reindex(big.index)).abs()
        assert diff.max(skipna=True) == pytest.approx(0.0, abs=1e-9), col


def test_adding_a_league_does_not_change_big5_reputation(big5_only, with_leagues):
    a = add_decision_score(with_leagues, 0.7)
    a = a[a["Is_Reference"]].set_index("Row_Key")
    b = add_decision_score(big5_only, 0.7).set_index("Row_Key")
    diff = (a["Reputation_Score"] - b["Reputation_Score"].reindex(a.index)).abs()
    assert diff.max(skipna=True) == pytest.approx(0.0, abs=1e-9)


def test_league_factor_lowers_weaker_league_scores_only():
    if not os.path.isdir(LEAGUES_DIR):
        pytest.skip("ek lig dosyasi yok")
    proc = FootballDataProcessor()
    full = proc.clean_and_engineer(270, {"Süper Lig": 1.0})
    weak = proc.clean_and_engineer(270, {"Süper Lig": 0.75})
    sl_full = full.loc[~full["Is_Reference"], "OptiScout_Score"].mean()
    sl_weak = weak.loc[~weak["Is_Reference"], "OptiScout_Score"].mean()
    assert sl_weak < sl_full - 5
    big_full = full[full["Is_Reference"]].set_index("Row_Key")["OptiScout_Score"]
    big_weak = weak[weak["Is_Reference"]].set_index("Row_Key")["OptiScout_Score"]
    assert np.allclose(big_full.sort_index(), big_weak.sort_index())
    assert (weak.loc[~weak["Is_Reference"], "League_Factor"] == 0.75).all()


def test_league_factor_scales_goals_and_goals_conceded_in_right_direction():
    proc = FootballDataProcessor()
    df = proc.clean_and_engineer(270, {"Süper Lig": 1.0})
    if df["Is_Reference"].all():
        pytest.skip("ek lig dosyasi yok")
    scaled = FootballDataProcessor()._league_adjust(df.copy(), {"Süper Lig": 0.8})
    sl = ~df["Is_Reference"]
    assert np.allclose(scaled.loc[sl, "Gls_adj"], df.loc[sl, "Gls_adj"] * 0.8)
    assert np.allclose(scaled.loc[sl, "GA90_adj"], df.loc[sl, "GA90_adj"] / 0.8)
    # Kurtaris yuzdesi duser ama 0-100 araliginda kalir
    gk = sl & (df["Score_Group"] == "GK")
    assert (scaled.loc[gk, "SavePct_adj"] < df.loc[gk, "SavePct_adj"] + 1e-9).all()
    assert scaled.loc[gk, "SavePct_adj"].between(0, 100).all()
    # Big-5 satirlari hic degismez
    assert np.allclose(scaled.loc[~sl, "Gls_adj"], df.loc[~sl, "Gls_adj"])
