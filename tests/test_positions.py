"""Mevki verisi: Transfermarkt eşleştirme kontrolü, elle düzeltme ve bekin ters kanat cezası."""
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_processor import FootballDataProcessor  # noqa: E402
from scripts.build_market_values import positions_compatible  # noqa: E402
from squad_optimizer import CROSS_SIDE_FULLBACK_FIT, SquadOptimizer  # noqa: E402


@pytest.mark.parametrize("fbref,sub,expected", [
    ("MF", "Goalkeeper", False),        # orta saha bir kaleciyle eşleşmiş -> başka kişi
    ("GK", "Centre-Back", False),
    ("FW", "Centre-Back", False),       # yalnızca forvet oynayan biri stoper olamaz
    ("DF", "Centre-Forward", False),
    ("DF", "Left Winger", True),        # kanat bek olarak oynayabilir
    ("MF", "Centre-Forward", True),     # FBref'te MF görünen santrforlar yaygın
    ("DF,MF", "Left-Back", True),
    ("", "Left-Back", True),            # FBref mevkisi yoksa reddedilmez
    ("MF", None, True),
])
def test_positions_compatible(fbref, sub, expected):
    assert positions_compatible(fbref, sub) is expected


def test_fullback_does_not_switch_sides_with_wrong_foot():
    left_footed_lb = {"TM_Sub_Position": "Left-Back", "TM_Foot": "left"}
    roles = SquadOptimizer.eligible_roles(left_footed_lb)
    assert roles["LB"] == 1.0
    assert roles["RB"] == CROSS_SIDE_FULLBACK_FIT
    # Sağ ayaklı sol bek sağ bekte oynayabilir: ROLE_FIT değeri korunur
    right_footed_lb = {"TM_Sub_Position": "Left-Back", "TM_Foot": "right"}
    assert SquadOptimizer.eligible_roles(right_footed_lb)["RB"] > CROSS_SIDE_FULLBACK_FIT
    both = {"TM_Sub_Position": "Right-Back", "TM_Foot": "both"}
    assert SquadOptimizer.eligible_roles(both)["LB"] > CROSS_SIDE_FULLBACK_FIT
    # Kanat oyuncuları ters tarafta oynayabilir (ters ayaklı kanat yaygındır)
    winger = {"TM_Sub_Position": "Left Winger", "TM_Foot": "left"}
    assert SquadOptimizer.eligible_roles(winger)["RW"] >= 0.9


def test_position_overrides_are_applied(tmp_path):
    path = tmp_path / "overrides.csv"
    base = FootballDataProcessor(overrides_path=str(path)).clean_and_engineer(min_minutes=270)
    target = base.sort_values("Min", ascending=False).iloc[0]
    new_sub = "Left-Back" if target["TM_Sub_Position"] != "Left-Back" else "Right-Back"
    pd.DataFrame([{"Player": target["Player"], "Squad": target["Squad"], "Sub_Position": new_sub,
                   "Note": "test"}]).to_csv(path, index=False)
    df = FootballDataProcessor(overrides_path=str(path)).clean_and_engineer(min_minutes=270)
    row = df[(df["Player"] == target["Player"]) & (df["Squad"] == target["Squad"])].iloc[0]
    assert row["TM_Sub_Position"] == new_sub
    assert row["Position_Source"] == "Elle düzeltme"
    assert row["Score_Group"] == "FB"
    # Diğer oyuncular etkilenmez
    assert (df["Position_Source"] == "Elle düzeltme").sum() == ((df["Player"] == target["Player"])
                                                               & (df["Squad"] == target["Squad"])).sum()


def test_invalid_override_position_is_rejected(tmp_path):
    path = tmp_path / "overrides.csv"
    pd.DataFrame([{"Player": "X", "Squad": "", "Sub_Position": "Sol Bek"}]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        FootballDataProcessor(overrides_path=str(path)).clean_and_engineer(min_minutes=270)


def test_reported_players_have_correct_roles():
    df = FootballDataProcessor().clean_and_engineer(min_minutes=270)
    def sub(name):
        rows = df[df["Player"] == name]
        if rows.empty:
            pytest.skip(f"{name} veride yok")
        return rows.iloc[0]["TM_Sub_Position"]
    assert sub("Levent Mercan") == "Left-Back"
    assert sub("Ilhan Fakili") in ("Left Winger", "Right Winger")
