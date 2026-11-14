import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_processor as dp  # noqa: E402
from data_processor import FootballDataProcessor, add_decision_score  # noqa: E402
from squad_optimizer import SquadOptimizer  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return FootballDataProcessor().clean_and_engineer(min_minutes=270)


def test_possession_adjustment_lifts_dominant_teams(df):
    f = df.drop_duplicates("Squad").set_index("Squad")["PAdj_Factor"]
    assert f.between(dp.PADJ_MIN, dp.PADJ_MAX).all()
    # Topu cok elinde tutan takimlarin savunma aksiyonlari yukari olceklenir
    assert f["Arsenal"] > 1.1 and f["Paris SG"] > 1.1
    assert f["Arsenal"] > f["Crystal Palace"]
    row = df[df["Squad"] == "Arsenal"].iloc[0]
    assert row["Def_Actions"] == pytest.approx(row["Def_Actions_raw"] * row["PAdj_Factor"])


def test_impact_score_exists_and_rewards_team_impact(df):
    assert df["Impact_Score"].between(0, 100).all()
    cb = df[df["Score_Group"] == "CB"]
    # Takima etki skoru, sahadayken takimin gol farki ile pozitif iliskili olmali
    assert cb["Impact_Score"].corr(cb["PlusMinus_adj"]) > 0.5


def test_decision_score_blends_form_and_reputation(df):
    full_form = add_decision_score(df, 1.0)
    assert (full_form["Decision_Score"] == full_form["OptiScout_Score"].round(1)).all()
    blended = add_decision_score(df, 0.6)
    has_rep = blended["Reputation_Score"].notna()
    expected = (0.6 * blended["OptiScout_Score"] + 0.4 * blended["Reputation_Score"]).round(1)
    assert (blended.loc[has_rep, "Decision_Score"] - expected[has_rep]).abs().max() <= 0.051
    # Piyasa degeri olmayan oyuncuda karar skoru yalnizca formdur
    assert (blended.loc[~has_rep, "Decision_Score"] == blended.loc[~has_rep, "OptiScout_Score"].round(1)).all()


def test_key_players_are_protected_from_sale_by_default(df):
    opt = SquadOptimizer(add_decision_score(df, 0.7), score_column="Decision_Score")
    for club in ["Arsenal", "Paris SG", "Tottenham"]:
        r = opt.plan_transfers(club, budget_eur=0, max_signings=2, cost_column="Est_Fee_EUR",
                               allow_sales=True, max_sales=2)
        key_ids = set(df[(df["Squad"] == club) & df["Key_Player"]]["Player_ID"])
        assert key_ids, club
        assert not set(r["sold"]["Player_ID"]) & key_ids


def test_manual_protection_list_is_respected(df):
    opt = SquadOptimizer(df)
    free = opt.plan_transfers("Real Madrid", budget_eur=0, max_signings=2, allow_sales=True,
                             max_sales=2, protected_ids=[])
    if free["sold"].empty:
        pytest.skip("Bu veride satis onerilmedi")
    blocked = free["sold"]["Player_ID"].tolist()
    r = opt.plan_transfers("Real Madrid", budget_eur=0, max_signings=2, allow_sales=True,
                           max_sales=2, protected_ids=blocked)
    assert not set(r["sold"]["Player_ID"]) & set(blocked)
