"""Hoca tercihi bileseni ve ilk 11'de kilitle/cikar."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor import FootballDataProcessor, add_decision_score  # noqa: E402
from squad_optimizer import SquadOptimizer  # noqa: E402


@pytest.fixture(scope="module")
def base():
    return FootballDataProcessor().clean_and_engineer(min_minutes=450)


def test_coach_weight_blends_minutes_share(base):
    q = add_decision_score(base, 0.7, 0.0)
    c = add_decision_score(base, 0.7, 0.3)
    assert (q["Decision_Score"] == add_decision_score(base, 0.7)["Decision_Score"]).all()  # varsayilan 0
    ok = c["Coach_Score"].notna()
    expected = (0.7 * q.loc[ok, "Decision_Score"] + 0.3 * c.loc[ok, "Coach_Score"]).round(1)
    assert (c.loc[ok, "Decision_Score"] - expected).abs().max() <= 0.11
    # Dakika payi bilinmeyen oyuncuda karar skoru kaliteye esit
    assert (c.loc[~ok, "Decision_Score"] == q.loc[~ok, "Decision_Score"]).all()


def test_regular_starter_gains_over_benched_player(base):
    c = add_decision_score(base, 0.7, 0.3)
    q = add_decision_score(base, 0.7, 0.0)
    gain = c["Decision_Score"] - q["Decision_Score"]
    regular = c["Coach_Score"] >= 90
    fringe = c["Coach_Score"] <= 50
    assert gain[regular].mean() > gain[fringe].mean()


@pytest.fixture(scope="module")
def opt(base):
    return SquadOptimizer(add_decision_score(base, 0.7, 0.2), score_column="Decision_Score")


def test_locked_players_play_and_banned_players_do_not(opt):
    plain = opt.optimize_starting_xi(max_avg_age=40)
    assert plain["status"] == "Optimal"
    banned = plain["squad"]["Player_ID"].iloc[:3].tolist()
    df = opt.df
    # Ilk 11'e normalde girmeyen, dusuk skorlu iki saha oyuncusu
    outside = df[~df["Player_ID"].isin(plain["squad"]["Player_ID"])].sort_values("Decision_Score")
    weak_cb = outside[outside["TM_Sub_Position"] == "Centre-Back"]["Player_ID"].iloc[0]
    best_cf = outside[outside["TM_Sub_Position"] == "Centre-Forward"]["Player_ID"].iloc[-1]
    locked = [weak_cb, best_cf]

    res = opt.optimize_starting_xi(max_avg_age=40, locked_ids=locked, banned_ids=banned)
    assert res["status"] == "Optimal"
    xi = set(res["squad"]["Player_ID"])
    assert set(locked) <= xi
    assert not (set(banned) & xi)
    assert res["squad"].loc[res["squad"]["Player_ID"].isin(locked), "Locked"].all()
    assert res["avg_effective_score"] <= plain["avg_effective_score"]


def test_locked_player_ignores_league_filter(opt):
    df = opt.df
    other = df[df["League"] != "Premier League"].sort_values("Decision_Score")["Player_ID"].iloc[0]
    res = opt.optimize_starting_xi(max_avg_age=40, selected_leagues=["Premier League"], locked_ids=[other])
    assert res["status"] == "Optimal"
    assert other in set(res["squad"]["Player_ID"])


def test_impossible_locks_explain_why(opt):
    # 4-3-3'te tek forvet yeri var: iki santrforu kilitlemek mumkun degil
    cfs = opt.df[opt.df["TM_Sub_Position"] == "Centre-Forward"]["Player_ID"].drop_duplicates().iloc[:2].tolist()
    res = opt.optimize_starting_xi(max_avg_age=40, allow_out_of_position=False, locked_ids=cfs)
    assert res["status"] == "Infeasible" and res["lock_issue"]

    gks = opt.df[opt.df["Primary_Pos"] == "GK"]["Player_ID"].drop_duplicates().iloc[:2].tolist()
    res = opt.optimize_starting_xi(max_avg_age=40, locked_ids=gks)
    assert res["status"] == "Infeasible"
    assert res["lock_issue"]
    with pytest.raises(ValueError):
        opt.optimize_starting_xi(locked_ids=[gks[0]], banned_ids=[gks[0]])
