import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor import FootballDataProcessor  # noqa: E402
from squad_optimizer import SquadOptimizer  # noqa: E402


@pytest.fixture(scope="module")
def opt():
    return SquadOptimizer(FootballDataProcessor().clean_and_engineer(min_minutes=270))


@pytest.mark.parametrize("club", ["Juventus", "Tottenham", "Real Madrid"])
@pytest.mark.parametrize("budget,n", [(0, 2), (30e6, 1), (100e6, 3)])
def test_plan_respects_budget_and_signing_limit(opt, club, budget, n):
    r = opt.plan_transfers(club, budget_eur=budget, max_signings=n, max_signing_age=28)
    assert r["baseline"]["status"] == "Optimal"
    assert r["plan"]["status"] == "Optimal"
    sig = r["signings"]
    assert len(sig) <= n
    assert sig["Market_Value_EUR"].sum() <= budget + 1e-6
    assert (sig["Age"] <= 28).all()
    assert (sig["Squad"] != club).all()
    assert sig["Squad"].value_counts().max() <= 1 if not sig.empty else True
    # Kulubun kendi oyunculari kulup kotasina takilmaz
    xi = r["plan"]["squad"]
    assert len(xi) == 11 and not xi["Player_ID"].duplicated().any()
    assert xi["Assigned_Role"].value_counts().to_dict() == SquadOptimizer.FORMATIONS["4-3-3"]
    # Transfer takimi kotulestiremez
    assert r["plan"]["avg_effective_score"] >= r["baseline"]["avg_effective_score"] - 0.1


def test_zero_budget_means_no_signings_except_free_none(opt):
    r = opt.plan_transfers("Juventus", budget_eur=0, max_signings=3)
    assert r["signings"]["Market_Value_EUR"].sum() == 0
    assert abs(r["plan"]["avg_effective_score"] - r["baseline"]["avg_effective_score"]) < 0.2 or not r["signings"].empty


def test_more_budget_never_hurts(opt):
    scores = [opt.plan_transfers("Tottenham", budget_eur=b, max_signings=2)["plan"]["avg_effective_score"]
              for b in (0, 10e6, 40e6, 150e6)]
    assert all(b >= a - 0.1 for a, b in zip(scores, scores[1:]))


def test_baseline_uses_only_own_players_and_fills_missing_roles(opt):
    r = opt.plan_transfers("Juventus", budget_eur=0, max_signings=0)
    base = r["baseline"]["squad"]
    assert (base["Squad"] == "Juventus").all()
    # Kadroda dogal sol bek yoksa saha oyuncusu 0.5 uygunlukla yerlestirilir
    assert base["Role_Fit"].min() >= 0.5


def test_gaps_are_weakest_and_candidates_improve_them(opt):
    r = opt.plan_transfers("Tottenham", budget_eur=60e6, max_signings=2, n_gaps=3)
    base = r["baseline"]["squad"]
    gaps = r["gaps"]
    assert len(gaps) == 3
    assert gaps["Effective_Score"].max() <= base["Effective_Score"].nlargest(8).min() + 1e-9
    for key, cand in r["candidates"].items():
        row_key = key.split("|", 1)[1]
        current = gaps.loc[gaps["Row_Key"] == row_key, "Effective_Score"].iloc[0]
        if not cand.empty:
            assert (cand["Effective_Score"] > current).all()
            assert (cand["Market_Value_EUR"] <= 60e6).all()


def test_unknown_club_raises(opt):
    with pytest.raises(ValueError):
        opt.plan_transfers("Nonexistent FC")
