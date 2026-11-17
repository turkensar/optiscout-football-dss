"""Transfer gerçekçiliği: kulüp katsayısı (kadro değeri) ve oyuncu katsayısı (piyasa değeri)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor import FootballDataProcessor, add_decision_score  # noqa: E402
from squad_optimizer import SquadOptimizer  # noqa: E402


@pytest.fixture(scope="module")
def opt():
    df = add_decision_score(FootballDataProcessor().clean_and_engineer(min_minutes=270), 0.7)
    return SquadOptimizer(df, score_column="Decision_Score")


def _small_club(opt):
    values = opt.df.groupby("Squad")["Club_Value_EUR"].first().dropna()
    counts = opt.df["Squad"].value_counts()
    values = values[counts.reindex(values.index) >= 13]
    return values.idxmin(), values.min()


def test_club_value_is_squad_total(opt):
    v = opt.df.groupby("Squad")["Club_Value_EUR"].nunique()
    assert (v <= 1).all()  # kulüp başına tek değer
    assert opt.club_value("Real Madrid") > opt.club_value("Rennes") > 0


def test_mask_rules(opt):
    ext = pd.DataFrame({
        "Squad": ["Big", "Big", "Mid", "Small"],
        "Age": [25, 33, 25, 25],
        "Club_Value_EUR": [1000e6, 1000e6, 100e6, 20e6],
        "Market_Value_EUR": [30e6, 10e6, 20e6, 2e6],
    })
    mask = opt.realistic_target_mask(ext, buyer_value=200e6)
    # 25 yaşında, 1 milyarlık kulüpten 200 milyonluk kulübe: hayır (eşik %50)
    # 33 yaşında aynı kulüpten: veteran eşiği %35 -> 350M gerekir
    assert list(mask) == [False, False, True, True]
    mask = opt.realistic_target_mask(ext, buyer_value=300e6)
    assert list(mask) == [False, False, True, True]
    mask = opt.realistic_target_mask(ext, buyer_value=400e6)
    assert list(mask) == [False, True, True, True]
    # Oyuncu değeri alıcı kadro değerinin %25'ini geçemez
    mask = opt.realistic_target_mask(ext.iloc[[2]], buyer_value=60e6)
    assert not mask.iloc[0]
    # Alıcı değeri bilinmiyorsa kural uygulanmaz
    assert opt.realistic_target_mask(ext, buyer_value=float("nan")).all()


def test_small_club_gets_only_realistic_targets(opt):
    club, value = _small_club(opt)
    plan = opt.plan_transfers(club, budget_eur=30e6, max_signings=2, cost_column="Est_Fee_EUR")
    assert plan["excluded_unrealistic"] > 0
    for _, s in plan["signings"].iterrows():
        threshold = SquadOptimizer.VETERAN_STEP_DOWN if s["Age"] >= SquadOptimizer.VETERAN_AGE else SquadOptimizer.STEP_DOWN
        assert value >= threshold * s["Club_Value_EUR"]
        assert s["Market_Value_EUR"] <= SquadOptimizer.MAX_VALUE_SHARE * value
    for cand in plan["candidates"].values():
        if not cand.empty:
            assert (cand["Club_Value_EUR"] * SquadOptimizer.VETERAN_STEP_DOWN <= value).all()


def test_rennes_fullback_not_offered_to_alanyaspor(opt):
    if "Alanyaspor" not in set(opt.df["Squad"]):
        pytest.skip("Süper Lig verisi yok")
    plan = opt.plan_transfers("Alanyaspor", formation="4-3-3", budget_eur=30e6, max_signings=3,
                              cost_column="Est_Fee_EUR")
    names = set(plan["signings"]["Player"])
    for cand in plan["candidates"].values():
        if not cand.empty:
            names |= set(cand["Player"])
    assert "Mahamadou Nagida" not in names


def test_realism_can_be_turned_off(opt):
    club, _ = _small_club(opt)
    plan = opt.plan_transfers(club, budget_eur=30e6, max_signings=2, realistic=False)
    assert plan["excluded_unrealistic"] == 0


def test_realism_reduces_repeated_recommendations(opt):
    counts = opt.df["Squad"].value_counts()
    clubs = [c for c in counts.index if counts[c] >= 13][:20]
    def most_common(realistic):
        seen = {}
        for c in clubs:
            p = opt.plan_transfers(c, budget_eur=30e6, max_signings=2, cost_column="Est_Fee_EUR",
                                   realistic=realistic)
            for pid in p["signings"].get("Player_ID", []):
                seen[pid] = seen.get(pid, 0) + 1
        return max(seen.values()) if seen else 0
    assert most_common(True) < most_common(False)
