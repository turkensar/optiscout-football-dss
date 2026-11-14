import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_processor as dp  # noqa: E402
from data_processor import FootballDataProcessor  # noqa: E402
from squad_optimizer import SquadOptimizer  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return FootballDataProcessor().clean_and_engineer(min_minutes=270)


def test_contract_classification_and_fee_factors():
    t = pd.DataFrame({
        "Min%": [50.0] * 5,  # rotasyon oyuncusu: rol carpani 1.0
        "Market_Value_EUR": [10e6, 10e6, 10e6, 10e6, 10e6],
        "TM_Contract_Expiry": ["2027-06-30", "2028-06-30", "2030-06-30", None, "2025-06-30"],
    })
    out = FootballDataProcessor._attach_contract_costs(t)
    assert out["Contract_Status"].tolist() == ["free", "last_year", "long", "unknown", "unknown"]
    assert out["Est_Fee_EUR"].tolist() == pytest.approx(
        [10e6 * dp.FEE_FACTOR_FREE, 10e6 * dp.FEE_FACTOR_LAST_YEAR, 10e6, 10e6, 10e6])
    # Veri tarihinden once bitmis gorunen sozlesme eskimis kayittir -> bilinmiyor (tam bedel)


def test_estimated_fee_bounds_and_key_player_premium(df):
    v = df[df["Market_Value_EUR"].notna()]
    max_factor = max(dp.IMPORTANCE_FACTORS.values())
    assert (v["Est_Fee_EUR"] <= v["Market_Value_EUR"] * max_factor + 1e-6).all()
    assert set(df["Contract_Status"]) <= {"free", "last_year", "long", "unknown"}
    # Uzun sozlesmeli kilit oyuncu piyasa degerinin ustunde, kenardaki oyuncu altinda satilir
    key_long = v[(v["Importance"] == "key") & (v["Contract_Status"] == "long")]
    fringe_long = v[(v["Importance"] == "fringe") & (v["Contract_Status"] == "long")]
    assert (key_long["Est_Fee_EUR"] > key_long["Market_Value_EUR"]).all()
    assert (fringe_long["Est_Fee_EUR"] < fringe_long["Market_Value_EUR"]).all()
    # Bedelsiz oyuncuya prim uygulanmaz
    free = v[v["Contract_Status"] == "free"]
    assert (free["Est_Fee_EUR"] <= free["Market_Value_EUR"] * dp.FEE_FACTOR_FREE + 1e-6).all()


@pytest.mark.parametrize("budget", [30e6, 100e6])
def test_optimizer_budget_uses_selected_cost_column(df, budget):
    opt = SquadOptimizer(df)
    r = opt.optimize_starting_xi(max_avg_age=26.5, budget_eur=budget, cost_column="Est_Fee_EUR")
    assert r["status"] == "Optimal"
    assert r["squad"]["Est_Fee_EUR"].sum() <= budget + 1e-6
    assert r["total_cost_eur"] == pytest.approx(r["squad"]["Est_Fee_EUR"].sum())
    # Indirimli maliyetle ayni butce en az piyasa degeri kadar iyi kadro kurar
    base = opt.optimize_starting_xi(max_avg_age=26.5, budget_eur=budget)
    assert r["avg_effective_score"] >= base["avg_effective_score"] - 0.1


def test_transfer_plan_with_contract_costs(df):
    opt = SquadOptimizer(df)
    mv = opt.plan_transfers("Juventus", budget_eur=20e6, max_signings=2, cost_column="Market_Value_EUR")
    fee = opt.plan_transfers("Juventus", budget_eur=20e6, max_signings=2, cost_column="Est_Fee_EUR")
    assert fee["signings"]["Est_Fee_EUR"].sum() <= 20e6 + 1e-6
    assert fee["spent_eur"] == pytest.approx(fee["signings"]["Est_Fee_EUR"].sum())
    assert fee["plan"]["avg_effective_score"] >= mv["plan"]["avg_effective_score"] - 0.1
