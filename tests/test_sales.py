import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor import FootballDataProcessor  # noqa: E402
from squad_optimizer import SquadOptimizer  # noqa: E402


@pytest.fixture(scope="module")
def opt():
    return SquadOptimizer(FootballDataProcessor().clean_and_engineer(min_minutes=270))


@pytest.mark.parametrize("club,budget", [("Juventus", 10e6), ("Tottenham", 20e6), ("Real Madrid", 0)])
@pytest.mark.parametrize("cost_col", ["Market_Value_EUR", "Est_Fee_EUR"])
def test_sales_fund_purchases_within_budget(opt, club, budget, cost_col):
    r = opt.plan_transfers(club, budget_eur=budget, max_signings=3, cost_column=cost_col,
                           allow_sales=True, max_sales=2)
    assert r["plan"]["status"] == "Optimal"
    # Net harcama butceyi asmaz: alim <= butce + satis geliri
    assert r["spent_eur"] <= budget + r["sale_revenue_eur"] + 1e-6
    assert len(r["sold"]) <= 2
    # Satilan oyuncu kulubun oyuncusudur ve sahada degildir
    assert (r["sold"]["Squad"] == club).all()
    assert not set(r["sold"]["Player_ID"]) & set(r["plan"]["squad"]["Player_ID"])
    assert r["sale_revenue_eur"] == pytest.approx(r["sold"]["Sale_Value_EUR"].sum())


def test_sales_never_make_the_team_worse(opt):
    without = opt.plan_transfers("Juventus", budget_eur=10e6, max_signings=3)
    with_sales = opt.plan_transfers("Juventus", budget_eur=10e6, max_signings=3, allow_sales=True)
    assert with_sales["plan"]["avg_effective_score"] >= without["plan"]["avg_effective_score"] - 0.1


def test_no_pointless_sales_when_budget_is_enough(opt):
    # Butce zaten yeterliyse satis gelirine ihtiyac yoktur; model gereksiz satis yapmamali
    r = opt.plan_transfers("Juventus", budget_eur=1e9, max_signings=2, allow_sales=True, max_sales=2)
    assert r["sold"].empty


def test_sales_disabled_by_default(opt):
    r = opt.plan_transfers("Real Madrid", budget_eur=0, max_signings=2)
    assert r["sold"].empty and r["sale_revenue_eur"] == 0
    assert r["spent_eur"] == 0
