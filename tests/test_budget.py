import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_market_values import match, normalize  # noqa: E402
from squad_optimizer import SquadOptimizer  # noqa: E402

POS_CYCLE = ["GK", "DF", "DF", "DF", "DF", "MF", "MF", "MF", "FW", "FW", "FW"]


def make_pool(n=66, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        score = float(rng.uniform(40, 95))
        rows.append(dict(
            Player=f"P{i}", Player_ID=f"P{i}|2000", Squad=f"S{i}", League="L",
            Primary_Pos=POS_CYCLE[i % len(POS_CYCLE)], Secondary_Pos="", Age=25, Min=900,
            OptiScout_Score=score,
            # Iyi oyuncu daha pahali (gercekci varsayim)
            Market_Value_EUR=float(score ** 2 * 10_000),
        ))
    return pd.DataFrame(rows)


def test_budget_is_respected():
    opt = SquadOptimizer(make_pool())
    unlimited = opt.optimize_starting_xi(max_avg_age=30)
    budget = unlimited["total_value_eur"] * 0.6
    limited = opt.optimize_starting_xi(max_avg_age=30, budget_eur=budget)
    assert limited["status"] == "Optimal"
    assert limited["total_value_eur"] <= budget + 1e-6
    assert limited["avg_score"] <= unlimited["avg_score"]


def test_budget_too_small_is_infeasible():
    res = SquadOptimizer(make_pool()).optimize_starting_xi(max_avg_age=30, budget_eur=1.0)
    assert res["status"] == "Infeasible"


def test_players_without_value_are_excluded_under_budget():
    pool = make_pool()
    pool.loc[0:5, "Market_Value_EUR"] = np.nan
    res = SquadOptimizer(pool).optimize_starting_xi(max_avg_age=30, budget_eur=1e12)
    assert res["excluded_no_value"] == 6
    assert res["squad"]["Market_Value_EUR"].notna().all()


def test_frontier_is_monotonic():
    opt = SquadOptimizer(make_pool())
    full_cost = opt.optimize_starting_xi(max_avg_age=30)["total_value_eur"]
    fr = opt.budget_frontier([full_cost * f for f in (0.6, 0.7, 0.8, 0.9, 1.0)], max_avg_age=30)
    scores = fr.loc[fr["Status"] == "Optimal", "Avg_Score"].tolist()
    assert len(scores) >= 2
    assert all(b >= a - 1e-9 for a, b in zip(scores, scores[1:]))


def test_normalize_handles_accents_and_special_letters():
    assert normalize("Vinícius Júnior") == "vinicius junior"
    assert normalize("Martin Ødegaard") == "martin odegaard"
    assert normalize("Kylian Mbappé-Lottin") == "kylian mbappe lottin"


def test_matcher_uses_birth_year_and_club():
    fb = pd.DataFrame([
        dict(Player="Vinícius Júnior", Squad="Real Madrid", Born=2000),
        dict(Player="Vitinha", Squad="Paris SG", Born=2000),
        dict(Player="Vitinha", Squad="Genoa", Born=2000),
        dict(Player="Nobody Known", Squad="X", Born=1999),
    ])
    fb["norm_name"] = fb["Player"].map(normalize)
    fb["Row_Key"] = fb["Player"] + "|" + fb["Born"].astype(str) + "|" + fb["Squad"]
    tm = pd.DataFrame([
        dict(player_id=1, name="Vinicius Junior", current_club_name="Real Madrid Club de Fútbol",
             birth_year=2000, market_value_in_eur=150_000_000, sub_position="Left Winger"),
        dict(player_id=2, name="Vitinha", current_club_name="Paris Saint-Germain Football Club",
             birth_year=2000, market_value_in_eur=90_000_000, sub_position="Central Midfield"),
        dict(player_id=3, name="Vitinha", current_club_name="Genoa Cricket and Football Club",
             birth_year=2000, market_value_in_eur=8_000_000, sub_position="Centre-Forward"),
        dict(player_id=4, name="Nobody Known", current_club_name="X",
             birth_year=1990, market_value_in_eur=1_000_000, sub_position=None),
    ])
    tm["norm_name"] = tm["name"].map(normalize)
    out = match(fb, tm).set_index("Row_Key")
    assert out.loc["Vinícius Júnior|2000|Real Madrid", "Market_Value_EUR"] == 150_000_000
    assert out.loc["Vitinha|2000|Paris SG", "Market_Value_EUR"] == 90_000_000
    assert out.loc["Vitinha|2000|Genoa", "Market_Value_EUR"] == 8_000_000
    assert "Nobody Known|1999|X" not in out.index  # dogum yili tutmuyor
