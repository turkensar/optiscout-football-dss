import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor import SEASON_START_YEAR, FootballDataProcessor  # noqa: E402
from similarity_engine import PlayerSimilarityEngine  # noqa: E402
from squad_optimizer import SquadOptimizer  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return FootballDataProcessor().clean_and_engineer(min_minutes=270)


def test_no_zero_or_missing_age(df):
    assert df["Age"].notna().all()
    assert (df["Age"] > 0).all()


def test_missing_age_is_derived_from_birth_year_or_dropped(monkeypatch):
    proc = FootballDataProcessor()
    raw = proc.load_raw_data()
    raw = raw[raw["Pos"] == "MF"].head(2).copy()
    raw["Player"] = ["Has Born", "No Age Info"]
    raw["Age"] = None
    raw["Born"] = [2004, None]
    raw["Min"] = 900
    monkeypatch.setattr(proc, "load_raw_data", lambda: raw.copy())
    out = proc.clean_and_engineer(min_minutes=0)
    assert out["Player"].tolist() == ["Has Born"]
    assert out["Age"].iloc[0] == SEASON_START_YEAR - 2004


def test_row_key_is_unique(df):
    assert not df["Row_Key"].duplicated().any()


def test_shrinkage_pulls_low_minute_players_towards_mean(df):
    fw = df[df["Primary_Pos"] == "FW"]
    low = fw.nsmallest(20, "Min")
    raw_spread = (low["Gls_p90"] - fw["Gls_p90"].mean()).abs().mean()
    adj_spread = (low["Gls_adj"] - fw["Gls_adj"].mean()).abs().mean()
    assert adj_spread < raw_spread


def test_similarity_uses_selected_row_and_excludes_same_person(df):
    engine = PlayerSimilarityEngine(df)
    target = df.iloc[0]
    res = engine.find_similar_players(target["Row_Key"], top_n=50, max_age=99, exclude_same_squad=False)
    assert target["Player_ID"] not in set(res["Player_ID"])
    assert res["Score_Group"].isin([target["Score_Group"], "DF", "MF", "FW"]).all()
    assert res["Similarity_Score"].between(-100, 100).all()


def test_similarity_empty_league_list_returns_empty(df):
    engine = PlayerSimilarityEngine(df)
    res = engine.find_similar_players(df.iloc[0]["Row_Key"], leagues=[])
    assert res.empty


@pytest.mark.parametrize("formation", list(SquadOptimizer.FORMATIONS))
@pytest.mark.parametrize("allow_oop", [True, False])
def test_optimizer_respects_constraints(df, formation, allow_oop):
    max_club, max_age = 2, 26.5
    res = SquadOptimizer(df).optimize_starting_xi(
        formation=formation,
        max_players_per_club=max_club,
        max_avg_age=max_age,
        allow_out_of_position=allow_oop,
    )
    assert res["status"] == "Optimal"
    xi = res["squad"]
    assert len(xi) == 11
    assert not xi["Player_ID"].duplicated().any()
    assert xi["Squad"].value_counts().max() <= max_club
    assert xi["Age"].mean() <= max_age + 1e-9
    assert xi["Assigned_Role"].value_counts().to_dict() == SquadOptimizer.FORMATIONS[formation]
    if not allow_oop:
        assert (xi["Role_Fit"] == 1.0).all()


def test_role_eligibility_comes_from_transfermarkt_sub_position():
    roles = SquadOptimizer.eligible_roles({"TM_Sub_Position": "Left-Back", "Primary_Pos": "MF"})
    assert roles["LB"] == 1.0 and roles["LWB"] == 1.0
    assert "ST" not in roles and "CM" not in roles
    # Alt mevki yoksa FBref ana mevkisine dusulur, indirimli
    roles = SquadOptimizer.eligible_roles({"TM_Sub_Position": None, "Primary_Pos": "FW", "Secondary_Pos": "MF"})
    assert roles["ST"] == 0.9 and roles["CM"] == 0.85


def test_formations_differ_by_role():
    f = SquadOptimizer.FORMATIONS
    assert f["4-2-3-1"] != f["4-4-2"]
    assert all(sum(v.values()) == 11 for v in f.values())


def test_optimizer_picks_same_person_only_once():
    rows = []
    # Ayni kisi (ayni Player_ID) iki kulupte, ikisi de en yuksek skorlu
    for squad in ["A", "B"]:
        rows.append(dict(Player="Star", Player_ID="Star|2000", Squad=squad, League="L",
                         Primary_Pos="FW", Secondary_Pos="", Age=25, Min=900, OptiScout_Score=99))
    for i in range(30):
        pos = ["GK", "DF", "DF", "DF", "DF", "MF", "MF", "MF", "FW", "FW"][i % 10]
        rows.append(dict(Player=f"P{i}", Player_ID=f"P{i}|2000", Squad=f"S{i}", League="L",
                         Primary_Pos=pos, Secondary_Pos="", Age=25, Min=900, OptiScout_Score=50))
    res = SquadOptimizer(pd.DataFrame(rows)).optimize_starting_xi(max_avg_age=30)
    assert res["status"] == "Optimal"
    assert (res["squad"]["Player"] == "Star").sum() == 1


def test_optimizer_infeasible_when_age_limit_impossible(df):
    res = SquadOptimizer(df).optimize_starting_xi(max_avg_age=15.0)
    assert res["status"] == "Infeasible"


def test_budget_frontier_is_monotonic_on_real_data(df):
    fr = SquadOptimizer(df).budget_frontier(
        [b * 1e6 for b in (25, 50, 100, 200, 400, 800)], max_avg_age=26.5
    )
    scores = fr.loc[fr["Status"] == "Optimal", "Avg_Score"].tolist()
    assert len(scores) == 6
    assert all(b >= a - 0.05 for a, b in zip(scores, scores[1:]))  # yuvarlama payi


def test_role_based_scores(df):
    from data_processor import ROLE_WEIGHTS, BROAD_WEIGHTS
    for w in list(ROLE_WEIGHTS.values()) + list(BROAD_WEIGHTS.values()):
        assert abs(sum(w.values()) - 1.0) < 1e-9
    assert df["OptiScout_Score"].between(0, 100).all()
    # Transfermarkt'ta kanat olan oyuncu, FBref onu MF dese bile kanatlarla kıyaslanır
    wingers = df[df["TM_Sub_Position"].isin(["Left Winger", "Right Winger"])]
    assert (wingers["Score_Group"] == "W").all()
    assert wingers["Primary_Pos"].nunique() > 1


def test_small_role_groups_fall_back_to_broad_group():
    from data_processor import MIN_ROLE_GROUP
    df = FootballDataProcessor().clean_and_engineer(min_minutes=720)
    counts = df["TM_Sub_Position"].map(
        __import__("data_processor").SUB_POSITION_TO_ROLE_GROUP).value_counts()
    small = counts[counts < MIN_ROLE_GROUP].index
    assert not df["Score_Group"].isin(small).any()


def test_similarity_for_winger_includes_mf_and_fw_wingers(df):
    target = df[(df["Score_Group"] == "W")].nlargest(1, "Min").iloc[0]
    res = PlayerSimilarityEngine(df).find_similar_players(target["Row_Key"], top_n=200, max_age=99)
    assert set(res.loc[res["Score_Group"] == "W", "Primary_Pos"]) >= {"MF", "FW"}


def test_non_penalty_ga_is_not_divided_twice(df):
    # FBref "G+A-PK" zaten 90 dk basina oran; bizim p90 degerimiz onunla ayni olmali
    played = df[df["90s"] >= 3]
    expected = (played["G-PK"] + played["Ast"]) / played["90s_safe"]
    assert ((played["GA_non_pk_p90"] - expected).abs() < 0.011).all()
    assert ((played["GA_non_pk_p90"] - played["G+A-PK"]).abs() < 0.05).mean() > 0.95


def test_z_blend_keeps_magnitude_differences():
    import data_processor as dp
    s = pd.DataFrame({"Primary_Pos": ["FW"] * 50, "x": [0.1] * 48 + [1.0, 5.0]})
    df = s.copy()
    for col in ["GA_non_pk_adj", "Gls_adj", "SoT_adj", "G_Sh_adj"]:
        df[col] = s["x"]
    mask = pd.Series(True, index=df.index)
    for col in ["Attacking_Score", "Creativity_Score", "Defensive_Score",
                "Goalkeeping_Score", "Discipline_Score", "OptiScout_Score"]:
        df[col] = 0.0
    dp.FootballDataProcessor._score_group(df, mask, mask, {"att": 1.0}, "FW")
    # Yuzdelik dilim tek basina 1.0 ile 5.0 arasinda ~2 puan fark birakirdi; z-skor farki buyutur
    assert df.loc[49, "Attacking_Score"] - df.loc[48, "Attacking_Score"] > 10


def test_sub_scores_only_exist_where_meaningful(df):
    from data_processor import applicable_sub_scores
    gk = df[df["Score_Group"] == "GK"]
    outfield = df[df["Score_Group"] != "GK"]
    assert gk[["Attacking_Score", "Creativity_Score", "Defensive_Score"]].isna().all().all()
    assert gk["Goalkeeping_Score"].notna().all()
    assert outfield["Goalkeeping_Score"].isna().all()
    assert outfield[["Attacking_Score", "Creativity_Score", "Defensive_Score"]].notna().all().all()
    # Roller arasi gorunumde alt skor siralamasi sunulmaz
    assert applicable_sub_scores("ALL") == []
    assert "Goalkeeping_Score" in applicable_sub_scores("GK")
    assert "Attacking_Score" in applicable_sub_scores("W")


def test_goalkeeper_never_tops_outfield_rankings(df):
    for col in ["Attacking_Score", "Creativity_Score", "Defensive_Score"]:
        top = df.sort_values(col, ascending=False, na_position="last").head(20)
        assert (top["Score_Group"] != "GK").all()
