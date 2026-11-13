import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor import FootballDataProcessor  # noqa: E402
from value_model import MarketValueModel, NUM_FEATURES, CAT_FEATURES, _make_model  # noqa: E402


@pytest.fixture(scope="module")
def vm():
    return MarketValueModel(FootballDataProcessor().clean_and_engineer(min_minutes=270)).fit()


def test_model_beats_naive_baseline_and_reports_honest_metrics(vm):
    m = vm.metrics
    assert m["n_train"] > 500
    assert 0.15 < m["r2"] < 0.9  # ortalamayi tahmin etmekten (R2=0) belirgin sekilde iyi
    assert 0 < m["median_pct_error"] < 1.5
    # Bulgu: kulup gucu varyansin buyuk kismini aciklar; performans tek basina az
    assert m["r2_with_club"] > m["r2"] > m["r2_context_only"] - 0.02


def test_predictions_are_out_of_fold_not_memorised(vm):
    # Tum veriyle egitilmis model egitim verisine daha iyi uyar; out-of-fold tahminler
    # bundan belirgin sekilde daha "durust" (daha yuksek hatali) olmali.
    X = vm.train[NUM_FEATURES + CAT_FEATURES]
    in_sample = _make_model(NUM_FEATURES, CAT_FEATURES).fit(X, vm.y).predict(X)
    oof = np.log(vm.predictions["Pred_Value_EUR"])
    assert np.mean(np.abs(oof - vm.y)) > np.mean(np.abs(in_sample - vm.y))
    assert len(vm.predictions) == vm.metrics["n_train"]


def test_opportunities_respect_filters(vm):
    opp = vm.opportunities(min_value_eur=2e6, min_minutes=450, performance_quantile=0.75,
                           max_age=25, roles=["W", "ST"], top_n=50)
    assert not opp.empty
    assert (opp["Market_Value_EUR"] >= 2e6).all()
    assert (opp["Min"] >= 450).all()
    assert (opp["Age"] <= 25).all()
    assert set(opp["Score_Group"]) <= {"W", "ST"}
    # Sirali: en buyuk firsat katsayisi en ustte
    assert opp["Value_Ratio"].is_monotonic_decreasing
    # Performans esigi: her oyuncu rolunun ust %25'inde
    p = vm.predictions
    for _, row in opp.iterrows():
        q = p.loc[p["Score_Group"] == row["Score_Group"], "OptiScout_Score"].quantile(0.75)
        assert row["OptiScout_Score"] >= q - 1e-9


def test_effects_are_sensible(vm):
    eff = vm.effects.set_index("Etken")["Etki_%"]
    assert eff["+10 OptiScout puanı"] > 0          # daha iyi performans -> daha degerli
    assert eff["30 yaş (26 yaşa göre)"] < 0        # yas ilerledikce deger duser
    assert eff["Lig: Premier League"] == eff[eff.index.str.startswith("Lig:")].max()
