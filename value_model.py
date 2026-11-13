"""
Piyasa değeri tahmini ve "değerinin altındaki oyuncu" (fırsat) tespiti.

Fikir: Oyuncunun performansından, yaşından, ligi ve rolünden "piyasa değeri ne
olmalı?" tahmin edilir. Gerçek Transfermarkt değeri tahmininin çok altında kalan,
yani performansına göre ucuz görünen oyuncular fırsat olarak listelenir.

Önemli tasarım kararları
------------------------
1. Çapraz doğrulama (out-of-fold) tahmini: Her oyuncunun tahmini, o oyuncuyu eğitimde
   hiç görmemiş bir modelden gelir. Aksi halde model, ezberlediği değeri geri verir ve
   "fırsat" sinyali kaybolur.
2. Kulüp gücü bilerek modele KONMAZ: Takım arkadaşlarının değerini eklemek açıklanan
   varyansı ~2 katına çıkarır, ama model "küçük kulüpteki iyi oyuncu ucuzdur" diye
   öğrenir ve tam da aranan gizli cevherleri ıskalar. Kulüp gücüyle model yalnızca
   karşılaştırma (piyasanın ne kadarını itibarın açıkladığı) için raporlanır.
3. Ridge regresyon: Bu veri boyutunda gradient boosting'den daha iyi genelleşti ve
   katsayıları yorumlanabilir ("+10 skor puanı değeri yüzde kaç artırır?").
4. Hedef log(değer): Piyasa değerleri çok çarpık (€0.5M - €200M); hata yüzdesel ölçülür.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_FEATURES = [
    "OptiScout_Score", "Attacking_Score", "Creativity_Score", "Defensive_Score",
    "Goalkeeping_Score", "Age", "Age2", "Min", "Start_Share",
]
CAT_FEATURES = ["League", "Score_Group"]
SUB_SCORES = ["Attacking_Score", "Creativity_Score", "Defensive_Score", "Goalkeeping_Score"]
N_FOLDS = 5
RANDOM_STATE = 42


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Age2"] = d["Age"] ** 2
    d["Start_Share"] = (d["Starts"] / d["MP"].clip(lower=1)).clip(0, 1)
    # Role uygulanamayan alt skorlar (ör. kalecide hücum) nötr değerle doldurulur
    d[SUB_SCORES] = d[SUB_SCORES].fillna(50.0)
    return d


def _make_model(num: list, cat: list):
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return make_pipeline(pre, RidgeCV(alphas=np.logspace(-2, 3, 30)))


class MarketValueModel:
    def __init__(self, df: pd.DataFrame):
        d = _prepare(df)
        self.train = d[d["Market_Value_EUR"].notna() & (d["Market_Value_EUR"] > 0)].copy()
        self.y = np.log(self.train["Market_Value_EUR"])
        self.metrics = {}
        self.predictions = pd.DataFrame()
        self.effects = pd.DataFrame()

    def fit(self) -> "MarketValueModel":
        if len(self.train) < 50:
            raise ValueError("Model için yeterli piyasa değeri eşleşmesi yok")
        X = self.train[NUM_FEATURES + CAT_FEATURES]
        kf = KFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        # 1) Out-of-fold tahminler (fırsat tespiti bunlarla yapılır)
        oof = cross_val_predict(_make_model(NUM_FEATURES, CAT_FEATURES), X, self.y, cv=kf)
        pct_err = np.abs(np.exp(oof - self.y) - 1)

        # 2) Karşılaştırma modelleri: yalnızca lig+rol+yaş (performans yok) ve + kulüp gücü
        ctx_num = ["Age", "Age2"]
        oof_ctx = cross_val_predict(_make_model(ctx_num, CAT_FEATURES),
                                    self.train[ctx_num + CAT_FEATURES], self.y, cv=kf)
        club = self._leave_one_out_club_strength()
        X_club = self.train[NUM_FEATURES + CAT_FEATURES].assign(Club_Strength=club)
        oof_club = cross_val_predict(_make_model(NUM_FEATURES + ["Club_Strength"], CAT_FEATURES),
                                     X_club, self.y, cv=kf)

        self.metrics = {
            "n_train": int(len(self.train)),
            "r2": float(r2_score(self.y, oof)),
            "median_pct_error": float(np.median(pct_err)),
            "r2_context_only": float(r2_score(self.y, oof_ctx)),
            "r2_with_club": float(r2_score(self.y, oof_club)),
        }

        pred = np.exp(oof)
        self.predictions = self.train.assign(
            Pred_Value_EUR=pred,
            Pred_Value_M=(pred / 1e6).round(1),
            Value_Ratio=(pred / self.train["Market_Value_EUR"]).round(2),
        )

        # 3) Yorumlanabilir etkiler: tüm veriyle eğitilmiş model
        full = _make_model(NUM_FEATURES, CAT_FEATURES).fit(X, self.y)
        self.effects = self._effects(full)
        return self

    def _leave_one_out_club_strength(self) -> pd.Series:
        """Takım arkadaşlarının ortalama log değeri (oyuncunun kendisi hariç)."""
        logv = self.y
        g = logv.groupby(self.train["Squad"])
        s, n = g.transform("sum"), g.transform("count")
        loo = (s - logv) / (n - 1).replace(0, np.nan)
        return loo.fillna(logv.mean())

    def _effects(self, model) -> pd.DataFrame:
        """Katsayıları yüzdesel etkiye çevirir: değer kaç kat/yüzde değişir."""
        pre, ridge = model.steps[0][1], model.steps[1][1]
        scaler = pre.named_transformers_["num"]
        coefs = ridge.coef_
        num_coef = dict(zip(NUM_FEATURES, coefs[: len(NUM_FEATURES)] / scaler.scale_))
        # Not: Alt skorlar, dakika ve ilk 11 oranı birbiriyle güçlü ilişkili olduğundan
        # tek tek katsayıları yanıltıcı işaret alabilir (çoklu doğrusallık); bu yüzden
        # yalnızca bağımsız yorumlanabilen etkenler raporlanır.
        rows = [
            {"Etken": "+10 OptiScout puanı", "Etki_%": (np.exp(10 * num_coef["OptiScout_Score"]) - 1) * 100},
        ]
        # Yaş: karesel terim nedeniyle 22 ve 30 yaş, 26 yaşa göre
        for age in (22, 30):
            delta = num_coef["Age"] * (age - 26) + num_coef["Age2"] * (age ** 2 - 26 ** 2)
            rows.append({"Etken": f"{age} yaş (26 yaşa göre)", "Etki_%": (np.exp(delta) - 1) * 100})
        # Lig: ortalama lige göre prim
        cat_names = pre.named_transformers_["cat"].get_feature_names_out(CAT_FEATURES)
        cat_coef = pd.Series(coefs[len(NUM_FEATURES):], index=cat_names)
        leagues = cat_coef[cat_coef.index.str.startswith("League_")]
        for name, c in (leagues - leagues.mean()).items():
            rows.append({"Etken": f"Lig: {name.replace('League_', '')}", "Etki_%": (np.exp(c) - 1) * 100})
        return pd.DataFrame(rows).assign(**{"Etki_%": lambda x: x["Etki_%"].round(1)})

    def opportunities(
        self,
        min_value_eur: float = 1e6,
        min_minutes: int = 450,
        performance_quantile: float = 0.75,
        max_age: float = 40,
        roles: list = None,
        top_n: int = 25,
    ) -> pd.DataFrame:
        """
        Fırsat oyuncuları: rolünde üst dilimde performans gösteren (varsayılan: rol içi
        en iyi %25) ama piyasa değeri tahminin çok altında kalan oyuncular.
        `min_value_eur` çok düşük ve güvenilmez değerleri (ör. €200K) dışlar.
        """
        p = self.predictions
        threshold = p.groupby("Score_Group")["OptiScout_Score"].transform(
            lambda s: s.quantile(performance_quantile)
        )
        mask = (
            (p["Market_Value_EUR"] >= min_value_eur)
            & (p["Min"] >= min_minutes)
            & (p["Age"] <= max_age)
            & (p["OptiScout_Score"] >= threshold)
        )
        if roles:
            mask &= p["Score_Group"].isin(roles)
        return p[mask].sort_values("Value_Ratio", ascending=False).head(top_n)
