import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


# Transfermarkt alt mevkileri (sub_position) -> taktik rol uygunluk katsayısı.
# 1.0 = doğal rolü, < 1.0 = oynayabilir ama skoru o oranda indirimli sayılır.
ROLE_FIT = {
    "GK":  {"Goalkeeper": 1.0},
    "CB":  {"Centre-Back": 1.0, "Defensive Midfield": 0.85},
    "LB":  {"Left-Back": 1.0, "Right-Back": 0.85, "Left Midfield": 0.85},
    "RB":  {"Right-Back": 1.0, "Left-Back": 0.85, "Right Midfield": 0.85},
    "LWB": {"Left-Back": 1.0, "Left Midfield": 1.0, "Left Winger": 0.9, "Right-Back": 0.8},
    "RWB": {"Right-Back": 1.0, "Right Midfield": 1.0, "Right Winger": 0.9, "Left-Back": 0.8},
    "DM":  {"Defensive Midfield": 1.0, "Central Midfield": 0.9, "Centre-Back": 0.8},
    "CM":  {"Central Midfield": 1.0, "Defensive Midfield": 0.9, "Attacking Midfield": 0.9},
    "AM":  {"Attacking Midfield": 1.0, "Second Striker": 0.9, "Central Midfield": 0.85,
            "Left Winger": 0.85, "Right Winger": 0.85},
    "LM":  {"Left Midfield": 1.0, "Left Winger": 1.0, "Right Midfield": 0.9, "Right Winger": 0.85,
            "Left-Back": 0.8},
    "RM":  {"Right Midfield": 1.0, "Right Winger": 1.0, "Left Midfield": 0.9, "Left Winger": 0.85,
            "Right-Back": 0.8},
    "LW":  {"Left Winger": 1.0, "Left Midfield": 0.95, "Right Winger": 0.9,
            "Attacking Midfield": 0.8, "Second Striker": 0.8},
    "RW":  {"Right Winger": 1.0, "Right Midfield": 0.95, "Left Winger": 0.9,
            "Attacking Midfield": 0.8, "Second Striker": 0.8},
    "ST":  {"Centre-Forward": 1.0, "Second Striker": 0.95, "Left Winger": 0.8, "Right Winger": 0.8},
}

# Alt mevkisi bilinmeyen oyuncular (Transfermarkt eşleşmesi yok) için FBref ana
# mevkisinden türetilen roller. Rolü kesin bilinmediği için katsayı 0.9.
FALLBACK_ROLES = {
    "GK": {"GK": 1.0},
    "DF": {"CB": 0.9, "LB": 0.9, "RB": 0.9, "LWB": 0.9, "RWB": 0.9},
    "MF": {"DM": 0.9, "CM": 0.9, "AM": 0.9, "LM": 0.9, "RM": 0.9},
    "FW": {"ST": 0.9, "LW": 0.9, "RW": 0.9},
}
SECONDARY_FALLBACK_FACTOR = 0.85

ROLE_GROUP = {
    "GK": "GK", "CB": "DF", "LB": "DF", "RB": "DF", "LWB": "DF", "RWB": "DF",
    "DM": "MF", "CM": "MF", "AM": "MF", "LM": "MF", "RM": "MF",
    "LW": "FW", "RW": "FW", "ST": "FW",
}
ROLE_ORDER = ["GK", "RB", "RWB", "CB", "LB", "LWB", "DM", "CM", "RM", "AM", "LM", "RW", "ST", "LW"]


class SquadOptimizer:
    # Her formasyon, Transfermarkt alt mevkilerine dayanan rol kotalarıyla tanımlanır.
    FORMATIONS = {
        "4-3-3":   {"GK": 1, "LB": 1, "CB": 2, "RB": 1, "DM": 1, "CM": 2, "LW": 1, "RW": 1, "ST": 1},
        "4-2-3-1": {"GK": 1, "LB": 1, "CB": 2, "RB": 1, "DM": 2, "AM": 1, "LW": 1, "RW": 1, "ST": 1},
        "4-4-2":   {"GK": 1, "LB": 1, "CB": 2, "RB": 1, "CM": 2, "LM": 1, "RM": 1, "ST": 2},
        "3-5-2":   {"GK": 1, "CB": 3, "LWB": 1, "RWB": 1, "DM": 1, "CM": 2, "ST": 2},
        "3-4-3":   {"GK": 1, "CB": 3, "LWB": 1, "RWB": 1, "CM": 2, "LW": 1, "RW": 1, "ST": 1},
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    @staticmethod
    def eligible_roles(row) -> dict:
        """Bir oyuncunun oynayabileceği roller ve uygunluk katsayıları."""
        sub = row.get("TM_Sub_Position")
        if isinstance(sub, str) and sub:
            return {role: fits[sub] for role, fits in ROLE_FIT.items() if sub in fits}
        roles = dict(FALLBACK_ROLES.get(row.get("Primary_Pos"), {}))
        sec = row.get("Secondary_Pos")
        if isinstance(sec, str) and sec in FALLBACK_ROLES:
            for role, fit in FALLBACK_ROLES[sec].items():
                roles.setdefault(role, SECONDARY_FALLBACK_FACTOR)
        return roles

    def optimize_starting_xi(
        self,
        formation: str = "4-3-3",
        max_players_per_club: int = 2,
        max_avg_age: float = 28.0,
        max_individual_age: float = 40.0,
        selected_leagues: list = None,
        excluded_squads: list = None,
        allow_out_of_position: bool = True,
        budget_eur: float = None,
    ) -> dict:
        """
        Karar değişkeni x[i, r] = 1 -> i. oyuncu r rolünde (ör. LB, DM, RW) oynar.
        Oyuncunun rol uygunluğu Transfermarkt alt mevkisinden gelir (ROLE_FIT);
        doğal rolü dışındaki bir rolde skoru uygunluk katsayısıyla çarpılır.
        `allow_out_of_position=False` ise yalnızca katsayısı 1.0 olan roller kullanılır.

        `budget_eur` verilirse toplam Transfermarkt piyasa değeri bütçeyi
        aşamaz. Piyasa değeri bilinmeyen oyuncular bu durumda havuzdan çıkarılır
        (maliyeti bilinmeyen oyuncu bütçeye "bedava" girmesin diye).
        """
        if formation not in self.FORMATIONS:
            raise ValueError(f"Gecersiz formasyon: {formation}")

        excluded_no_value = 0
        infeasible = {"status": "Infeasible", "squad": pd.DataFrame(), "excluded_no_value": 0}

        pool = self.df.copy()
        pool = pool[pool["Age"] <= max_individual_age]

        if selected_leagues is not None:
            pool = pool[pool["League"].isin(selected_leagues)]

        if excluded_squads:
            pool = pool[~pool["Squad"].isin(excluded_squads)]

        if budget_eur is not None:
            has_value = pool["Market_Value_EUR"].notna() if "Market_Value_EUR" in pool.columns else pd.Series(False, index=pool.index)
            excluded_no_value = int((~has_value).sum())
            pool = pool[has_value]
            infeasible["excluded_no_value"] = excluded_no_value

        pool = pool.reset_index(drop=True)
        if len(pool) < 11:
            return infeasible

        # Kucuk orneklem etkisi skorlamada (shrinkage) giderildigi icin ayrica
        # dakika agirligi uygulanmaz; amac = sum(skor x rol uygunlugu).
        weighted_scores = pool["OptiScout_Score"].to_numpy(dtype=float)

        req_roles = self.FORMATIONS[formation]

        # Değişken listesi: (oyuncu satırı, rol, uygunluk, katsayı)
        var_player, var_role, var_fit, var_gain = [], [], [], []
        for i, row in enumerate(pool.to_dict("records")):
            for role, fit in self.eligible_roles(row).items():
                if role not in req_roles:
                    continue
                if not allow_out_of_position and fit < 1.0:
                    continue
                var_player.append(i)
                var_role.append(role)
                var_fit.append(fit)
                var_gain.append(weighted_scores[i] * fit)

        if not var_player:
            return infeasible

        var_player = np.array(var_player)
        var_role = np.array(var_role)
        var_fit = np.array(var_fit)
        n_vars = len(var_player)
        c = -1.0 * np.array(var_gain)

        # Esitlik bozucu: skor neredeyse ayniysa ucuz kadroyu sec. 1e-12/€ ile
        # €1 milyarlik kadronun cezasi 0.001 puan; skor siralamasini degistirmez.
        if budget_eur is not None:
            c = c + 1e-12 * pool["Market_Value_EUR"].to_numpy(dtype=float)[var_player]

        def player_row(mask_per_player: np.ndarray) -> np.ndarray:
            """Oyuncu bazlı 0/1 maskesini değişken uzayına taşır."""
            return mask_per_player[var_player].astype(float)

        A_rows, lb_list, ub_list = [], [], []

        # Kisit 1: Formasyon Rol Kisitlari (toplam 11'i de garanti eder)
        for role, count in req_roles.items():
            A_rows.append((var_role == role).astype(float))
            lb_list.append(float(count))
            ub_list.append(float(count))

        # Kisit 2: Ayni kisi en fazla bir kez secilir (birden fazla rol degiskeni
        # veya sezon icinde transfer nedeniyle iki kulup satiri olsa bile)
        for _, idx in pool.groupby("Player_ID").groups.items():
            mask = np.zeros(len(pool), dtype=bool)
            mask[list(idx)] = True
            A_rows.append(player_row(mask))
            lb_list.append(0.0)
            ub_list.append(1.0)

        # Kisit 3: Ayni Kulupten Maksimum Oyuncu Kisiti
        for squad_name in pool["Squad"].unique():
            A_rows.append(player_row((pool["Squad"] == squad_name).to_numpy()))
            lb_list.append(0.0)
            ub_list.append(float(max_players_per_club))

        # Kisit 4: Kadro Maksimum Yas Ortalamasi Kisiti
        A_rows.append(pool["Age"].to_numpy(dtype=float)[var_player])
        lb_list.append(0.0)
        ub_list.append(11.0 * float(max_avg_age))

        # Kisit 5: Butce (toplam piyasa degeri)
        if budget_eur is not None:
            A_rows.append(pool["Market_Value_EUR"].to_numpy(dtype=float)[var_player])
            lb_list.append(0.0)
            ub_list.append(float(budget_eur))

        constraints = LinearConstraint(
            np.array(A_rows), np.array(lb_list), np.array(ub_list)
        )

        res = milp(
            c=c,
            integrality=np.ones(n_vars),
            bounds=Bounds(0, 1),
            constraints=constraints,
        )

        if not res.success or res.x is None:
            return infeasible

        chosen = np.where(res.x > 0.5)[0]
        selected_df = pool.iloc[var_player[chosen]].copy()
        selected_df["Assigned_Role"] = var_role[chosen]
        selected_df["Assigned_Pos"] = selected_df["Assigned_Role"].map(ROLE_GROUP)
        selected_df["Role_Fit"] = var_fit[chosen]
        selected_df["Out_Of_Position"] = selected_df["Role_Fit"] < 1.0

        role_order = {r: i for i, r in enumerate(ROLE_ORDER)}
        selected_df["Role_Order"] = selected_df["Assigned_Role"].map(role_order)
        selected_df = selected_df.sort_values(
            by=["Role_Order", "OptiScout_Score"], ascending=[True, False]
        ).drop(columns=["Role_Order"])

        total_value = (
            float(selected_df["Market_Value_EUR"].sum())
            if "Market_Value_EUR" in selected_df.columns and selected_df["Market_Value_EUR"].notna().any()
            else None
        )

        return {
            "status": "Optimal",
            "squad": selected_df,
            "avg_score": round(selected_df["OptiScout_Score"].mean(), 1),
            # Modelin maksimize ettigi deger: rol uygunlugu ile duzeltilmis ortalama skor
            "avg_effective_score": round((selected_df["OptiScout_Score"] * selected_df["Role_Fit"]).mean(), 1),
            "avg_age": round(selected_df["Age"].mean(), 1),
            "total_value_eur": total_value,
            "excluded_no_value": excluded_no_value,
        }

    def budget_frontier(self, budgets_eur: list, **kwargs) -> pd.DataFrame:
        """
        Farklı bütçe seviyeleri için modeli tekrar çözer ve
        bütçe -> ulaşılabilen en yüksek ortalama skor eğrisini döndürür.
        "Bütçeyi X artırırsam kadro ne kadar iyileşir?" sorusunun cevabı.
        """
        rows = []
        for budget in budgets_eur:
            res = self.optimize_starting_xi(budget_eur=budget, **kwargs)
            rows.append({
                "Budget_M": budget / 1e6,
                "Status": res["status"],
                "Avg_Score": res.get("avg_effective_score"),
                "Spent_M": (res["total_value_eur"] / 1e6) if res.get("total_value_eur") is not None else None,
            })
        return pd.DataFrame(rows)
