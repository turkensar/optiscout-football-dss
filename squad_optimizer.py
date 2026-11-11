import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


class SquadOptimizer:
    FORMATIONS = {
        "4-3-3": {"GK": 1, "DF": 4, "MF": 3, "FW": 3},
        "4-2-3-1": {"GK": 1, "DF": 4, "MF": 4, "FW": 2},
        "3-5-2": {"GK": 1, "DF": 3, "MF": 5, "FW": 2},
        "4-4-2": {"GK": 1, "DF": 4, "MF": 4, "FW": 2},
        "3-4-3": {"GK": 1, "DF": 3, "MF": 4, "FW": 3},
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def optimize_starting_xi(
        self,
        formation: str = "4-3-3",
        max_players_per_club: int = 2,
        max_avg_age: float = 28.0,
        max_individual_age: float = 40.0,
        selected_leagues: list = None,
        excluded_Squads: list = None,
    ) -> dict:
        if formation not in self.FORMATIONS:
            raise ValueError(f"Gecersiz formasyon: {formation}")

        pool = self.df.copy()
        pool = pool[pool["Age"] <= max_individual_age]

        if selected_leagues:
            pool = pool[pool["League"].isin(selected_leagues)]

        if excluded_Squads:
            pool = pool[~pool["Squad"].isin(excluded_Squads)]

        pool = pool.reset_index(drop=True)
        n = len(pool)
        if n < 11:
            return {"status": "Infeasible", "squad": pd.DataFrame()}

        weighted_scores = (
            pool["OptiScout_Score"]
            * (0.85 + 0.15 * (pool["Min"].clip(upper=900) / 900.0))
        ).to_numpy()
        c = -1.0 * weighted_scores

        integrality = np.ones(n)
        bounds = Bounds(0, 1)

        A_rows, lb_list, ub_list = [], [], []

        # Kisit 1: Toplam 11 Oyuncu
        A_rows.append(np.ones(n))
        lb_list.append(11.0)
        ub_list.append(11.0)

        # Kisit 2: Formasyon Mevki Kisitlari
        req_pos = self.FORMATIONS[formation]
        for pos, count in req_pos.items():
            row = (pool["Primary_Pos"] == pos).astype(float).to_numpy()
            A_rows.append(row)
            lb_list.append(float(count))
            ub_list.append(float(count))

        # Kisit 3: Ayni Kulupten Maksimum Oyuncu Kisiti
        for squad_name in pool["Squad"].unique():
            row = (pool["Squad"] == squad_name).astype(float).to_numpy()
            A_rows.append(row)
            lb_list.append(0.0)
            ub_list.append(float(max_players_per_club))

        # Kisit 4: Kadro Maksimum Yas Ortalamasi Kisiti
        A_rows.append(pool["Age"].to_numpy(dtype=float))
        lb_list.append(0.0)
        ub_list.append(11.0 * float(max_avg_age))

        constraints = LinearConstraint(
            np.array(A_rows), np.array(lb_list), np.array(ub_list)
        )

        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)

        if not res.success or res.x is None:
            return {"status": "Infeasible", "squad": pd.DataFrame()}

        selected_idx = np.where(res.x > 0.5)[0]
        selected_df = pool.iloc[selected_idx].copy()

        pos_order = {"GK": 1, "DF": 2, "MF": 3, "FW": 4}
        selected_df["Pos_Order"] = selected_df["Primary_Pos"].map(pos_order)
        selected_df = selected_df.sort_values(
            by=["Pos_Order", "OptiScout_Score"], ascending=[True, False]
        ).drop(columns=["Pos_Order"])

        return {
            "status": "Optimal",
            "squad": selected_df,
            "avg_score": round(selected_df["OptiScout_Score"].mean(), 1),
            "avg_age": round(selected_df["Age"].mean(), 1),
        }
