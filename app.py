import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_processor import ROLE_GROUP_LABELS, FootballDataProcessor, applicable_sub_scores
from similarity_engine import PlayerSimilarityEngine
from squad_optimizer import SquadOptimizer

st.set_page_config(
    page_title="OptiScout | Football Scouting DSS",
    page_icon="⚽",
    layout="wide",
)


@st.cache_data
def load_processed_data(min_minutes: int):
    processor = FootballDataProcessor()
    df = processor.clean_and_engineer(min_minutes=min_minutes)
    return df, processor.source_file, processor.has_market_values


# Streamlit her etkileşimde (kaydırıcı, filtre) bütün sayfayı baştan çalıştırır.
# Aşağıdaki fonksiyonlar sonuçlarını girdilerine göre önbelleğe alır; böylece örneğin
# 1. sekmedeki yaş filtresi değişince MILP ve benzerlik hesabı tekrar yapılmaz.
# Listeler önbellek anahtarı olabilmesi için tuple olarak geçirilir.
@st.cache_data(max_entries=64, show_spinner=False)
def cached_similar_players(min_minutes: int, target_key: str, top_n: int, max_age: int, leagues: tuple):
    df, _, _ = load_processed_data(min_minutes)
    return PlayerSimilarityEngine(df).find_similar_players(
        target_key=target_key, top_n=top_n, max_age=max_age, leagues=list(leagues)
    )


def _league_pool(min_minutes: int, leagues: tuple):
    df, _, _ = load_processed_data(min_minutes)
    return df[df["League"].isin(leagues)]


@st.cache_data(max_entries=64, show_spinner="Optimum 11 hesaplanıyor...")
def cached_optimize(min_minutes: int, leagues: tuple, excluded_squads: tuple, **kwargs):
    return SquadOptimizer(_league_pool(min_minutes, leagues)).optimize_starting_xi(
        selected_leagues=list(leagues), excluded_squads=list(excluded_squads), **kwargs
    )


@st.cache_data(max_entries=16, show_spinner="Bütçe eğrisi hesaplanıyor...")
def cached_frontier(min_minutes: int, leagues: tuple, excluded_squads: tuple, budgets: tuple, **kwargs):
    return SquadOptimizer(_league_pool(min_minutes, leagues)).budget_frontier(
        list(budgets), selected_leagues=list(leagues), excluded_squads=list(excluded_squads), **kwargs
    )


def fmt_value(v) -> str:
    """Piyasa değerini okunabilir biçimde yazar: 45.0 -> '€45M'."""
    if v is None or v != v:  # None veya NaN
        return "?"
    return f"€{v:,.0f}M" if v >= 1 else f"€{v * 1000:,.0f}K"


def draw_tactical_pitch(xi_df, has_values: bool = False) -> go.Figure:
    fig = go.Figure()

    pitch_shapes = [
        dict(type="rect", x0=0, y0=0, x1=100, y1=100, line=dict(color="white", width=2), fillcolor="#1b4d3e", layer="below"),
        dict(type="line", x0=50, y0=0, x1=50, y1=100, line=dict(color="white", width=2), layer="below"),
        dict(type="circle", x0=40, y0=38, x1=60, y1=62, line=dict(color="white", width=2), layer="below"),
        dict(type="rect", x0=0, y0=22, x1=16, y1=78, line=dict(color="white", width=2), layer="below"),
        dict(type="rect", x0=84, y0=22, x1=100, y1=78, line=dict(color="white", width=2), layer="below"),
    ]
    fig.update_layout(shapes=pitch_shapes)

    # Takım soldan sağa hücum eder: x = derinlik, y = genişlik (y yüksek = sol kanat)
    role_xy = {
        "GK": (6, [50]),
        "CB": (22, None), "LB": (27, [88]), "RB": (27, [12]),
        "LWB": (40, [86]), "RWB": (40, [14]),
        "DM": (40, None), "CM": (53, None), "AM": (66, None),
        "LM": (58, [86]), "RM": (58, [14]),
        "LW": (77, [84]), "RW": (77, [16]),
        "ST": (86, None),
    }
    central_spread = {1: [50], 2: [36, 64], 3: [26, 50, 74]}
    group_colors = {"GK": "#f59e0b", "DF": "#3b82f6", "MF": "#10b981", "FW": "#ef4444"}

    for group in ["GK", "DF", "MF", "FW"]:
        xs, ys, labels = [], [], []
        group_df = xi_df[xi_df["Assigned_Pos"] == group]
        for role, role_df in group_df.groupby("Assigned_Role", sort=False):
            x, fixed_y = role_xy[role]
            y_list = fixed_y or central_spread.get(len(role_df), [50] * len(role_df))
            for (_, row), y in zip(role_df.iterrows(), y_list):
                xs.append(x)
                ys.append(y)
                labels.append(
                    f"<b>{row['Player']}</b>{'*' if row['Out_Of_Position'] else ''}"
                    f"<br>{role} · {row['Squad']} ({row['OptiScout_Score']})"
                    f"{' · ' + fmt_value(row['Market_Value_M']) if has_values else ''}"
                )
        if not xs:
            continue
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                marker=dict(size=26, color=group_colors[group], line=dict(width=2, color="white")),
                text=labels,
                # Kaleci etiketi altta: 3'lü savunmada ortadaki stoperin etiketiyle çakışmasın
                textposition="bottom center" if group == "GK" else "top center",
                textfont=dict(color="white", size=11),
                name=group,
            )
        )

    fig.update_xaxes(range=[-5, 105], visible=False)
    fig.update_yaxes(range=[-5, 105], visible=False)
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
        title=dict(text="⚽ Optimum İlk 11 Saha Dizilişi (Taktik Tahtası)", font=dict(color="white")),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="white")),
    )
    return fig


st.sidebar.title("⚽ OptiScout DSS")
st.sidebar.caption("2026-2027 Avrupa 5 Büyük Lig Karar Destek Sistemi")

min_mins = st.sidebar.slider(
    "Minimum Süre Filtresi (Dakika)", min_value=90, max_value=720, value=270, step=45
)
df, source_file, has_values = load_processed_data(min_mins)
st.sidebar.caption(f"Veri kaynağı: `{source_file}`")
if has_values:
    matched_share = df["Market_Value_EUR"].notna().mean() * 100
    st.sidebar.caption(f"Transfermarkt piyasa değeri eşleşmesi: %{matched_share:.0f}")

all_leagues = sorted(df["League"].unique().tolist())
selected_leagues = st.sidebar.multiselect(
    "Lig Filtresi", options=all_leagues, default=all_leagues
)

if not selected_leagues:
    st.warning("En az bir lig seçin.")
    st.stop()

filtered_df = df[df["League"].isin(selected_leagues)].copy()

st.sidebar.markdown("---")
st.sidebar.metric("Analiz Edilen Oyuncu Havuzu", f"{len(filtered_df)} Oyuncu")

st.title("⚽ OptiScout: Oyuncu İzleme (Scouting) & Kadro Optimizasyon KDS")
st.markdown(
    "FBref **2026-2027** verileri üzerinde **Yüzdelik Dilim (Percentile) Skorlaması**, "
    "**Kosinüs Benzerliği (Cosine Similarity)** ve **Doğrusal Programlama (MILP)** "
    "ile çalışan karar destek platformu."
)

tab1, tab2, tab3 = st.tabs([
    "📊 1. Oyuncu Sıralaması & Kadran Analizi",
    "🔍 2. İkame Oyuncu Bulucu (Kosinüs Benzerliği)",
    "🧠 3. Optimum İlk 11 Çözücü (MILP)"
])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        role_options = ["Tümü"] + [g for g in ROLE_GROUP_LABELS if g in set(df["Score_Group"])]
        pos_filter = st.selectbox(
            "Rol Seçin",
            role_options,
            format_func=lambda g: "Tümü" if g == "Tümü" else ROLE_GROUP_LABELS[g],
            help="Skorlar rol grubu içinde hesaplanır: bekler beklerle, kanatlar kanatlarla kıyaslanır.",
        )
    with col2:
        max_age_tab1 = st.slider("Maksimum Yaş", 16, 40, 25, key="age_t1")
    with col3:
        # Alt skorlar rol içinde hesaplandığı için yalnızca tek bir rol seçiliyken
        # anlamlıdır; "Tümü" görünümünde genel skor ve ham per-90 değerler sunulur.
        raw_metrics = ["Gls_p90", "Ast_p90", "GA_non_pk_p90", "Def_Actions_p90"]
        if pos_filter == "GK":
            raw_metrics = ["Saves_p90"]
        sort_options = (
            ["OptiScout_Score"]
            + applicable_sub_scores("ALL" if pos_filter == "Tümü" else pos_filter)
            + raw_metrics
        )
        sort_metric = st.selectbox(
            "Sıralama Ölçütü",
            sort_options,
            help="Hücum/Yaratıcılık/Savunma gibi alt skorlar rol içinde hesaplanır; "
                 "roller arası karşılaştırılamadığı için yalnızca bir rol seçiliyken listelenir.",
        )

    view_df = filtered_df[filtered_df["Age"] <= max_age_tab1].copy()
    if pos_filter != "Tümü":
        view_df = view_df[view_df["Score_Group"] == pos_filter]

    view_df = view_df.sort_values(by=sort_metric, ascending=False, na_position="last")

    role_title = "Tümü" if pos_filter == "Tümü" else ROLE_GROUP_LABELS[pos_filter]
    st.subheader(f"🏆 En Yüksek Performanslı Oyuncular ({role_title} | Yaş <= {max_age_tab1})")
    display_cols = [
        "Player", "Squad", "League", "Score_Group", "TM_Sub_Position", "Age", "Min",
        "Gls", "Ast", "Gls_p90", "Ast_p90", "Def_Actions_p90", "OptiScout_Score"
    ] + (["Market_Value_M"] if has_values else [])
    st.dataframe(view_df[display_cols].head(25), hide_index=True)

    st.markdown("---")
    st.subheader("🎯 Dört Kadran Performans Haritası (Outlier Tespiti)")
    c_x, c_y = st.columns(2)
    with c_x:
        x_axis_col = st.selectbox("X Ekseni", ["Sh/90", "Gls_p90", "Ast_p90", "Crs_p90", "TklW_p90"], index=0)
    with c_y:
        y_axis_col = st.selectbox("Y Ekseni", ["Gls_p90", "SoT/90", "GA_non_pk_p90", "Int_p90", "Def_Actions_p90"], index=0)

    fig_scatter = px.scatter(
        view_df.head(150),
        x=x_axis_col,
        y=y_axis_col,
        color="Score_Group",
        size="OptiScout_Score",
        hover_name="Player",
        hover_data=["Squad", "League", "Age", "Min", "OptiScout_Score"],
        text="Player",
        template="plotly_dark",
        height=550,
        title=f"{x_axis_col} vs {y_axis_col} (İlk 150 Oyuncu)"
    )
    fig_scatter.update_traces(textposition="top center", textfont_size=10)
    fig_scatter.add_vline(x=view_df[x_axis_col].median(), line_dash="dash", line_color="gray")
    fig_scatter.add_hline(y=view_df[y_axis_col].median(), line_dash="dash", line_color="gray")
    st.plotly_chart(fig_scatter)

with tab2:
    st.subheader("🔍 Yıldız Oyuncu İkamesi & Genç Yetenek (Wonderkid) Keşfi")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        # İsim benzersiz değil: aynı isimli farklı oyuncular ve iki kulüpte
        # görünen transfer oyuncuları var. Bu yüzden Row_Key ile seçiyoruz.
        key_to_label = dict(zip(df["Row_Key"], df["Label"]))
        key_list = sorted(key_to_label, key=lambda k: key_to_label[k])
        yamal_keys = [k for k in key_list if k.startswith("Lamine Yamal|")]
        default_idx = key_list.index(yamal_keys[0]) if yamal_keys else 0
        target_key = st.selectbox(
            "Referans Alınacak Yıldız Oyuncu",
            key_list,
            index=default_idx,
            format_func=lambda k: key_to_label[k],
        )
    with c2:
        sim_max_age = st.slider("Alternatif Oyuncu Maksimum Yaş", 17, 38, 23)
    with c3:
        top_n_sim = st.slider("Listelenecek Oyuncu Sayısı", 5, 20, 10)

    similar_df = cached_similar_players(
        min_mins, target_key, top_n_sim, sim_max_age, tuple(selected_leagues)
    )

    ref_row = df[df["Row_Key"] == target_key].iloc[0]
    st.info(
        f"**Referans Oyuncu:** {ref_row['Player']} ({ref_row['Squad']} - {ref_row['League']}) | "
        f"**Rol:** {ROLE_GROUP_LABELS.get(ref_row['Score_Group'], ref_row['Score_Group'])} | **Yaş:** {int(ref_row['Age'])} | "
        f"**OptiScout Skoru:** {ref_row['OptiScout_Score']}"
    )

    col_table, col_radar = st.columns([3, 2])
    with col_table:
        sim_cols = [
            "Player", "Squad", "League", "Age", "Min",
            "Similarity_Score", "OptiScout_Score", "Gls_p90", "Ast_p90", "Def_Actions_p90"
        ] + (["Market_Value_M"] if has_values else [])
        st.dataframe(similar_df[sim_cols], hide_index=True)
        st.caption(
            "Benzerlik Skoru = kosinüs benzerliği × 100. 100: aynı profil, "
            "0: ilişkisiz, negatif: zıt profil."
        )

    with col_radar:
        if not similar_df.empty:
            compare_key = st.selectbox(
                "Radar Grafiğinde Karşılaştırılacak Aday:",
                similar_df["Row_Key"].tolist(),
                format_func=lambda k: key_to_label[k],
            )
            comp_row = similar_df[similar_df["Row_Key"] == compare_key].iloc[0]

            if ref_row["Score_Group"] == "GK":
                # Kaleci: alt metrikler rol içi metrik puanları (yüzdelik + z-skor karışımı)
                radar_cats = ["SavePct_score", "CSPct_score", "GA90_score", "Saves_score", "Discipline_Score", "OptiScout_Score"]
                radar_labels = ["Kurtarış %", "Gol Yemeden %", "Az Gol Yeme", "Kurtarış Sayısı", "Disiplin", "Genel Skor"]
                gk_pool = df[df["Score_Group"] == "GK"].set_index("Row_Key")
                ref_row, comp_row = ref_row.copy(), comp_row.copy()
                for src, dst, asc in [("SavePct_adj", "SavePct_score", True), ("CSPct_adj", "CSPct_score", True),
                                      ("GA90_adj", "GA90_score", False), ("Saves_adj", "Saves_score", True)]:
                    pct_rank = (gk_pool[src].rank(pct=True, ascending=asc) * 100).round(1)
                    ref_row[dst] = pct_rank.get(ref_row["Row_Key"], 50.0)
                    comp_row[dst] = pct_rank.get(comp_row["Row_Key"], 50.0)
            else:
                radar_cats = ["Attacking_Score", "Creativity_Score", "Defensive_Score", "Discipline_Score", "OptiScout_Score"]
                radar_labels = ["Hücum", "Yaratıcılık", "Savunma", "Disiplin", "Genel Skor"]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[ref_row[c] for c in radar_cats],
                theta=radar_labels,
                fill="toself",
                name=ref_row["Player"]
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[comp_row[c] for c in radar_cats],
                theta=radar_labels,
                fill="toself",
                name=f"{comp_row['Player']} (Benzerlik: {comp_row['Similarity_Score']})"
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                template="plotly_dark",
                title="Yüzdelik Dilim (0-100) Profil Karşılaştırması"
            )
            st.plotly_chart(fig_radar)

with tab3:
    st.subheader("🧠 Doğrusal Programlama (MILP) ile Kısıtlı Optimum İlk 11")

    o1, o2, o3, o4, o5 = st.columns(5)
    with o1:
        formation = st.selectbox("Taktik Diziliş", ["4-3-3", "4-2-3-1", "3-5-2", "3-4-3", "4-4-2"])
    with o2:
        max_club = st.slider("Kulüp Başına Maksimum Oyuncu", 1, 5, 2)
    with o3:
        max_avg_age = st.slider("Kadro Maksimum Yaş Ortalaması", 20.0, 32.0, 26.5, 0.5)
    with o4:
        exclude_giants = st.checkbox("Dev Kulüpleri Dışla (Fırsat 11'i)", value=False)
    with o5:
        allow_oop = st.checkbox(
            "Mevki dışı oyunculara izin ver",
            value=True,
            help="Açıkken bir oyuncu doğal rolüne yakın bir rolde de oynatılabilir "
                 "(ör. sol bek sol kanat bekinde, 10 numara kanatta). Skoru rol uygunluk "
                 "katsayısıyla (0.80–0.95) çarpılır.",
        )
    st.caption(
        "Roller Transfermarkt alt mevkilerinden gelir (stoper, sol bek, ön libero, sağ kanat...). "
        "Alt mevkisi bilinmeyen oyuncular FBref ana mevkisine göre, skoru %10 indirimli olarak değerlendirilir."
    )

    budget_eur = None
    if has_values:
        b1, b2 = st.columns([1, 3])
        with b1:
            use_budget = st.checkbox("💶 Bütçe kısıtı", value=True)
        with b2:
            budget_m = st.slider(
                "Toplam Transfer Bütçesi (milyon €, Transfermarkt piyasa değeri)",
                min_value=10, max_value=1500, value=250, step=10,
                disabled=not use_budget,
            )
        if use_budget:
            budget_eur = budget_m * 1e6
    else:
        st.info(
            "Bütçe kısıtı için `data/market_values.csv` gerekli. "
            "Oluşturmak için: `python scripts/build_market_values.py`"
        )

    giant_clubs = [
        "Barcelona", "Real Madrid", "Atlético Madrid", "Manchester City",
        "Arsenal", "Liverpool", "Chelsea", "Paris SG", "Bayern Munich", "Inter", "Juventus"
    ] if exclude_giants else []

    result = cached_optimize(
        min_mins,
        tuple(selected_leagues),
        tuple(giant_clubs),
        formation=formation,
        max_players_per_club=max_club,
        max_avg_age=max_avg_age,
        allow_out_of_position=allow_oop,
        budget_eur=budget_eur,
    )

    if result["status"] == "Optimal":
        xi_df = result["squad"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Çözüm Durumu (HiGHS MILP Solver)", "OPTİMAL (MILP)")
        m2.metric(
            "İlk 11 Rol Uyumlu Ortalama Skor",
            f"{result['avg_effective_score']} / 100",
            help="OptiScout skoru × rol uygunluk katsayısı. Modelin maksimize ettiği değer. "
                 f"Ham ortalama: {result['avg_score']}",
        )
        m3.metric("İlk 11 Yaş Ortalaması", f"{result['avg_age']} Yaş")
        if result.get("total_value_eur") is not None:
            spent_m = result["total_value_eur"] / 1e6
            m4.metric(
                "İlk 11 Toplam Piyasa Değeri",
                fmt_value(spent_m),
                delta=f"{fmt_value(budget_eur / 1e6 - spent_m)} bütçe kaldı" if budget_eur else None,
                delta_color="off",
            )
        if budget_eur and result.get("excluded_no_value"):
            st.caption(
                f"Piyasa değeri bulunamayan {result['excluded_no_value']} oyuncu, "
                "maliyeti bilinmediği için bütçeli çözümde havuz dışı bırakıldı."
            )

        col_pitch, col_xi_table = st.columns([3, 2])
        with col_pitch:
            st.plotly_chart(draw_tactical_pitch(xi_df, has_values))
            if xi_df["Out_Of_Position"].any():
                st.caption("* Doğal rolü dışında oynatılan oyuncu (skoru uygunluk katsayısıyla indirimli)")
        with col_xi_table:
            st.dataframe(
                xi_df[[
                    "Assigned_Role", "Player", "TM_Sub_Position", "Squad", "League", "Age",
                    "Min", "Gls", "Ast", "OptiScout_Score"
                ] + (["Market_Value_M"] if has_values else [])],
                hide_index=True,
                height=480,
            )
    else:
        st.error(
            "Seçilen kısıtlarla uygun bir 11 bulunamadı (Infeasible). "
            "Bütçe, yaş veya kulüp sınırını esnetin."
        )

    if has_values:
        st.markdown("---")
        st.subheader("📈 Bütçe - Performans Eğrisi")
        st.caption(
            "Aynı kısıtlarla model farklı bütçeler için yeniden çözülür. Eğrinin düzleştiği "
            "nokta, ek harcamanın kadroyu artık anlamlı iyileştirmediği bütçedir."
        )
        if st.button("Eğriyi hesapla"):
            budgets = [b * 1e6 for b in (25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000, 1500)]
            frontier = cached_frontier(
                min_mins,
                tuple(selected_leagues),
                tuple(giant_clubs),
                tuple(budgets),
                formation=formation,
                max_players_per_club=max_club,
                max_avg_age=max_avg_age,
                allow_out_of_position=allow_oop,
            )
            ok = frontier[frontier["Status"] == "Optimal"]
            if ok.empty:
                st.warning("Bu kısıtlarla hiçbir bütçe seviyesinde çözüm bulunamadı.")
            else:
                fig_frontier = px.line(
                    ok, x="Budget_M", y="Avg_Score", markers=True,
                    hover_data={"Spent_M": ":.0f"},
                    labels={"Budget_M": "Bütçe (milyon €)", "Avg_Score": "Rol Uyumlu Ortalama Skor",
                            "Spent_M": "Harcanan (milyon €)"},
                    template="plotly_dark", height=420,
                )
                st.plotly_chart(fig_frontier)
