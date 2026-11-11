import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_processor import FootballDataProcessor
from similarity_engine import PlayerSimilarityEngine
from squad_optimizer import SquadOptimizer

st.set_page_config(
    page_title="OptiScout | AI Football DSS",
    page_icon="⚽",
    layout="wide",
)


@st.cache_data
def load_processed_data(min_minutes: int):
    processor = FootballDataProcessor()
    return processor.clean_and_engineer(min_minutes=min_minutes)


def draw_tactical_pitch(xi_df) -> go.Figure:
    fig = go.Figure()

    pitch_shapes = [
        dict(type="rect", x0=0, y0=0, x1=100, y1=100, line=dict(color="white", width=2), fillcolor="#1b4d3e", layer="below"),
        dict(type="line", x0=50, y0=0, x1=50, y1=100, line=dict(color="white", width=2), layer="below"),
        dict(type="circle", x0=40, y0=38, x1=60, y1=62, line=dict(color="white", width=2), layer="below"),
        dict(type="rect", x0=0, y0=22, x1=16, y1=78, line=dict(color="white", width=2), layer="below"),
        dict(type="rect", x0=84, y0=22, x1=100, y1=78, line=dict(color="white", width=2), layer="below"),
    ]
    fig.update_layout(shapes=pitch_shapes)

    pos_x = {"GK": 8, "DF": 28, "MF": 56, "FW": 82}
    pos_colors = {"GK": "#f59e0b", "DF": "#3b82f6", "MF": "#10b981", "FW": "#ef4444"}

    for pos in ["GK", "DF", "MF", "FW"]:
        sub = xi_df[xi_df["Primary_Pos"] == pos].reset_index(drop=True)
        count = len(sub)
        if count == 0:
            continue

        y_coords = [50] if count == 1 else [15 + i * (70 / (count - 1)) for i in range(count)]
        x_coords = [pos_x[pos]] * count

        labels = [
            f"<b>{row['Player']}</b><br>{row['Squad']} ({row['OptiScout_Score']})"
            for _, row in sub.iterrows()
        ]

        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers+text",
                marker=dict(size=26, color=pos_colors[pos], line=dict(width=2, color="white")),
                text=labels,
                textposition="top center",
                textfont=dict(color="white", size=11),
                name=pos,
            )
        )

    fig.update_xaxes(range=[-5, 105], visible=False)
    fig.update_yaxes(range=[-5, 105], visible=False)
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
        title="⚽ Optimum İlk 11 Saha Dizilişi (Taktik Tahtası)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="white")),
    )
    return fig


st.sidebar.title("⚽ OptiScout DSS")
st.sidebar.caption("2026-2027 Avrupa 5 Büyük Lig Karar Destek Sistemi")

min_mins = st.sidebar.slider(
    "Minimum Süre Filtresi (Dakika)", min_value=90, max_value=720, value=270, step=45
)
df = load_processed_data(min_mins)

all_leagues = sorted(df["League"].unique().tolist())
selected_leagues = st.sidebar.multiselect(
    "Lig Filtresi", options=all_leagues, default=all_leagues
)

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
    "🔍 2. İkame Oyuncu Bulucu (ML Cosine Similarity)",
    "🧠 3. Optimum İlk 11 Çözücü (MILP)"
])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        pos_filter = st.selectbox("Mevki Seçin", ["Tümü", "FW", "MF", "DF", "GK"])
    with col2:
        max_age_tab1 = st.slider("Maksimum Yaş", 16, 40, 25, key="age_t1")
    with col3:
        sort_metric = st.selectbox(
            "Sıralama Ölçütü",
            ["OptiScout_Score", "Attacking_Score", "Creativity_Score", "Defensive_Score", "Gls_p90", "Ast_p90"]
        )

    view_df = filtered_df[filtered_df["Age"] <= max_age_tab1].copy()
    if pos_filter != "Tümü":
        view_df = view_df[view_df["Primary_Pos"] == pos_filter]

    view_df = view_df.sort_values(by=sort_metric, ascending=False)

    st.subheader(f"🏆 En Yüksek Performanslı Oyuncular ({pos_filter} | Yaş <= {max_age_tab1})")
    display_cols = [
        "Player", "Squad", "League", "Primary_Pos", "Age", "Min",
        "Gls", "Ast", "Gls_p90", "Ast_p90", "Def_Actions_p90", "OptiScout_Score"
    ]
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
        color="Primary_Pos",
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
    sim_engine = PlayerSimilarityEngine(df)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        player_list = sorted(df["Player"].unique().tolist())
        default_idx = player_list.index("Lamine Yamal") if "Lamine Yamal" in player_list else 0
        target_player = st.selectbox("Referans Alınacak Yıldız Oyuncu", player_list, index=default_idx)
    with c2:
        sim_max_age = st.slider("Alternatif Oyuncu Maksimum Yaş", 17, 38, 23)
    with c3:
        top_n_sim = st.slider("Listelenecek Oyuncu Sayısı", 5, 20, 10)

    similar_df = sim_engine.find_similar_players(
        target_player=target_player,
        top_n=top_n_sim,
        max_age=sim_max_age,
        leagues=selected_leagues,
    )

    ref_row = df[df["Player"] == target_player].iloc[0]
    st.info(
        f"**Referans Oyuncu:** {ref_row['Player']} ({ref_row['Squad']} - {ref_row['League']}) | "
        f"**Mevki:** {ref_row['Primary_Pos']} | **Yaş:** {int(ref_row['Age'])} | "
        f"**OptiScout Skoru:** {ref_row['OptiScout_Score']}"
    )

    col_table, col_radar = st.columns([3, 2])
    with col_table:
        sim_cols = [
            "Player", "Squad", "League", "Age", "Min",
            "Similarity_%", "OptiScout_Score", "Gls_p90", "Ast_p90", "Def_Actions_p90"
        ]
        st.dataframe(similar_df[sim_cols], hide_index=True)

    with col_radar:
        if not similar_df.empty:
            compare_player = st.selectbox(
                "Radar Grafiğinde Karşılaştırılacak Aday:",
                similar_df["Player"].tolist()
            )
            comp_row = similar_df[similar_df["Player"] == compare_player].iloc[0]

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
                name=f"{comp_row['Player']} (%{comp_row['Similarity_%']} Benzer)"
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                template="plotly_dark",
                title="Yüzdelik Dilim (0-100) Profil Karşılaştırması"
            )
            st.plotly_chart(fig_radar)

with tab3:
    st.subheader("🧠 Doğrusal Programlama (MILP) ile Kısıtlı Optimum İlk 11")
    optimizer = SquadOptimizer(filtered_df)

    o1, o2, o3, o4 = st.columns(4)
    with o1:
        formation = st.selectbox("Taktik Diziliş", ["4-3-3", "4-2-3-1", "3-5-2", "3-4-3", "4-4-2"])
    with o2:
        max_club = st.slider("Kulüp Başına Maksimum Oyuncu", 1, 5, 2)
    with o3:
        max_avg_age = st.slider("Kadro Maksimum Yaş Ortalaması", 20.0, 32.0, 26.5, 0.5)
    with o4:
        exclude_giants = st.checkbox("Dev Kulüpleri Dışla (Fırsat 11'i)", value=False)

    giant_clubs = [
        "Barcelona", "Real Madrid", "Atlético Madrid", "Manchester City",
        "Arsenal", "Liverpool", "Chelsea", "Paris SG", "Bayern Munich", "Inter", "Juventus"
    ] if exclude_giants else []

    result = optimizer.optimize_starting_xi(
        formation=formation,
        max_players_per_club=max_club,
        max_avg_age=max_avg_age,
        selected_leagues=selected_leagues,
        excluded_Squads=giant_clubs,
    )

    if result["status"] == "Optimal":
        xi_df = result["squad"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Çözüm Durumu (HiGHS MILP Solver)", "OPTİMAL (MILP)")
        m2.metric("İlk 11 Ortalama OptiScout Puanı", f"{result['avg_score']} / 100")
        m3.metric("İlk 11 Yaş Ortalaması", f"{result['avg_age']} Yaş")

        col_pitch, col_xi_table = st.columns([3, 2])
        with col_pitch:
            st.plotly_chart(draw_tactical_pitch(xi_df))
        with col_xi_table:
            st.dataframe(
                xi_df[[
                    "Primary_Pos", "Player", "Squad", "League", "Age",
                    "Min", "Gls", "Ast", "OptiScout_Score"
                ]],
                hide_index=True,
                height=480,
            )
    else:
        st.error("Seçilen kısıtlarla uygun bir 11 bulunamadı (Infeasible). Yaş veya kulüp sınırını esnetin.")
