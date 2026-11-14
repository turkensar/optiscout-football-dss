import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_processor import ROLE_GROUP_LABELS, FootballDataProcessor, applicable_sub_scores
from similarity_engine import PlayerSimilarityEngine
from squad_optimizer import SquadOptimizer
from value_model import MarketValueModel

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


@st.cache_data(max_entries=32, show_spinner="Transfer planı hesaplanıyor...")
def cached_transfer_plan(min_minutes: int, leagues: tuple, club: str, **kwargs):
    df, _, _ = load_processed_data(min_minutes)
    return SquadOptimizer(df).plan_transfers(club, target_leagues=list(leagues), **kwargs)


# Tablolarda kod adları yerine okunur Türkçe başlıklar gösterilir
COLUMN_LABELS = {
    "Player": "Oyuncu", "Squad": "Kulüp", "League": "Lig", "Age": "Yaş", "Min": "Dakika",
    "Score_Group": "Rol", "TM_Sub_Position": "Alt Mevki (TM)", "Assigned_Role": "Görev",
    "Gls": "Gol", "Ast": "Asist", "Gls_p90": "Gol/90", "Ast_p90": "Asist/90",
    "Def_Actions_p90": "Savunma Aks./90", "GA_non_pk_p90": "PS'siz G+A/90",
    "OptiScout_Score": "OptiScout", "Effective_Score": "Rol Skoru",
    "Similarity_Score": "Benzerlik", "Market_Value_M": "Piyasa Değeri",
    "Role_Fit": "Rol Uyumu", "Improvement": "Kazanç",
    "Pred_Value_M": "Tahmini Değer", "Value_Ratio": "Fırsat Katsayısı",
    "Contract_Label": "Sözleşme", "Contract_Expiry": "Sözleşme Bitişi", "Est_Fee_M": "Tahmini Bedel",
    "Attacking_Score": "Hücum", "Creativity_Score": "Yaratıcılık", "Defensive_Score": "Savunma",
    "Goalkeeping_Score": "Kalecilik", "Discipline_Score": "Disiplin", "Saves_p90": "Kurtarış/90",
}
SUB_POSITION_TR = {
    "Goalkeeper": "Kaleci", "Centre-Back": "Stoper", "Left-Back": "Sol Bek", "Right-Back": "Sağ Bek",
    "Defensive Midfield": "Ön Libero", "Central Midfield": "Merkez Orta Saha",
    "Attacking Midfield": "10 Numara", "Left Midfield": "Sol Orta Saha", "Right Midfield": "Sağ Orta Saha",
    "Left Winger": "Sol Kanat", "Right Winger": "Sağ Kanat", "Second Striker": "İkinci Forvet",
    "Centre-Forward": "Santrfor",
}


AXIS_LABELS = {
    "Sh/90": "Şut/90", "SoT/90": "İsabetli Şut/90", "Gls_p90": "Gol/90", "Ast_p90": "Asist/90",
    "Crs_p90": "Orta/90", "TklW_p90": "Top Kapma/90", "Int_p90": "Araya Girme/90",
    "GA_non_pk_p90": "Penaltısız Gol+Asist/90", "Def_Actions_p90": "Savunma Aksiyonu/90",
}


def show_table(frame, **kwargs):
    """st.dataframe sarmalayıcısı: Türkçe başlıklar, rol adları ve sayı biçimleri."""
    t = frame.copy()
    if "Score_Group" in t.columns:
        t["Score_Group"] = t["Score_Group"].map(lambda g: ROLE_GROUP_LABELS.get(g, g))
    if "TM_Sub_Position" in t.columns:
        t["TM_Sub_Position"] = t["TM_Sub_Position"].map(lambda v: SUB_POSITION_TR.get(v, v) if isinstance(v, str) else "—")
    if "Age" in t.columns:
        t["Age"] = t["Age"].round(0).astype("Int64")
    t = t.rename(columns=COLUMN_LABELS)
    config = {
        "Piyasa Değeri": st.column_config.NumberColumn(format="€%.1fM"),
        "Tahmini Değer": st.column_config.NumberColumn(format="€%.1fM"),
        "Tahmini Bedel": st.column_config.NumberColumn(format="€%.1fM"),
        "Fırsat Katsayısı": st.column_config.NumberColumn(format="%.1fx"),
        "Rol Uyumu": st.column_config.NumberColumn(format="%.2f"),
    }
    for col in ["OptiScout", "Rol Skoru", "Benzerlik", "Kazanç", "Hücum", "Yaratıcılık",
                "Savunma", "Kalecilik", "Disiplin"]:
        config[col] = st.column_config.NumberColumn(format="%.1f")
    for col in ["Gol/90", "Asist/90", "Savunma Aks./90", "PS'siz G+A/90", "Kurtarış/90"]:
        config[col] = st.column_config.NumberColumn(format="%.2f")
    kwargs.setdefault("hide_index", True)
    kwargs.setdefault("width", "stretch")
    return st.dataframe(t, column_config=config, **kwargs)


@st.cache_data(max_entries=8, show_spinner="Değer modeli eğitiliyor...")
def cached_value_model(min_minutes: int):
    df, _, _ = load_processed_data(min_minutes)
    return MarketValueModel(df).fit()


def fmt_value(v) -> str:
    """Piyasa değerini okunabilir biçimde yazar: 45.0 -> '€45M'."""
    if v is None or v != v:  # None veya NaN
        return "?"
    if abs(v) < 0.0005:
        return "€0"
    return f"€{v:,.0f}M" if v >= 1 else f"€{v * 1000:,.0f}K"


def draw_tactical_pitch(xi_df, has_values: bool = False,
                        title: str = "⚽ Optimum İlk 11 Saha Dizilişi (Taktik Tahtası)",
                        home_club: str = None) -> go.Figure:
    """home_club verilirse o kulübün oyuncularında kulüp adı yazılmaz (dar sahada etiketler kısalır)."""
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
        "GK": (5, [50]),
        "CB": (22, None), "LB": (27, [88]), "RB": (27, [12]),
        "LWB": (40, [86]), "RWB": (40, [14]),
        "DM": (40, None), "CM": (53, None), "AM": (66, None),
        "LM": (58, [86]), "RM": (58, [14]),
        "LW": (77, [84]), "RW": (77, [16]),
        "ST": (86, None),
    }
    central_spread = {1: [50], 2: [33, 67], 3: [26, 50, 74]}
    group_colors = {"GK": "#f59e0b", "DF": "#3b82f6", "MF": "#10b981", "FW": "#ef4444"}

    three_cb = int((xi_df["Assigned_Role"] == "CB").sum()) >= 3
    for group in ["GK", "DF", "MF", "FW"]:
        xs, ys, labels, ring_colors, ring_widths = [], [], [], [], []
        group_df = xi_df[xi_df["Assigned_Pos"] == group]
        for role, role_df in group_df.groupby("Assigned_Role", sort=False):
            x, fixed_y = role_xy[role]
            y_list = fixed_y or central_spread.get(len(role_df), [50] * len(role_df))
            for (_, row), y in zip(role_df.iterrows(), y_list):
                is_new = bool(row.get("New_Signing", False))
                xs.append(x)
                ys.append(y)
                ring_colors.append("#facc15" if is_new else "white")
                ring_widths.append(5 if is_new else 2)
                labels.append(
                    f"<b>{'★ ' if is_new else ''}{row['Player']}</b>{'*' if row['Out_Of_Position'] else ''}"
                    f"<br>{role}{'' if row['Squad'] == home_club else ' · ' + row['Squad']} ("
                    f"{row['OptiScout_Score']}"
                    f"{'→' + str(row['Effective_Score']) if row['Out_Of_Position'] and 'Effective_Score' in row else ''})"
                    f"{' · ' + fmt_value(row['Market_Value_M']) if has_values else ''}"
                )
        if not xs:
            continue
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                marker=dict(size=26, color=group_colors[group], line=dict(width=ring_widths, color=ring_colors)),
                text=labels,
                # Kaleci etiketi: 3'lü savunmada ortadaki stoperin etiketiyle çakışmasın diye
                # altta, 4'lü savunmada stoper etiketleriyle çakışmasın diye üstte
                textposition=("bottom center" if three_cb else "top center") if group == "GK" else "top center",
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
        title=dict(text=title, font=dict(color="white")),
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
    use_contracts = st.sidebar.checkbox(
        "📝 Sözleşme durumunu maliyete yansıt",
        value=True,
        help="Açıkken transfer maliyeti olarak piyasa değeri yerine tahmini bedel kullanılır: "
             "2027 yazında sözleşmesi bitecek oyuncu piyasa değerinin %15'i (imza ücreti vb.), "
             "son yılına giren oyuncu %60'ı, diğerleri tam değeri.",
    )
else:
    use_contracts = False
cost_col = "Est_Fee_EUR" if use_contracts else "Market_Value_EUR"
cost_label = "Tahmini Transfer Bedeli" if use_contracts else "Piyasa Değeri"
cost_m_col = "Est_Fee_M" if use_contracts else "Market_Value_M"

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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 1. Oyuncu Sıralaması & Kadran Analizi",
    "🔍 2. İkame Oyuncu Bulucu (Kosinüs Benzerliği)",
    "🧠 3. Optimum İlk 11 Çözücü (MILP)",
    "🏟️ 4. Kulüp Boşluk Analizi & Transfer Planı",
    "💎 5. Fırsat Oyuncular (Değer Tahmini)",
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
            format_func=lambda c: COLUMN_LABELS.get(c, c),
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
    show_table(view_df[display_cols].head(25), hide_index=True)

    st.markdown("---")
    st.subheader("🎯 Dört Kadran Performans Haritası (Outlier Tespiti)")
    c_x, c_y = st.columns(2)
    with c_x:
        x_axis_col = st.selectbox(
            "X Ekseni", ["Sh/90", "Gls_p90", "Ast_p90", "Crs_p90", "TklW_p90"], index=0,
            format_func=lambda c: AXIS_LABELS.get(c, c),
        )
    with c_y:
        y_axis_col = st.selectbox(
            "Y Ekseni", ["Gls_p90", "SoT/90", "GA_non_pk_p90", "Int_p90", "Def_Actions_p90"], index=0,
            format_func=lambda c: AXIS_LABELS.get(c, c),
        )

    # Okunabilirlik: yalnızca en yüksek skorlu oyuncuların adı yazılır, diğerleri fareyle görülür
    plot_df = view_df.head(150).copy()
    plot_df["Rol"] = plot_df["Score_Group"].map(lambda g: ROLE_GROUP_LABELS.get(g, g))
    top_keys = set(plot_df.nlargest(15, "OptiScout_Score")["Row_Key"])
    plot_df["Etiket"] = plot_df.apply(lambda r: r["Player"] if r["Row_Key"] in top_keys else "", axis=1)
    x_label, y_label = AXIS_LABELS.get(x_axis_col, x_axis_col), AXIS_LABELS.get(y_axis_col, y_axis_col)

    fig_scatter = px.scatter(
        plot_df,
        x=x_axis_col,
        y=y_axis_col,
        color="Rol",
        size="OptiScout_Score",
        hover_name="Player",
        hover_data={"Squad": True, "League": True, "Age": True, "Min": True,
                    "OptiScout_Score": True, "Rol": False, "Etiket": False},
        text="Etiket",
        labels={x_axis_col: x_label, y_axis_col: y_label, **COLUMN_LABELS},
        template="plotly_dark",
        height=550,
        title=f"{x_label} vs {y_label} · ilk {len(plot_df)} oyuncu (en iyi 15'in adı gösteriliyor)",
    )
    fig_scatter.update_traces(textposition="top center", textfont_size=11)
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
            "Player", "Squad", "Age", "Similarity_Score", "OptiScout_Score", "Gls_p90", "Ast_p90",
        ] + (["Market_Value_M"] if has_values else [])
        show_table(similar_df[sim_cols], hide_index=True)
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
                title="Profil Karşılaştırması (rol içi 0-100)",
                legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
                margin=dict(l=60, r=60, t=60, b=40),
            )
            st.plotly_chart(fig_radar)

with tab3:
    st.subheader("🧠 Doğrusal Programlama (MILP) ile Kısıtlı Optimum İlk 11")

    o1, o2, o3 = st.columns(3)
    with o1:
        formation = st.selectbox("Taktik Diziliş", ["4-3-3", "4-2-3-1", "3-5-2", "3-4-3", "4-4-2"])
    with o2:
        max_club = st.slider("Kulüp Başına Maksimum Oyuncu", 1, 5, 2)
    with o3:
        max_avg_age = st.slider("Kadro Maksimum Yaş Ortalaması", 20.0, 32.0, 26.5, 0.5)
    o4, o5 = st.columns(2)
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
                f"Toplam Transfer Bütçesi (milyon €, {cost_label.lower()})",
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
        cost_column=cost_col,
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
        spent_eur = result.get("total_cost_eur") if budget_eur else result.get("total_value_eur")
        if spent_eur is not None:
            spent_m = spent_eur / 1e6
            m4.metric(
                f"İlk 11 Toplam {cost_label if budget_eur else 'Piyasa Değeri'}",
                fmt_value(spent_m),
                delta=f"{fmt_value(budget_eur / 1e6 - spent_m)} bütçe kaldı" if budget_eur else None,
                delta_color="off",
            )
        if budget_eur and result.get("excluded_no_value"):
            st.caption(
                f"Piyasa değeri bulunamayan {result['excluded_no_value']} oyuncu, "
                "maliyeti bilinmediği için bütçeli çözümde havuz dışı bırakıldı."
            )

        col_pitch, col_xi_table = st.columns([11, 9])
        with col_pitch:
            st.plotly_chart(draw_tactical_pitch(xi_df, has_values))
            if xi_df["Out_Of_Position"].any():
                st.caption("* Doğal rolü dışında oynatılan oyuncu (skoru uygunluk katsayısıyla indirimli)")
        with col_xi_table:
            show_table(
                xi_df[["Assigned_Role", "Player", "Squad", "OptiScout_Score"]
                      + ([cost_m_col] + (["Contract_Label"] if use_contracts else []) if has_values else [])],
                hide_index=True,
                height=(len(xi_df) + 1) * 35 + 3,
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
            budgets = [b * 1e6 for b in (10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000, 1500)]
            common = dict(formation=formation, max_players_per_club=max_club,
                          max_avg_age=max_avg_age, allow_out_of_position=allow_oop)
            curves = [("Piyasa değeri ile", "Market_Value_EUR")]
            if use_contracts:
                curves.append(("Sözleşme durumu dahil", "Est_Fee_EUR"))
            parts = []
            for name, col in curves:
                fr = cached_frontier(min_mins, tuple(selected_leagues), tuple(giant_clubs), tuple(budgets),
                                     cost_column=col, **common)
                parts.append(fr.assign(Maliyet=name))
            frontier = pd.concat(parts, ignore_index=True)
            ok = frontier[frontier["Status"] == "Optimal"]
            if ok.empty:
                st.warning("Bu kısıtlarla hiçbir bütçe seviyesinde çözüm bulunamadı.")
            else:
                fig_frontier = px.line(
                    ok, x="Budget_M", y="Avg_Score", color="Maliyet", markers=True,
                    hover_data={"Spent_M": ":.0f", "Maliyet": False},
                    labels={"Budget_M": "Bütçe (milyon €)", "Avg_Score": "Rol Uyumlu Ortalama Skor",
                            "Spent_M": "Harcanan (milyon €)", "Maliyet": "Maliyet hesabı"},
                    color_discrete_map={"Piyasa değeri ile": "#64748b", "Sözleşme durumu dahil": "#facc15"},
                    template="plotly_dark", height=420, log_x=True,
                )
                fig_frontier.update_layout(legend=dict(orientation="h", y=1.08, x=0))
                st.plotly_chart(fig_frontier)
                if use_contracts:
                    st.caption(
                        "Sözleşmesi biten oyuncuları hesaba katmak en çok küçük bütçelerde fark yaratır: "
                        "aynı parayla belirgin şekilde daha iyi bir kadro kurulabilir."
                    )

with tab4:
    st.subheader("🏟️ Kulüp Boşluk Analizi & Transfer Planı")
    st.markdown(
        "Seçilen kulübün **mevcut oyuncularıyla kurabileceği en iyi 11** bulunur, en zayıf "
        "mevkiler (boşluklar) belirlenir ve bütçe içinde **hangi transferlerin takımı en çok "
        "iyileştireceği** MILP ile hesaplanır. Kulübün kendi oyuncuları maliyetsiz, dış oyuncular "
        "piyasa değerleriyle modele girer; kimin alınacağına ve kimin yedeğe düşeceğine model karar verir."
    )

    if not has_values:
        st.info(
            "Transfer planı için piyasa değerleri gerekli (`data/market_values.csv`). "
            "Oluşturmak için: `python scripts/build_market_values.py`"
        )
    else:
        clubs = sorted(df["Squad"].unique().tolist())
        p1, p2, p3 = st.columns([2, 1, 1])
        with p1:
            club = st.selectbox(
                "Kulüp", clubs,
                index=clubs.index("Tottenham") if "Tottenham" in clubs else 0,
            )
        with p2:
            plan_formation = st.selectbox(
                "Diziliş", list(SquadOptimizer.FORMATIONS), key="plan_formation"
            )
        with p3:
            max_signings = st.slider("Maksimum Transfer Sayısı", 1, 5, 2)
        q1, q2 = st.columns([3, 1])
        with q1:
            plan_budget_m = st.slider("Transfer Bütçesi (milyon €)", 0, 300, 50, 5)
        with q2:
            max_signing_age = st.slider("Transferlerde Maks. Yaş", 18, 36, 28)

        plan = cached_transfer_plan(
            min_mins,
            tuple(selected_leagues),
            club,
            formation=plan_formation,
            budget_eur=plan_budget_m * 1e6,
            cost_column=cost_col,
            max_signings=max_signings,
            max_signing_age=max_signing_age,
        )
        st.caption(
            f"{club} kadrosu, kenar çubuğundaki minimum süre filtresini (≥ {min_mins} dk) geçen "
            f"**{plan['own_players']} oyuncudan** oluşur. Sakat veya az süre almış oyuncuları da "
            "katmak için filtreyi düşürün. Transfer hedefleri seçili liglerden aranır; "
            "aynı kulüpten en fazla 1 oyuncu alınır."
        )

        base, after = plan["baseline"], plan["plan"]
        if base["status"] != "Optimal":
            st.error(
                f"{club} için bu dizilişte 11 kurulamadı (ör. süre filtresini geçen kaleci yok). "
                "Süre filtresini düşürmeyi deneyin."
            )
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Mevcut 11 (rol uyumlu skor)", f"{base['avg_effective_score']}")
            if after["status"] == "Optimal":
                k2.metric(
                    "Transfer Sonrası 11",
                    f"{after['avg_effective_score']}",
                    delta=f"{after['avg_effective_score'] - base['avg_effective_score']:+.1f}",
                )
                k3.metric("Harcanan", fmt_value(plan["spent_eur"] / 1e6),
                          delta=f"{fmt_value(plan_budget_m - plan['spent_eur'] / 1e6)} kaldı",
                          delta_color="off")
                k4.metric("Transfer", f"{len(plan['signings'])} / {max_signings}")
            else:
                k2.metric("Transfer Sonrası 11", "—")

            pc1, pc2 = st.columns(2)
            with pc1:
                st.plotly_chart(draw_tactical_pitch(base["squad"], True, f"Mevcut 11 · {club}", home_club=club))
            with pc2:
                if after["status"] == "Optimal":
                    st.plotly_chart(draw_tactical_pitch(after["squad"], True, "Transfer Sonrası 11 (★ = yeni)", home_club=club))

            if base["squad"]["Role_Fit"].le(0.5).any():
                st.caption(
                    "Mevcut 11'de skoru yarıya indirilmiş oyuncular, kadroda o rol için doğal "
                    "oyuncu olmadığı için zorunlu olarak yerleştirilmiştir (rol uygunluğu 0.5)."
                )

            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**🔻 En zayıf mevkiler (boşluklar)**")
                show_table(
                    plan["gaps"][["Assigned_Role", "Player", "TM_Sub_Position", "Role_Fit",
                                  "Effective_Score"]],
                    hide_index=True,
                )
            with g2:
                st.markdown("**✍️ Önerilen transferler**")
                if plan["signings"].empty:
                    st.write("Bu bütçeyle mevcut 11'i iyileştiren bir transfer bulunamadı.")
                else:
                    show_table(
                        plan["signings"][["Assigned_Role", "Player", "Squad"]
                                         + (["Contract_Label", "Est_Fee_M"] if use_contracts else ["Market_Value_M"])
                                         + ["Effective_Score"]],
                        hide_index=True,
                    )
                if not plan["dropped"].empty:
                    st.markdown("**↘️ İlk 11'den çıkanlar**")
                    show_table(
                        plan["dropped"][["Assigned_Role", "Player", "Effective_Score"]],
                        hide_index=True,
                    )

            st.markdown("**🔎 Boşluk başına alternatif adaylar** (bütçeye sığan, mevcut oyuncudan daha iyi)")
            for key, cand in plan["candidates"].items():
                role, row_key = key.split("|", 1)
                current = plan["gaps"][plan["gaps"]["Row_Key"] == row_key].iloc[0]
                with st.expander(
                    f"{role} · şu an: {current['Player']} ({current['Effective_Score']})"
                ):
                    if cand.empty:
                        st.write("Bütçeye sığan daha iyi bir aday yok.")
                    else:
                        show_table(
                            cand[["Player", "Squad", "League", "Age", "TM_Sub_Position", "Market_Value_M"]
                                 + (["Contract_Label", "Est_Fee_M"] if use_contracts else [])
                                 + ["Effective_Score", "Improvement"]],
                            hide_index=True,
                        )

            st.caption(
                "Varsayımlar: Oyuncu satışları ve maaşlar modele dahil değildir. Maliyet olarak "
                + ("sözleşme durumuna göre tahmini bedel (2027 yazında bedelsiz: değerin %15'i, son yılı: %60'ı) "
                   if use_contracts else "Transfermarkt piyasa değeri ")
                + "kullanılır; gerçek bonservis bedeli farklı olabilir."
            )

with tab5:
    st.subheader("💎 Fırsat Oyuncular: Performansına Göre Ucuz Kalanlar")
    st.markdown(
        "Bir regresyon modeli, oyuncunun **performansından, yaşından, ligi ve rolünden** "
        "piyasa değerinin ne olması gerektiğini tahmin eder. Rolünde üst dilimde oynayan ama "
        "gerçek piyasa değeri tahminin çok altında kalan oyuncular **fırsat** olarak listelenir. "
        "Her oyuncunun tahmini, o oyuncuyu eğitimde hiç görmemiş bir modelden gelir "
        "(5 katlı çapraz doğrulama); böylece model ezberlediği değeri geri vermez."
    )

    if not has_values:
        st.info("Değer modeli için piyasa değerleri gerekli (`data/market_values.csv`).")
    else:
        vm = cached_value_model(min_mins)
        mt = vm.metrics
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Açıklanan Varyans (R², çapraz doğrulama)", f"{mt['r2']:.2f}")
        v2.metric("Tipik Tahmin Hatası (medyan)", f"±%{mt['median_pct_error'] * 100:.0f}")
        v3.metric("Eğitim Örneği", f"{mt['n_train']} oyuncu")
        v4.metric("Kulüp Gücü Eklenince R²", f"{mt['r2_with_club']:.2f}",
                  help="Takım arkadaşlarının ortalama değeri eklendiğinde (oyuncunun kendisi hariç).")

        st.info(
            f"**Bulgu:** Yalnızca yaş, lig ve rol bilgisi değerin R² = {mt['r2_context_only']:.2f} kadarını "
            f"açıklıyor; bu sezonun performansı bunu sadece {mt['r2']:.2f}'ye çıkarıyor. Kulüp gücü eklenince "
            f"{mt['r2_with_club']:.2f}'ye fırlıyor. Yani piyasa, sahadaki güncel performanstan çok kulübü ve "
            "itibarı fiyatlıyor; fırsatlar tam bu boşlukta. Kulüp gücü fırsat modeline bilerek konmadı: "
            "konsaydı model \"küçük kulüpteki iyi oyuncu ucuzdur\" diye öğrenip aranan oyuncuları ıskalardı."
        )

        f1, f2, f3, f4, f5 = st.columns([3, 2, 2, 2, 2])
        with f1:
            role_codes = [g for g in ["GK", "CB", "FB", "DM", "CM", "AM", "W", "ST"]
                          if g in set(vm.predictions["Score_Group"])]
            opp_roles = st.multiselect("Rol", role_codes, format_func=lambda g: ROLE_GROUP_LABELS[g],
                                       placeholder="Tüm roller")
        with f2:
            opp_max_age = st.slider("Maksimum Yaş", 17, 36, 27, key="opp_age")
        with f3:
            opp_min_value = st.slider("Min. Piyasa Değeri (milyon €)", 0.5, 20.0, 1.0, 0.5,
                                      help="Çok düşük değerler (€0.2M gibi) çoğu zaman eski veya güvenilmez kayıttır.")
        with f4:
            perf_label = st.selectbox("Performans Eşiği", ["Rolünde en iyi %10", "Rolünde en iyi %25",
                                                           "Rolünde en iyi %50"], index=1)
        with f5:
            only_expiring = st.checkbox(
                "Sözleşmesi 2028'e kadar bitenler",
                help="Yalnızca önümüzdeki yaz bedelsiz olacak ya da son yılına girecek oyuncular: "
                     "bunlar tahminen çok daha ucuza alınabilir.",
            )
        quantile = {"Rolünde en iyi %10": 0.9, "Rolünde en iyi %25": 0.75, "Rolünde en iyi %50": 0.5}[perf_label]

        opp = vm.opportunities(
            min_value_eur=opp_min_value * 1e6,
            min_minutes=450,
            performance_quantile=quantile,
            max_age=opp_max_age,
            roles=opp_roles or None,
            top_n=200,
        )
        opp = opp[opp["League"].isin(selected_leagues)]
        if only_expiring:
            opp = opp[opp["Contract_Status"].isin(["free", "last_year"])]
        opp = opp.head(25)

        st.markdown(f"**{len(opp)} fırsat oyuncu** (en az 450 dakika; fırsat katsayısı = tahmini değer / gerçek değer)")
        if opp.empty:
            st.write("Bu filtrelerle fırsat oyuncu bulunamadı.")
        else:
            show_table(opp[["Player", "Squad", "Score_Group", "Age", "OptiScout_Score",
                            "Market_Value_M", "Pred_Value_M", "Value_Ratio", "Contract_Label", "Est_Fee_M"]])

        ch1, ch2 = st.columns([3, 2])
        with ch1:
            pr = vm.predictions.copy()
            pr["Durum"] = np.where(pr["Row_Key"].isin(opp["Row_Key"]), "Fırsat", "Diğer")
            pr["Rol"] = pr["Score_Group"].map(lambda g: ROLE_GROUP_LABELS.get(g, g))
            fig_val = px.scatter(
                pr.sort_values("Durum", ascending=False),
                x="Market_Value_M", y="Pred_Value_M", color="Durum",
                color_discrete_map={"Fırsat": "#facc15", "Diğer": "#475569"},
                hover_name="Player",
                hover_data={"Squad": True, "Rol": True, "Age": True, "OptiScout_Score": True,
                            "Market_Value_M": ":.1f", "Pred_Value_M": ":.1f", "Durum": False},
                log_x=True, log_y=True, template="plotly_dark", height=460,
                labels={**COLUMN_LABELS,
                        "Market_Value_M": "Gerçek piyasa değeri (milyon €, log)",
                        "Pred_Value_M": "Tahmini değer (milyon €, log)"},
                title="Gerçek vs Tahmini Değer (çizginin üstü: performansına göre ucuz)",
            )
            lo = float(min(pr["Market_Value_M"].min(), pr["Pred_Value_M"].min()))
            hi = float(max(pr["Market_Value_M"].max(), pr["Pred_Value_M"].max()))
            fig_val.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Tahmin = Gerçek",
                                         line=dict(color="gray", dash="dash")))
            st.plotly_chart(fig_val)
        with ch2:
            eff = vm.effects.copy().sort_values("Etki_%")
            eff["Yön"] = np.where(eff["Etki_%"] >= 0, "Artırır", "Azaltır")
            eff["Etiket"] = eff["Etki_%"].map(lambda v: f"{v:+.0f}%")
            fig_eff = px.bar(
                eff, x="Etki_%", y="Etken", orientation="h", color="Yön", text="Etiket",
                color_discrete_map={"Artırır": "#10b981", "Azaltır": "#ef4444"},
                template="plotly_dark", height=460, title="Değeri Ne Belirliyor? (diğer her şey sabitken)",
                labels={"Etki_%": "Piyasa değerine etkisi (%)", "Etken": ""},
            )
            fig_eff.update_traces(textposition="outside", cliponaxis=False)
            fig_eff.update_layout(showlegend=False, margin=dict(r=40))
            # Dış etiketler için iki yanda boşluk: negatif etiketler eksen yazılarıyla çakışmasın
            fig_eff.update_xaxes(range=[min(eff["Etki_%"].min(), 0) - 30, max(eff["Etki_%"].max(), 0) + 30])
            st.plotly_chart(fig_eff)

        st.caption(
            "Sınırlamalar: Model yalnızca bu sezonun ilk haftalarındaki performansı görür; sakatlık geçmişi, "
            "sözleşme süresi, potansiyel ve itibar modelde yoktur. Bu yüzden liste bir karar değil, "
            "scout'ların daha yakından izlemesi gereken oyuncular için bir ön elemedir."
        )

