# --------------------------------------------------
#  🎾  TENNIS SET PREDICTOR – ATP / WTA  (v2025-07-12)
# --------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np

# ══════  VERSIONE / RESET SESSIONE  ══════
APP_VERSION = "2025-07-12-live-set"
if st.session_state.get("version") != APP_VERSION:
    st.session_state.clear()
    st.session_state["version"] = APP_VERSION

# ══════  FUNZIONI BACKEND  ══════
@st.cache_data(show_spinner=False)
def scarica_e_aggiorna_stats():
    # dataset Jeff Sackmann (2023-2024)
    urls = {
        "ATP23": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2023.csv",
        "ATP24": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2024.csv",
        "WTA23": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_2023.csv",
        "WTA24": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_2024.csv",
    }
    df_atp = pd.concat([pd.read_csv(urls["ATP23"]), pd.read_csv(urls["ATP24"])])
    df_wta = pd.concat([pd.read_csv(urls["WTA23"]), pd.read_csv(urls["WTA24"])])

    def make_stats(df, gender):
        df["gender"] = gender
        rows = []
        for surface in ["Hard", "Clay", "Grass"]:
            for player in pd.unique(pd.Series(df["winner_name"].tolist() + df["loser_name"].tolist())):
                mask = (((df["winner_name"] == player) | (df["loser_name"] == player)) &
                        (df["surface"] == surface))
                sub = df[mask]
                if len(sub) < 8:
                    continue
                sv_won, rt_won = [], []
                for _, r in sub.iterrows():
                    if r["winner_name"] == player:
                        if r["w_svpt"] > 0:
                            sv_won.append(r["w_1stWon"] / r["w_svpt"])
                        if r["l_svpt"] > 0:
                            rt_won.append(r["l_1stWon"] / r["l_svpt"])
                    else:
                        if r["l_svpt"] > 0:
                            sv_won.append(r["l_1stWon"] / r["l_svpt"])
                        if r["w_svpt"] > 0:
                            rt_won.append(r["w_1stWon"] / r["w_svpt"])
                if not sv_won or not rt_won:
                    continue
                rows.append({
                    "player": player,
                    "surface": surface,
                    "serve_pts_won": sum(sv_won) / len(sv_won),
                    "return_pts_won": sum(rt_won) / len(rt_won),
                    "gender": gender
                })
        return pd.DataFrame(rows)

    stats_df = pd.concat([make_stats(df_atp, "M"), make_stats(df_wta, "F")]).reset_index(drop=True)
    matches_df = pd.concat([df_atp, df_wta])
    return stats_df, matches_df

def hold_prob(player, surface, gender, stats):
    row = stats.query("player==@player and surface==@surface and gender==@gender")
    if row.empty:
        return 0.75
    sv, rt = row.iloc[0][["serve_pts_won", "return_pts_won"]]
    β0, β1, β2 = -4.0, 12.0, -10.0
    return 1 / (1 + np.exp(-(β0 + β1 * sv + β2 * rt)))

def simulate_set(pA, pB, first_srv="A", sims=20000):
    out = {}
    for _ in range(sims):
        A = B = 0
        srv_A = (first_srv == "A")
        while True:
            if (A >= 6 or B >= 6) and abs(A - B) >= 2:
                break
            if A == 6 and B == 6:               # tie-break
                A += np.random.rand() < 0.52
                B = 7 - A
                break
            if srv_A:
                A += np.random.rand() < pA
                B += np.random.rand() >= pA
            else:
                B += np.random.rand() < pB
                A += np.random.rand() >= pB
            srv_A = not srv_A
        out[f"{A}-{B}"] = out.get(f"{A}-{B}", 0) + 1
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()}

# ══════  UI BASE  ══════
st.set_page_config(page_title="Tennis Predictor", page_icon="🎾", layout="centered")
st.title("🎾 Tennis Set Predictor (ATP/WTA)")

# prima volta: scarica dati
if "stats" not in st.session_state:
    stats_df, matches_df = scarica_e_aggiorna_stats()
    st.session_state["stats"] = stats_df
    st.session_state["matches"] = matches_df
else:
    stats_df = st.session_state["stats"]
    matches_df = st.session_state["matches"]

# cache per simulazioni
if "outcomes" not in st.session_state:
    st.session_state["outcomes"] = None
    st.session_state["results_table"] = None

# ══════  TABS  ══════
tab_generic, tab_live, tab_manage = st.tabs(
    ["🎾 Predizione generica", "⚡ Live set", "🛠️ Gestione database"]
)

# ---------- TAB GENERICO ----------
with tab_generic:
    st.markdown("ℹ️ *Simulazione media pre-match: usa dati storici*")
    gen = st.radio("Circuito", ["ATP (M)", "WTA (F)"], horizontal=True)
    gcode = "M" if gen.startswith("ATP") else "F"
    surf = st.selectbox("Superficie", ["Hard", "Clay", "Grass"])
    players = sorted(stats_df.query("gender==@gcode")["player"].unique())
    colA, colB = st.columns(2)
    A = colA.selectbox("Giocatore A", players)
    B = colB.selectbox("Giocatore B", players)
    pA = hold_prob(A, surf, gcode, stats_df)
    pB = hold_prob(B, surf, gcode, stats_df)
    st.write(f"Probabilità hold **{A}**: {pA*100:.1f}%  |  **{B}**: {pB*100:.1f}%")
    if st.button("Calcola probabilità set"):
        st.session_state["outcomes"] = simulate_set(pA, pB, first_srv="A")
        st.session_state["results_table"] = sorted(
            st.session_state["outcomes"].items(), key=lambda x: -x[1]
        )
        st.table(pd.DataFrame(st.session_state["results_table"][:10],
                              columns=["Risultato", "Probabilità"]))
    if st.session_state["outcomes"]:
        st.markdown("---")
        with st.form("quote"):
            quota = st.number_input("Quota bookmaker", 1.01, 1000.0, 3.50, 0.01)
            score = st.selectbox("Punteggio", [r for r, _ in st.session_state["results_table"]])
            sub = st.form_submit_button("Confronta")
        if sub:
            p = st.session_state["outcomes"][score]
            fair = 1 / p
            ev = quota * p - 1
            st.metric("Quota fair", f"{fair:.2f}")
            st.metric("Valore atteso", f"{ev*100:.1f}%")
# ---------- TAB LIVE ----------
with tab_live:
    st.markdown("### Simula il **set successivo** con dati LIVE")
    col1, col2 = st.columns(2)
    prev_set = col1.text_input("Punteggio set appena concluso (es. 6-4)")
    first_srv_next = col2.radio("Chi serve per primo nel set +1?", ["A", "B"])
    pA_live = st.slider("Probabilità live: A tiene il servizio", 0.45, 0.95, 0.80, 0.01)
    pB_live = st.slider("Probabilità live: B tiene il servizio", 0.45, 0.95, 0.78, 0.01)
    if st.button("Simula prossimo set"):
        live_out = simulate_set(pA_live, pB_live, first_srv_next)
        st.table(pd.DataFrame(sorted(live_out.items(), key=lambda x: -x[1])[:10],
                              columns=["Risultato", "Probabilità"]))

# ---------- TAB DB ----------
with tab_manage:
    st.markdown("*(Qui rimane il tuo blocco per aggiungere / editare statistiche)*")

st.caption("© 2025 – Demo integrata con simulazione generica + live | dati Jeff Sackmann")
