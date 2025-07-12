import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components

GA_SCRIPT = """
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-2FEMB1XB39"></script>
<script>
window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());
gtag('config', 'G-2FEMB1XB39');
</script>
"""

components.html(GA_SCRIPT, height=0)

# Versione: aggiorna ogni volta che modifichi il codice
APP_VERSION = "2025-07-12-live-set-v1.5.6"
if st.session_state.get("version") != APP_VERSION:
    st.session_state.clear()
    st.session_state["version"] = APP_VERSION

# =========== FUNZIONI BACKEND ===========

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
            # Win by 2, 6+ games
            if (A >= 6 or B >= 6) and abs(A - B) >= 2:
                break
            # Tie-break at 6-6
            if A == 6 and B == 6:
                tb = np.random.rand() < 0.52
                if tb:
                    A += 1
                else:
                    B += 1
                break
            # Game
            if srv_A:
                if np.random.rand() < pA:
                    A += 1
                else:
                    B += 1
            else:
                if np.random.rand() < pB:
                    B += 1
                else:
                    A += 1
            srv_A = not srv_A
        # Rimuove punteggi impossibili tipo 7-0, 8-6, ecc.
        if (A >= 6 or B >= 6) and abs(A - B) >= 2 and (A <= 7 and B <= 7):
            out[f"{A}-{B}"] = out.get(f"{A}-{B}", 0) + 1
        elif (A == 7 and B == 6) or (B == 7 and A == 6):
            out[f"{A}-{B}"] = out.get(f"{A}-{B}", 0) + 1
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()}

def calcola_hold_percent(prime_in, first_won, second_won):
    # Modello semplificato: percentuale di game vinti al servizio ≈
    #   prime_in * first_won + (1 - prime_in) * second_won
    p_hold = (prime_in / 100) * (first_won / 100) + (1 - prime_in / 100) * (second_won / 100)
    return p_hold

# ========== APP ==========
st.set_page_config(page_title="Tennis Set Predictor", page_icon="🎾", layout="centered")
st.title("🎾 Tennis Set Predictor (ATP/WTA)")

if "stats" not in st.session_state:
    stats_df, matches_df = scarica_e_aggiorna_stats()
    st.session_state["stats"] = stats_df
    st.session_state["matches"] = matches_df
else:
    stats_df = st.session_state["stats"]
    matches_df = st.session_state["matches"]

if "outcomes" not in st.session_state:
    st.session_state["outcomes"] = None
    st.session_state["results_table"] = None

tab_generic, tab_live, tab_manage = st.tabs(
    ["🎾 Predizione generica", "⚡ Live set", "🛠️ Gestione database"]
)

# ---------- TAB GENERICO ----------
with tab_generic:
    st.markdown("ℹ️ <b>Simulazione media pre-match:</b> usa dati storici", unsafe_allow_html=True)
    gen = st.radio("Circuito", ["ATP (M)", "WTA (F)"], horizontal=True)
    gcode = "M" if gen.startswith("ATP") else "F"
    surf = st.selectbox("Superficie", ["Hard", "Clay", "Grass"])
    players = sorted(stats_df.query("gender==@gcode")["player"].unique())
    colA, colB = st.columns(2)
    A = colA.selectbox("Giocatore A", players)
    B = colB.selectbox("Giocatore B", players)
    pA = hold_prob(A, surf, gcode, stats_df)
    pB = hold_prob(B, surf, gcode, stats_df)
    st.write(f"Probabilità hold <b>{A}</b>: <b>{pA*100:.1f}%</b>  |  <b>{B}</b>: <b>{pB*100:.1f}%</b>", unsafe_allow_html=True)
    if st.button("Calcola probabilità set"):
        st.session_state["outcomes"] = simulate_set(pA, pB, first_srv="A")
        st.session_state["results_table"] = [
            {"Risultato": k, "Probabilità": f"{v*100:.2f}%"}
            for k, v in sorted(st.session_state["outcomes"].items(), key=lambda x: -x[1])
        ]
        st.table(pd.DataFrame(st.session_state["results_table"][:10]))
    if st.session_state["outcomes"]:
        st.markdown("---")
        st.markdown("### 🎰 Confronta le quote dei bookmaker")
        with st.form("quote"):
            quota = st.number_input("Quota bookmaker", 1.01, 1000.0, 3.50, 0.01)
            score = st.selectbox("Punteggio", [r["Risultato"] for r in st.session_state["results_table"]])
            sub = st.form_submit_button("Confronta")
        if sub:
            p = st.session_state["outcomes"][score]
            fair = 1 / p
            ev = quota * p - 1
            colq, colv = st.columns(2)
            with colq:
                st.markdown("#### 💸 Quota fair")
                st.markdown(f"<div style='font-size:2.2em; font-weight:700; color:#1a66ff'>{fair:.2f}</div>", unsafe_allow_html=True)
                st.caption("Quota reale calcolata")
            with colv:
                st.markdown("#### 📈 Valore atteso")
                st.markdown(f"<div style='font-size:2.2em; font-weight:700; color:{'green' if ev>0 else 'red'}'>{ev*100:.1f}%</div>", unsafe_allow_html=True)
                st.caption("Scommessa vantaggiosa" if ev>0 else "Scommessa NON vantaggiosa")
            if ev > 0:
                st.success("👍 Value bet!")
            else:
                st.warning("👎 Non conviene")

# ---------- TAB LIVE ----------
with tab_live:
    st.markdown("### ⚡ Simula il <b>prossimo set</b> con dati live", unsafe_allow_html=True)
    mode = st.radio("Scegli la modalità di input",
                    ["Probabilità di hold", "Statistiche live (prime %, punti vinti)"])
    first_srv_next = st.radio("Chi serve per primo nel set +1?", ["A", "B"], horizontal=True)

    if mode == "Probabilità di hold":
        pA_live = st.slider("Probabilità live: A tiene il servizio (%)", 45, 95, 80, 1) / 100
        pB_live = st.slider("Probabilità live: B tiene il servizio (%)", 45, 95, 78, 1) / 100

    else:  # Statistiche dettagliate
        st.markdown("#### Inserisci le statistiche del set appena finito")
        col1, col2 = st.columns(2)
        a_1st_in  = col1.slider("A – % prime in campo", 40, 90, 65, 1)
        a_1st_won = col1.slider("A – % punti vinti su 1ª", 50, 90, 75, 1)
        a_2nd_won = col1.slider("A – % punti vinti su 2ª", 30, 70, 50, 1)
        b_1st_in  = col2.slider("B – % prime in campo", 40, 90, 63, 1)
        b_1st_won = col2.slider("B – % punti vinti su 1ª", 50, 90, 72, 1)
        b_2nd_won = col2.slider("B – % punti vinti su 2ª", 30, 70, 48, 1)

        pA_live = calcola_hold_percent(a_1st_in, a_1st_won, a_2nd_won)
        pB_live = calcola_hold_percent(b_1st_in, b_1st_won, b_2nd_won)
        st.info(f"Probabilità calcolata:  A hold ≈ {pA_live*100:.1f}%  |  B hold ≈ {pB_live*100:.1f}%")

    if st.button("Simula prossimo set"):
        live_out = simulate_set(pA_live, pB_live, first_srv_next)
        st.table(pd.DataFrame([
            {"Risultato": k, "Probabilità": f"{v*100:.2f}%"} 
            for k, v in sorted(live_out.items(), key=lambda x: -x[1])[:12]
        ]))

# ---------- TAB DB ----------
with tab_manage:
    st.markdown("### ➕ Aggiungi o aggiorna statistiche giocatore")
    with st.form(key="add_stat"):
        player_name = st.text_input("Nome giocatore")
        surf = st.selectbox("Superficie", ["Hard", "Clay", "Grass"], key="db_surf")
        sv = st.number_input("Serve pts won (es: 0.50–0.90)", 0.50, 0.95, 0.75, 0.01, key="db_sv")
        rt = st.number_input("Return pts won (es: 0.20–0.50)", 0.20, 0.60, 0.35, 0.01, key="db_rt")
        gen = st.radio("Genere", ["M", "F"], horizontal=True, key="db_gen")
        add_btn = st.form_submit_button("Aggiungi o aggiorna")
    if add_btn:
        cond = (stats_df.player == player_name) & (stats_df.surface == surf) & (stats_df.gender == gen)
        if stats_df[cond].empty:
            stats_df = pd.concat([
                stats_df,
                pd.DataFrame([{
                    "player": player_name, "surface": surf,
                    "serve_pts_won": sv, "return_pts_won": rt,
                    "gender": gen
                }])
            ], ignore_index=True)
            st.success("Nuova statistica aggiunta!")
        else:
            stats_df.loc[cond, ["serve_pts_won", "return_pts_won"]] = [sv, rt]
            st.success("Statistica aggiornata!")
        st.session_state["stats"] = stats_df

    st.markdown("---")
    st.markdown("### 📝 Database completo (editabile)")
    edited_df = st.data_editor(
        stats_df.sort_values(["gender", "player", "surface"]),
        num_rows="dynamic", use_container_width=True, key="main_db"
    )
    if st.button("💾 Salva tutte le modifiche"):
        st.session_state["stats"] = edited_df
        st.success("Statistiche manuali salvate!")

    st.info(
        "Usa questo tab per aggiungere/aggiornare statistiche mancanti (giocatori, superfici, genere). "
        "Per la predizione aggiorna sempre le statistiche online, oppure completa manualmente qui."
    )

st.caption("© 2025 – Simulatore ATP/WTA con quote e calcolo live | Dati Jeff Sackmann")
