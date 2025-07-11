import streamlit as st
import pandas as pd
import numpy as np

# Se la versione del codice cambia, resetta la vecchia sessione
APP_VERSION = "2025-07-11-fix65"

if st.session_state.get("version") != APP_VERSION:
    st.session_state.clear()
    st.session_state["version"] = APP_VERSION

# =========== FUNZIONI BACKEND ===========

@st.cache_data(show_spinner=False)
def aggiorna_stats_online():
    # Scarica dati ATP
    atp_2024 = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2024.csv"
    atp_2023 = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2023.csv"
    wta_2024 = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_2024.csv"
    wta_2023 = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_2023.csv"
    df_atp = pd.concat([pd.read_csv(atp_2023), pd.read_csv(atp_2024)])
    df_wta = pd.concat([pd.read_csv(wta_2023), pd.read_csv(wta_2024)])

    def make_stats(df, gender):
        df['gender'] = gender
        surface_types = ["Hard", "Clay", "Grass"]
        players = pd.unique(pd.Series(df['winner_name'].tolist() + df['loser_name'].tolist()))
        rows = []
        for surface in surface_types:
            for player in players:
                mask = (
                    ((df['winner_name'] == player) | (df['loser_name'] == player)) &
                    (df['surface'] == surface)
                )
                player_df = df[mask]
                if len(player_df) > 8:
                    sv_won = []
                    ret_won = []
                    for _, row in player_df.iterrows():
                        if row['winner_name'] == player:
                            if row['w_svpt'] and row['w_svpt'] > 0:
                                sv_won.append(row['w_1stWon'] / row['w_svpt'])
                            if row['l_svpt'] and row['l_svpt'] > 0:
                                ret_won.append(row['l_1stWon'] / row['l_svpt'])
                        elif row['loser_name'] == player:
                            if row['l_svpt'] and row['l_svpt'] > 0:
                                sv_won.append(row['l_1stWon'] / row['l_svpt'])
                            if row['w_svpt'] and row['w_svpt'] > 0:
                                ret_won.append(row['w_1stWon'] / row['w_svpt'])
                    serve_pts_won = sum(sv_won) / len(sv_won) if sv_won else np.nan
                    return_pts_won = sum(ret_won) / len(ret_won) if ret_won else np.nan
                    rows.append({
                        "player": player,
                        "surface": surface,
                        "serve_pts_won": serve_pts_won,
                        "return_pts_won": return_pts_won,
                        "gender": gender
                    })
        stats = pd.DataFrame(rows)
        stats = stats.dropna()
        return stats

    stats_atp = make_stats(df_atp, "M")
    stats_wta = make_stats(df_wta, "F")
    stats_df = pd.concat([stats_atp, stats_wta]).reset_index(drop=True)
    return stats_df, pd.concat([df_atp, df_wta])

def estimate_hold_prob(player, surface, gender, df_stats):
    row = df_stats.query("player==@player and surface==@surface and gender==@gender")
    if row.empty:
        return 0.75
    sv, ret = row.iloc[0][["serve_pts_won", "return_pts_won"]]
    beta0, beta1, beta2 = -4.0, 12.0, -10.0
    logit = beta0 + beta1 * sv + beta2 * ret
    return float(1 / (1 + np.exp(-logit)))

def simulate_set(pA, pB, first_server, n_sim=20000):
    outcomes = {}
    for _ in range(n_sim):
        gA = gB = 0
        server_A = (first_server == "A")
        while True:
            # Termina se uno ha almeno 6 e almeno 2 di vantaggio
            if (gA >= 6 or gB >= 6) and abs(gA - gB) >= 2:
                break
            if gA == 6 and gB == 6:
                # Tie-break
                if np.random.rand() < 0.52:
                    gA += 1
                else:
                    gB += 1
                break
            # Simula il game
            if server_A:
                if np.random.rand() < pA:
                    gA += 1
                else:
                    gB += 1
            else:
                if np.random.rand() < pB:
                    gB += 1
                else:
                    gA += 1
            server_A = not server_A
        # Registrazione risultato finale valido (es: 7-6, 6-3, ecc.)
        score = f"{gA}-{gB}"
        outcomes[score] = outcomes.get(score, 0) + 1
    tot = sum(outcomes.values())
    return {k: v / tot for k, v in outcomes.items()}

# ========== APP ==========

st.set_page_config(page_title="Tennis Set Predictor", page_icon="🎾", layout="centered")
st.title("🎾 Tennis Set Predictor (ATP/WTA)")

# --- Session state inizializzazione ---
if "stats" not in st.session_state or "matches" not in st.session_state:
    stats_df, matches_df = aggiorna_stats_online()
    st.session_state["stats"] = stats_df
    st.session_state["matches"] = matches_df
else:
    stats_df = st.session_state["stats"]
    matches_df = st.session_state["matches"]

# Per mantenere la simulazione anche quando cambi le quote!
if "outcomes" not in st.session_state:
    st.session_state["outcomes"] = None
    st.session_state["results_table"] = None

# --- Sidebar: aggiornamento dati e info ---
if st.sidebar.button("🔄 Aggiorna statistiche online"):
    with st.spinner("Download e aggiornamento statistiche in corso..."):
        stats_df, matches_df = aggiorna_stats_online()
        st.session_state["stats"] = stats_df
        st.session_state["matches"] = matches_df
        st.sidebar.success("Statistiche aggiornate da internet!")

if not matches_df.empty:
    try:
        last_date = pd.to_datetime(matches_df['tourney_date'], format='%Y%m%d').max()
        last_tourney = matches_df[matches_df['tourney_date'] == last_date]['tourney_name'].values[0]
        st.sidebar.info(f"Statistiche aggiornate a: {last_tourney} ({last_date.strftime('%d %b %Y')})")
    except Exception:
        st.sidebar.info("Statistiche aggiornate agli ultimi tornei disponibili.")

tab_predict, tab_manage = st.tabs(["🎾 Predizione", "🛠️ Gestione database"])

# ========== TAB PREVISIONE ==========
with tab_predict:
    st.subheader("1️⃣  Parametri di simulazione")
    gender = st.radio("Circuito", ["Maschile (ATP)", "Femminile (WTA)"], horizontal=True)
    gender_code = "M" if gender.startswith("M") else "F"
    surface_options = ["Hard", "Clay", "Grass"]

    # Ricerca integrata nel selectbox, niente campo aggiuntivo
    players = sorted(stats_df.query("gender == @gender_code")["player"].unique())
    colA, colB = st.columns(2)
    player_a = colA.selectbox("Giocatore A", players, key="A")
    player_b = colB.selectbox("Giocatore B", players, key="B")
    surface = st.selectbox("Superficie", surface_options)
    first_server = st.radio("Chi serve per primo?", [player_a, player_b])

    st.markdown("---")

    # -- Simulazione set --
    if st.button("Calcola probabilità set"):
        pA = estimate_hold_prob(player_a, surface, gender_code, stats_df)
        pB = estimate_hold_prob(player_b, surface, gender_code, stats_df)

        outcomes = simulate_set(
            pA, pB,
            first_server="A" if first_server == player_a else "B",
            n_sim=20000
        )
        st.session_state["outcomes"] = outcomes
        st.session_state["results_table"] = [
            {"Risultato": k, "Probabilità": f"{v*100:.2f}%"}
            for k, v in sorted(outcomes.items(), key=lambda x: -x[1])
        ]

        st.info(
            f"Probabilità hold {player_a}: **{pA*100:.1f}%** ‒ "
            f"{player_b}: **{pB*100:.1f}%**"
        )
        st.header("2️⃣  Probabilità risultati esatti del set")
        st.table(st.session_state["results_table"][:12])

    # -- Form quote bookmaker --
    if st.session_state["outcomes"]:
        st.markdown("---")
        st.header("3️⃣  Confronta con le quote bookmaker")

        with st.form("quote_form"):
            quota = st.number_input("Quota bookmaker", 1.01, 1000.0, 3.50, 0.01)
            sel_score = st.selectbox(
                "Scegli il punteggio",
                [r["Risultato"] for r in st.session_state["results_table"]],
                key="sel_score"
            )
            ok = st.form_submit_button("Calcola valore")

        if ok:
            p = st.session_state["outcomes"][sel_score]
            fair_odds = 1 / p
            ev = quota * p - 1
            col1, col2 = st.columns(2)
            col1.metric("Quota fair", f"{fair_odds:.2f}")
            col2.metric("Valore atteso", f"{ev*100:.1f}%")
            st.success("👍 Value bet!" if ev > 0 else "👎 Non conviene")
    else:
        st.info("Prima calcola le probabilità del set per confrontare le quote.")

# ========== TAB GESTIONE DB ==========
with tab_manage:
    st.markdown("### Aggiungi o aggiorna statistiche di un giocatore")
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
        stats_df.to_csv("data/stats.csv", index=False)

    st.markdown("---")
    st.markdown("### Database completo (editabile)")
    edited_df = st.data_editor(
        stats_df.sort_values(["gender", "player", "surface"]),
        num_rows="dynamic", use_container_width=True, key="main_db"
    )
    if st.button("💾 Salva tutte le modifiche"):
        st.session_state["stats"] = edited_df
        edited_df.to_csv("data/stats.csv", index=False)
        st.success("Statistiche manuali salvate!")

    st.info(
        "Usa questo tab per aggiungere/aggiornare statistiche mancanti (giocatori, superfici, genere). "
        "Per la predizione aggiorna sempre le statistiche online, oppure completa manualmente qui."
    )

st.markdown("---")
st.caption("App tennis predictor - powered by Matteo D'Anella & dati Jeff Sackmann | ATP & WTA | Editing statistica integrato")
