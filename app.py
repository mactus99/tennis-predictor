import streamlit as st
import pandas as pd
import numpy as np

# ======== FUNZIONI BACKEND =========

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
        while gA < 6 and gB < 6:
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
        if gA == gB:  # 6-6 tie-break
            if np.random.rand() < 0.52:
                gA += 1
            else:
                gB += 1
        score = f"{max(gA,gB)}-{min(gA,gB)}"
        outcomes[score] = outcomes.get(score, 0) + 1
    tot = sum(outcomes.values())
    return {k: v / tot for k, v in outcomes.items()}

# ======== APP FRONTEND =========

st.set_page_config(page_title="Tennis Set Predictor", page_icon="🎾", layout="centered")
st.title("🎾 Tennis Set Predictor (ATP/WTA)")

@st.cache_data(show_spinner=False)
def get_stats_and_matches():
    return aggiorna_stats_online()

# Stato sessione
if "stats" not in st.session_state or "matches" not in st.session_state:
    stats_df, matches_df = get_stats_and_matches()
    st.session_state["stats"] = stats_df
    st.session_state["matches"] = matches_df
else:
    stats_df = st.session_state["stats"]
    matches_df = st.session_state["matches"]

if st.sidebar.button("🔄 Aggiorna statistiche online"):
    with st.spinner("Download e aggiornamento statistiche in corso..."):
        stats_df, matches_df = aggiorna_stats_online()
        st.session_state["stats"] = stats_df
        st.session_state["matches"] = matches_df
        st.sidebar.success("Statistiche aggiornate da internet!")

# Info aggiornamento
if not matches_df.empty:
    try:
        last_date = pd.to_datetime(matches_df['tourney_date'], format='%Y%m%d').max()
        last_tourney = matches_df[matches_df['tourney_date'] == last_date]['tourney_name'].values[0]
        st.sidebar.info(f"Statistiche aggiornate a: {last_tourney} ({last_date.strftime('%d %b %Y')})")
    except Exception:
        st.sidebar.info("Statistiche aggiornate agli ultimi tornei disponibili.")

tab_predict, tab_manage = st.tabs(["🎾 Predizione", "🛠️ Gestione database"])

# ========== TAB PREVISIONE ==============
with tab_predict:
    st.subheader("1️⃣  Parametri di simulazione")
    gender = st.radio("Circuito", ["Maschile (ATP)", "Femminile (WTA)"], horizontal=True)
    gender_code = "M" if gender.startswith("M") else "F"
    surface_options = ["Hard", "Clay", "Grass"]

    filter_name = st.text_input("Cerca giocatore per nome o parte di nome", "")
    all_players = sorted(stats_df.query("gender == @gender_code")["player"].unique())
    filtered_players = [p for p in all_players if filter_name.lower() in p.lower()] if filter_name else all_players

    colA, colB = st.columns(2)
    player_a = colA.selectbox("Giocatore A", filtered_players, key="A")
    player_b = colB.selectbox("Giocatore B", filtered_players, key="B")
    surface = st.selectbox("Superficie", surface_options)
    first_server = st.radio("Chi serve per primo?", [player_a, player_b])

    st.markdown("---")

    if st.button("Calcola probabilità set"):
        pA = estimate_hold_prob(player_a, surface, gender_code, stats_df)
        pB = estimate_hold_prob(player_b, surface, gender_code, stats_df)
        st.info(
            f"Probabilità che **{player_a}** tenga il servizio: **{pA*100:.1f}%**\n\n"
            f"Probabilità che **{player_b}** tenga il servizio: **{pB*100:.1f}%**"
        )
        outcomes = simulate_set(pA, pB, first_server="A" if first_server == player_a else "B", n_sim=20000)
        sorted_outcomes = sorted(outcomes.items(), key=lambda x: -x[1])
        results_table = [{"Risultato": k, "Probabilità": f"{v*100:.2f}%"} for k, v in sorted_outcomes]
        st.header("2️⃣  Probabilità risultati esatti del set")
        st.table(results_table[:12])

        st.markdown("---")
        st.header("3️⃣  Confronta con le quote bookmaker")
        with st.form(key="quote_form"):
            quote = st.number_input(
                "Quota bookmaker", min_value=1.01, max_value=1000.0,
                value=3.50, step=0.01
            )
            sel_score = st.selectbox(
                "Scegli il punteggio", [r["Risultato"] for r in results_table],
                key="score_sel"
            )
            submit_q = st.form_submit_button("Calcola valore")
        if submit_q:
            prob = outcomes.get(sel_score, 0)
            if prob:
                fair_odds = 1 / prob
                ev = quote * prob - 1
                st.metric("Quota fair", f"{fair_odds:.2f}")
                st.metric("Valore atteso", f"{ev*100:.1f}%")
                st.success("👍 Value bet!" if ev > 0 else "👎 Non conviene")
            else:
                st.error("Punteggio troppo raro nella simulazione.")
    else:
        st.info("Scegli i parametri e clicca su **Calcola probabilità set**")

# ========== TAB DATABASE ==============
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
