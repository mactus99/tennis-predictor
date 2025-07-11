import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

# --- Funzione per aggiornare e calcolare statistiche online ---
def aggiorna_stats_online():
    st.info("Scaricamento dati ATP da GitHub... attendi circa 30 secondi.")
    # Scarica gli ultimi 2 anni
    url_2024 = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2024.csv"
    url_2023 = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2023.csv"
    df_2024 = pd.read_csv(url_2024)
    df_2023 = pd.read_csv(url_2023)
    df = pd.concat([df_2024, df_2023])
    surface_types = ["Hard", "Clay", "Grass"]
    players = pd.unique(df['winner_name'].tolist() + df['loser_name'].tolist())
    rows = []
    for surface in surface_types:
        for player in players:
            mask = (
                ((df['winner_name'] == player) | (df['loser_name'] == player)) &
                (df['surface'] == surface)
            )
            player_df = df[mask]
            if len(player_df) > 8:
                # Serve % e Return %
                sv_won = []
                ret_won = []
                for _, row in player_df.iterrows():
                    if row['winner_name'] == player:
                        if row['w_svpt'] and row['w_svpt'] > 0:
                            sv_won.append(row['w_svpt'] and row['w_1stWon'] / row['w_svpt'])
                        if row['l_svpt'] and row['l_svpt'] > 0:
                            ret_won.append(row['l_1stWon'] / row['l_svpt'])
                    elif row['loser_name'] == player:
                        if row['l_svpt'] and row['l_svpt'] > 0:
                            sv_won.append(row['l_svpt'] and row['l_1stWon'] / row['l_svpt'])
                        if row['w_svpt'] and row['w_svpt'] > 0:
                            ret_won.append(row['w_1stWon'] / row['w_svpt'])
                serve_pts_won = sum(sv_won) / len(sv_won) if sv_won else np.nan
                return_pts_won = sum(ret_won) / len(ret_won) if ret_won else np.nan
                rows.append({
                    "player": player,
                    "surface": surface,
                    "serve_pts_won": serve_pts_won,
                    "return_pts_won": return_pts_won
                })
    stats = pd.DataFrame(rows)
    stats = stats.dropna()
    return stats, df

# --- Funzione per stimare probabilità di tenere il servizio ---
def estimate_hold_prob(player, surface, df_stats):
    row = df_stats.query("player==@player and surface==@surface")
    if row.empty:
        return 0.75
    sv, ret = row.iloc[0][["serve_pts_won", "return_pts_won"]]
    beta0, beta1, beta2 = -4.0, 12.0, -10.0
    logit = beta0 + beta1 * sv + beta2 * ret
    return float(1 / (1 + np.exp(-logit)))

# --- Simulazione punteggi set ---
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

# --- APP STREAMLIT ---
st.set_page_config(page_title="Tennis Set Predictor", page_icon="🎾", layout="centered")
st.title("🎾 Tennis Set Predictor - Risultato Esatto & Value Bet")

# --- Stato della cache (mantiene statistiche in sessione) ---
@st.cache_data(show_spinner=False)
def get_stats_and_matches():
    return aggiorna_stats_online()

# --- Pulsante per aggiornare dati online ---
if st.sidebar.button("🔄 Aggiorna statistiche online"):
    with st.spinner("Download e aggiornamento statistiche in corso..."):
        stats_df, match_df = aggiorna_stats_online()
        st.session_state["stats"] = stats_df
        st.session_state["match"] = match_df
        st.sidebar.success("Statistiche aggiornate da internet!")
        # Salva anche localmente, se vuoi
        stats_df.to_csv("data/stats.csv", index=False)
else:
    if "stats" not in st.session_state:
        stats_df, match_df = get_stats_and_matches()
        st.session_state["stats"] = stats_df
        st.session_state["match"] = match_df
    else:
        stats_df = st.session_state["stats"]
        match_df = st.session_state["match"]

# --- Mostra livello aggiornamento ---
if "match" in st.session_state:
    try:
        last_date = pd.to_datetime(st.session_state["match"]['tourney_date'], format='%Y%m%d').max()
        last_tourney = st.session_state["match"][st.session_state["match"]['tourney_date'] == last_date]['tourney_name'].values[0]
        st.sidebar.info(f"Statistiche aggiornate a: {last_tourney} ({last_date.strftime('%d %b %Y')})")
    except Exception:
        st.sidebar.info("Statistiche aggiornate agli ultimi tornei disponibili.")

# --- (Optional) Editing manuale delle statistiche ---
with st.sidebar.expander("✏️ Modifica statistiche manualmente (avanzato)"):
    edited_df = st.data_editor(stats_df, num_rows="dynamic", use_container_width=True)
    if st.button("💾 Salva modifiche manuali"):
        st.session_state["stats"] = edited_df
        edited_df.to_csv("data/stats.csv", index=False)
        st.success("Statistiche manuali salvate!")

st.sidebar.markdown("""
<small>
*Aggiorna statistiche online per i dati più freschi.<br>
Puoi aggiungere/correggere statistiche manualmente in caso di errori o per Challenger/qualificazioni, etc.*
</small>
""", unsafe_allow_html=True)

# --- Selezione giocatori e superficie ---
players = sorted(stats_df["player"].unique())
surface_options = sorted(stats_df["surface"].unique())
if not players or not surface_options:
    st.error("Non sono presenti statistiche, aggiorna online oppure aggiungi dati manualmente nella sidebar.")
    st.stop()
st.header("1️⃣  Scegli i giocatori e la superficie")
colA, colB = st.columns(2)
player_a = colA.selectbox("Giocatore A", players, key="A")
player_b = colB.selectbox("Giocatore B", players, key="B")
surface = st.selectbox("Superficie", surface_options)
first_server = st.radio("Chi serve per primo?", [player_a, player_b])

st.markdown("---")

# --- Calcolo probabilità ---
if st.button("Calcola probabilità set"):
    pA = estimate_hold_prob(player_a, surface, stats_df)
    pB = estimate_hold_prob(player_b, surface, stats_df)
    st.info(f"Probabilità che **{player_a}** tenga il servizio: **{pA*100:.1f}%**\n\n"
            f"Probabilità che **{player_b}** tenga il servizio: **{pB*100:.1f}%**")
    outcomes = simulate_set(pA, pB, first_server="A" if first_server == player_a else "B", n_sim=20000)
    sorted_outcomes = sorted(outcomes.items(), key=lambda x: -x[1])
    results_table = [{"Risultato": k, "Probabilità": f"{v*100:.2f}%"} for k, v in sorted_outcomes]
    st.header("2️⃣  Probabilità risultati esatti del set")
    st.table(results_table[:12])

    # Quote bookmaker e value bet
    st.markdown("---")
    st.header("3️⃣  Confronta con le quote bookmaker")
    quote = st.number_input("Quota bookmaker (es. 6-4)", min_value=1.01, max_value=1000.0, value=3.50, step=0.01)
    sel_score = st.selectbox("Seleziona il punteggio da confrontare", [r["Risultato"] for r in results_table])
    prob = outcomes.get(sel_score, 0)
    if prob > 0:
        fair_odds = 1 / prob
        ev = quote * prob - 1
        col1, col2 = st.columns(2)
        col1.metric("Quota fair", f"{fair_odds:.2f}")
        col2.metric("Valore Atteso (EV)", f"{ev*100:.1f}%")
        if ev > 0:
            st.success("🔥 Scommessa di valore! (EV > 0)")
        else:
            st.warning("Nessun valore sulla quota scelta (EV < 0)")
    else:
        st.error("Il punteggio scelto non è presente nella simulazione.")
else:
    st.info("Scegli i parametri e clicca su **Calcola probabilità set**")

st.markdown("---")
st.caption("Statistiche sempre aggiornate online (fonte: Jeff Sackmann ATP) | Editing manuale avanzato disponibile.")
