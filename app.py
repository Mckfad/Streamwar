"""
StreamWar — Plateforme de recommandation ALS
Lance avec : streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import requests
import re
import os
from functools import lru_cache

# ══════════════════════════════════════════════
# CONFIG PAGE
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="StreamWar",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e0d5;
}
.stApp { background: #0a0a0f; }
h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 2px; }

section[data-testid="stSidebar"] {
    background: #12121a !important;
    border-right: 1px solid #2a2a3a;
}

/* ── Auth page ── */
.auth-container {
    max-width: 420px;
    margin: 40px auto;
    background: linear-gradient(160deg, #12121a, #16161f);
    border: 1px solid #2a2a3a;
    border-radius: 18px;
    padding: 36px 36px 28px;
}
.auth-logo {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    color: #ff4d4d;
    letter-spacing: 6px;
    text-align: center;
    margin-bottom: 4px;
}
.auth-sub {
    text-align: center;
    color: #5050a0;
    font-size: 0.82rem;
    margin-bottom: 28px;
}
.demo-box {
    background: #0e0e1a;
    border: 1px solid #2a2a4a;
    border-left: 3px solid #6060ff;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 20px;
    font-size: 0.82rem;
    color: #8080c0;
    line-height: 1.7;
}
.demo-box strong { color: #a0a0ff; }
.demo-cred {
    display: inline-block;
    background: #1a1a3a;
    border: 1px solid #3a3aff44;
    border-radius: 6px;
    padding: 2px 10px;
    font-family: monospace;
    color: #c0c0ff;
    letter-spacing: 2px;
}
.rule-hint {
    font-size: 0.72rem;
    color: #4a4a7a;
    margin-top: 2px;
}

/* ── Movie cards ── */
.movie-card {
    background: linear-gradient(180deg, #1a1a2e 0%, #16161f 100%);
    border: 1px solid #2e2e45;
    border-radius: 14px;
    overflow: hidden;
    transition: transform .25s, border-color .25s, box-shadow .25s;
    margin-bottom: 6px;
    height: 100%;
    display: flex;
    flex-direction: column;
}
.movie-card:hover {
    transform: translateY(-6px);
    border-color: #ff4d4d;
    box-shadow: 0 12px 32px rgba(255,77,77,0.18);
}
.movie-poster { width:100%; aspect-ratio:2/3; object-fit:cover; display:block; }
.movie-poster-placeholder {
    width:100%; aspect-ratio:2/3;
    background: linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
    display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    gap:8px; font-size:2.8rem;
}
.movie-poster-placeholder span {
    font-family:'DM Sans',sans-serif; font-size:0.65rem;
    color:#3a3a6a; letter-spacing:1px; text-transform:uppercase;
}
.movie-info { padding:10px 12px 12px; flex:1; display:flex; flex-direction:column; justify-content:space-between; }
.movie-title {
    font-weight:600; font-size:0.9rem; color:#e8e0d5;
    margin:0 0 6px; line-height:1.4; min-height:2.52rem;
}
.genre-badge {
    display:inline-block; background:#1e1e2e; border:1px solid #3a3a5a;
    border-radius:20px; padding:2px 8px; font-size:0.62rem;
    color:#7070a0; margin:2px 2px 0 0;
}
.rated-stars { color:#ff4d4d; font-size:0.75rem; }

/* ── Reco card ── */
.reco-card {
    background:linear-gradient(135deg,#16161f,#1a1a2e);
    border:1px solid #2a2a3a; border-left:3px solid #ff4d4d;
    border-radius:12px; padding:18px; margin-bottom:18px;
}
.reco-verdict { font-size:0.9rem; line-height:1.6; color:#b0b0c0; margin-bottom:8px; }
.verdict-liked    { color:#4caf50; font-weight:600; }
.verdict-disliked { color:#ff4d4d; font-weight:600; }
.verdict-neutral  { color:#ff9800; font-weight:600; }

/* ── Header ── */
.main-header {
    text-align:center; padding:20px 0 10px;
    border-bottom:1px solid #2a2a3a; margin-bottom:24px;
}
.main-header h1 {
    font-family:'Bebas Neue',sans-serif; font-size:3.5rem;
    color:#ff4d4d; margin:0; letter-spacing:6px;
}
.main-header p { color:#6060a0; font-size:0.9rem; margin:4px 0 0; }

/* ── Historique ── */
.hist-item {
    display:flex; justify-content:space-between; align-items:center;
    padding:5px 0; border-bottom:1px solid #1e1e2e; font-size:0.78rem;
}
.hist-title { color:#c0c0e0; flex:1; margin-right:8px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.hist-stars { color:#ff4d4d; white-space:nowrap; }

/* ── User badge ── */
.user-badge {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.8rem;
    color: #8080c0;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.user-badge strong { color: #ff4d4d; font-size: 0.95rem; letter-spacing: 1px; }

/* ── Boutons ── */
.stButton > button {
    background:#ff4d4d; color:white; border:none;
    border-radius:8px; font-weight:600; cursor:pointer;
}
.stButton > button:hover { background:#e03030; }
div[data-testid="stMetricValue"] { color:#ff4d4d !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# GESTION DES UTILISATEURS (fichier users.pkl)
# ══════════════════════════════════════════════
USERS_FILE = "users.pkl"
DEMO_USER  = "DEMO"
DEMO_PASS  = "1234"

def load_users() -> dict:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "rb") as f:
            return pickle.load(f)
    # Fichier inexistant → on crée avec le compte démo
    users = {DEMO_USER: {"password": DEMO_PASS, "ratings": {}}}
    save_users(users)
    return users

def save_users(users: dict):
    with open(USERS_FILE, "wb") as f:
        pickle.dump(users, f)

def save_current_ratings():
    """Sauvegarde les notations de l'utilisateur connecté dans users.pkl."""
    uid = st.session_state.get("user_id")
    if not uid:
        return
    users = load_users()
    if uid in users:
        users[uid]["ratings"] = dict(st.session_state.get("my_ratings", {}))
        save_users(users)

# ══════════════════════════════════════════════
# CHARGEMENT DU MODÈLE
# ══════════════════════════════════════════════
MODEL_PATH     = "model_als_custom.pkl1"
GDRIVE_FILE_ID = "1ZT_K3OWktNgVsRo4shwW-MuSWkGThoN1"

@st.cache_resource(show_spinner="⏳ Chargement du modèle…")
def load_model():
    # Sur Streamlit Cloud : télécharge le .pkl depuis Google Drive si absent
    if not os.path.exists(MODEL_PATH):
        try:
            import gdown
            gdown.download(
                f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
                MODEL_PATH,
                quiet=False,
                fuzzy=True,
            )
        except Exception as e:
            st.error(f"❌ Impossible de télécharger le modèle depuis Drive : {e}")
            st.stop()
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Erreur lors du chargement du modèle : {e}")
    st.stop()

U             = model['U']
V             = model['V']
movie_map     = model['movie_map']
inv_movie_map = model['inv_movie_map']
movies_df     = model['movies']

if 'popularity' in model:
    popularity = model['popularity']
else:
    pop_scores = np.linalg.norm(V, axis=1)
    popularity = pd.Series(pop_scores, index=np.arange(len(V))).sort_values(ascending=False)

def cold_start_diverse(top_k: int = 12) -> list:
    candidates  = popularity.index.tolist()
    selected    = []
    genre_count: dict = {}
    MAX_PER_GENRE = 2
    for m_idx in candidates:
        if len(selected) >= top_k:
            break
        mid = inv_movie_map.get(int(m_idx))
        if mid is None: continue
        row = movies_df[movies_df['movie_id'] == mid]
        if row.empty: continue
        dominant = row.iloc[0]['genres'].split('|')[0]
        if genre_count.get(dominant, 0) < MAX_PER_GENRE:
            selected.append(int(m_idx))
            genre_count[dominant] = genre_count.get(dominant, 0) + 1
    if len(selected) < top_k:
        for m_idx in candidates:
            if int(m_idx) not in selected:
                selected.append(int(m_idx))
            if len(selected) >= top_k:
                break
    return selected

# ══════════════════════════════════════════════
# TMDB
# ══════════════════════════════════════════════
TMDB_API_KEY = ""
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except Exception:
    pass


IMG_BASE = "https://image.tmdb.org/t/p/w342"

@lru_cache(maxsize=5000)
def get_poster_url(title_raw: str) -> str:
    if not TMDB_API_KEY:
        return ""
    title = re.sub(r'\s*\(\d{4}\)\s*$', '', title_raw).strip()
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": TMDB_API_KEY, "query": title, "language": "fr-FR"},
            timeout=3,
        )
        results = r.json().get("results", [])
        if results and results[0].get("poster_path"):
            return IMG_BASE + results[0]["poster_path"]
    except Exception:
        pass
    return ""

# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _cached_als(ratings_tuple: tuple, top_k: int) -> list:
    new_ratings = dict(ratings_tuple)
    idx   = [int(k) for k in new_ratings.keys()]
    r_vec = np.array(list(new_ratings.values()), dtype=np.float64)
    V_sub = V[idx]
    A     = V_sub.T @ V_sub + 0.1 * np.eye(V.shape[1])
    b     = V_sub.T @ r_vec
    u_vec = np.linalg.solve(A, b)
    scores      = V @ u_vec
    scores[idx] = -np.inf
    top         = np.argpartition(scores, -top_k)[-top_k:]
    top         = top[np.argsort(scores[top])[::-1]]
    return top.tolist()

def als_recommend_new_user(new_ratings: dict, top_k: int = 12) -> list:
    if not new_ratings:
        return cold_start_diverse(top_k)
    ratings_tuple = tuple(sorted((int(k), float(v)) for k, v in new_ratings.items()))
    return _cached_als(ratings_tuple, top_k)

def get_movie_info(m_idx: int) -> dict:
    mid = inv_movie_map.get(m_idx)
    row = movies_df[movies_df['movie_id'] == mid]
    if row.empty:
        return {"title": "Inconnu", "genres": ""}
    return {"title": row.iloc[0]['title'], "genres": row.iloc[0]['genres']}

def rating_to_verdict(r: float):
    stars = "★" * int(round(r)) + "☆" * (5 - int(round(r)))
    if r >= 4:   return "aimé",         "verdict-liked",    stars
    elif r <= 2: return "pas aimé",     "verdict-disliked", stars
    else:        return "trouvé moyen", "verdict-neutral",  stars

GENRE_FR = {
    "Action":"Action","Adventure":"Aventure","Animation":"Animation",
    "Children's":"Jeunesse","Comedy":"Comédie","Crime":"Policier",
    "Documentary":"Documentaire","Drama":"Drame","Fantasy":"Fantasy",
    "Film-Noir":"Film Noir","Horror":"Horreur","Musical":"Musical",
    "Mystery":"Mystère","Romance":"Romance","Sci-Fi":"Science-Fiction",
    "Thriller":"Thriller","War":"Guerre","Western":"Western",
}

STAR_LABELS = {
    1: "1 ★       — Mauvais",
    2: "2 ★★      — Moyen",
    3: "3 ★★★    — Bien",
    4: "4 ★★★★  — Très bien",
    5: "5 ★★★★★ — Excellent",
}

def genre_sentence(genres_str: str) -> str:
    parts = [GENRE_FR.get(g, g) for g in genres_str.split('|')]
    if len(parts) == 1: return f"un film de {parts[0]}"
    return f"un film de {', '.join(parts[:-1])} et {parts[-1]}"

def render_movie_card(title, genres_list, poster_url, rated_stars=""):
    badge_html = "".join(f"<span class='genre-badge'>{GENRE_FR.get(g,g)}</span>" for g in genres_list[:3])
    img_html   = (f"<img class='movie-poster' src='{poster_url}' loading='lazy'/>"
                  if poster_url
                  else "<div class='movie-poster-placeholder'>🎬<span>Affiche indisponible</span></div>")
    rated_html = f"<span class='rated-stars'>{rated_stars}</span>" if rated_stars else ""
    return f"""
    <div class='movie-card'>
        {img_html}
        <div class='movie-info'>
            <p class='movie-title'>{title} {rated_html}</p>
            <div>{badge_html}</div>
        </div>
    </div>"""

# ══════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════
for key, default in [
    ('user_id', None), ('my_ratings', {}), ('last_rated', None),
    ('show_recos', False), ('page', 1),
    ('prev_genre', 'Tous'), ('prev_search', ''),
    ('auth_tab', 'login'),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════════
# PAGE AUTHENTIFICATION
# ══════════════════════════════════════════════
if not st.session_state['user_id']:
    # Centrer avec colonnes vides
    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        st.markdown("<div class='auth-logo'>STREAMWAR</div>", unsafe_allow_html=True)
        st.markdown("<div class='auth-sub'>Moteur de recommandation · ALS Matrix Factorization</div>", unsafe_allow_html=True)

        # Compte démo
        st.markdown("""
        <div class='demo-box'>
            <strong>🎬 Compte démo disponible</strong><br/>
            Essaie la plateforme sans créer de compte :<br/><br/>
            &nbsp;&nbsp;Pseudo &nbsp;→ <span class='demo-cred'>DEMO</span><br/>
            &nbsp;&nbsp;Mot de passe → <span class='demo-cred'>1234</span><br/><br/>
            <span style='font-size:0.75rem; color:#5050a0;'>
            Le compte démo est partagé — tes notations y sont visibles par tous.
            Crée ton propre compte pour une expérience personnalisée.
            </span>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["🔑 Connexion", "✨ Créer un compte"])

        # ── CONNEXION ──
        with tab_login:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            login_user = st.text_input("Pseudo (4 lettres)", max_chars=4,
                                       placeholder="ex : ALEX", key="login_user").upper().strip()
            login_pass = st.text_input("Mot de passe (4 chiffres)", max_chars=4,
                                       type="password", placeholder="ex : 1234", key="login_pass")
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            if st.button("Se connecter", use_container_width=True, key="btn_login"):
                users = load_users()
                if not login_user or not login_pass:
                    st.error("Remplis les deux champs.")
                elif login_user not in users:
                    st.error(f"Compte « {login_user} » introuvable.")
                elif users[login_user]["password"] != login_pass:
                    st.error("Mot de passe incorrect.")
                else:
                    st.session_state['user_id']    = login_user
                    st.session_state['my_ratings'] = dict(users[login_user].get("ratings", {}))
                    st.success(f"Bienvenue {login_user} !")
                    st.rerun()

        # ── INSCRIPTION ──
        with tab_signup:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            new_user = st.text_input("Choisis un pseudo (4 lettres)", max_chars=4,
                                     placeholder="ex : MARC", key="new_user").upper().strip()
            st.markdown("<p class='rule-hint'>4 lettres exactement, majuscules automatiques.</p>",
                        unsafe_allow_html=True)
            new_pass = st.text_input("Choisis un mot de passe (4 chiffres)", max_chars=4,
                                     type="password", placeholder="ex : 5678", key="new_pass")
            st.markdown("<p class='rule-hint'>4 chiffres exactement.</p>", unsafe_allow_html=True)
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            if st.button("Créer mon compte", use_container_width=True, key="btn_signup"):
                users = load_users()
                err = None
                if not new_user or not new_pass:
                    err = "Remplis les deux champs."
                elif len(new_user) != 4 or not new_user.isalpha():
                    err = "Le pseudo doit contenir exactement 4 lettres."
                elif len(new_pass) != 4 or not new_pass.isdigit():
                    err = "Le mot de passe doit contenir exactement 4 chiffres."
                elif new_user in users:
                    err = f"Le pseudo « {new_user} » est déjà pris. Choisis-en un autre."
                if err:
                    st.error(err)
                else:
                    users[new_user] = {"password": new_pass, "ratings": {}}
                    save_users(users)
                    st.session_state['user_id']    = new_user
                    st.session_state['my_ratings'] = {}
                    st.success(f"Compte créé ! Bienvenue {new_user} 🎉")
                    st.rerun()

    st.stop()

# ══════════════════════════════════════════════
# SIDEBAR (utilisateur connecté)
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🎬 StreamWar")
    st.markdown("---")

    # Badge utilisateur
    uid = st.session_state['user_id']
    is_demo = (uid == DEMO_USER)
    st.markdown(f"""
    <div class='user-badge'>
        👤 &nbsp;<strong>{uid}</strong>
        {'&nbsp;<span style="color:#6060ff;font-size:0.7rem;">DÉMO</span>' if is_demo else ''}
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚪 Déconnexion", use_container_width=True):
        save_current_ratings()
        for k in ['user_id','my_ratings','last_rated','show_recos','page','prev_genre','prev_search']:
            st.session_state[k] = {'my_ratings':{}}.get(k, None if k!='page' else 1)
        st.session_state['my_ratings']   = {}
        st.session_state['show_recos']   = False
        st.session_state['page']         = 1
        st.session_state['prev_genre']   = 'Tous'
        st.session_state['prev_search']  = ''
        st.session_state['user_id']      = None
        st.rerun()

    if not TMDB_API_KEY:
        st.warning("🖼️ **Affiches désactivées**\nAjoute ta clé TMDB dans `.streamlit/secrets.toml` :\n```\nTMDB_API_KEY = 'xxxx'\n```")

    st.markdown("---")
    st.markdown("### 🗂️ Mes notations")
    my_ratings = st.session_state['my_ratings']

    if my_ratings:
        st.metric("Films notés", len(my_ratings))
        st.metric("Note moyenne", f"{np.mean(list(my_ratings.values())):.1f} / 5")

        if st.button("🔮 Mes recommandations", use_container_width=True):
            st.session_state['show_recos'] = True
            st.rerun()
        if st.button("🗑️ Réinitialiser mes notes", use_container_width=True):
            st.session_state.update({'my_ratings':{}, 'last_rated':None, 'show_recos':False, 'page':1})
            save_current_ratings()
            st.rerun()

        # ── Historique ──
        st.markdown("---")
        st.markdown("### 📋 Historique")
        hist_sorted = sorted(my_ratings.items(), key=lambda x: x[1], reverse=True)
        hist_html = ""
        for m_idx, rating in hist_sorted:
            info  = get_movie_info(int(m_idx))
            stars = "★" * int(rating) + "☆" * (5 - int(rating))
            hist_html += f"""
            <div class='hist-item'>
                <span class='hist-title'>{info['title']}</span>
                <span class='hist-stars'>{stars}</span>
            </div>"""
        st.markdown(hist_html, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='background:#0e0e1a; border:1px solid #2a2a4a; border-left:3px solid #6060ff;
                    border-radius:10px; padding:14px 16px; margin-top:8px;
                    font-size:0.82rem; color:#8080c0; line-height:1.65;'>
            <strong style='color:#a0a0ff;'>🤖 Comment ça marche ?</strong><br/><br/>
            Sans notation, StreamWar applique une stratégie
            <strong style='color:#c0c0ff;'>Cold Start</strong> :
            les films affichés sont sélectionnés parmi les
            <strong style='color:#c0c0ff;'>plus populaires</strong>
            du catalogue, avec une
            <strong style='color:#c0c0ff;'>diversité de genres garantie</strong>
            (max 2 films par genre).<br/><br/>
            Dès ta <strong style='color:#ff4d4d;'>première notation</strong>, l'algorithme
            <strong style='color:#c0c0ff;'>ALS</strong> prend le relais et calcule tes
            recommandations personnalisées en temps réel.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔍 Filtres")
    all_genres = sorted({g for gs in movies_df['genres'] for g in gs.split('|')})
    sel_genre  = st.selectbox("Genre", ["Tous"] + all_genres)
    search_q   = st.text_input("Rechercher", placeholder="Ex: Matrix…")

    if sel_genre != st.session_state['prev_genre'] or search_q != st.session_state['prev_search']:
        st.session_state['page'] = 1
        st.session_state['prev_genre']  = sel_genre
        st.session_state['prev_search'] = search_q

# ══════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════
st.markdown("""
<div class='main-header'>
    <h1>STREAMWAR</h1>
    <p>Moteur de recommandation · ALS Matrix Factorization · MovieLens 1M</p>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE RECOMMANDATIONS
# ══════════════════════════════════════════════
if st.session_state['show_recos'] and st.session_state['my_ratings']:
    st.markdown("## 🔮 Recommandations Personnalisées")

    last = st.session_state['last_rated']
    if last:
        info = get_movie_info(last['m_idx'])
        verdict_label, verdict_cls, stars = rating_to_verdict(last['rating'])
        gdesc = genre_sentence(info['genres'])
        if last['rating'] >= 4:
            follow = "Voici des films partageant la même nature cinématographique — vous devriez les adorer."
        elif last['rating'] <= 2:
            follow = "On a retenu que vous n'aimez pas vraiment. Voici des films bien différents qui devraient vous plaire davantage."
        else:
            follow = "Ce film vous a laissé mitigé. Voici des alternatives qui pourraient mieux vous correspondre."
        st.markdown(f"""
        <div class='reco-card'>
            <div class='reco-verdict'>
                Vous avez <span class='{verdict_cls}'>{verdict_label}</span>
                <strong style='color:#fff'>&nbsp;{info['title']}</strong>
                &nbsp;<span style='color:#ff4d4d'>{stars}</span><br/>
                C'est {gdesc}. {follow}
            </div>
        </div>""", unsafe_allow_html=True)

    reco_idxs = als_recommend_new_user(st.session_state['my_ratings'], top_k=12)
    cols = st.columns(4)
    for i, m_idx in enumerate(reco_idxs):
        info   = get_movie_info(m_idx)
        poster = get_poster_url(info['title'])
        genres = info['genres'].split('|')
        with cols[i % 4]:
            st.markdown(render_movie_card(info['title'], genres, poster), unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("← Retour au catalogue"):
        st.session_state['show_recos'] = False
        st.rerun()

# ══════════════════════════════════════════════
# PAGE CATALOGUE
# ══════════════════════════════════════════════
else:
    df = movies_df.copy()
    df['m_idx'] = df['movie_id'].map(movie_map)
    if search_q:
        df = df[df['title'].str.contains(search_q, case=False, na=False)]
    if sel_genre != "Tous":
        df = df[df['genres'].str.contains(sel_genre, na=False)]

    pop_order = {m: i for i, m in enumerate(popularity.index.tolist())}
    df['pop_rank'] = df['m_idx'].map(lambda x: pop_order.get(x, 99999))
    df = df.sort_values('pop_rank').reset_index(drop=True)

    total    = len(df)
    PAGE_SZ  = 20
    max_page = max(1, (total - 1) // PAGE_SZ + 1)

    st.markdown(f"### 🎥 Catalogue &mdash; {total:,} films", unsafe_allow_html=True)

    page    = st.session_state['page']
    page_df = df.iloc[(page - 1) * PAGE_SZ : page * PAGE_SZ]

    N_COLS = 4
    for start in range(0, len(page_df), N_COLS):
        row_slice = page_df.iloc[start:start + N_COLS]
        cols      = st.columns(N_COLS, gap="medium")
        for col, (_, film) in zip(cols, row_slice.iterrows()):
            m_idx  = film['m_idx']
            title  = film['title']
            genres = film['genres'].split('|')
            poster = get_poster_url(title)
            already = m_idx in st.session_state['my_ratings']
            rated_s = ("★" * int(st.session_state['my_ratings'].get(m_idx, 0))) if already else ""

            with col:
                st.markdown(render_movie_card(title, genres, poster, rated_s), unsafe_allow_html=True)
                with st.expander("⭐ Noter ce film"):
                    default_r = int(st.session_state['my_ratings'].get(m_idx, 3))
                    r_val = st.radio(
                        "Note",
                        options=[1, 2, 3, 4, 5],
                        index=default_r - 1,
                        key=f"sl_{m_idx}",
                        label_visibility="collapsed",
                        format_func=lambda x: STAR_LABELS[x],
                        horizontal=False,
                    )
                    if st.button("✓ Valider", key=f"btn_{m_idx}", use_container_width=True):
                        st.session_state['my_ratings'][int(m_idx)] = r_val
                        st.session_state['last_rated'] = {'m_idx': int(m_idx), 'rating': r_val}
                        st.session_state['show_recos'] = True
                        save_current_ratings()
                        st.rerun()

    # ── Pagination ──
    st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
    col_prev, col_mid, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("← Précédent", disabled=(page <= 1), use_container_width=True):
            st.session_state['page'] -= 1
            st.rerun()
    with col_mid:
        st.markdown(
            f"<div style='text-align:center; padding:8px 0; color:#e8e0d5; font-weight:600; font-size:1rem;'>"
            f"Page {page} / {max_page}</div>",
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button("Suivant →", disabled=(page >= max_page), use_container_width=True):
            st.session_state['page'] += 1
            st.rerun()
