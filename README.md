# 🎬 StreamWar — Moteur de Recommandation ALS

> Plateforme de recommandation de films basée sur l'algorithme **ALS (Alternating Least Squares)** — MovieLens 1M

🔗 **Demo live** : [streamwar-ma746fxx8p2r5ifxzwwvun.streamlit.app](https://streamwar-ma746fxx8p2r5ifxzwwvun.streamlit.app/)  
👤 **Compte démo** : pseudo `DEMO` / mot de passe `1234`

---

## 📌 Description

StreamWar est une application web interactive qui recommande des films personnalisés à chaque utilisateur. L'algorithme ALS factorise la matrice de notations en deux matrices latentes (utilisateurs × films) et résout un système linéaire par moindres carrés à chaque itération.

---

## 📊 Dataset & Performances

| Métrique | Valeur |
|---|---|
| Dataset | MovieLens 1M |
| Utilisateurs | 6 040 |
| Films | 3 706 |
| Notations | 1 000 209 |
| Taux de sparsité | 95.53 % |
| Facteurs latents | 32 |
| Itérations ALS | 15 |
| Régularisation λ | 0.1 |
| **RMSE final** | **0.6142** |

---

## 🗂️ Structure du projet

```
├── app.py                  # Application Streamlit
├── train.py                # Entraînement du modèle ALS
├── model_als_custom.pkl    # Modèle sauvegardé (généré par train.py)
├── users.pkl               # Base utilisateurs (généré au premier lancement)
├── ratings.dat             # Données MovieLens 1M
├── movies.dat
├── users.dat
└── .streamlit/
    └── secrets.toml        # Clé TMDB (optionnelle)
```

---

## 🚀 Installation & Lancement

```bash
# 1. Cloner le projet
git clone <repo-url>
cd streamwar

# 2. Installer les dépendances
pip install streamlit numpy pandas requests

# 3. Entraîner le modèle (une seule fois, ~3 min)
python train.py

# 4. Lancer l'application
streamlit run app.py
```

### Affiches TMDB (optionnel)

Créer `.streamlit/secrets.toml` :
```toml
TMDB_API_KEY = "ta_clé_ici"
```
Clé gratuite sur [themoviedb.org](https://www.themoviedb.org/signup).

---

## ⚙️ Fonctionnalités

- **Authentification** — Compte personnel (pseudo 4 lettres / mot de passe 4 chiffres) avec persistance des notations
- **Cold Start** — Recommandations diversifiées par genre pour les nouveaux utilisateurs (sans aucune notation)
- **Recommandations ALS** — Calcul instantané via résolution ALS en une passe dès la première notation
- **Catalogue** — 3 706 films filtrables par genre et recherche textuelle, paginés
- **Historique** — Suivi de toutes les notations dans la sidebar
- **Affiches TMDB** — Récupération automatique des posters si clé API configurée

---

## 🧠 Algorithme ALS

À chaque itération, on fixe alternativement **V** puis **U** et on résout :

```
u_u = (V_Iu^T · V_Iu + λI)^-1 · V_Iu^T · r_u
```

**Avantage sur SGD** : chaque mise à jour est indépendante → parallélisme total sur cluster, pas de learning rate à calibrer, convergence garantie à chaque étape.

**Avantage sur KNN** : avec 95.53 % de sparsité, la similarité cosinus devient instable. L'ALS projette utilisateurs et films dans un espace dense de 32 dimensions — les données manquantes sont complétées implicitement.

---

## 👤 Auteur

**FADEGNON Steeve** — Projet Advanced ML
