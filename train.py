import pandas as pd
import numpy as np
import pickle
import time

# ─────────────────────────────────────────────
# 1. Chargement des données MovieLens 1M
# ─────────────────────────────────────────────
print("Chargement des données MovieLens 1M...")

ratings = pd.read_csv(
    'ratings.dat', sep='::', engine='python',
    names=['user_id', 'movie_id', 'rating', 'timestamp']
)
movies = pd.read_csv(
    'movies.dat', sep='::', engine='python',
    encoding='latin-1', names=['movie_id', 'title', 'genres']
)
users = pd.read_csv(
    'users.dat', sep='::', engine='python',
    names=['user_id', 'gender', 'age', 'occupation', 'zip']
)

print(f"  ✓ {len(ratings):,} notations | {len(movies):,} films | {len(users):,} utilisateurs")

# ─────────────────────────────────────────────
# 2. Encodage des IDs (compacts, sans trous)
# ─────────────────────────────────────────────
unique_users  = ratings['user_id'].unique()
unique_movies = ratings['movie_id'].unique()

user_map      = {uid: i for i, uid in enumerate(unique_users)}
movie_map     = {mid: i for i, mid in enumerate(unique_movies)}
inv_user_map  = {i: uid for uid, i in user_map.items()}
inv_movie_map = {i: mid for mid, i in movie_map.items()}

ratings['u_idx'] = ratings['user_id'].map(user_map)
ratings['m_idx'] = ratings['movie_id'].map(movie_map)

n_users  = len(user_map)
n_items  = len(movie_map)

# ─────────────────────────────────────────────
# Calcul de la sparsité (pour le rapport)
# ─────────────────────────────────────────────
total_entries    = n_users * n_items
observed_entries = len(ratings)
sparsity         = 1 - observed_entries / total_entries
print(f"\n📊 Statistiques dataset :")
print(f"  Matrice : {n_users} × {n_items} = {total_entries:,} entrées")
print(f"  Notations observées : {observed_entries:,}")
print(f"  Taux de sparsité : {sparsity:.4%}")

# ─────────────────────────────────────────────
# 3. Construction des index rapides
# ─────────────────────────────────────────────
print("\n🔧 Construction des index...")

user_ratings = {}
for row in ratings.itertuples(index=False):
    u, m, r = row.u_idx, row.m_idx, row.rating
    if u not in user_ratings:
        user_ratings[u] = {}
    user_ratings[u][m] = float(r)

item_ratings = {}
for row in ratings.itertuples(index=False):
    u, m, r = row.u_idx, row.m_idx, row.rating
    if m not in item_ratings:
        item_ratings[m] = {}
    item_ratings[m][u] = float(r)

print("  ✓ Index construits")

# ─────────────────────────────────────────────
# 4. Paramètres ALS
# ─────────────────────────────────────────────
N_FACTORS  = 32
REG        = 0.1
ITERATIONS = 15

np.random.seed(42)
U = np.random.normal(0, 0.01, (n_users, N_FACTORS))
V = np.random.normal(0, 0.01, (n_items, N_FACTORS))

# ─────────────────────────────────────────────
# 5. Boucle ALS
# ─────────────────────────────────────────────
print(f"\n🚀 Entraînement ALS ({ITERATIONS} itérations, {N_FACTORS} facteurs, λ={REG})...")
I_k = np.eye(N_FACTORS)

for it in range(ITERATIONS):
    t0 = time.time()

    # Étape 1 : fixer V, optimiser U
    for u, ratings_u in user_ratings.items():
        idx   = list(ratings_u.keys())
        r_vec = np.array(list(ratings_u.values()), dtype=np.float64)
        V_sub = V[idx]
        A     = V_sub.T @ V_sub + REG * I_k
        b     = V_sub.T @ r_vec
        U[u]  = np.linalg.solve(A, b)

    # Étape 2 : fixer U, optimiser V
    for i, ratings_i in item_ratings.items():
        idx   = list(ratings_i.keys())
        r_vec = np.array(list(ratings_i.values()), dtype=np.float64)
        U_sub = U[idx]
        A     = U_sub.T @ U_sub + REG * I_k
        b     = U_sub.T @ r_vec
        V[i]  = np.linalg.solve(A, b)

    # RMSE sur échantillon aléatoire
    sample = ratings.sample(min(10_000, len(ratings)), random_state=it)
    preds  = np.einsum('ij,ij->i', U[sample['u_idx'].values], V[sample['m_idx'].values])
    preds  = np.clip(preds, 1, 5)
    rmse   = np.sqrt(np.mean((preds - sample['rating'].values) ** 2))
    print(f"  Iter {it+1:02d}/{ITERATIONS}  RMSE={rmse:.4f}  ({time.time()-t0:.1f}s)")

# ─────────────────────────────────────────────
# 6. Score de popularité (cold-start fallback)
# ─────────────────────────────────────────────
popularity = (
    ratings.groupby('m_idx')['rating']
    .agg(count='count', mean='mean')
    .assign(score=lambda d: d['mean'] * np.log1p(d['count']))
    .sort_values('score', ascending=False)
)

payload = {
    'U':             U,
    'V':             V,
    'user_map':      user_map,
    'movie_map':     movie_map,
    'inv_user_map':  inv_user_map,
    'inv_movie_map': inv_movie_map,
    'movies':        movies,
    'users':         users,
    'popularity':    popularity,
    'user_ratings':  user_ratings,
    'item_ratings':  item_ratings,
    'n_factors':     N_FACTORS,
}

with open('model_als_custom.pkl1', 'wb') as f:
    pickle.dump(payload, f)

print("\n✅ Modèle sauvegardé → model_als_custom.pkl")
print(f"   U: {U.shape}  |  V: {V.shape}")