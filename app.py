from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from flask import render_template




# Load dataset
df = pd.read_csv("spotify_songs.csv")
df_clean = df.dropna().drop_duplicates()
df_clean["playlist_genre"] = df_clean["playlist_genre"].str.strip().str.lower()
genre_encoded = pd.get_dummies(df_clean["playlist_genre"], prefix="genre")
df_clean = pd.concat([df_clean, genre_encoded], axis=1)

selected_features = [
    "energy", "danceability", "genre_edm", "genre_rap", "genre_pop", "genre_r&b",
    "genre_rock", "genre_latin", "key", "liveness", "tempo", "mode",
    "track_popularity", "instrumentalness"
]

X_selected = df_clean[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

kmeans = KMeans(n_clusters=10, random_state=42)
df_clean['cluster'] = kmeans.fit_predict(X_scaled)

# Recommendation function
def playlist_multi_cluster_recommendation(song_list, df, features, kmeans, scaler,
                                          n_recommendations=5, buffer_step=5, max_attempts=5,
                                          min_popularity=85):

    playlist_df = df[df['track_name'].isin(song_list)]
    if playlist_df.empty:
        return pd.DataFrame(), {}

    playlist_scaled = scaler.transform(playlist_df[features])
    playlist_centroid = np.mean(playlist_scaled, axis=0).reshape(1, -1)
    playlist_clusters = kmeans.predict(playlist_scaled)
    cluster_counts = Counter(playlist_clusters)
    total = sum(cluster_counts.values())

    cluster_distribution = {c: int((v / total) * n_recommendations) for c, v in cluster_counts.items()}
    while sum(cluster_distribution.values()) < n_recommendations:
        cluster_distribution[max(cluster_distribution, key=cluster_counts.get)] += 1

    recommendations = []
    used = set()

    for cluster_id, num_needed in cluster_distribution.items():
        candidates = df[
            (df['cluster'] == cluster_id) &
            (~df['track_name'].isin(song_list)) &
            (df['track_popularity'] > min_popularity)
        ].copy()

        if candidates.empty or num_needed == 0:
            continue

        candidates_scaled = scaler.transform(candidates[features])

        for attempt in range(max_attempts):
            k = min(num_needed + buffer_step * (attempt + 1), len(candidates))
            knn = NearestNeighbors(n_neighbors=k)
            knn.fit(candidates_scaled)
            _, indices = knn.kneighbors(playlist_centroid)
            recs = candidates.iloc[indices[0]]

            unique_artists = set()
            diverse = []
            for _, row in recs.iterrows():
                key = (row['track_name'], row['track_artist'])
                if key not in used and row['track_artist'] not in unique_artists:
                    diverse.append(row)
                    used.add(key)
                    unique_artists.add(row['track_artist'])
                if len(diverse) == num_needed:
                    break

            recommendations.extend(diverse)
            if len(diverse) >= num_needed:
                break

    rec_df = pd.DataFrame(recommendations)
    if rec_df.empty:
        return rec_df, {}

    rec_scaled = scaler.transform(rec_df[features])
    cosine_sim = cosine_similarity(rec_scaled, playlist_centroid).mean()

    ils = cosine_similarity(rec_scaled)
    ils_score = ils[np.triu_indices_from(ils, k=1)].mean()

    genre_cov = df[df['track_name'].isin(rec_df['track_name'])]['playlist_genre'].nunique() / df['playlist_genre'].nunique()
    repeat_rate = (rec_df['track_artist'].value_counts() > 1).sum() / len(rec_df)
    avg_pop = rec_df['track_popularity'].mean()

    metrics = {
        "cosine_similarity": round(cosine_sim, 3),
        "ils": round(ils_score, 3),
        "genre_coverage": round(genre_cov * 100, 1),
        "artist_repeat_rate": round(repeat_rate * 100, 1),
        "average_popularity": round(avg_pop, 1)
    }

    return rec_df.reset_index(drop=True).head(n_recommendations), metrics


# Flask setup
app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    song_list = data.get("songs", [])
    n = int(data.get("n", 10))
    min_pop = int(data.get("min_popularity", 85))

    results, metrics = playlist_multi_cluster_recommendation(
        song_list, df_clean, selected_features, kmeans, scaler,
        n_recommendations=n, min_popularity=min_pop
    )

    if results.empty:
        return jsonify({"error": "No valid recommendation found."}), 400

    return jsonify({
        "recommendations": results.to_dict(orient="records"),
        "metrics": metrics
    })

if __name__ == "__main__":
    app.run(debug=True)
