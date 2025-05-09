# 🎧 Spotiflopio – Music Recommendation System

**Spotiflopio** is a music recommendation system that suggests similar songs based on a user-provided playlist. It uses unsupervised machine learning techniques (KMeans clustering and KNN) to recommend tracks with high popularity and genre diversity. The app includes a simple HTML interface powered by a Flask backend.

## 🚀 Features

- Input your favorite songs as a custom playlist.
- Get personalized recommendations based on musical features and genres.
- Evaluation metrics provided:
  - Cosine similarity
  - Intra-list similarity (ILS)
  - Genre coverage
  - Artist repeat rate
  - Average popularity

## 🧠 Technologies Used

- Python
- Flask + Flask-CORS
- Scikit-learn
- Pandas & NumPy
- HTML/CSS (frontend interface)

## 📁 Project Structure

```
SPOTIFLOPIO/
├── app.py                 # Main Flask application
├── spotify_songs.csv      # Dataset with song features
├── requirements.txt       # Python dependencies
├── Procfile               # Deployment configuration for Render
└── templates/
    └── index.html         # HTML interface
```

## 🌐 Live Demo

👉 _Add your Render URL here once deployed_  
Example: `https://spotiflopio.onrender.com`

## 📦 Installation (Local)

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/spotiflopio.git
   cd spotiflopio
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   python app.py
   ```

4. **Open your browser**
   Visit: [http://localhost:5000](http://localhost:5000)

## ☁️ Deployment on Render

> This app is ready to deploy on [Render](https://render.com) using the free web service plan.

### Render setup:
- Add your GitHub repository
- Set the **Start Command**: `python app.py`
- Leave **Build Command** empty
- Use **Free instance type**
- Add `requirements.txt` and `Procfile` at the root level

## ✨ Example Input

```
Shape of You
Blinding Lights
Stay
Let Her Go
```

## 🛠 TODO

- Add Spotify API integration for real-time track info
- Include audio previews
- Add playlist saving/export feature

## 📄 License

MIT License. Feel free to fork, adapt, and share!
