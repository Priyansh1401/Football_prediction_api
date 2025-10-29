from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="GoalPredicta AI Model API")

# --- Load your trained models ---
clf = joblib.load("models_retrained_new/XGBClassifier_FTResult.joblib")
reg_fthg = joblib.load("models_retrained_new/XGBRegressor_FTHome.joblib")
reg_ftag = joblib.load("models_retrained_new/XGBRegressor_FTAway.joblib")
reg_hthg = joblib.load("models_retrained_new/XGBRegressor_HTHome.joblib")
reg_htag = joblib.load("models_retrained_new/XGBRegressor_HTAway.joblib")
scaler = joblib.load("models_retrained_new/scaler.joblib")

# --- Input schema ---
class MatchRequest(BaseModel):
    home: str
    away: str
    home_elo: float = 1600
    away_elo: float = 1600
    form_strength: float = 0.0
    elo_diff: float = 0.0
    home_xg: float = 1.2
    away_xg: float = 1.1
    shot_diff: float = 0.0
    corner_diff: float = 0.0

@app.post("/predict")
def predict_match(req: MatchRequest):
    try:
        # Prepare input
        X = np.array([[req.home_elo, req.away_elo, req.form_strength, req.elo_diff,
                       req.home_xg, req.away_xg, req.shot_diff, req.corner_diff]])
        X_scaled = scaler.transform(X)

        # Classification (Win/Draw/Loss)
        probs = clf.predict_proba(X_scaled)[0]

        # Full-time predictions
        pred_ft_home = reg_fthg.predict(X_scaled)[0]
        pred_ft_away = reg_ftag.predict(X_scaled)[0]

        # Half-time predictions
        pred_ht_home = reg_hthg.predict(X_scaled)[0]
        pred_ht_away = reg_htag.predict(X_scaled)[0]

        response = {
            "fixture": f"{req.home} vs {req.away}",
            "home_win_pct": round(probs[0] * 100, 2),
            "draw_pct": round(probs[1] * 100, 2),
            "away_win_pct": round(probs[2] * 100, 2),
            "half_time": {"home": int(pred_ht_home), "away": int(pred_ht_away)},
            "full_time": {"home": int(pred_ft_home), "away": int(pred_ft_away)},
            "model_used": "XGBoost Ensemble v2"
        }
        return response
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def root():
    return {"message": "GoalPredicta AI Model API is running!"}


