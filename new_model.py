import math
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import statsmodels.api as sm
from itertools import product


#Load and prep data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
files = ["2023_24.csv", "2024_25.csv", "2025_26.csv"]
paths = [os.path.join(BASE_DIR, "data", f) for f in files]

df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

rows = []
for _, r in df.iterrows():
    rows.append({"Date": r["Date"], "team": r["HomeTeam"], "opponent": r["AwayTeam"],
                 "venue": "Home", "goals_for": r["FTHG"], "goals_against": r["FTAG"], "shots_for": r["HS"], "shots_against": r["AS"], "target": r["FTR"]})
    away_target = "A" if r["FTR"]=="H" else "H" if r["FTR"]=="A" else "D"
    rows.append({"Date": r["Date"], "team": r["AwayTeam"], "opponent": r["HomeTeam"],
                 "venue": "Away", "goals_for": r["FTAG"], "goals_against": r["FTHG"], "shots_for": r["AS"], "shots_against": r["HS"], "target": away_target})

team_df = pd.DataFrame(rows)
team_df["venue_code"] = team_df["venue"].map({"Home":1,"Away":0})

def add_rolling(group):
    group = group.sort_values("Date")
    group["gf_avg"] = group["goals_for"].rolling(5).mean().shift(1)
    group["ga_avg"] = group["goals_against"].rolling(5).mean().shift(1)
    group["shots_avg"] = group["shots_for"].rolling(5).mean().shift(1)
    group["shots_against_avg"] = group["shots_against"].rolling(5).mean().shift(1)
    return group

team_df = team_df.groupby("team", group_keys=False).apply(add_rolling).dropna()
team_df["gd_avg"] = team_df["gf_avg"] - team_df["ga_avg"]
team_df["shot_conv"] = team_df["gf_avg"] / (team_df["shots_avg"]+1)

#One-hot team/opponent
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe_df = pd.DataFrame(ohe.fit_transform(team_df[["team","opponent"]]), columns=ohe.get_feature_names_out(["team","opponent"]))
team_df = pd.concat([team_df.reset_index(drop=True), ohe_df], axis=1)

#Train xG Boost
predictors = ["venue_code","gf_avg","ga_avg","shots_avg","shots_against_avg","gd_avg","shot_conv"] + list(ohe_df.columns)
train = team_df[team_df["Date"] < "2025-12-01"]

target_map = {"H":0,"D":1,"A":2}
inv_map = {v:k for k,v in target_map.items()}
train_y = train["target"].map(target_map)

xgb = XGBClassifier(objective="multi:softprob", num_class=3, n_estimators=300, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=1, use_label_encoder=False)
cal_xgb = CalibratedClassifierCV(xgb, method="isotonic", cv=3)
cal_xgb.fit(train[predictors], train_y)

#Poisson regression
X_pois = sm.add_constant(team_df[["venue_code","gf_avg","ga_avg","shots_avg","shots_against_avg","gd_avg","shot_conv"]])
y_pois = team_df["goals_for"]
poisson_model = sm.GLM(y_pois, X_pois, family=sm.families.Poisson()).fit()

poisson_pmf = lambda k, lam: (lam**k * np.exp(-lam)) / math.factorial(k)

#Prediction func
def predict_match(home_team, away_team):
    home = team_df[team_df["team"]==home_team].sort_values("Date").iloc[-1]
    away = team_df[team_df["team"]==away_team].sort_values("Date").iloc[-1]

    #xG Boost H/D/A
    row = {"venue_code":1,"gf_avg":home["gf_avg"],"ga_avg":home["ga_avg"],
           "shots_avg":home["shots_avg"],"shots_against_avg":home["shots_against_avg"],
           "gd_avg":home["gd_avg"],"shot_conv":home["shot_conv"]}
    df_input = pd.DataFrame([row])
    for col in ohe_df.columns: df_input[col]=0
    if f"team_{home_team}" in df_input.columns: df_input[f"team_{home_team}"]=1
    if f"opponent_{away_team}" in df_input.columns: df_input[f"opponent_{away_team}"]=1

    xgb_probs = {inv_map[i]:p for i,p in enumerate(cal_xgb.predict_proba(df_input[predictors])[0])}

    #Poisson exact scores
    features = ["venue_code","gf_avg","ga_avg","shots_avg","shots_against_avg","gd_avg","shot_conv"]
    lambda_home = poisson_model.predict(sm.add_constant(pd.DataFrame([{col: home[col] for col in features}]), has_constant='add'))[0]
    lambda_away = poisson_model.predict(sm.add_constant(pd.DataFrame([{col: away[col] for col in features}]), has_constant='add'))[0]

    scores = list(product(range(6), repeat=2))
    score_probs = {(h,a): poisson_pmf(h,lambda_home)*poisson_pmf(a,lambda_away) for h,a in scores}
    home_win = sum(p for (h,a),p in score_probs.items() if h>a)
    draw = sum(p for (h,a),p in score_probs.items() if h==a)
    away_win = sum(p for (h,a),p in score_probs.items() if h<a)

    return {"XGBoost_probs": xgb_probs, "Poisson_probs":{"H":home_win,"D":draw,"A":away_win}, "score_probs":score_probs}

#UI
teams = sorted(team_df["team"].unique())
print("Available teams:", teams)
home_team = input("Enter HOME team: ")
away_team = input("Enter AWAY team: ")

probs = predict_match(home_team, away_team)

print(f"\n{home_team} vs {away_team}")
print("\nXGBoost Probabilities")
for k,v in probs["XGBoost_probs"].items(): print(f"{k}: {v:.2%}")
print("\nPoisson H/D/A Probabilities")
for k,v in probs["Poisson_probs"].items(): print(f"{k}: {v:.2%}")

top_scores = sorted(probs["score_probs"].items(), key=lambda x:x[1], reverse=True)[:3]
print("\nTop 3 Likely Scores")
for s,p in top_scores: print(f"{s[0]}-{s[1]} : {p:.2%}")

blend = {k: (probs["XGBoost_probs"][k]+probs["Poisson_probs"][k])/2 for k in ["H","D","A"]}
fav = max(blend, key=blend.get)
print("\nMost likely to win:", home_team if fav=="H" else away_team if fav=="A" else "Draw")
