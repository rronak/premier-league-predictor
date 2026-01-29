# Football Match Predictor

This project predicts the outcome of football matches using a **Random Forest classifier** and team statistics.

---

## Features

- Uses **rolling averages** of goals, shots, and shots against for recent team form.
- **3-class model**: Home Win / Draw / Away Win.
- Interactive command-line interface for predicting matches:
  - User selects **Home Team** and **Away Team**.
  - Outputs **probabilities** and **model verdict**.
- Trains on multiple seasons of football data.

---

## How to Run

1. Get the data from: https://www.football-data.co.uk/englandm.php
2. Place your CSV files in the data folder.
3. Run the Python script:

```bash
python model.py
