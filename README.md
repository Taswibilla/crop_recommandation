# CropSense — AI Crop Recommendation System

An end-to-end machine learning web application that recommends the best crop to grow based on soil and climate parameters. Built with Python, Flask, MongoDB, and deployed on Render.com.

---

## Features

- **EDA** — 6 automated plots (distributions, heatmap, boxplots, pairplot, confusion matrix)
- **ML Model Comparison** — 6 models trained and evaluated; best model auto-selected
- **99.3% Accuracy** — Random Forest wins with F1-macro = 0.9932
- **User Auth** — Register/Login with bcrypt-hashed passwords stored in MongoDB
- **Prediction API** — REST JSON endpoints + HTML form interface
- **Render Ready** — Procfile + runtime.txt included for one-click deploy

---

## Project Structure

```
crop_app/
├── eda_train.py          # EDA + model training + pickle export
├── app.py                # Flask web application
├── data/
│   └── crop_recommendation.xlsx
├── models/               # Generated after running eda_train.py
│   ├── crop_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── eda_outputs/          # Generated EDA plots (PNG)
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   └── predict.html
├── static/
│   └── style.css
├── requirements.txt
├── Procfile
├── runtime.txt
├── .env                  # Local env vars (not committed)
└── .env.example          # Template for env vars
```

---

## Dataset

| Property | Value |
|---|---|
| File | `crop_recommendation.xlsx` |
| Rows | 2,200 |
| Features | N, P, K, temperature, humidity, ph, rainfall |
| Target | `label` (22 crop classes) |
| Balance | 100 samples per class |
| Missing | N(39), P(96), K(141), temp(1), humidity(1), ph(15) — filled with median |

---

## Model Results

| Model | Accuracy | F1 (macro) | CV F1 (5-fold) |
|---|---|---|---|
| **Random Forest** ✓ | **0.9932** | **0.9932** | **0.9868 ± 0.004** |
| SVM (RBF) | 0.9773 | 0.9773 | 0.9726 ± 0.008 |
| Gradient Boosting | 0.9727 | 0.9729 | 0.9793 ± 0.005 |
| KNN | 0.9636 | 0.9634 | 0.9553 ± 0.008 |
| Logistic Regression | 0.9386 | 0.9381 | 0.9443 ± 0.014 |
| Decision Tree | 0.9091 | 0.8994 | 0.9191 ± 0.010 |

---

## Local Setup & Commands

### 1. Install dependencies

```bash
python3.11 -m pip install -r requirements.txt
```

### 2. Run EDA + Train Models

```bash
cd crop_app
python3.11 eda_train.py
```

This will:
- Clean and impute the dataset
- Generate 6 EDA plots in `eda_outputs/`
- Train and compare 6 ML models
- Save the best model (Random Forest) as `models/crop_model.pkl`
- Save `models/scaler.pkl` and `models/label_encoder.pkl`

### 3. Configure environment

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:
```
SECRET_KEY=your-random-secret-key
MONGO_URI=mongodb://localhost:27017/cropdb
```

### 4. Start MongoDB (local)

```bash
brew services start mongodb/brew/mongodb-community@7.0
```

### 5. Run the Flask app

```bash
python3.11 app.py
```

App runs at: **http://127.0.0.1:5000**

> If port 5000 is taken (macOS AirPlay), use a different port:
> ```bash
> PORT=5001 python3.11 app.py
> ```

---

## API Reference

### Register

```bash
curl -X POST http://127.0.0.1:5000/api/register \
  -H "Content-Type: application/json" \
  -d '{"username":"Alice","email":"alice@example.com","password":"mypassword"}'
```

### Login

```bash
curl -c cookie.txt -X POST http://127.0.0.1:5000/api/login \
  -H "Content-Type: application/json" \
  -d '{"email":"alice@example.com","password":"mypassword"}'
```

### Predict (authenticated)

```bash
curl -b cookie.txt -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":20.9,"humidity":82.0,"ph":6.5,"rainfall":202.9}'
```

Response:
```json
{"crop": "rice", "confidence": 97.0}
```

---

## Deploy on Render.com

### Step 1 — MongoDB Atlas

1. Go to [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas) → create a free cluster
2. Create a database user (username + password)
3. Network Access → Add IP Address → `0.0.0.0/0`
4. Copy the connection string:
   ```
   mongodb+srv://<user>:<password>@cluster0.xxxxx.mongodb.net/cropdb?retryWrites=true&w=majority
   ```

### Step 2 — GitHub

```bash
cd crop_app
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

> Remove `models/` from `.gitignore` if you want to commit the pre-trained pickle files (recommended to avoid retraining on Render).

### Step 3 — Render

1. Go to [render.com](https://render.com) → **New Web Service**
2. Connect your GitHub repo
3. Set the following:

| Setting | Value |
|---|---|
| **Build Command** | `pip install -r requirements.txt && python eda_train.py` |
| **Start Command** | `gunicorn app:app` |
| **Environment Variables** | `MONGO_URI` = your Atlas URI |
| | `SECRET_KEY` = any long random string |

4. Click **Deploy** — Render will install deps, train the model, and launch the app.

---

## Supported Crops (22 classes)

apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, watermelon
