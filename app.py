import os, pickle, functools
from datetime import datetime, timezone
from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, jsonify)
from pymongo import MongoClient
import bcrypt
from dotenv import load_dotenv
import numpy as np

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

# ── MongoDB ──────────────────────────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/cropdb")
client = MongoClient(MONGO_URI)
db     = client.get_default_database() if "mongodb.net" in MONGO_URI else client["cropdb"]
users  = db["users"]
users.create_index("email", unique=True)

# ── Load ML artifacts ────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")

with open(os.path.join(MODEL_DIR, "crop_model.pkl"),    "rb") as f: model    = pickle.load(f)
with open(os.path.join(MODEL_DIR, "scaler.pkl"),        "rb") as f: scaler   = pickle.load(f)
with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f: le       = pickle.load(f)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# ── Auth helper ──────────────────────────────────────────────────────────────
def login_required(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

# ── HTML Routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not (username and email and password):
            flash("All fields are required.", "danger")
            return render_template("register.html")

        if users.find_one({"email": email}):
            flash("Email already registered.", "warning")
            return render_template("register.html")

        pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        users.insert_one({
            "username": username,
            "email":    email,
            "password": pw_hash,
            "created_at": datetime.now(timezone.utc),
        })
        flash("Account created! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = users.find_one({"email": email})
        if user and bcrypt.checkpw(password.encode(), user["password"]):
            session["user_id"] = str(user["_id"])
            session["username"] = user["username"]
            flash(f"Welcome back, {user['username']}!", "success")
            return redirect(url_for("predict"))

        flash("Invalid email or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("index"))


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    result = None
    if request.method == "POST":
        try:
            values = [float(request.form.get(f, 0)) for f in FEATURES]
            arr    = scaler.transform([values])
            proba  = model.predict_proba(arr)[0]
            idx    = int(np.argmax(proba))
            crop   = le.inverse_transform([idx])[0]
            conf   = round(float(proba[idx]) * 100, 2)
            result = {"crop": crop.title(), "confidence": conf, "values": dict(zip(FEATURES, values))}
        except Exception as e:
            flash(f"Prediction error: {e}", "danger")

    return render_template("predict.html", result=result)


# ── JSON API Endpoints ───────────────────────────────────────────────────────
@app.route("/api/register", methods=["POST"])
def api_register():
    data     = request.get_json(force=True) or {}
    username = data.get("username", "").strip()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not (username and email and password):
        return jsonify({"error": "username, email, and password required"}), 400
    if users.find_one({"email": email}):
        return jsonify({"error": "Email already registered"}), 409

    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    users.insert_one({
        "username": username, "email": email,
        "password": pw_hash,
        "created_at": datetime.now(timezone.utc),
    })
    return jsonify({"message": "User registered successfully"}), 201


@app.route("/api/login", methods=["POST"])
def api_login():
    data     = request.get_json(force=True) or {}
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    user = users.find_one({"email": email})
    if user and bcrypt.checkpw(password.encode(), user["password"]):
        session["user_id"]  = str(user["_id"])
        session["username"] = user["username"]
        return jsonify({"message": "Login successful", "username": user["username"]}), 200

    return jsonify({"error": "Invalid credentials"}), 401


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "user_id" not in session:
        return jsonify({"error": "Authentication required"}), 401

    data = request.get_json(force=True) or {}
    try:
        values = [float(data.get(f, 0)) for f in FEATURES]
    except (ValueError, TypeError):
        return jsonify({"error": "All feature values must be numeric"}), 400

    arr   = scaler.transform([values])
    proba = model.predict_proba(arr)[0]
    idx   = int(np.argmax(proba))
    crop  = le.inverse_transform([idx])[0]
    conf  = round(float(proba[idx]) * 100, 2)
    return jsonify({"crop": crop, "confidence": conf}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
