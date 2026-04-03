from pathlib import Path
import pickle

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Product details shown on the left side of the page.
PRODUCT = {
    "name": "EchoBuds Pro Wireless Earbuds",
    "image_url": "https://images.unsplash.com/photo-1606220838315-056192d5e927?auto=format&fit=crop&w=900&q=80",
}

MODEL_CANDIDATE_PATHS = [
    Path("sentiment_model.pkl"),
    Path("sentimental_model.pkl"),
    Path("model.pkl"),
    Path("model.pickle"),
    Path("static/model/sentiment_model.pkl"),
    Path("static/model/sentimental_model.pkl"),
    Path("static/model/model.pkl"),
    Path("static/model/model.pickle"),
]
VECTORIZER_CANDIDATE_PATHS = [
    Path("vocab.pkl"),
    Path("vectorizer.pkl"),
    Path("static/model/vocab.pkl"),
    Path("static/model/vectorizer.pkl"),
]


def load_pickle(path: Path):
    """Load a pickle file and return the Python object."""
    with path.open("rb") as file:
        return pickle.load(file)


def load_with_joblib(path: Path):
    """Load an artifact with joblib if available."""
    import joblib

    return joblib.load(path)


def load_first_supported_artifact(
    candidates: list[Path],
    required_attr: str | None = None,
):
    """Try candidate files with pickle and joblib and return the first valid artifact."""
    errors: list[str] = []
    loaders = [
        ("pickle", load_pickle),
        ("joblib", load_with_joblib),
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue

        for loader_name, loader in loaders:
            try:
                artifact = loader(candidate)
                if required_attr and not hasattr(artifact, required_attr):
                    errors.append(
                        f"{candidate} loaded via {loader_name} but missing '{required_attr}'"
                    )
                    continue
                return artifact, candidate, loader_name, errors
            except Exception as exc:
                errors.append(f"{candidate} via {loader_name}: {exc}")

    return None, None, None, errors


# Try to load trained artifacts. If they are not present,
# the app still runs with a beginner-friendly fallback prediction.
model = None
vectorizer = None
model_load_warning = None

try:
    model, model_path, model_loader, model_errors = load_first_supported_artifact(
        MODEL_CANDIDATE_PATHS
    )
    vectorizer, vectorizer_path, vectorizer_loader, vectorizer_errors = load_first_supported_artifact(
        VECTORIZER_CANDIDATE_PATHS,
        required_attr="transform",
    )

    if model is None or vectorizer is None:
        raise FileNotFoundError
except FileNotFoundError:
    model_load_warning = (
        "Model/vectorizer could not be loaded. "
        "Model can be pickle/joblib (sentiment_model.pkl, model.pkl, model.pickle). "
        "Vectorizer must support transform() (for example vectorizer.pkl). "
        "Using fallback prediction until files are added."
    )
except Exception as exc:
    model_load_warning = f"Could not load model files: {exc}"


# In-memory review list (resets when server restarts).
REVIEWS = [
    {"id": 1, "text": "Battery life is amazing and sound quality is super clear."},
    {"id": 2, "text": "The right earbud stops working sometimes. Very frustrating."},
    {"id": 3, "text": "Comfortable fit, fast pairing, and great value for the price."},
    {"id": 4, "text": "Mic quality is poor during calls in noisy places."},
]


# Tiny lexicon used only if model files are unavailable.
FALLBACK_POSITIVE_WORDS = {
    "amazing",
    "awesome",
    "best",
    "clear",
    "comfortable",
    "excellent",
    "fast",
    "good",
    "great",
    "love",
    "perfect",
    "super",
    "value",
}
FALLBACK_NEGATIVE_WORDS = {
    "bad",
    "broken",
    "frustrating",
    "hate",
    "issue",
    "lag",
    "poor",
    "problem",
    "stops",
    "terrible",
    "worse",
    "worst",
}


def normalize_prediction(raw_prediction) -> str:
    """Convert model output into 'Positive' or 'Negative'."""
    text = str(raw_prediction).strip().lower()

    if text in {"1", "positive", "pos", "p"}:
        return "Positive"
    if text in {"0", "negative", "neg", "n"}:
        return "Negative"

    # If model returns unexpected values, default to negative for safety.
    return "Negative"


def fallback_predict(review_text: str) -> str:
    """Simple fallback sentiment predictor based on keyword counting."""
    words = {w.strip(".,!?;:\"'()[]{}") for w in review_text.lower().split()}
    pos_score = len(words & FALLBACK_POSITIVE_WORDS)
    neg_score = len(words & FALLBACK_NEGATIVE_WORDS)
    return "Positive" if pos_score >= neg_score else "Negative"


def predict_sentiment(review_text: str) -> str:
    """Predict sentiment with trained model if available, otherwise fallback."""
    if model is not None and vectorizer is not None:
        features = vectorizer.transform([review_text])
        prediction = model.predict(features)[0]
        return normalize_prediction(prediction)

    return fallback_predict(review_text)


def decorate_review(review: dict, is_latest: bool = False) -> dict:
    """Attach sentiment details used by the frontend."""
    sentiment = predict_sentiment(review["text"])
    return {
        "id": review["id"],
        "text": review["text"],
        "sentiment": sentiment,
        "emoji": "😊" if sentiment == "Positive" else "😞",
        "css_class": "sentiment-positive" if sentiment == "Positive" else "sentiment-negative",
        "is_latest": is_latest,
    }


def calculate_summary(decorated_reviews: list[dict]) -> dict:
    """Count positive and negative reviews."""
    positive_count = sum(1 for item in decorated_reviews if item["sentiment"] == "Positive")
    negative_count = sum(1 for item in decorated_reviews if item["sentiment"] == "Negative")
    return {"positive": positive_count, "negative": negative_count}


@app.route("/", methods=["GET"])
def index():
    decorated_reviews = [decorate_review(item) for item in REVIEWS]
    summary = calculate_summary(decorated_reviews)

    return render_template(
        "index.html",
        product=PRODUCT,
        reviews=decorated_reviews,
        summary=summary,
        model_warning=model_load_warning,
    )


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    review_text = (payload.get("review") or "").strip()

    if not review_text:
        return jsonify({"success": False, "error": "Please enter a review before submitting."}), 400

    new_review = {
        "id": (REVIEWS[-1]["id"] + 1) if REVIEWS else 1,
        "text": review_text,
    }
    REVIEWS.append(new_review)

    decorated_reviews = [decorate_review(item) for item in REVIEWS]
    summary = calculate_summary(decorated_reviews)
    latest = decorate_review(new_review, is_latest=True)

    return jsonify(
        {
            "success": True,
            "review": latest,
            "summary": summary,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
