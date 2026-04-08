from pathlib import Path
import pickle

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


class CustomVectorizer:
    """Compatibility shim for pickled vectorizers created in notebooks.

    Some training notebooks save a custom class named `CustomVectorizer`
    under `__main__`. Defining it here allows pickle/joblib to deserialize
    those artifacts inside the Flask app.
    """

    def transform(self, texts):
        # Try common wrapped vectorizer attribute names.
        for attr in ("vectorizer", "tfidf", "cv", "base_vectorizer"):
            inner = getattr(self, attr, None)
            if inner is not None and hasattr(inner, "transform"):
                return inner.transform(texts)

        # Handle case where the object stores vocabulary directly.
        vocabulary = getattr(self, "vocabulary_", None) or getattr(self, "vocabulary", None)
        if vocabulary:
            from sklearn.feature_extraction.text import CountVectorizer

            if not hasattr(self, "_fallback_vectorizer"):
                self._fallback_vectorizer = CountVectorizer(vocabulary=vocabulary)
            return self._fallback_vectorizer.transform(texts)

        raise AttributeError("CustomVectorizer has no usable transform() backend")


# Expose compatibility class where legacy pickles expect it.
import __main__

__main__.CustomVectorizer = CustomVectorizer

# Product details shown on the left side of the page.
PRODUCT = {
    "name": "EchoBuds Pro Wireless Earbuds",
    "image_url": "/static/images/echobuds.jpg",
}

MODEL_CANDIDATE_PATHS = [
    Path("static/model/sentiment_model.pkl"),
    Path("static/model/model.pkl"),
    Path("static/model/model.pickle"),
    Path("notebooks/static/model/model.pickle"),
]
VECTORIZER_CANDIDATE_PATHS = [
    Path("static/model/vectorizer.pkl"),
    Path("static/model/vectorizer.pickle"),
    Path("static/model/tfidf_vectorizer.pkl"),
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
PREDICTION_LABEL_MAP = {
    "1": "Positive",
    "0": "Negative",
}

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

if model is None or vectorizer is None:
    error_details = []
    if "model_errors" in locals() and model_errors:
        error_details.append(f"model errors: {' | '.join(model_errors)}")
    if "vectorizer_errors" in locals() and vectorizer_errors:
        error_details.append(f"vectorizer errors: {' | '.join(vectorizer_errors)}")
    if error_details:
        model_load_warning = (model_load_warning or "") + " " + " ".join(error_details)


def infer_prediction_label_map(loaded_model, loaded_vectorizer) -> dict[str, str]:
    """Infer how binary model class labels map to sentiment words.

    Many sentiment models are trained with either 1=positive or 1=negative.
    This function probes the loaded pipeline with strong anchor sentences and
    derives a reliable mapping when possible.
    """
    default_map = {"1": "Positive", "0": "Negative"}

    if loaded_model is None or loaded_vectorizer is None:
        return default_map

    try:
        if not hasattr(loaded_model, "predict"):
            return default_map

        classes = list(getattr(loaded_model, "classes_", []))
        if len(classes) != 2:
            return default_map

        positive_probe = "excellent great best awesome value comfortable clear sound"
        negative_probe = "worst broken terrible bad poor frustrating issue"
        features = loaded_vectorizer.transform([positive_probe, negative_probe])
        pos_raw, neg_raw = loaded_model.predict(features)

        # If probes separate cleanly, map their predicted labels directly.
        if str(pos_raw) != str(neg_raw):
            inferred = {
                str(pos_raw): "Positive",
                str(neg_raw): "Negative",
            }
            # Include numeric aliases so values like 0/1 and bools still resolve.
            if str(pos_raw) == "0":
                inferred["False"] = "Positive"
            if str(pos_raw) == "1":
                inferred["True"] = "Positive"
            if str(neg_raw) == "0":
                inferred["False"] = "Negative"
            if str(neg_raw) == "1":
                inferred["True"] = "Negative"
            return inferred

        # If both probes land in one class, try coefficient semantics for linear models.
        if hasattr(loaded_model, "coef_") and len(getattr(loaded_model, "coef_", [])) > 0:
            coef = loaded_model.coef_[0]

            positive_terms = {
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
            negative_terms = {
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

            vocab_index = getattr(loaded_vectorizer, "vocab_index", None)
            if not isinstance(vocab_index, dict):
                vocab_index = getattr(loaded_vectorizer, "vocabulary_", None)

            if isinstance(vocab_index, dict):
                pos_weights = [float(coef[idx]) for term, idx in vocab_index.items() if term in positive_terms]
                neg_weights = [float(coef[idx]) for term, idx in vocab_index.items() if term in negative_terms]

                if pos_weights and neg_weights:
                    pos_mean = sum(pos_weights) / len(pos_weights)
                    neg_mean = sum(neg_weights) / len(neg_weights)
                    class_one_sentiment = "Negative" if neg_mean > pos_mean else "Positive"
                    class_zero_sentiment = "Positive" if class_one_sentiment == "Negative" else "Negative"
                    return {"1": class_one_sentiment, "0": class_zero_sentiment}

        # If still unresolved, fall back to explicit class names.
        class_names = {str(c).strip().lower() for c in classes}
        if class_names & {"positive", "pos", "p"} and class_names & {
            "negative",
            "neg",
            "n",
        }:
            return {
                str(c): "Positive"
                if str(c).strip().lower() in {"positive", "pos", "p"}
                else "Negative"
                for c in classes
            }
    except Exception:
        return default_map

    return default_map


PREDICTION_LABEL_MAP = infer_prediction_label_map(model, vectorizer)


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

    mapped = PREDICTION_LABEL_MAP.get(str(raw_prediction)) or PREDICTION_LABEL_MAP.get(text)
    if mapped in {"Positive", "Negative"}:
        return mapped

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
        normalized = normalize_prediction(prediction)

        # Fall back to lexicon if model confidence is too low for this sentence.
        confidence = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(features)[0]
                confidence = float(max(probs))
            except Exception:
                confidence = None

        if confidence is not None and confidence < 0.80:
            return fallback_predict(review_text)

        return normalized

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
