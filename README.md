# Sentimental_Analysis

Flask web app for product review sentiment analysis using a trained model.

## Project Structure

```text
Sentimental_Analysis/
|-- app.py
|-- model.pkl
|-- vectorizer.pkl
|-- static/
|   |-- style.css
|   |-- model/
|       |-- model.pkl (optional alternate location)
|       |-- vectorizer.pkl (optional alternate location)
`-- templates/
	 `-- index.html
```

## Features

- Product details panel (image + name)
- Customer reviews panel with sentiment tags
- Green/Red sentiment coloring (Positive/Negative)
- Live sentiment summary counters
- Add new review with dynamic UI update
- Empty input validation
- Latest review highlight animation

## Run Instructions

1. Open the project folder in VS Code.
2. Activate your virtual environment:
	- Windows PowerShell: `./env/Scripts/Activate.ps1`
3. Install dependencies:
	- `pip install flask`
4. Place your trained files in one of these locations:
	- `model.pkl` and `vectorizer.pkl` in project root (recommended)
	- OR inside `static/model/`
5. Start the app:
	- `python app.py`
6. Open browser:
	- `http://127.0.0.1:5000/`

## Notes

- If model files are missing, the app still runs using a simple fallback predictor.
- Review data is stored in memory and resets when the server restarts.
