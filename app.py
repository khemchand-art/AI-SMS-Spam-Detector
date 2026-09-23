from flask import Flask, render_template, request
import os
import joblib

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "improved_sms_spam_model.pkl")
)

features = joblib.load(
    os.path.join(BASE_DIR, "improved_tfidf_features.pkl")
)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    score = None
    message = ""

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if message:
            X = features.transform([message])

            prediction = model.predict(X)[0]

            score = round(
                float(model.decision_function(X)[0]), 3
            )

            result = "SPAM" if prediction == 1 else "HAM"

    return render_template(
        "index.html",
        result=result,
        score=score,
        message=message
    )


if __name__ == "__main__":
    app.run(
        debug=False,
        use_reloader=False,
        host="127.0.0.1",
        port=5000
    )
