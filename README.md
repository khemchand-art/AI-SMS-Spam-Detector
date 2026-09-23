# AI Based Real-Time SMS Spam Detection System

Class 12 project using TF-IDF features and a Linear SVM classifier.

Required model files in the project root:
- improved_sms_spam_model.pkl
- improved_tfidf_features.pkl

Run locally:
pip install -r requirements.txt
python app.py

Then open http://127.0.0.1:5000

Production start command:
gunicorn app:app

The SVM decision score displayed by the app is a decision-function score, not a calibrated probability.
