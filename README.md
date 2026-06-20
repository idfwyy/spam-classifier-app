# 📧 SMS Spam Classifier — Live App

An interactive web app that classifies SMS messages as spam or ham, powered by a Multinomial Naive Bayes model trained on the SMS Spam Collection dataset.

**👉 Try it live:** 

## What it does

- Type or paste a message.
- Click Predict.
- The model returns SPAM or HAM with a confidence score.
- Also shows the probability breakdown for both classes.

## Tech stack

- Python
- scikit-learn (Multinomial Naive Bayes, TF-IDF)
- Streamlit (web UI + deployment)
- joblib (model persistence)

## Model details

- **Accuracy:** 96.9% on held-out test data
- **Spam precision:** 1.00 (zero false alarms)
- **Spam recall:** 0.76 (catches 76% of spam)
- **Dataset:** SMS Spam Collection (~5,500 messages, 2002 British SMS)

## Known limitations

The model was trained on 2002-era SMS data. It struggles with modern spam patterns (iPhone scams, phishing URLs, account verification cons) because those vocabularies don't exist in the training data. This is a real ML phenomenon called **distribution shift** — addressed in production by continuously retraining on fresh data.

The app surfaces this honestly in its UI so users understand what the model can and can't do.

## Companion project

The full training notebook (data exploration, model training, evaluation) is in a separate repo: [email_spam_classifier](https://github.com/idfwyy/email_spam_classifier)

## Run locally

```bash
git clone https://github.com/idfwyy/spam-classifier-app.git
cd spam-classifier-app
pip install -r requirements.txt
streamlit run app.py
```

---

Built by Rahul. Part of my journey toward AI engineering. 🚀