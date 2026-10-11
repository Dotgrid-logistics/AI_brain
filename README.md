# 🧠 AI Brain: Banking Intent Chatbot
 
A simple AI "brain" that **understands** what a customer is asking and **responds**. It classifies each message into one of 77 banking intents (for example `card_arrival`, `lost_or_stolen_card`, `refund_not_showing_up`), then replies using either a template or the Gemini API.
 
**Live demo:** `PASTE-YOUR-STREAMLIT-LINK-HERE`
 
> ⚠️ This is a learning project. Don't enter real bank details or personal information.
 
---
 
## What it does
 
- Takes a free-text message such as *"my card still hasn't arrived"*
- Predicts the customer's **intent** with a classic machine-learning model
- Reports how **confident** it is, and asks the user to rephrase if it isn't confident enough
- Generates a reply, either from a template or a natural-sounding answer from Gemini (toggle in the sidebar)
## How it works
 
```
User message
     │
     ▼
 Clean text  (lowercase, strip punctuation)
     │
     ▼
 TF-IDF vectorizer  (words + word pairs → numbers)
     │
     ▼
 Classifier  (Logistic Regression or Naive Bayes, whichever scored best on validation)
     │
     ▼
 Confidence check ── below threshold ──► "Sorry, I didn't catch that."
     │
     ▼
 Predicted intent ──► template reply  or  Gemini reply
```
 
| Stage | What it does |
|---|---|
| **Cleaning** | Lowercases text, removes punctuation, drops duplicates and empty rows |
| **TF-IDF** | Turns each sentence into numbers, giving more weight to distinctive words. Uses unigrams and bigrams |
| **Model selection** | Trains Logistic Regression and Naive Bayes, keeps the better one on a validation split |
| **Confidence threshold** | Tuned on validation data to trade off how often the bot answers against how often it is right |
| **Final evaluation** | The model is retrained on all training data and tested **once** on the untouched `test.csv` |
| **Replies** | Template replies by default; Gemini generates a natural reply when enabled |
 
## Results
 
| Metric | Value |
|---|---|
| Intents | 77 |
| Validation accuracy | `88.5%` |
| Final test accuracy | `89.3%` |
| Confidence threshold | `0.4` |
 
## Dataset
 
[**Banking77**](https://huggingface.co/datasets/PolyAI/banking77) by PolyAI: about 13,000 customer-service queries labeled with 77 intents.
 
Casanueva et al. (2020), *Efficient Intent Detection with Dual Sentence Encoders.* Check the dataset page for its license and citation details.
 
## Tech stack
 
- Python, pandas, NumPy
- scikit-learn (TF-IDF, Logistic Regression, Naive Bayes)
- Streamlit (web interface and hosting)
- Google Gemini API (optional natural replies)
- Google Colab (training)
## Project structure
 
```
ai-brain/
├── app.py              # Streamlit app
├── ai_brain.joblib     # Saved vectorizer + model + threshold
├── requirements.txt    # Dependencies (scikit-learn version is pinned)
├── notebook/           # (optional) the Colab notebook used for training
└── README.md
```
 
## Run it locally
 
```bash
git clone https://github.com/YOUR-USERNAME/ai-brain.git
cd ai-brain
pip install -r requirements.txt
streamlit run app.py
```
 
To enable Gemini replies locally, create `.streamlit/secrets.toml`:
 
```toml
GEMINI_API_KEY = "A generated Gemini API key"
I can't share mine.
```
 
**Never commit this file.** Add `.streamlit/secrets.toml` to your `.gitignore`. Without a key, the app still works using template replies.
 
## Retrain the model
 
1. Open the notebook in Google Colab.
2. Put `train.csv` and `test.csv` in your Google Drive and set the folder path in the notebook.
3. Run all cells. The last saving step writes a new `ai_brain.joblib`.
4. Replace the file in this repo. Make sure the `scikit-learn` version in `requirements.txt` matches the one used in Colab.
## Deploy
 
Hosted on [Streamlit Community Cloud](https://share.streamlit.io). Add `GEMINI_API_KEY` under the app's **Secrets** settings.
 
## Limitations
 
- Only recognizes the 77 banking intents it was trained on. Anything else should fall below the threshold and trigger the fallback message.
- TF-IDF matches words, not meaning, so unusual phrasing can fool it.
- No conversation memory. Each message is classified on its own.
- Banking77 has some noisy labels, which caps achievable accuracy.
- Gemini replies depend on API availability and free-tier rate limits. The app falls back to template replies if the call fails.
## Ideas for improvement
 
- Replace TF-IDF with sentence embeddings or fine-tune a small BERT model
- Add conversation memory
- Train on out-of-scope examples (for example from CLINC150) to improve the fallback
- Write specific, hand-crafted replies for the most common intents

 
Built by **Praise**.
`ADD-YOUR-GITHUB / LINKEDIN / EMAIL HERE`
 
