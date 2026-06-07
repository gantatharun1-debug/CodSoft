# 🤖 RuleBot — Rule-Based Chatbot

A complete, fine-tuned rule-based chatbot built with **Python** and **NLTK**, available as both an interactive Jupyter Notebook and a production-ready **Streamlit** web application.

---

## 📁 Project Structure

```
├── RuleBot_Complete.ipynb      # Master notebook — NLP concepts + chatbot engine + UI widget
└── RuleBot_Streamlit_App.py    # Full-featured web application
```

---

## ✨ Features

- **NLP Pipeline** — tokenization, stemming, lemmatization, and stop word removal (demonstrated step-by-step in the notebook)
- **Intent Matching Engine** — keyword overlap scoring with a configurable confidence threshold
- **Regex-Based Matching** — alternative advanced matching using compiled regular expressions
- **14 Built-in Intents** — greeting, farewell, name, how_are_you, age, creator, weather, joke, help, thanks, time, python, about_ai, compliment, insult, and a default fallback
- **Fine-Tuning Support** — accuracy tested before and after adding patterns; evaluation report included
- **Chat History Logger** — logs every message with timestamps, exportable to `.txt`
- **Streamlit Web UI** — chat bubble interface, quick-reply buttons, session stats sidebar, accuracy panel, and intent explorer
- **Jupyter Widget** — interactive chat UI inside Colab/Jupyter (no terminal needed)

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| NLP Processing | `nltk` (punkt, wordnet, stopwords) |
| Text Lemmatization | `WordNetLemmatizer` |
| Web Application | `streamlit` |
| Notebook UI | `ipywidgets` |
| Data Export | `json`, built-in Python I/O |

---

## ⚙️ Installation

### 1. Clone / download the files

```bash
git clone <your-repo-url>
cd rulebot
```

### 2. Install dependencies

```bash
pip install streamlit nltk ipywidgets gtts
```

NLTK data packages are downloaded automatically on first run (`punkt`, `punkt_tab`, `wordnet`, `stopwords`, `omw-1.4`).

---

## 🚀 Usage

### Option A — Streamlit Web App

```bash
streamlit run RuleBot_Streamlit_App.py
```

Opens automatically at **http://localhost:8501**

**What you get:**
- Messenger-style chat bubbles (blue for user, grey for bot)
- Quick-reply buttons: Hello, Tell a joke, What is Python?, What is AI?, Help
- Sidebar with session stats and list of known topics
- Download chat history as `.txt`
- Expandable accuracy report (runs 15 test cases live)
- Expandable intent explorer to browse all patterns and responses

### Option B — Jupyter / Google Colab Notebook

Open `RuleBot_Complete.ipynb` and run all cells top to bottom with `Shift+Enter`.

| Step Range | What It Covers |
|------------|----------------|
| Steps 1–2 | Install libraries + download NLTK data |
| Steps 3–6 | NLP concepts: tokenization, stemming, lemmatization, stop word removal |
| Steps 7–11 | Build the full chatbot engine (preprocessor → intents → matcher → responder → pipeline) |
| Steps 12–14 | Accuracy test → fine-tune → re-test |
| Step 15 | Save intents to `intents.json` |
| Step 16 | Live terminal-style chat loop |
| Steps 17–18 | Advanced: regex matching + chat history logger |
| Steps 19–20 | Full evaluation report + interactive widget UI |

---

## 🧠 How It Works

### Text Preprocessing

Every user message is cleaned through a five-stage pipeline before matching:

1. Lowercase
2. Remove URLs (`http://...`, `www....`)
3. Strip punctuation and special characters
4. Tokenize into individual words
5. Lemmatize to root form (optionally remove stop words)

### Intent Matching

```
score = 0
for each pattern in the intent:
    if pattern is a substring of the raw input  → +10 points
    score += number of lemma tokens shared with the pattern
```

The intent with the highest score wins. If the score is below the confidence threshold (`CONFIDENCE_MIN = 3`), the bot falls back to the `default` intent.

### Response Generation

Once an intent is matched, a response is chosen at **random** from that intent's `responses` list, giving natural variety to repeated questions.

---

## 📊 Accuracy

The built-in evaluation runs **15 labeled test cases** and reports pass/fail per case. After fine-tuning in the notebook, typical accuracy reaches **93%+**. The Streamlit app surfaces this report under the *"View Chatbot Accuracy Report"* expander.

---

## 🗂️ Adding New Intents

To teach RuleBot a new topic, add an entry to the `INTENTS` dictionary (in either file):

```python
'your_topic': {
    'patterns': [
        'phrase one', 'phrase two', 'keyword'
    ],
    'responses': [
        'First possible reply.',
        'Second possible reply.',
    ]
},
```

More patterns = higher matching accuracy. Run the evaluation after adding patterns to verify.

---

## 💾 Chat History Export

In the **notebook**, call `save_history()` after a chat session to write `chat_history.txt`.

In the **Streamlit app**, click **📥 Download Chat** in the sidebar (appears after the first message is sent). The file is named `rulebot_chat_YYYYMMDD_HHMM.txt`.

---

## 📌 Known Limitations

- No memory between sessions — the bot does not remember previous conversations
- No live data access — weather, time, and stock queries are redirected to external sources
- Pattern matching only — the bot does not understand context or multi-turn dialogue
- Regex matching (Step 17 in notebook) is a separate, independent engine and is not used by the Streamlit app by default

---

## 🚀 Next Steps / Ideas

- Integrate a knowledge base from `intents.json` for dynamic intent loading
- Add a database or file-backed session store for conversation history
- Swap the rule-based matcher for a lightweight ML classifier (e.g., scikit-learn `TfidfVectorizer` + `LogisticRegression`)
- Deploy to Streamlit Cloud, Heroku, or a Docker container

---

## 📄 License

This project is for educational purposes. Feel free to extend and adapt it for your own learning or projects.
