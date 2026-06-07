# ============================================================
#  RuleBot — Streamlit Web Application
#  Fine-tuned Rule-Based Chatbot with full UI
# ============================================================
#
#  HOW TO RUN:
#  1. Install dependencies:
#       pip install streamlit nltk
#  2. Run in terminal:
#       streamlit run RuleBot_Streamlit_App.py
#  3. A browser window opens automatically at http://localhost:8501
#
# ============================================================

# ============================================================
#  SECTION 1 — IMPORTS & SETUP
# ============================================================

import streamlit as st
import random
import re
import json
import os
import nltk
from datetime import datetime

# Download all required NLTK packages silently on first run
for pkg in ['punkt', 'punkt_tab', 'wordnet', 'stopwords', 'omw-1.4']:
    nltk.download(pkg, quiet=True)

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# ============================================================
#  SECTION 2 — INTENTS DICTIONARY (Fine-Tuned)
#  Add more patterns to any intent to improve accuracy
# ============================================================

INTENTS = {

    'greeting': {
        'patterns': [
            'hello', 'hi', 'hey', 'good morning', 'good evening', 'good afternoon',
            'howdy', 'greetings', 'whats up', 'sup', 'hiya', 'yo', 'ello',
            'morning', 'afternoon', 'evening', 'hi there', 'hello there',
            'hey there', 'good day', 'what is up', 'hi everyone', 'good to see you'
        ],
        'responses': [
            'Hello! 👋 How can I help you today?',
            'Hi there! What can I do for you? 😊',
            'Hey! Great to see you. How can I assist?',
            'Greetings! I am RuleBot — ask me anything!'
        ]
    },

    'farewell': {
        'patterns': [
            'bye', 'goodbye', 'see you', 'see ya', 'take care', 'later',
            'ciao', 'farewell', 'ttyl', 'talk later', 'going now',
            'leaving', 'im off', 'see you soon', 'good night', 'goodnight',
            'night', 'catch you later', 'see you around'
        ],
        'responses': [
            'Goodbye! 👋 Have a great day!',
            'See you later! Take care 😊',
            'Bye! Come back anytime you need help!',
            'Farewell! It was nice chatting with you.'
        ]
    },

    'name': {
        'patterns': [
            'what is your name', 'who are you', 'your name',
            'what should i call you', 'introduce yourself',
            'what do i call you', 'tell me your name',
            'what are you called', 'may i know your name',
            'your identity', 'tell me who you are', 'what are you'
        ],
        'responses': [
            'I am RuleBot 🤖 — your rule-based assistant!',
            'My name is RuleBot. Nice to meet you!',
            'You can call me RuleBot. How can I help?'
        ]
    },

    'how_are_you': {
        'patterns': [
            'how are you', 'how do you do', 'hows it going',
            'how are you doing', 'are you okay', 'you good',
            'hows things', 'feeling okay', 'you alright',
            'doing okay', 'how have you been', 'how is it going',
            'how are things', 'how is life', 'all good with you'
        ],
        'responses': [
            'I am doing great, thanks for asking! 😄',
            'All systems running perfectly! How about you?',
            'I am just a bot, but I feel wonderful! 🤖'
        ]
    },

    'age': {
        'patterns': [
            'how old are you', 'what is your age', 'your age',
            'how young are you', 'when were you born', 'age'
        ],
        'responses': [
            'I am ageless — I was just created! 🤖',
            'Age does not apply to bots, but I am brand new!'
        ]
    },

    'creator': {
        'patterns': [
            'who made you', 'who created you', 'who built you',
            'who is your creator', 'who developed you',
            'who programmed you', 'who designed you'
        ],
        'responses': [
            'I was built by a Python developer learning AI! 🐍',
            'A smart programmer created me using Python and NLTK!'
        ]
    },

    'weather': {
        'patterns': [
            'weather', 'temperature', 'is it raining',
            'hows the weather', 'forecast', 'sunny', 'cloudy',
            'will it rain', 'is it hot', 'is it cold'
        ],
        'responses': [
            'I cannot check live weather, but try weather.com! 🌤',
            'I do not have internet access, but Google Weather works great! ☀'
        ]
    },

    'joke': {
        'patterns': [
            'tell me a joke', 'joke', 'make me laugh', 'say something funny',
            'humor me', 'funny', 'comedy', 'laugh', 'entertain me',
            'crack a joke', 'be funny', 'got any jokes',
            'got a joke', 'share a joke', 'any jokes', 'cheer me up'
        ],
        'responses': [
            'Why do programmers prefer dark mode? Because light attracts bugs! 🐛😂',
            'Why was the computer cold? It left its Windows open! 😄',
            'I told my computer I needed a break. Now it keeps sending me Kit Kat ads! 🍫',
            'Why do Java developers wear glasses? Because they do not C#! 😂',
            'What do you call a fish with no eyes? A fsh! 🐟😂'
        ]
    },

    'help': {
        'patterns': [
            'help', 'what can you do', 'assist me', 'what do you know',
            'capabilities', 'features', 'how can you help',
            'what topics do you know', 'what do you cover'
        ],
        'responses': [
            'I can chat, tell jokes, answer basic questions, and more! Try asking me anything 😊',
            'Here is what I can do: greet you, tell jokes, answer FAQs, and chat!'
        ]
    },

    'thanks': {
        'patterns': [
            'thank you', 'thanks', 'thank you so much', 'thx', 'ty',
            'appreciate it', 'many thanks', 'much appreciated',
            'appreciate that', 'cheers', 'thanks a lot', 'thanks a bunch',
            'thank you very much', 'so grateful'
        ],
        'responses': [
            'You are welcome! 😊',
            'Happy to help! Anything else?',
            'No problem at all! 🤖'
        ]
    },

    'time': {
        'patterns': [
            'what time is it', 'current time', 'tell me the time',
            'time now', 'whats the time', 'do you know the time'
        ],
        'responses': [
            'I do not have access to real-time data, but your device clock does! ⏰'
        ]
    },

    'python': {
        'patterns': [
            'python', 'what is python', 'tell me about python',
            'python programming', 'learn python', 'python language',
            'is python good', 'why python', 'python basics',
            'python tutorial', 'how to learn python'
        ],
        'responses': [
            'Python is a powerful, beginner-friendly language used in AI and data science! 🐍',
            'Python is great for AI! It has tons of libraries like NumPy and TensorFlow!'
        ]
    },

    'about_ai': {
        'patterns': [
            'what is ai', 'artificial intelligence', 'machine learning',
            'deep learning', 'neural network', 'tell me about ai',
            'what is machine learning', 'what is deep learning',
            'tell me about machine learning', 'explain ai',
            'what are neural networks', 'how does ai work'
        ],
        'responses': [
            'AI is the simulation of human intelligence by machines! 🤖',
            'Machine Learning is a branch of AI where machines learn from data!',
            'Deep Learning uses neural networks to solve complex problems like image recognition!'
        ]
    },

    'compliment': {
        'patterns': [
            'you are great', 'well done', 'good job', 'you are awesome',
            'nice work', 'brilliant', 'impressive', 'you are smart',
            'you are the best', 'love you', 'amazing'
        ],
        'responses': [
            'Thank you so much! You are very kind 😊',
            'That means a lot! I try my best 🤖',
            'Aw, thanks! You made my circuits smile 😄'
        ]
    },

    'insult': {
        'patterns': [
            'you are stupid', 'you are dumb', 'you are useless',
            'you are bad', 'you are terrible', 'i hate you',
            'you are trash', 'you are garbage'
        ],
        'responses': [
            'I am sorry to hear that. I am still learning! 😔',
            'That is okay, I will try to do better! 🙂',
            'I understand. Let me know how I can improve!'
        ]
    },

    'default': {
        'patterns': [],
        'responses': [
            'Hmm, I did not understand that. Try asking something else! 🤔',
            'I am not sure about that. Could you rephrase? 😅',
            'That is beyond my rules. Try: say hello, ask for a joke, or ask about Python!'
        ]
    }
}


# ============================================================
#  SECTION 3 — NLP ENGINE
# ============================================================

lemmatizer     = WordNetLemmatizer()
STOP_WORDS     = set(stopwords.words('english'))
CONFIDENCE_MIN = 3  # Minimum score to accept a match


def preprocess(text, remove_stopwords=False):
    """
    Full text preprocessing pipeline:
    1. Lowercase
    2. Remove URLs and special characters
    3. Tokenize
    4. Optionally remove stop words
    5. Lemmatize
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def match_intent(user_input):
    """
    Scores all intents against the user input.
    Returns (best_intent, score).
    Score < CONFIDENCE_MIN → returns 'default'.
    """
    tokens    = preprocess(user_input)
    user_lower = user_input.lower()

    best_intent = 'default'
    best_score  = 0

    for intent, data in INTENTS.items():
        if intent == 'default':
            continue
        score = 0
        for pattern in data['patterns']:
            if pattern in user_lower:
                score += 10
            score += len(set(preprocess(pattern)) & set(tokens))
        if score > best_score:
            best_score  = score
            best_intent = intent

    if best_score < CONFIDENCE_MIN:
        best_intent = 'default'

    return best_intent, best_score


def get_bot_response(user_input):
    """Main chatbot function — input string → reply string."""
    if not user_input.strip():
        return 'Please type something! 👂'
    intent, _ = match_intent(user_input)
    return random.choice(INTENTS[intent]['responses'])


# ============================================================
#  SECTION 4 — STREAMLIT PAGE CONFIG & STYLING
# ============================================================

st.set_page_config(
    page_title='RuleBot — AI Chatbot',
    page_icon='🤖',
    layout='centered'
)

# Custom CSS for chat bubble styling
st.markdown("""
<style>
    /* Overall page */
    .main { background-color: #f0f2f6; }

    /* User bubble — right aligned, blue */
    .user-bubble {
        background: #0084ff;
        color: white;
        padding: 10px 16px;
        border-radius: 20px 20px 4px 20px;
        max-width: 75%;
        margin: 6px 0 6px auto;
        text-align: right;
        font-size: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.15);
        display: inline-block;
        float: right;
        clear: both;
    }

    /* Bot bubble — left aligned, light grey */
    .bot-bubble {
        background: #e4e6eb;
        color: #1c1e21;
        padding: 10px 16px;
        border-radius: 20px 20px 20px 4px;
        max-width: 75%;
        margin: 6px auto 6px 0;
        font-size: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.10);
        display: inline-block;
        float: left;
        clear: both;
    }

    /* Timestamp label */
    .msg-time {
        font-size: 10px;
        color: #999;
        clear: both;
        padding: 0 4px;
    }

    /* Metric card */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        text-align: center;
    }

    /* Sidebar header */
    .sidebar-title {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    /* Input box tweaks */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #0084ff;
        padding: 10px 18px;
        font-size: 15px;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
#  SECTION 5 — SESSION STATE INITIALIZATION
# ============================================================

if 'messages' not in st.session_state:
    # Pre-load with bot welcome message
    st.session_state.messages = [
        {
            'role': 'bot',
            'text': 'Hi! I am RuleBot 🤖 — your rule-based assistant. Say hello, ask for a joke, ask about Python, or type **help** to see what I can do!',
            'time': datetime.now().strftime('%H:%M')
        }
    ]

if 'msg_count' not in st.session_state:
    st.session_state.msg_count = 0

if 'last_intent' not in st.session_state:
    st.session_state.last_intent = '-'


# ============================================================
#  SECTION 6 — SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('## 🤖 RuleBot')
    st.markdown('A rule-based chatbot built with Python + NLTK')
    st.divider()

    # Stats
    st.markdown('### 📊 Session Stats')
    col1, col2 = st.columns(2)
    with col1:
        st.metric('Messages', st.session_state.msg_count)
    with col2:
        st.metric('Intents', len(INTENTS) - 1)

    st.markdown(f'**Last intent matched:** `{st.session_state.last_intent}`')
    st.divider()

    # Topics the bot knows
    st.markdown('### 💬 Topics I Know')
    topics = [k for k in INTENTS.keys() if k != 'default']
    for t in topics:
        st.markdown(f'- {t.replace("_", " ").title()}')
    st.divider()

    # Clear chat button
    if st.button('🗑️ Clear Chat', use_container_width=True):
        st.session_state.messages = [
            {
                'role': 'bot',
                'text': 'Chat cleared! Start fresh — say hello! 👋',
                'time': datetime.now().strftime('%H:%M')
            }
        ]
        st.session_state.msg_count   = 0
        st.session_state.last_intent = '-'
        st.rerun()

    # Download chat history
    if st.session_state.msg_count > 0:
        history_text = '=== RuleBot Chat History ===\n\n'
        for msg in st.session_state.messages:
            role = 'You' if msg['role'] == 'user' else 'Bot'
            history_text += '[{}] {}: {}\n'.format(msg['time'], role, msg['text'])

        st.download_button(
            label='📥 Download Chat',
            data=history_text,
            file_name='rulebot_chat_{}.txt'.format(datetime.now().strftime('%Y%m%d_%H%M')),
            mime='text/plain',
            use_container_width=True
        )

    st.divider()
    st.caption('Built with Python · NLTK · Streamlit')


# ============================================================
#  SECTION 7 — MAIN CHAT AREA
# ============================================================

st.title('🤖 RuleBot Chat')
st.caption('A fine-tuned rule-based chatbot — powered by Python and NLTK')
st.divider()

# ---- Render all chat messages ----
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            st.markdown(
                "<div class='user-bubble'>{}</div>"
                "<div class='msg-time' style='text-align:right'>{}</div>".format(
                    msg['text'], msg['time']
                ),
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='bot-bubble'>{}</div>"
                "<div class='msg-time'>{}</div>".format(
                    msg['text'], msg['time']
                ),
                unsafe_allow_html=True
            )


# ============================================================
#  SECTION 8 — INPUT BOX & SEND LOGIC
# ============================================================

st.divider()

# Quick-reply buttons
st.markdown('**Quick replies:**')
quick_cols = st.columns(5)
quick_replies = ['Hello 👋', 'Tell a joke 😂', 'What is Python? 🐍', 'What is AI? 🤖', 'Help 🆘']

for i, label in enumerate(quick_replies):
    if quick_cols[i].button(label, use_container_width=True):
        # Strip emoji for the actual message
        clean = label.split(' ')[0] + (' ' + ' '.join(label.split(' ')[1:-1]) if len(label.split(' ')) > 2 else '')
        clean = re.sub(r'[^\w\s\?]', '', label).strip()

        # Save user message
        st.session_state.messages.append({
            'role': 'user',
            'text': clean,
            'time': datetime.now().strftime('%H:%M')
        })

        # Get and save bot reply
        intent, score = match_intent(clean)
        reply = get_bot_response(clean)
        st.session_state.messages.append({
            'role': 'bot',
            'text': reply,
            'time': datetime.now().strftime('%H:%M')
        })
        st.session_state.msg_count   += 1
        st.session_state.last_intent  = intent
        st.rerun()


# Text input form — pressing Enter submits it
with st.form(key='chat_form', clear_on_submit=True):
    col_input, col_send = st.columns([5, 1])
    with col_input:
        user_text = st.text_input(
            label='Message',
            placeholder='Type your message here...',
            label_visibility='collapsed'
        )
    with col_send:
        submitted = st.form_submit_button('Send', use_container_width=True)

if submitted and user_text.strip():
    # Save user message
    st.session_state.messages.append({
        'role': 'user',
        'text': user_text.strip(),
        'time': datetime.now().strftime('%H:%M')
    })

    # Match intent and get reply
    intent, score = match_intent(user_text.strip())
    reply = get_bot_response(user_text.strip())

    # Save bot reply
    st.session_state.messages.append({
        'role': 'bot',
        'text': reply,
        'time': datetime.now().strftime('%H:%M')
    })

    # Update session stats
    st.session_state.msg_count   += 1
    st.session_state.last_intent  = intent

    st.rerun()


# ============================================================
#  SECTION 9 — ACCURACY PANEL (Expandable)
# ============================================================

with st.expander('🎯 View Chatbot Accuracy Report'):
    test_cases = [
        ('hello',                 'greeting'),
        ('good morning',         'greeting'),
        ('bye',                  'farewell'),
        ('see you later',        'farewell'),
        ('what is your name',    'name'),
        ('who are you',          'name'),
        ('how are you doing',    'how_are_you'),
        ('are you okay',         'how_are_you'),
        ('tell me a joke',       'joke'),
        ('make me laugh',        'joke'),
        ('thank you',            'thanks'),
        ('what can you do',      'help'),
        ('tell me about python', 'python'),
        ('what time is it',      'time'),
        ('xyzabc random stuff',  'default'),
    ]

    correct = 0
    rows    = []
    for msg, expected in test_cases:
        predicted, score = match_intent(msg)
        ok = (predicted == expected)
        if ok:
            correct += 1
        rows.append({
            'Input'    : msg,
            'Expected' : expected,
            'Got'      : predicted,
            'Score'    : score,
            'Result'   : '✅ PASS' if ok else '❌ FAIL'
        })

    accuracy = correct / len(test_cases) * 100

    st.metric('Accuracy', '{:.1f}%'.format(accuracy),
              delta='{}/{} passed'.format(correct, len(test_cases)))

    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if accuracy >= 90:
        st.success('Excellent! The chatbot is well-tuned.')
    elif accuracy >= 75:
        st.warning('Good — add more patterns to the FAIL rows to push higher.')
    else:
        st.error('Needs improvement — add patterns for every FAIL row.')


# ============================================================
#  SECTION 10 — INTENT EXPLORER (Expandable)
# ============================================================

with st.expander('🗂️ Browse All Intents & Patterns'):
    intent_filter = st.selectbox(
        'Select an intent to inspect:',
        options=[k for k in INTENTS.keys() if k != 'default']
    )
    data = INTENTS[intent_filter]

    col_p, col_r = st.columns(2)
    with col_p:
        st.markdown('**Patterns ({}):**'.format(len(data['patterns'])))
        for p in data['patterns']:
            st.markdown('- `{}`'.format(p))
    with col_r:
        st.markdown('**Responses ({}):**'.format(len(data['responses'])))
        for r in data['responses']:
            st.markdown('- {}'.format(r))
