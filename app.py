import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Next Word Prediction | RNN & LSTM",
    page_icon="🧠",
    layout="centered",
)

st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #000000;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #333333;
    }

    div.stButton > button:first-child {
        background-color: #2563EB;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: background-color 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #1D4ED8;
        color: #FFFFFF;
    }

    .result-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0 1.25rem 0;
        font-size: 1.05rem;
    }
    .result-card b {
        color: #2563EB;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    lstm_model = tf.keras.models.load_model("lstm_model.keras")
    rnn_model = tf.keras.models.load_model("rnn_model.keras")
    return rnn_model, lstm_model


@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


@st.cache_resource
def load_max_len():
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    return max_len


rnn_model, lstm_model = load_models()
tokenizer = load_tokenizer()
max_len = load_max_len()

index_to_word = {index: word for word, index in tokenizer.word_index.items()}
MODELS = {"LSTM": lstm_model, "SimpleRNN": rnn_model}

def predict_next_word(model, text: str, top_k: int = 1):
    text = text.lower()
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding="pre")

    probs = model.predict(seq, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_words = [(index_to_word.get(i, ""), float(probs[i])) for i in top_indices]
    best_word = top_words[0][0] if top_words else ""
    return best_word, top_words


def generate_text(model, seed_text: str, n_words: int) -> str:
    text = seed_text
    for _ in range(n_words):
        next_word, _ = predict_next_word(model, text, top_k=1)
        if not next_word:
            break
        text += " " + next_word
    return text


with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.radio("Choose a model", list(MODELS.keys()), index=0)
    num_words = st.slider("Words to generate", min_value=1, max_value=20, value=5)
    show_probs = st.checkbox("Show top-5 word probabilities", value=True)
    st.markdown("---")
    st.caption(
        "Both models were trained on a dataset of famous quotes to predict "
        "the next word given some starting text."
    )

model = MODELS[model_choice]

st.title("🧠 Next Word Prediction")
st.caption("Predict the next word(s) using a SimpleRNN or LSTM model trained on quotes.")

seed_text = st.text_input(
    "Enter your starting text",
    value="The world as we",
    placeholder="Type a phrase...",
)

col1, col2 = st.columns(2)
predict_clicked = col1.button("🔮 Predict Next Word", use_container_width=True)
generate_clicked = col2.button("✨ Generate Text", use_container_width=True)

st.markdown("---")

if predict_clicked:
    if not seed_text.strip():
        st.warning("Please enter some text first.")
    else:
        best_word, top_words = predict_next_word(model, seed_text, top_k=5)

        st.markdown(
            f'<div class="result-card">{seed_text.strip()} → <b>{best_word}</b></div>',
            unsafe_allow_html=True,
        )

        if show_probs and top_words:
            st.subheader("Top 5 candidate words")

            df = pd.DataFrame(top_words, columns=["Word", "Probability"])
            df["Probability"] = (df["Probability"] * 100).round(3).astype(str) + "%"
            df.index = df.index + 1
            st.table(df)

if generate_clicked:
    if not seed_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner(f"Generating {num_words} word(s) with {model_choice}..."):
            result = generate_text(model, seed_text, num_words)

        st.markdown(
            f'<div class="result-card">{result}</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")
st.caption(
    f"Model input length: {max_len} tokens (padded) · Vocabulary size: "
    f"{tokenizer.num_words or len(tokenizer.word_index)} words"
)