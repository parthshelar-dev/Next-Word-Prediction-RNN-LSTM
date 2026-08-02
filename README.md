# 🧠 Next Word Prediction — RNN & LSTM

A Streamlit web app that predicts the next word (or generates full phrases) from a starting piece of text, using two custom-trained deep learning models — a **SimpleRNN** and an **LSTM** — trained on a dataset of famous quotes.

**🔗 Live App:** [next-word-prediction-rnn-lstm.streamlit.app](https://next-word-prediction-rnn-lstm.streamlit.app/)

---

## 📸 Preview

**Text Generation**

![Generate Text](images/generate%20text.png)

**Next Word Prediction**

![Next Word Predict](images/next%20word%20predict.png)

---

## ✨ Features

- 🔀 **Switch between models** — compare predictions from a SimpleRNN and an LSTM, both trained on the same dataset
- 🔮 **Predict Next Word** — get the single most likely next word, plus a ranked table of the top 5 candidate words with probabilities
- ✨ **Generate Text** — greedily extend your input by a configurable number of words (1–20)
- 🎨 Clean, custom-themed UI (dark sidebar, styled buttons, card-based output)
- ⚡ Models, tokenizer, and config are cached with `st.cache_resource` so they load once per session, not on every interaction

---

## 🧩 How It Works

1. Input text is lowercased and converted to a sequence of token IDs using a Keras `Tokenizer` fit on the quotes dataset (vocabulary capped at top 10,000 words).
2. The sequence is left-padded (`padding="pre"`) to a fixed length of 745 tokens, matching the shape the models were trained on.
3. The padded sequence is passed through the selected model, which outputs a probability distribution over the vocabulary via a softmax layer.
4. The word with the highest probability (or top-5, for the prediction view) is mapped back from token ID to text using an index-to-word lookup.
5. For text generation, this process repeats — each predicted word is appended to the input and fed back into the model for the next prediction.

### Model Architecture

Both models share the same shape, differing only in their recurrent layer:

| Layer | SimpleRNN Model | LSTM Model |
|---|---|---|
| Embedding | `(745, 50)` | `(745, 50)` |
| Recurrent layer | `SimpleRNN(128)` | `LSTM(128)` |
| Dense (output) | `10,000` units, softmax | `10,000` units, softmax |

---

## 🗂️ Project Structure

```
Next-Word-Prediction-RNN-LSTM/
├── app.py                      # Streamlit app
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python version for deployment
├── lstm_model.keras            # Trained LSTM model
├── rnn_model.keras             # Trained SimpleRNN model
├── tokenizer.pkl               # Fitted Keras tokenizer
├── max_len.pkl                 # Max sequence length used for padding
├── qoute_dataset.csv           # Training dataset (quotes + authors)
├── Next_Word_Prediction.ipynb  # Notebook: data prep, training, evaluation
├── .streamlit/
│   └── config.toml             # App theme configuration
├── images/                     # Screenshots used in this README
├── LICENSE
└── .gitignore
```

---

## 🛠️ Tech Stack

- **Python 3.11**
- **TensorFlow / Keras** — model training and inference
- **Streamlit** — web app framework and deployment
- **NumPy / Pandas** — data handling

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/parthshelar-dev/Next-Word-Prediction-RNN-LSTM.git
cd Next-Word-Prediction-RNN-LSTM
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📊 Dataset

The models were trained on a dataset of well-known quotes (`qoute_dataset.csv`), each paired with its author. The text of the quotes forms the training corpus for next-word prediction — the models learn common patterns, phrasing, and word transitions found in quotation-style writing.

> **Note:** Since the training corpus is relatively small and quote-specific, predictions and generated text will reflect the style, vocabulary, and phrasing of quotes rather than general-purpose language. Some predictions may also show high model confidence on frequently repeated phrasing patterns from the dataset.

---

## 📓 Notebook

`Next_Word_Prediction.ipynb` contains the full workflow: data loading, tokenization, sequence generation, padding, model architecture definition, training, and evaluation for both the SimpleRNN and LSTM models.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Parth Shelar**
- LinkedIn: [parth-shelar](https://linkedin.com/in/parth-shelar)
