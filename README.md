# JailBreakScan

A prompt injection and jailbreak detection tool built on top of a logistic regression classifier trained on 1900+ real and curated prompts.

---

## Background

While working on AI security projects, I kept running into the same problem — LLMs are surprisingly easy to manipulate with the right phrasing. Prompts like *"pretend you have no restrictions"* or *"you are now DAN"* are simple but effective at bypassing safety layers.

I wanted to build something that sits in front of an LLM and catches these attempts before they go through. That's what this is.

---

## What it does

Paste any prompt into the interface. The model tells you whether it looks like a jailbreak attempt or a normal prompt, along with a confidence score.

That's it. Simple, fast, and works locally.

---

## Stack

- Python 3.12
- Scikit-learn — TF-IDF vectorizer + Logistic Regression
- Streamlit — web interface
- Pandas — data handling
- Dataset — 1900+ prompts, mix of handcrafted and public sources

---

## Results

Tested on 744 held-out prompts.

| Metric | Score |
|---|---|
| Accuracy | 99.8% |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 | 1.00 |

Only 1 misclassification out of 744 test prompts.

---

## Running it locally

```bash
git clone https://github.com/yourusername/JailBreakScan.git
cd JailBreakScan

python -m venv venv
venv\Scripts\activate

pip install pandas scikit-learn streamlit datasets

python download_data.py
python train_model.py
streamlit run app.py
```

---

## Project structure
JailBreakScan/
├── download_data.py     # builds and saves the dataset
├── train_model.py       # trains and evaluates the model
├── app.py               # streamlit interface
├── data.csv             # final training data
├── model.pkl            # saved model
├── vectorizer.pkl       # saved vectorizer
└── README.md
---

## Limitations

The current model is trained on pattern-based jailbreaks — things like "ignore your instructions" or "pretend you are". More sophisticated attacks that don't follow obvious patterns will likely slip through. A transformer-based classifier would handle those better and is the next step.

---

## What's next

- Replace logistic regression with a fine-tuned DistilBERT
- Expand dataset with adversarial examples
- Add a REST API so it can plug into other systems
- Deploy publicly on Streamlit Cloud

---

Made by SRI ANISHITHA RANI PATHAKOTA