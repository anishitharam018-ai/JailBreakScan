import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import pickle

# Load the dataset
print("Loading data...")
df = pd.read_csv("data.csv")

# ── Fix imbalance by upsampling jailbreak prompts ────────────────────
print("\nFixing class imbalance...")
df_normal = df[df["type"] == "normal"]
df_jailbreak = df[df["type"] == "jailbreak"]

df_jailbreak_upsampled = resample(
    df_jailbreak,
    replace=True,
    n_samples=len(df_normal),
    random_state=42
)

df_balanced = pd.concat([df_normal, df_jailbreak_upsampled]).reset_index(drop=True)

print("After balancing:")
print(df_balanced["type"].value_counts())

# Separate prompts and labels
X = df_balanced["prompt"]
y = df_balanced["type"]

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining on {len(X_train)} prompts")
print(f"Testing on {len(X_test)} prompts")

# Convert text to numbers using TF-IDF
print("\nConverting text to numbers...")
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
print("Training the model...")
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train_vec, y_train)

# Test the model
print("\nEvaluating the model...")
y_pred = model.predict(X_test_vec)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model and vectorizer
print("\nSaving model...")
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model saved successfully!")