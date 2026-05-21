import streamlit as st
import pickle

# Load the saved model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Page config
st.set_page_config(
    page_title="JailBreakScan",
    page_icon="🔐",
    layout="centered"
)

# Header
st.title("🔐 JailBreakScan")
st.subheader("AI Prompt Injection & Jailbreak Detector")
st.write("Paste any prompt below to instantly detect if it's a jailbreak attempt or a safe prompt.")

st.divider()

# Input box
prompt = st.text_area("🧠 Enter a prompt to scan:", height=150, placeholder="Type or paste a prompt here...")

col1, col2 = st.columns([1, 3])

with col1:
    scan = st.button("🔍 Scan Prompt", use_container_width=True)

with col2:
    clear = st.button("🗑️ Clear", use_container_width=True)

if clear:
    st.rerun()

st.divider()

# Scan logic
if scan:
    if prompt.strip() == "":
        st.warning("⚠️ Please enter a prompt first.")
    else:
        prompt_vec = vectorizer.transform([prompt])
        prediction = model.predict(prompt_vec)[0]
        confidence = model.predict_proba(prompt_vec).max() * 100

        if prediction == "jailbreak":
            st.error("🚨 JAILBREAK DETECTED")
            st.metric(label="Threat Confidence", value=f"{confidence:.1f}%")
            st.markdown("**This prompt appears to be attempting to bypass AI safety guidelines.**")

            st.divider()
            st.markdown("### 🔎 Why is this flagged?")
            st.markdown("""
            This prompt contains patterns commonly used in jailbreak attacks such as:
            - Asking the AI to ignore its instructions
            - Roleplaying as an unrestricted AI
            - Bypassing content policies through fictional framing
            """)

        else:
            st.success("✅ SAFE PROMPT")
            st.metric(label="Safety Confidence", value=f"{confidence:.1f}%")
            st.markdown("**This prompt appears to be a normal, safe interaction.**")

st.divider()

# Footer
st.markdown("### 📊 How it works")
col1, col2, col3 = st.columns(3)

with col1:
    st.info("**Step 1**\n\nPrompt is converted to numbers using TF-IDF Vectorizer")

with col2:
    st.info("**Step 2**\n\nLogistic Regression model analyzes the pattern")

with col3:
    st.info("**Step 3**\n\nResult is returned with confidence score")

st.caption("Built with Python · Scikit-learn · Streamlit | JailBreakScan v1.0")