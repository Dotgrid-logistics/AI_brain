import re, random
import joblib
import streamlit as st

st.set_page_config(page_title="AI Brain", page_icon="🧠")
st.title("🧠 My AI Brain")
st.caption("An intent classifier trained on Banking77, with optional Gemini replies.")

@st.cache_resource
def load_brain():
    b = joblib.load("ai_brain.joblib")
    return b["vec"], b["model"], b["threshold"]

vec, model, THRESHOLD = load_brain()

def clean(t):
    t = str(t).lower()
    t = re.sub(r"[^a-z0-9\s']", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def pretty(label):
    return str(label).replace("_", " ").lower()

def classify(text):
    p = model.predict_proba(vec.transform([clean(text)]))[0]
    i = p.argmax()
    if p[i] < THRESHOLD:
        return None, float(p[i])
    return model.classes_[i], float(p[i])

def canned_reply(label):
    return random.choice([
        f"It sounds like you're asking about '{pretty(label)}'. Let me help with that.",
        f"Got it, this is about '{pretty(label)}'. What would you like to know?",
    ])

@st.cache_resource
def get_client():
    try:
        from google import genai
        return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        return None

def gemini_reply(text, label):
    client = get_client()
    if client is None:
        return canned_reply(label)
    try:
        prompt = (f"You are a friendly banking support bot. The user's intent was "
                  f"classified as '{pretty(label)}'. Reply briefly and naturally to: {text}")
        return client.models.generate_content(
            model="gemini-3.8-flash", contents=prompt).text
    except Exception:
        return canned_reply(label)

use_gemini = st.sidebar.checkbox("Natural replies (Gemini)", value=True)
st.sidebar.write("Confidence threshold:", THRESHOLD)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything about your bank account."}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if user := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user})
    with st.chat_message("user"):
        st.write(user)

    label, conf = classify(user)
    if label is None:
        reply = "Sorry, I didn't catch that. Could you rephrase?"
        tag = f"no confident match ({conf:.2f})"
    else:
        reply = gemini_reply(user, label) if use_gemini else canned_reply(label)
        tag = f"intent: {pretty(label)} ({conf:.2f})"

    with st.chat_message("assistant"):
        st.write(reply)
        st.caption(tag)
    st.session_state.messages.append({"role": "assistant", "content": reply})
