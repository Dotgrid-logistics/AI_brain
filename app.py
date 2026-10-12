import re
import joblib
import streamlit as st
from google import genai
from google.genai import types

MODEL = "gemini-3.8-flash"
MAX_TURNS = 40          # messages per browser session (protects your free quota)
HISTORY_WINDOW = 20     # recent messages Gemini can see

SYSTEM_PROMPT = """You are Atlas, a calm, quick-witted AI assistant with a dry sense of humor.
- Converse naturally, like a smart friend. Keep replies concise unless asked for detail.
- Be honest: if you don't know something or can't verify it, say so. You have no live internet access.
- Ask a short clarifying question when a request is ambiguous.
- For banking questions, give general guidance only. Never ask for card numbers, PINs, passwords or other sensitive details.
"""

st.set_page_config(page_title="Atlas", page_icon="🧠")
st.title("🧠 Atlas")
st.caption("A conversational AI assistant: a trained intent classifier plus Gemini for open conversation.")

# ---------- classifier (your trained brain) ----------
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

# ---------- Gemini (the conversational brain) ----------
@st.cache_resource
def get_client():
    try:
        return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        return None

def to_contents(messages):
    msgs = messages[-HISTORY_WINDOW:]
    while msgs and msgs[0]["role"] != "user":   # Gemini wants history to start with the user
        msgs = msgs[1:]
    return [types.Content(role="user" if m["role"] == "user" else "model",
                          parts=[types.Part(text=m["content"])]) for m in msgs]

def stream_reply(messages, intent_hint):
    client = get_client()
    if client is None:
        yield "My language brain is offline (no API key is set)."
        return
    system = SYSTEM_PROMPT
    if intent_hint:
        system += (f"\nNote: a classifier thinks the user's latest message relates to the "
                   f"banking topic '{intent_hint}'. Treat this only as a hint.")
    try:
        for chunk in client.models.generate_content_stream(
            model=MODEL,
            contents=to_contents(messages),
            config=types.GenerateContentConfig(
                system_instruction=system, temperature=0.8, max_output_tokens=600),
        ):
            if chunk.text:
                yield chunk.text
    except Exception:
        yield "Sorry, I hit a problem reaching my language model. Please try again in a moment."

# ---------- chat UI ----------
GREETING = "Hi, I'm Atlas. Ask me anything, from banking questions to random curiosities."

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]
    st.session_state.turns = 0

with st.sidebar:
    show_intent = st.checkbox("Show detected intent", value=True)
    st.write("Classifier threshold:", THRESHOLD)
    if st.button("Clear chat"):
        st.session_state.messages = [{"role": "assistant", "content": GREETING}]
        st.session_state.turns = 0
        st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if user := st.chat_input("Message Atlas..."):
    if st.session_state.turns >= MAX_TURNS:
        st.warning("Session limit reached. Press 'Clear chat' to start a new conversation.")
        st.stop()
    st.session_state.turns += 1

    st.session_state.messages.append({"role": "user", "content": user})
    with st.chat_message("user"):
        st.write(user)

    label, conf = classify(user)
    hint = pretty(label) if label else None

    with st.chat_message("assistant"):
        reply = st.write_stream(stream_reply(st.session_state.messages, hint))
        if show_intent:
            st.caption(f"intent: {hint} ({conf:.2f})" if hint
                       else f"general conversation (classifier confidence {conf:.2f})")
    st.session_state.messages.append({"role": "assistant", "content": reply})
