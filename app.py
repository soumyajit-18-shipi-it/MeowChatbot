# ============================================================
#  MeowBot AI — Futuristic PDF Chatbot
#  Stack : Streamlit · Groq (OpenAI-compat) · PyMuPDF · dotenv
# ============================================================

import os
import streamlit as st
from openai import OpenAI, AuthenticationError, APIConnectionError, APIStatusError
from dotenv import load_dotenv
import fitz  # PyMuPDF

# ── env ─────────────────────────────────────────────────────
load_dotenv()


def get_default_api_key() -> str:
    try:
        groq = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
        openai_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
    except Exception:
        groq = None
        openai_key = None

    if groq:
        return str(groq).strip()
    if openai_key:
        return str(openai_key).strip()

    return os.getenv("GROQ_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()


# ── page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="MeowBot AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ════════════════════════════════════════════════════════════

GLOBAL_CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

:root {
  --navy: #020818;
  --panel: #0d1f3c;
  --cyan: #00f5ff;
  --blue: #0066ff;
  --purple: #7b2fff;
  --text: #c8d8f0;
  --text-dim: #6a8aaa;
  --border: rgba(0,245,255,0.18);
  --glass: rgba(13,31,60,0.55);
}

html, body, [data-testid="stApp"] {
  background: var(--navy) !important;
  font-family: 'Rajdhani', sans-serif !important;
  color: var(--text) !important;
}

[data-testid="stApp"]::before {
  content: '';
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background-image:
    linear-gradient(rgba(0,245,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,245,255,0.03) 1px, transparent 1px);
  background-size: 48px 48px;
}

[data-testid="stMainBlockContainer"] {
  position: relative;
  z-index: 1;
  max-width: 1200px !important;
  padding: 1.5rem 2rem 4rem !important;
}

[data-testid="stChatMessage"] {
  background: var(--glass) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 1rem 1.2rem !important;
  margin-bottom: 1rem !important;
  overflow-wrap: break-word !important;
  word-break: break-word !important;
}

[data-testid="stChatMessage"] * {
  overflow-wrap: break-word !important;
  word-break: break-word !important;
  white-space: pre-wrap !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
  border-left: 3px solid var(--cyan) !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  border-left: 3px solid var(--blue) !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
  line-height: 1.8 !important;
  font-size: 1rem !important;
  color: var(--text) !important;
}

[data-testid="stChatMessage"] pre {
  overflow-x: auto !important;
  white-space: pre-wrap !important;
}

[data-testid="stChatInput"] {
  border-radius: 14px !important;
  border: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] {
  background: rgba(10,22,40,0.9) !important;
  border-right: 1px solid var(--border) !important;
}

.hero {
  text-align: center;
  padding: 2rem 1rem;
}

.hero-title {
  font-family: 'Orbitron', monospace;
  font-size: clamp(2rem, 5vw, 3.5rem);
  font-weight: 900;
  background: linear-gradient(135deg, #fff 0%, var(--cyan) 40%, var(--blue) 80%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.hero-sub {
  color: var(--text-dim);
  margin-top: 0.7rem;
}

.welcome-card {
  padding: 2rem;
  border-radius: 18px;
  border: 1px solid var(--border);
  background: var(--glass);
  text-align: center;
  margin-top: 2rem;
}

.sidebar-logo {
  font-family: 'Orbitron', monospace;
  font-size: 1.2rem;
  font-weight: 800;
  color: var(--cyan);
}

.sidebar-section {
  margin-top: 1.5rem;
  margin-bottom: 0.6rem;
  font-size: 0.8rem;
  color: var(--cyan);
  letter-spacing: 0.15em;
}

.pdf-info-box {
  padding: 0.8rem;
  border-radius: 10px;
  border: 1px dashed var(--border);
  background: rgba(0,245,255,0.04);
  margin-top: 1rem;
}

</style>
"""

# ════════════════════════════════════════════════════════════
# PDF HELPERS
# ════════════════════════════════════════════════════════════

def extract_pdf_text(uploaded_file) -> str:
    try:
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(pages)
    except Exception as exc:
        st.error(f"PDF parse error: {exc}")
        return ""


# ════════════════════════════════════════════════════════════
# GROQ CLIENT
# ════════════════════════════════════════════════════════════

def build_client(api_key: str) -> OpenAI:
    api_key = (api_key or "").strip()
    if not api_key:
        raise ValueError("API key is required to build client")

    if api_key.startswith("gsk_"):
        return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    return OpenAI(api_key=api_key)


# ════════════════════════════════════════════════════════════
# PROMPT
# ════════════════════════════════════════════════════════════

def build_prompt(user_input: str, pdf_text: str) -> str:
    if pdf_text.strip():
        return f"""You are a helpful AI assistant.

Answer using the PDF content below.

PDF CONTENT:
{pdf_text[:12000]}

USER QUESTION:
{user_input}
"""
    return user_input


# ════════════════════════════════════════════════════════════
# CHAT RENDER
# ════════════════════════════════════════════════════════════

def render_chat_content(text: str) -> None:
    st.markdown(text)


# ════════════════════════════════════════════════════════════
# SESSION
# ════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "messages": [],
        "pdf_text": "",
        "pdf_name": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ════════════════════════════════════════════════════════════
# SIDEBAR  ← KEY FIX IS HERE
# ════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">🤖 MEOWBOT AI</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">API CONFIG</div>', unsafe_allow_html=True)

        # Use index-based default so session state doesn't fight the widget
        api_mode = st.radio(
            "API Mode",
            ["Use Default API Key", "Bring Your Own API Key"],
            label_visibility="collapsed",
        )

        # ── FIX: always initialise user_api_key, only overwrite when BYOK ──
        user_api_key = ""
        # allow a temporary session-scoped key (entered on main page)
        if "user_api_key" in st.session_state and not user_api_key:
            user_api_key = st.session_state.get("user_api_key", "")
        if api_mode == "Bring Your Own API Key":
            user_api_key = st.text_input(
                "Groq / OpenAI API Key",
                type="password",
                placeholder="gsk_... or sk-...",
            )

        # ── FIX: resolve final key correctly ──
        default_api_key = get_default_api_key()
        if api_mode == "Use Default API Key":
            final_api_key = default_api_key
        else:
            # BYOK: prefer what the user typed; fall back to default
            final_api_key = user_api_key.strip() if user_api_key.strip() else default_api_key

        # ── optional status hint ──
        if api_mode == "Use Default API Key":
            if default_api_key:
                st.success("✅ Default key loaded", icon=None)
            else:
                st.warning("⚠️ No default key found. Add GROQ_API_KEY to .env or Streamlit secrets.")

        st.markdown('<div class="sidebar-section">MODEL</div>', unsafe_allow_html=True)

        model = st.selectbox(
            "Model",
            [
                "llama-3.3-70b-versatile",
                "llama3-8b-8192",
                "gemma2-9b-it",
            ],
            label_visibility="collapsed",
        )

        st.markdown('<div class="sidebar-section">PDF UPLOAD</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            label_visibility="collapsed",
        )

        if uploaded and uploaded.name != st.session_state.pdf_name:
            with st.spinner("Parsing PDF..."):
                text = extract_pdf_text(uploaded)
            if text:
                st.session_state.pdf_text = text
                st.session_state.pdf_name = uploaded.name
                st.success(f"Loaded: {uploaded.name}")

        if st.session_state.pdf_name:
            chars = len(st.session_state.pdf_text)
            st.markdown(
                f"""
                <div class="pdf-info-box">
                📄 {st.session_state.pdf_name}<br>
                {chars:,} characters
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Remove PDF", use_container_width=True):
                st.session_state.pdf_text = ""
                st.session_state.pdf_name = ""
                st.rerun()

        # If no API key is available, allow a quick session-scoped paste here
        if not final_api_key:
            session_tmp = st.text_input(
                "Paste Groq/OpenAI API key for this session (won't be saved to repo)",
                type="password",
                key="session_api_key",
                placeholder="gsk_... or sk-...",
            )

            if session_tmp and session_tmp.strip():
                st.session_state["user_api_key"] = session_tmp.strip()
                final_api_key = session_tmp.strip()

        st.markdown('<div class="sidebar-section">CONTROLS</div>', unsafe_allow_html=True)

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return final_api_key, model


# ════════════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════════════

def render_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">MeowBot AI</div>
            <div class="hero-sub">Futuristic AI Chatbot with PDF Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════
# WELCOME
# ════════════════════════════════════════════════════════════

def render_welcome():
    st.markdown(
        """
        <div class="welcome-card">
            <h2>🤖 Neural Core Ready</h2>
            <br>
            <p>Upload a PDF and ask questions, or chat normally with MeowBot AI.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════
# CHAT HISTORY
# ════════════════════════════════════════════════════════════

def render_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            render_chat_content(msg["content"])


# ════════════════════════════════════════════════════════════
# RESPONSE
# ════════════════════════════════════════════════════════════

def get_response(client, model, messages):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=2048,
    )
    return response.choices[0].message.content


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    init_session()
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    final_api_key, model = render_sidebar()

    render_hero()

    if not st.session_state.messages:
        render_welcome()
    else:
        render_history()

    prompt = st.chat_input("Send a message to MeowBot AI...")

    if prompt:
        if not final_api_key:
            st.error("❌ No API key configured. Add GROQ_API_KEY to Streamlit Secrets or enter a key below for this session.")

            with st.form("enter_key_form"):
                temp = st.text_input("Enter Groq/OpenAI API key for this session", type="password", key="temp_api_key")
                submit = st.form_submit_button("Use key for this session")

                if submit:
                    if temp and temp.strip():
                        st.session_state["user_api_key"] = temp.strip()
                        st.success("Using provided API key for this session. Re-running...")
                        st.experimental_rerun()
                    else:
                        st.warning("Please paste a valid API key before submitting.")

            return

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            render_chat_content(prompt)

        final_prompt = build_prompt(prompt, st.session_state.pdf_text)

        api_messages = [
            {
                "role": "system",
                "content": "You are MeowBot AI. Reply clearly and professionally.",
            },
            # history excluding the last user message (which we send as final_prompt)
            *[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ],
            {"role": "user", "content": final_prompt},
        ]

        try:
            with st.spinner("Thinking..."):
                client = build_client(final_api_key)
                reply = get_response(client, model, api_messages)

            st.session_state.messages.append({"role": "assistant", "content": reply})

            with st.chat_message("assistant"):
                render_chat_content(reply)

        except AuthenticationError:
            st.error("❌ Authentication failed. Check your API key.")
        except APIConnectionError:
            st.error("❌ Connection error. Check your internet connection.")
        except APIStatusError as exc:
            st.error(f"❌ API error [{exc.status_code}]: {exc.message}")
        except Exception as exc:
            st.error(f"❌ Unexpected error: {exc}")


if __name__ == "__main__":
    main()