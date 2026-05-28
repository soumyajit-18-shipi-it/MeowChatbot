# 🤖 MeowBot AI

MeowBot AI is a futuristic AI chatbot built using Streamlit and Groq LLMs with PDF document intelligence support.

Users can upload PDFs, ask questions about documents, and interact with an advanced cyberpunk-themed chatbot UI.

---

# ✨ Features

## ⚡ Groq-Powered AI Responses

MeowBot AI uses Groq’s ultra-fast inference engine with OpenAI-compatible APIs to generate extremely fast AI responses. The app supports multiple models including:

* `llama-3.3-70b-versatile`
* `llama3-8b-8192`
* `gemma2-9b-it`

The selected model processes user prompts and generates conversational responses in real time.

---

## 📄 PDF Intelligence System

Users can upload PDF documents directly into the chatbot.

### How it works

1. The uploaded PDF is processed using **PyMuPDF**
2. Text is extracted from every page
3. The extracted text is stored temporarily in session memory
4. When the user asks a question, the PDF content is injected into the AI prompt
5. The AI answers using the document context

This enables:

* PDF summarization
* Question answering
* Context-aware responses
* Document understanding

---

## 💬 Persistent Chat History

The chatbot stores conversation history using Streamlit session state.

### How it works

* Every user and assistant message is stored inside:

```python id="31mevb"
st.session_state.messages
```

* Previous messages are sent back to the model during each API call
* This allows contextual multi-turn conversations

---

## 🎨 Futuristic Cyberpunk UI

The interface is fully customized using advanced CSS styling.

### Includes

* Glassmorphism effects
* Neon glow aesthetics
* Animated grid backgrounds
* Responsive layout
* Modern chat bubbles
* Gradient typography
* Custom sidebar styling

The design is inspired by futuristic AI operating systems and cyberpunk interfaces.

---

## 🔑 Bring Your Own API Key (BYOK)

Users can either:

* Use the default API key from Streamlit Secrets or local environment variables
* Enter their own Groq API key directly in the sidebar

### How it works

The app dynamically switches between:

```python id="0vy4np"
GROQ_API_KEY
```

and user-provided keys from the sidebar input field.

### Deployment note

For Streamlit Cloud, do not commit a `.env` file. Add secrets in the app settings instead:

* `GROQ_API_KEY`
* `OPENAI_API_KEY` (optional fallback)

The app reads Streamlit Secrets first, then falls back to local environment variables for development.

---

## 🧠 Multi-Model Support

Users can dynamically switch between multiple LLMs directly from the sidebar.

### Benefits

* Faster responses
* Different reasoning styles
* Better experimentation
* Lower latency options

The selected model is passed directly into the Groq API request.

---

## 📂 PDF Upload System

The PDF uploader supports drag-and-drop uploads through Streamlit’s file uploader component.

### Workflow

1. Upload PDF
2. Extract text
3. Save extracted text in session
4. Enable PDF-aware AI responses

The uploaded document remains active until removed manually.

---

## 🛡 Error Handling

The application includes robust exception handling for:

* Invalid API keys
* API connection failures
* Rate limits
* PDF parsing errors
* Unexpected runtime exceptions

This prevents crashes and improves stability.

---

# 🛠 Tech Stack

| Technology | Purpose                |
| ---------- | ---------------------- |
| Python     | Backend logic          |
| Streamlit  | Frontend framework     |
| Groq API   | AI inference           |
| OpenAI SDK | API compatibility      |
| PyMuPDF    | PDF text extraction    |
| dotenv     | Environment management |

---

# 📦 Installation

Clone the repository:

```bash id="7xv7vt"
git clone <your-repo-link>
cd meowbot-ai
```

Install dependencies:

```bash id="jlwmcz"
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a local `.env` file for development only:

```env id="r4x6pr"
GROQ_API_KEY=your_groq_api_key
```

Get your API key from:

https://console.groq.com/keys

On Streamlit Cloud, configure the same key names in the app's Secrets panel instead of using `.env`.

---

# ▶️ Run the App

```bash id="b3a3lz"
streamlit run app.py
```

---

# 📁 Project Structure

```bash id="t2m4hm"
├── app.py
├── requirements.txt
├── .env
└── README.md
```

---

# 🚀 Future Improvements

* RAG pipelines
* Vector database support
* OCR support
* Multi-document querying
* Voice assistant integration
* Memory persistence
* Authentication system
* Cloud deployment support

---

# 📜 License

MIT License

---

# 👨‍💻 Author

Soumyajit Rout
BITS Pilani — Computer Science
