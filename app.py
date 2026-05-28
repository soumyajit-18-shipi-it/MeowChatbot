import streamlit as st
from openai import OpenAI

# Page Config
st.set_page_config(
    page_title="BYOK AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Bring Your Own Key Chatbot")

st.markdown("Enter your API key and start chatting.")

# Sidebar
st.sidebar.header("Settings")

# User API Key Input
user_api_key = st.sidebar.text_input(
    "Enter OpenAI API Key",
    type="password"
)

# Model Selection
model = st.sidebar.selectbox(
    "Choose Model",
    [
        "gpt-4.1-mini",
        "gpt-4o-mini"
    ]
)

# Clear Chat Button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Previous Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
prompt = st.chat_input("Type your message here...")

# Check if API key exists
if prompt:

    if not user_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        st.stop()

    # Create OpenAI Client
    client = OpenAI(api_key=user_api_key)

    # Store User Message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant Response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=st.session_state.messages
                )

                reply = response.choices[0].message.content

                st.markdown(reply)

                # Save Assistant Response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply
                })

            except Exception as e:
                st.error(f"Error: {e}")