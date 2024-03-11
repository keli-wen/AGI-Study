import json
import logging

import numpy as np
import streamlit as st
from pytriton.client import ModelClient


st.title("Chat-LLM Whisper 🎵")
# Default Model Related Setting.
st.session_state["MODEL-NAME"] = "deepseek-llm-7b-chat"
st.session_state["URL"] = "grpc://127.0.0.1:8001"


@st.cache_resource
def build_client(url: str, model_name: str) -> ModelClient:
    """streamlit-cached function to build client."""
    return ModelClient(url, model_name)


# Get the logging.
logger = logging.getLogger("examples.chat-llm.client")


# Create the cached client.
client = build_client(st.session_state["URL"], st.session_state["MODEL-NAME"])


def chat_page():
    """Chat page for ChatGPT Whisper."""
    log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
    )

    # Use `st.session_state.history_messages` to store the chat history.
    if "history_messages" not in st.session_state:
        st.session_state.history_messages = []

    # Show the chat history in the front-end.
    for message in st.session_state.history_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Message GPT..."):
        # Update the chat history with the user's prompt.
        st.session_state.history_messages.append(
            {"role": "user", "content": prompt}
        )

        # Show the user's prompt in the front-end.
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Step1. Convert the list of dict to np.ndarray for PyTriton transport.
            np_messages = np.frombuffer(
                json.dumps(st.session_state.history_messages).encode("utf-8"),
                dtype=np.uint8,
            )

            # Step2. Send the inference request to the triton inference server.
            response_dict = client.infer_sample(messages=np_messages)

            # Step3. Decode the response and show it in the front-end.
            response = response_dict["response"][0].decode("utf-8")
            st.markdown(response)

        st.session_state.history_messages.append(
            {"role": "assistant", "content": response}
        )


if __name__ == "__main__":
    chat_page()
