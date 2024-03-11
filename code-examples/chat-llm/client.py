import json

import numpy as np
import streamlit as st
from pytriton.client import ModelClient


st.title("Chat-LLM Whisper 🎵")
# Default Model Related Setting.
st.session_state["MODEL-NAME"] = "deepseek-llm-7b-chat"
st.session_state["URL"] = "grpc://127.0.0.1:8001"


# Create the PyTriton client.
st.session_state["client"] = ModelClient(
    url=st.session_state["URL"], model_name=st.session_state["MODEL-NAME"]
)


def chat_page():
    """Chat page for ChatGPT Whisper."""
    # Use `st.session_state.history_messages` to store the chat history.
    if "history_messages" not in st.session_state:
        st.session_state.history_messages = []

    # Show the chat history in the front-end.
    for message in st.session_state.history_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get the user's prompt. := is the walrus operator.
    if prompt := st.chat_input("Message Chat-bot..."):
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
            response_dict = st.session_state["client"].infer_sample(
                messages=np_messages
            )

            # Step3. Decode the response and show it in the front-end.
            response = response_dict["response"][0].decode("utf-8")
            st.markdown(response)

        st.session_state.history_messages.append(
            {"role": "assistant", "content": response}
        )


if __name__ == "__main__":
    chat_page()
