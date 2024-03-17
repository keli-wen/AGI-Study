import json

import numpy as np
import streamlit as st
from pytriton.client import ModelClient


st.title("Chat-LLM Whisper 🎵")
# Default Frontend Related Setting.
if "chat_counter" not in st.session_state:
    st.session_state["chat_counter"] = 0
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

    st.sidebar.title("🎁 More Extra Options")
    st.sidebar.divider()

    st.sidebar.markdown("### Custom :violet[Instructions]")
    custom_instructions = st.sidebar.text_area(
        label="Please enter custom instructions here...",
        # label_visibility="collapsed",
    )

    st.sidebar.markdown("### Chat with :orange[Image]")
    image = st.sidebar.file_uploader(
        key=f"{st.session_state['chat_counter']}_image_uploader",
        label="Upload Image",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
    st.sidebar.markdown("### Chat with :gray[Audio]")
    voice_return = st.sidebar.checkbox(
        "Enable Voice Return",
        key=f"enable_Voice_return",
    )

    # Show the chat history in the front-end.
    for message in st.session_state.history_messages:
        with st.chat_message(message["role"]):
            if "image" in message:
                st.image(message["image"])
            st.markdown(message["content"])

    # Get the user's prompt. := is the walrus operator.
    if prompt := st.chat_input("Message Chat-bot..."):
        # Update the chat history with the user's prompt.
        user_message = {"role": "user", "content": prompt}
        if image is not None:
            user_message["image"] = image
        st.session_state.history_messages.append(user_message)

        # Show the user's prompt in the front-end.
        with st.chat_message("user"):
            if image:
                st.image(image)
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
            
            # Step4. Update the chat counter to remove the sended image.
            st.session_state["chat_counter"] += 1

        st.session_state.history_messages.append(
            {"role": "assistant", "content": response}
        )


if __name__ == "__main__":
    chat_page()
