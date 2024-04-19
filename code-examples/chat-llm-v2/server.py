import base64
import hashlib
import json
import logging

import os
import tempfile
from io import BytesIO

import numpy as np
import torch

from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek_vl.utils.io import load_pil_images
from pytriton.decorators import sample
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
from transformers import AutoModelForCausalLM

# Get the logger.
logger = logging.getLogger("code-examples.chat-llm.server")

# The basic process of loading model.
model_name = "deepseek-ai/deepseek-vl-7b-chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_name)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
temporal_image_cache = tempfile.TemporaryDirectory()


def decode_message_images(message):
    """Decode the base64 images in the message and save them in the cache."""
    if message.get("images", None) is not None:
        post_images = []
        for image in message["images"]:
            # Decode the base64 string to get the bytes
            image_bytes = base64.b64decode(image.encode("utf-8"))

            # Create a BytesIO object from the decoded bytes
            image = BytesIO(image_bytes)

            # Use a cache to save the image and get the path.
            # First, use hash to get the image name.
            image_name = hashlib.md5(image_bytes).hexdigest()

            # Check if the image exists in the cache otherwise save it.
            if not os.path.exists(
                f"{temporal_image_cache.name}/{image_name}.png"
            ):
                with open(
                    f"{temporal_image_cache.name}/{image_name}.png", "wb"
                ) as f:
                    f.write(image_bytes)

            post_images.append(f"{temporal_image_cache.name}/{image_name}.png")
        message["images"] = post_images
    return message


@sample
def _infer_fn(messages: np.ndarray) -> np.ndarray:
    """Inference function for the DeepSeek LLM-7B model."""
    # Step1. Convert the np.ndarray to list of dict.
    # e.g. [{"role": "user", "content": "Hello"}]
    history_messages = json.loads(messages.tobytes().decode("utf-8"))

    # Step2. Decode the base64 images in the message.
    history_messages = list(map(decode_message_images, history_messages))
    history_messages.append({"role": "Assistant", "content": ""})
    print(history_messages)

    # Step3. Load images and prepare for inputs.
    pil_images = load_pil_images(history_messages)
    prepare_inputs = vl_chat_processor(
        conversations=history_messages, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    # Step4. Run image encoder to get the image embeddings.
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # Step5. Run the model to get the response.
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )

    # Step6. Return the response as np.ndarray.
    result = tokenizer.decode(
        outputs[0].cpu().tolist(), skip_special_tokens=True
    )

    logger.debug(result, type(result))
    return {"response": np.char.encode([result], "utf-8")}


def main():
    """Main function for the triton inference server."""
    log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
    )

    # Define the TritonConfig.
    triton_config = TritonConfig(
        http_address="127.0.0.1", http_port=8000, grpc_port=8001
    )

    with Triton(config=triton_config) as triton:
        logger.info("Loading DeepSeek LLM-7B model.")

        # The key step for triton inference server.
        triton.bind(
            model_name=model_name.split("/")[-1],
            infer_func=_infer_fn,
            inputs=[
                Tensor(name="messages", dtype=np.uint8, shape=(-1,)),
            ],
            outputs=[
                Tensor(name="response", dtype=bytes, shape=(1,)),
            ],
            config=ModelConfig(batching=False),
            strict=True,
        )
        triton.serve()


if __name__ == "__main__":
    main()
