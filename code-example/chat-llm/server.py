import json
import logging

import numpy as np
import torch
from pytriton.decorators import sample
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Get the logger.
logger = logging.getLogger("code-examples.chat-llm.server")

# The basic process of loading model.
model_name = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id


@sample
def _infer_fn(messages: np.ndarray) -> np.ndarray:
    """Inference function for the DeepSeek LLM-7B model."""
    # Step1. Convert the np.ndarray to list of dict.
    # e.g. [{"role": "user", "content": "Hello"}]
    history_messages = json.loads(messages.tobytes().decode("utf-8"))

    # Step2. Apply the chat template and return the tensor.
    input_tensor = tokenizer.apply_chat_template(
        history_messages, add_generation_prompt=True, return_tensors="pt"
    )

    # Step3. Generate the response and decode it.
    outputs = model.generate(input_tensor.to(model.device), max_new_tokens=500)
    result = tokenizer.decode(
        outputs[0][input_tensor.shape[1] :], skip_special_tokens=True
    )

    # Step4. Return the response as np.ndarray.
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
