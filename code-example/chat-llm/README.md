## Chat-LLM V1

旨在给出一个比较简洁，但是完整度比较高的 example，包括：
- [x] `transformers` for model.
- [x] `streamlit` for front-end.
- [x] `pytriton` for model deployment.
- [ ] `vLLM` for inference optimization. (maybe in the V2)

### Model Selection
模型选择的是 `deepseek` 的 [`llm-7b-chat`](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat)。在当前配置下大约需要 15GB 的显存。


### Installation

> ⚠️ 如果存在网络问题 `export HF_ENDPOINT=https://hf-mirror.com`。

```bash
# For front-end.
pip install streamlit

# For model.
pip install accelerate
pip install --upgrade transformers

# For NVIDIA-PyTriton.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
pip install -U nvidia-pytriton
```

### Running example locally
```bash
# In one terminal.
streamlit run client.py --server.port 8080 --server.address 127.0.0.1

# In another terminal.
python server.py
```

### Client (frontend)
TODO 


### Server (model backend)
TODO