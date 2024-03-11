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
cd code-examples/chat-llm

# In one terminal. The CUDA_VISIBLE_DEVICES is optional.
CUDA_VISIBLE_DEVICES=7 python server.py

# In another terminal. 
streamlit run client.py --server.port 8080 --server.address 127.0.0.1
```

### Client (frontend)

Client 端主要负责 UI 展示和与 Server 端的交互。本人旨在做一个 minimum 的 example，所以一切从简。并没有尝试集成诸如 streaming ouput 等功能。

类 ChatGPT 的页面主要使用 Streamlit 框架的组件实现。整体实现非常简单，如果尝试 follow 只需要阅读该文档 [Streamlit: Build a basic LLM chat app](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps)，大约只需要 `30` 分钟。

这里为了不 block 大家的理解，我会尝试用最简单，最直接，最不容易 confused 的语言进行介绍。

首先，你必须理解 Streamlit 的一个特性。Streamlit 是一个基于 Python 的前端框架，它的特点是：**每次用户的交互都会触发整个页面的重新渲染**。简单的理解：它是从头重新执行这个脚本来进行交互的，这样的特性使得 Streamlit 的开发非常简单。**显然，如果只是从头执行脚本，那么所有的状态都会丢失**。代码中的 `st.session_state` 会在每次重新渲染的时候保持状态。

对于一些特殊的，诸如数据库，网络连接，`streamlit` 支持 cache 操作。所以，我们把我们的 `ModelClient` 进行 cache。这样我们就可以避免重新渲染时需要重新连接 Server。`PyTriton` 的 `ModelClient` 的定义是非常简单的，只需要我们传入对应的 `url` 和 `model_name` 即可。

我们在 `st.session_state` 中的  `history_message` 中保存用户的输入。但是由于 `PyTriton` 仅支持传输 `numpy.ndarray` 格式，所以我们需要进行一个简单的编码操作后发送到 Server 端。这里我们使用的是 `client.infer_sample` 代表我们传入的是一个单个（而非批处理）的推理请求。


### Server (model backend)

Server 端的内容更为核心，主要包括：
- HuggingFace Packages 的使用。
- PyTriton 中 `infer_fn` 的定义与绑定。

TODO