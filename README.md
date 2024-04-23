# AGI-Study


🎯 Be a good Deep Learning Engineer.

## Code-Examples

- [x] `chat-llm-v1`：基于 `PyTriton`，`Streamlit` 和 `DeepSeek` 制作的最简化 Chat Project。
- [ ] `chat-llm-v2`：基于 `chat-llm-v1` 制作的 `vision language` 版本，并优化了多模型选择，dynamic batching 和 streaming output 等新特性。（施工中）

## Environment

> 这部分主要介绍 DL 环境配置相关的内容。

- [x] **CUDA** Related Env Config：介绍 GPU Driver Version，Cuda Toolkit Version 的更新。后续将包括多 Cuda 版本管理等。

## Train

> 这部分主要介绍当前 LLM 中常用的 Training 框架以及相关知识点。

- [ ] `PYTORCH LIGHTNING` 入门介绍（低优先级）
- [ ] DeepSpeed 介绍：
  - [ ] DeepSpeed -- ZeRO 原理介绍（见知乎，待搬运）。
  - [ ] DeepSpeed 实战（环境配置，Example）（TODO，Low Priority）

## Inference & Deploy

> 这部分主要介绍推理优化和部署相关的内容。
>
> - **🤔Q: What's the Inference Optimization?**
> - **📖A:** Inference optimization refers to **the process of enhancing the efficiency and speed at which LLMs analyze data and generate responses**. This process is crucial for practical applications, as it directly impacts the model's performance and usability.

- [x] [`Basic-LLM-Inference.md`](https://github.com/keli-wen/AGI-Study/blob/master/inference/Basic-LLM-Inference.md)：基于 meta-llama 介绍基础的 LLM Inference pipeline。
- [ ] `Batch-Inference-Optimization.md`：（施工中）Basic 的进阶版。
- [x] `vLLM`: 介绍 `vLLM` 的使用，**以及后续的 `vLLM` 核心原理和代码的探索。**
- [x] `TensorRT-LLM`：目前是非常简单的介绍了 `TensorRT-LLM` 的使用信息。
- [ ] `Mixture of Depth`：（施工中）关于 MoD 的最新介绍。
- [ ] `Nvidia Triton Inference Server`：首先进行工具扫盲，然后主要从应用的角度介绍这个工具的使用。
- [ ] `Quantization in LLM`：（施工中） 

## Demo

> 这部分主要介绍 DEMO 制作相关的经验。

- [x] `FastAPI`: 介绍 `FastAPI` 的基本信息，以及它如何应用在 LLM 相关的 DEMO 原型中。
- [ ] `Streamlit`：介绍如何 `Streamlit` 如何使用，并定制化自己的 DEMO 前端。

