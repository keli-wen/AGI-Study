# AGI-Study


🎯 Be a good Deep Learning Engineer. (大量施工👷)

## Code-Examples

- [x] [`chat-llm-v1`](https://github.com/keli-wen/AGI-Study/tree/master/code-examples/chat-llm-v1)：基于 `PyTriton`，`Streamlit` 和 `DeepSeek` 制作的最简化 Chat Project。
- [ ] `chat-llm-v2`：基于 `chat-llm-v1` 制作的 `vision language` 版本，并优化了多模型选择，dynamic batching 和 streaming output 等新特性。（施工中）

## 🔥 LLM Dev Best-Practice

由于我认为 LLM Dev 才是我等普通人能做的事情，我最近在全力学习一些 Agentic System / RAG / Prompt Engineering 的最佳实践（基于 OpenAI / Anthropic / Google 等公司的技术博客），以及如何从 experimental 到 production 的最佳实践。这部分预计包括：

- [ ] **Agentic System**：关于如何构建有效的 Agent 的最佳实践。
  - [x] [Anthropic - Building Effective Agents](best-practice/Anthropic%20-%20Building%20effective%20agents/README.md) 
- [ ] **RAG**：如何设计和优化 RAG 的最佳实践。
- [ ] **Prompt Engineering**：如何设计和优化 Prompt 的最佳实践。
- [ ] **LLM Dev**：如何从实验到生产的最佳实践。

## Environment

> 这部分主要介绍 DL 环境配置相关的内容。

- [x] [**CUDA** Related Env Config](https://github.com/keli-wen/AGI-Study/blob/master/env/cuda-related/)：介绍 GPU Driver Version，Cuda Toolkit Version 的更新。包括多 Cuda 版本管理等。
- [ ] [**Docker** Related Env Config](https://github.com/keli-wen/AGI-Study/blob/master/env/docker-related/)：Docker 的基本使用教程（菜鸟教程）。

## Train

> 这部分主要介绍当前 LLM 中常用的 Training 框架以及相关知识点。

- [ ] `PYTORCH LIGHTNING` 入门介绍（低优先级）
- [ ] DeepSpeed 介绍：
  - [ ] DeepSpeed -- ZeRO 原理介绍（见知乎，待搬运）。
  - [ ] DeepSpeed 实战（环境配置，Example）（TODO，Low Priority）[Refer: DeepSpeed PR](https://github.com/microsoft/DeepSpeedExamples/pull/843).

## Tokenizer
- [x] Byte-Pair Encoding 算法解读。
- [ ] Google SentencePiece 库使用介绍。

## Inference & Deploy

> 这部分主要介绍推理优化和部署相关的内容。
>
> - **🤔Q: What's the Inference Optimization?**
> - **📖A:** Inference optimization refers to **the process of enhancing the efficiency and speed at which LLMs analyze data and generate responses**. This process is crucial for practical applications, as it directly impacts the model's performance and usability.

- [x] [`Basic-LLM-Inference.md`](https://github.com/keli-wen/AGI-Study/blob/master/inference/Basic-LLM-Inference.md)：基于 meta-llama 介绍基础的 LLM Inference pipeline。
- [ ] `Batch-Inference-Optimization.md`：（施工中）Basic 的进阶版。
- [ ] `vLLM`: （施工中）介绍 `vLLM` 的使用，**以及后续的 `vLLM` 核心原理和代码的探索。**
- [ ] `TensorRT-LLM`：目前是非常简单的介绍了 `TensorRT-LLM` 的使用信息。
- [x] `Mixture of Depth`：关于 MoD 的最新介绍，Transformer-based 模型的动态算力分配。
- [ ] `Nvidia Triton Inference Server`：首先进行工具扫盲，然后主要从应用的角度介绍这个工具的使用。
- [ ] `Quantization in LLM`：（施工中） 

## Demo

> 这部分主要介绍 DEMO 制作相关的经验。

- [x] `FastAPI`: 介绍 `FastAPI` 的基本信息，以及它如何应用在 LLM 相关的 DEMO 原型中。
- [ ] `Streamlit`：介绍如何 `Streamlit` 如何使用，并定制化自己的 DEMO 前端。

## Visualization

开源一些可视化的资源。
