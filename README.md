# 🤖AGI-Study

- [AGI-Study](#agi-study)
  - [Environment](#environment)
  - [Train](#train)
  - [Optimization](#optimization)
  - [Inference](#inference)
  - [Demo](#demo)
  - [Basic](#basic)


我建立这个 Repo 的初衷是以 Code 指导理论学习，所以我会尽量以 Code 为主，以应用为主。

## Environment

> 这部分主要介绍 DL 环境配置相关的内容。

- [x] **CUDA** Related Env Config：介绍 GPU Driver Version，Cuda Toolkit Version 的更新。后续将包括多 Cuda 版本管理等。

## Train

> 这部分主要介绍当前 LLM 中常用的 Training 框架以及相关知识点。

- [ ] `PYTORCH LIGHTNING` 入门介绍
- [ ] DeepSpeed 介绍：
  - [ ] DeepSpeed -- ZeRO 原理介绍（见知乎，待搬运）。
  - [ ] DeepSpeed 实战（环境配置，Example）（TODO，Low Priority）

## Optimization

> 可能就是 CUDA 相关的内容。

- [ ] `CUDA` 基本知识入门

## Inference

> 这部分主要介绍推理优化相关的内容。
>
> - **🤔Q: What's the Inference Optimization?**
> - **📖A:** Inference optimization refers to **the process of enhancing the efficiency and speed at which LLMs analyze data and generate responses**. This process is crucial for practical applications, as it directly impacts the model's performance and usability.

- [x] `vLLM`: 介绍 `vLLM` 的使用，**以及后续的 `vLLM` 核心原理和代码的探索。**
- [x] `TensorRT-LLM`：目前是非常简单的介绍了 `TensorRT-LLM` 的使用信息。
- [ ] `Nvidia Triton Inference Server`：首先进行工具扫盲，然后主要从应用的角度介绍这个工具的使用。

## Demo

> 这部分主要介绍 DEMO 制作相关的经验。

- [x] `FastAPI`: 介绍 `FastAPI` 的基本信息，以及它如何应用在 LLM 相关的 DEMO 原型中。

- [ ] `Streamlit`：介绍如何 `Streamlit` 如何使用，并定制化自己的 DEMO 前端。

## Basic

> 这个部分主要介绍一些重要的基础知识。（后续可能会进行进一步的细分）
