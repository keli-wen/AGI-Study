# 基于 PyTriton 的模型部署，一篇就够了

## Why PyTriton?

如何漂亮的回答这个问题我觉得很有价值。部署有很多种框架值得选择，为什么我会选择介绍 PyTriton 这个框架呢？以及为什么需要部署框架呢？思考这些问题能让你对整个 DL 生产 pipeline 有更透彻的理解。

**>_: 什么是部署（Deployment）？**

部署有多种形式的部署，一般来讲（尤其是现在 LLM 领域），我们所说的部署就是 Service 部署，让模型允许在特定的服务器上接受用户的输入，高性能完成推理任务，并返回输出。

**>_: 为什么需要专门的部署框架？**

在我最初做 DEMO 的时候我发现一般有两种部署方案：

- [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)，最简单的部署方案。
- [Triton Inference Server](https://github.com/triton-inference-server/server) + 各种 backend （`C++`, `Python`, `TensorRT`, ...)。

方案一属于个人开发者做做小玩具，小 DEMO 时用的。第二种方案是比较成熟的专业方案，很多大公司内部都在使用（包括我之前的 Team）。可以提供**更好的性能**（Dynamic Batching），对不同 Device 更自动化更优秀的支持（CPU 层面执行，多 Device 多模型支持）还会提供一些专业的性能工具用来获取诸如吞吐量等性能指标。

看起来使用第二种方案一定是更好的选择。但关于 Triton 的使用/学习博客却不是很多（相比于 FastAPI）这是为什么呢？

**>_: 为什么是 `PyTriton` ?**

在最开始尝试使用 Triton Inference Server 时，我就觉得这玩意有点太复杂了。属于我一看就想睡觉。我甚至连如何安装都看了半天。相较于方案一，Triton 的学习成本有一些陡峭，你需要熟悉后才能体验到它的强大。打个不太恰当的比方，类似于 C++ 初学者还在学习怎么编译时，Python 初学者的程序就已经跑起来了。**略过高昂的学习成本似乎和崇尚快速原型实现的 Scientist 之间有所冲突，这也是我认为 Triton Inference Server 在普通开发者中用的并不多的原因。**

说回来，Triton 你可以理解为**支持多种框架的统一推理服务化工具**，支持例如 `C++`，`Python`，`TensorRT` 等不同的后端，虽然强大但是复杂。而 `PyTriton` 是 NVIDIA 实现的一个类似于 Flask/FastAPI 的接口，它大大简化了 Triton 在 Python 环境中的部署。

**所以我认为 `PyTriton` 很好的平衡了部署方案一和方案二之间的优缺点。提供了相比 Triton Inference Server 更平滑的学习曲线，更良好的开发效率，也提供了相比 FastAPI/Flask 更优秀的推理性能，更适配深度学习任务的相关工具库（多设备管理等）。**

综上， `PyTriton` 是一个挺不错的部署方案，兼而有之也比较专业。缺点是项目处于开发的前期，文档不是很全面（有时甚至得跳转到 Triton 的文档里去），再加上 NVIDIA 做文档的水平本来就“比较一般”（哈哈），中文社区的相关资料更是少之又少，遂抛砖引玉写一个入门的中文文档。

## Installation

PyTriton 的安装不算太麻烦（虽然会有一些小坑），整体还是常规的。主要内容可以参考 NVIDAI 的官方文档：[Offical Installation](https://triton-inference-server.github.io/pytriton/latest/installation/)。我推荐参考 [Creating virtualenv using `miniconda`](https://triton-inference-server.github.io/pytriton/latest/installation/#creating-virtualenv-using-miniconda) 这个 Section。大致流程如下：

```bash
# Create your virtual environment.

# Export the conda library path.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

# Install library for interpreter.
pip install nvidia-pytriton
```

> 注意，`export` 是必须的，否则可能出现找不到某些动态链接库（`libpython`）的报错。
>
> 其中 `CONDA_PREFIX` 是一个环境变量，它表示当前激活的 conda 环境的路径。当你使用 conda 激活一个特定的环境时，conda会自动设置 `CONDA_PREFIX` 变量来指向那个环境的根目录。而关于 **LD_LIBRARY_PATH** 是什么可以参考我的仓库中的 [`env/cuda-related.md`](https://github.com/keli-wen/AGI-Study/blob/master/env/cuda-related.md#3-multi-cuda-management)。

## Quick Start