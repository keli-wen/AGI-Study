# CUDA-Related Env Config

在进行 CUDA 相关的环境配置前，我们需要先搞清楚，CUDA-Related Env 的配置的基础知识。

我们通常会使用两个命令： `nvcc --version` 和 `nvidia-smi`。我这里用我自己的服务器举例：

```
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243
$ nvidia-smi
Wed Jan 24 11:43:58 2024       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.182.03   Driver Version: 470.182.03   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100 80G...  On   | 00000001:00:00.0 Off |                    0 |
| N/A   30C    P0    41W / 300W |     47MiB / 80994MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1472      G   /usr/lib/xorg/Xorg                 46MiB |
+-----------------------------------------------------------------------------+
```

第一个命令返回的文本说明我们 nvcc 的版本是 `10.1`。

第二个命令返回的有用信息主要是这一行：`NVIDIA-SMI 470.182.03   Driver Version: 470.182.03   CUDA Version: 11.4`。

我们惊讶的发现，具有有两个不同的 CUDA Version，他们有什么区别呢？具体的区别有很多博客/回答中有很优秀的回复，可以参考：[StackOverflow - Different CUDA versions shown by nvcc and NVIDIA-smi](https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi).

我这里提供**通俗的解释**。NVCC 是 Nvidia Cuda Compiler 的缩写，NVIDIA-SMI 是 NVIDIA System Management Interface 的缩写。

- NVCC 代表的是当前计算机中 CUDA 编译器，那么 `nvcc -v` 得到的是你当前计算机中**安装和使用的** CUDA Compiler 的版本或者说 CUDA Toolkit 的版本。
- NVIDIA-SMI 实际上展示的是 GPU Driver 版本，例如 `Driver Version: 470.182.03`，它后面跟随的 CUDA Version 代表的是当前 GPU Driver 所能**支持的最高 CUDA 版本。**

**一般情况下，**我们可以认为 NVCC 得到的 CUDA Version 会**小于等于** NVIDIA-SMI 得到的 CUDA Version。因为一个是实际使用的 CUDA Version，一个是最高可支持的 CUDA Version。

假设你的期望 CUDA 版本是 $X$，NVCC 得到的版本是 $Y$，NVIDIA-SMI 得到的版本是 $Z$。如果你需要进行 CUDA-Related Env 配置，大致流程为：

1. 首先应该看，$X$ 是否大于 $Z$。如果大于 $Z$，则需要先**更新 GPU Driver**，然后跳转至步骤 2。
2. 如果 $X$ 小于 $Z$，那么我们只需要**更新 NVCC 即可**。

## Update GPU Driver

TODO

## Re-Install `NVCC`

TODO