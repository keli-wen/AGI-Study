# Mixture-of-Depths: 动态算力分配，一篇就够了。

- author: @keli-wen

本文是我拜读的第一篇 Conditional Computation 领域的 paper。我会尽量深入，保证知其所以然，任何可能存在困惑的地方我都尝试在一篇文章内解决。因为我本人也是小白，有什么理解上的问题欢迎大家指正。

> Why `Depths`?

## Introduction



## Implementing Mixture-of-Depths Transformers

![Figure1](./assets/MoD-Figure1.png)

图片难懂的地方在于右图如何理解，应该是按行，从下往上看。MoD 的优势是，tokens 可以在不同 layer 中有选择的被处理（尽管总共被处理的次数不多）。而在使用了 Early-Exit Conditional Computation 技术的 Transformer 中，tokens 要么被连续处理，要么永远不会再被处理，在普通的 Transformer 中 tokens 则会一直被处理。

### Sampling

> 这部分是我开始困惑最久的内容。我完全无法理解为什么存在这个限制："While expert-choice routing has a number of advantages, it has one distinct problem: the top-$k$ operation is non-causal. **This means that whether a given token’s routing weight is among the top-$k$​ for the sequence depends on the values of the routing weights for tokens that come after it, which we don’t have access to when autoregressively sampling."**



### Training methods

所有模型均采用相同的基本超参数配置，例如训练步骤的余弦调度、128的批次大小、2048的序列长度，唯一的变化是在等效浮点操作（isoFLOP）分析期间调整了模型的层数、头数和嵌入大小，以生成不同规模的模型。