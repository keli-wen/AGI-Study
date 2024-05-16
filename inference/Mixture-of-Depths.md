# Mixture-of-Depths: 动态算力分配，一篇就够了。

- author: @LastWhisper

本文是我拜读的第一篇 Conditional Computation 领域的 paper。我会尽量深入，保证知其所以然，任何可能存在困惑的地方我都尝试在一篇文章内解决。因为我本人也是小白，有什么理解上的问题欢迎大家指正。

> **Q ：Why name `Depths`?**
>
> **A** ：TODO

## 1. Introduction



## 2. Implementing Mixture-of-Depths Transformers

我们整体的设计策略如下：

- 通过限制可以参与块计算（即 self-attention 和随后的 MLP）的序列中的 tokens 数量，设置一个低于普通 transformer 的静态计算预算。例如，正常的 transformer 允许序列中的所有 tokens 进行 self-attention 的计算，我们则限制只有 50% 的 tokens 能进行 self-attention 的计算。
- 利用一个逐块的路由来估计为每个 token 计算一个标量权重（==为什么要强调标量 Scalar 呢？可能是因为 Scalar 代表一个单一的数值权重，而非向量或矩阵==），用于表示路由器对于该 token 的偏好（是参与计算，或者残差链接）。
- 确定每一个序列，在每一个块的 top-k 标量权重，以选择将会参与到块计算的 tokens。由于只会有确定的 k 个 token 将参与块的计算，因此计算图和 tensor 大小在整个训练过程中保持不变；仅仅只有哪些 tokens 会参与是动态的和上下文敏感的。（由路由器决定）

### 2.1 Defining a compute budget

为了在每次前向传播控制总的计算预算，我们需要利用容量这个概念。**容量（capacity）** 定义了每次输入中参与计算的总 tokens 数量。例如，假设在普通的 transformer 中，所有 batch 中所有序列总参与计算的 tokens 数则可以被定义为容量 $T$。在使用了 Mixture of Experts 的 transformer 中，每个专家 MLP 的容量是小于 $T$​，但是由于每个块存在多个专家，所以 MoE 的总计算量与普通 transformer 是接近等价的。

> 这里有两个容易混淆的概念，分别是：**容量**和计算预算。
>
> - 静态计算预算（static computation budget）可以理解为整个模型的总计算量或 FLOP 数。
> - 容量（Capacity）代表整个 transformer 中参与计算的总 tokens 数。
>
> 依据我的理解，在该文章中，我们可以这样理解：“通过修改容量修改了计算预算“。这对于后文实验部分的理解也许有帮助。

通常情况下，使用条件计算的 transformer 的总 FLOPs 是由 token capacity 决定的，而非任何路由的结果。这是因为**静态图的实现**考虑的是最差情况。例如，如果路由之后只有少部分的 tokens，我们会填充到对应的容量大小，或者会删除超出容量部分的 tokens。

> **Q：为什么静态图的实现是这样的呢？简单描述下动态和静态图之间的区别？**
>
> **A：**TODO

我们可以通过降低计算的**容量**来实现相比标准 transformer 每次前向传递使用更少的计算预算。然而，如果不加选择地减少计算预算，可能会导致性能恶化。我们的假设某些特定的 tokens 可能不需要像其他 tokens 那样被多次处理，并且这些 tokens 可以通过学习被识别出来。因此，如果网络能够学习如何选取合适的 tokens 用来填充容量，则有可能保持模型性能不变。下面将介绍为此目的而设计的 routing scheme。

### 2.2 Routing around transformer blocks

路由实际上是把 tokens 在该层的行为进行二分类：

- 通过 self-attention 和 MLP blocks 进行计算。
- 残差链接。

如果我们将路径 (1) 的容量设置为小于 $T$（序列和 batch 中的总 tokens 数）的任何值，那么每次前向传递的总 FLOP 数将少于标准 transformer。例如，如果我们将块的容量设置为 $\dfrac{T}{2}$ （即与标准 transformer 相比只有一半的标记数），那么在self-attention 过程中的 query 与 key 矩阵乘法将只需要普通 transformer 的 $25$% 的 FLOP 计算量（ $\dfrac{T^2}{4}$ 与 $T^2$ ​​)。类似的计算可以确定 MLP 对应的 FLOP 减少量。

从直观上来说，随着我们不断缩小块容量，每次前向传递的总 FLOP 数会减少（并且完成 forward 所需的时间也会减少）。但是，下游的性能也会受到我们缩小块容量的激进程度的影响，以及我们实现的路由算法的影响。

分析极端情况，如果我们保持每个块的容量为 $T$，并将每个 tokens 都正常通过块进行计算，我们就恢复成了标准的 transformer。另一个极端情况下，如果我们将每个块的容量设置为 $0$，并将所有 tokens 都路由为不参与块的计算，那么我们最终将得到一个非常快的模型，它不会参与 transformer 绝大部分参数的计算，显然这会导致很差的下游性能。文章的假设为：**在这两个极端之间存在一个最优模型，它比普通 transformer 更快，同时性能却不差甚至更好，而且 step 的速度更快。**

### 2.3 Routing schemes

**如何路由 token 决定了该方法的性能**。举一个不恰当的例子，假设有一位学生正在学习英语，老师每次只是随机地从句子中挑选几个词来教学。如果我们考虑这样的一个句子：`She is a very good teacher`，如果老师随机挑选词汇，比如只教授 `she, a, good`，那么学生最终可能会对句子的真实含义产生误解，或者根本无法理解完整的语义结构。（好吧，我承认这个例子不是很恰当，大家意会即可）

最基础的 routing 方案就是在 tokens 中**随机选择并丢弃**（类似于 `dropout`）。后续实验将以该方式作为对比，展示过分简单的 routing 实现会导致性能急剧下降（相较于标准 transformer ）。

与随机丢弃对应的是可学习的路由（learnable routing）。我们首先假设 learnable routing 是可行的，即假定网络应该能学到哪些 tokens 需要更多或更少的处理。如果 Transformer 确实做了很多无用的计算，那么问题便从 **“Transformer 是否存在冗余计算”** 转化为 **“Transformer 冗余了多少计算量”**，这是一个经验问题，我们通过实验来确定减少 Transformer 计算容量的激进程度，以及我们最多能允许多少 tokens 在每层 block 中不被计算。

假设完毕开始实践，目前有两种主流可学习的路由架构（见图），token-choice（自选）和 expert-choice（反选）。

在 token-choice 中，会训练一个 router 为每个 token 生成到所有可能的 **（计算）路径** 上的概率分布（例如，在 MoE 中选择 token 经过哪个 expert)。然后，tokens 会被发送到概率最高的那条路径。并且为避免所有的 tokens 选择相同的路径，可以引入辅助 loss。但是，**即使引入辅助 loss，token-choice 路由架构仍可能因为大部分 token 仅选择趋同的少部分路径，从而出现负载均衡问题。**

在 expert-choice 中，每条路径通过 route 挑选 Top-k 个 tokens。这种架构保证不会出现负载均衡问题，因为每个路径分配且仅分配 $k$ 个 tokens。但是，这种架构同样存在局限，因为有些 tokens 可能出现在所有路径的 Top-k 集合从而被过度使用，而有些 token 则不出现在任何路径中。

> 思考接下来几个小问题会对理解该小节更有帮助：
>
> - 在 token-choice 架构中，**如何保证计算容量的准确利用**？每个 token 都存在一个概率分布并将选择一条路径，由于任意路径容量有限，tokens 之间存在竞争，如何保证计算容量被充分的利用呢？
> - 对于任意一条路径，如果选择该路径的 tokens 超过它的容量，我们如何确定哪些 tokens 要被丢弃呢？

![Figure1](./assets/MoD-Figure2.png)

***Figure2.Explanation*** 当使用 token-choice routing 架构时（左图），tokens 会选择它们倾向的路径。如果一个路径中的 tokens 超过了其容量，那么超过容量的 tokens 将被丢弃（例如图左中的 Expert1 的容量为 2，紫色的 token 便无法再选择 Expert1 而被丢弃）。同时论文中提到，当选择某条计算路径的 token 数超过其容量时，**由底层实现决定具体是哪些 token 被丢弃**。例如，优先级可能按先来后到决定（序列中的先后关系或者是**batch 中的先后关系**）。当使用 expert-choice routing 架构时（中间图），我们则能依据 tokens 的 router 权重和 top-k 为每个路径**选择恰好 $k$ 个 tokens**。没有被任何路径选择的 tokens 将被丢弃（橘色）同时有些 tokens 会被重复选择（黄色）。在 MoD 中，我们将采用 expert-choice routing 架构（右图）。在 MoD 中，仅有一条计算路径（即参与 Transformer Block 的计算），那么没有被选择的 tokens 会被直接残差连接，从而减少前向传播中 FLOPs 的消耗。

MoD 选择 expert-choice routing 架构基于以下几个原因：

- 不需要引入额外的辅助损失来解决可能的负载均衡问题。
- top-k 操作暗含了相对大小的比较，router 可以通过比较所有 tokens 的权重来选择它认为最需要计算的 tokens。如果 router 能得到（学习到）正确权重，那么该架构可以保证最重要的 tokens 一定位于 top-k 集合中，而 token-choice 架构则无法做到这一点。因为每个 token 的选择仅取决于权重，无法进行相对大小的比较。对于我们这个特定的设计，实际上另外一条计算路径是**空操作**，所以确保重要的 tokens 参与计算至关重要。所以 top-k 操作更能满足这个需求。
- 最后，**参考前文我提出的小问题，token-choice 架构难以保证使用准确的计算量**，可能会出现过处理和欠处理的情况，这是因为不同 tokens 选择路径的情况，参与计算的 tokens 数可能不同，计算量自然也不同，并且还需要一些额外的优先级设计来判断抛弃哪些多余的 tokens，**因此 token-choice 的实现会不必要的复杂**。反观 expert-choice 可以**仅用一个 top-k 操作划分出两个不相交的结合，并且能保证计算量的精准控制**。

### 2.4 Routing Implementation

在进行 Routing 架构实现前，可以简单的回顾一下 routing 的流程。简单流程示意：

```
(tokens) - router -> (tokens router weight) - TopK -> (top-k tokens router weight) - SelfAttention, MLP -> (output)
```

tokens 经过 router 得到对应的权重，然后结合 top-k 选择对应的 tokens 集合作为 Transformer block 的输入。

假设，对于给定的 layer $l$ ，我们有长度为 $S$ 的序列中所有 token 的 embedding， 即 $X^{l} = \{ x_{i}^{l} | i \text{ is an integer, } 1\le i \le S\}$ 。那么对于给定的 token embedding 其 router 权重是一个通过线性层计算得到的标量（Scalar）值： $r_{i}^{l} = w_\theta^{T} x_{i}^{l}$​ 。

除了使用 router 权重进行 top-k 集合的选择，**我们的目标是使用 router 权重来决定每个 token 通过 transformer block 的计算输出**。假设 $P_{\beta}(R^l)$ 表示 router 权重集合 $R^l$ 中的 $\beta$  分位数，其中 $\beta = 1 - \frac{C}{S}$ 且 $C$ 是用户定义好的每个 batch 中的计算容量（计算容量就是用来定义将会被处理的 tokens 数，显然我们需要保证 $C < S$ ） 。对于给定的 token，transformer block 输出为：

$$
x_i^{l+1}= \begin{cases}  r_i^l f_i (\tilde{X}^l)+x_i^l, & \text{ if } r_i^l>P_\beta (R^l) \\\\ x_i^l, & \text{ if } r_i^l < P_\beta ( R^l ) \end{cases}
$$

其中 $\widetilde{X}^l$ 是指对应 router 权值为 top-k 的 tokens embedding 集合（即所有满足 $r_i^l > P_\beta(R^l)$ 的 tokens embedding）， $f$ 则代表 transformer block 中的算子（Self-Attention 和 MLP）。你会发现 tokens $i$ 的结果 $x_i^{l+1}$ 同时受到其他 tokens $x_{i\neq j}^{l}$ 的影响， 这是因为 self-attention 操作的存在。由于输入到 $f$ 的 tokens 数量更少（因为 $\widetilde{X}^l$ 的集合大小为 $C$ 且 $C < S$ ），从而降低了用于 Self-Attention 和 MLP 的计算开销，因此 MoD Transformer 仅需要比标准 Transformer 更少的计算量。

你还会发现，我们会将 $f$ 的输出乘上 router 的权重 $r_i^l$ 。**这样会将 router 权重置于梯度路径上，从而在语言建模的过程中受到梯度下降的影响**（也就是 train router）。Google 也说他们尝试了**不同的版本的实现**，即在残差链接的计算路径中也考虑引入 router 权重的影响（可以理解为 $x_i^{l+1} = r_i^l x_i^l$ ）但是后面发现，目前的版本已经能取得更好的效果， 并且实现起来更为简单。

### 2.5 Sampling

> 这部分是我开始困惑最久的内容。我完全无法理解为什么存在这个限制："While expert-choice routing has a number of advantages, it has one distinct problem: the top-$k$ operation is non-causal. **This means that whether a given token’s routing weight is among the top-$k$​ for the sequence depends on the values of the routing weights for tokens that come after it, which we don’t have access to when autoregressively sampling."**

### 2.6 Training methods

所有模型均采用相同的基本超参数配置，例如训练步骤的余弦调度、128的批次大小、2048的序列长度，唯一的变化是在等效浮点操作（isoFLOP）分析期间调整了模型的层数、头数和嵌入大小，以生成不同规模的模型。

## 3. Result

### 3.1 Training, isoFLOP comparisons

选用较小的 FLOP budget(6e18) 来进行最优超参的确定。可以发现在 Figure3 左图中，最优的 MoD 模型有更低的 Loss 同时有更多的参数量（向下且向右）。这个现象的结果是：存在一个更小的 MoD 模型，能达到接近 optimal baseline 的性能，同时有更优的 step 速度。

Figure3 右图便是在解释这种可能。我们选择了 MoD Model#3 （大约有 220M 的参数，图中的3）和 isoFLOP optimal baseline Model#1（也是 220M 的参数，图中的1）进行比较。Model#3 在性能上能略好于 Model#1，但在训练时的 step 性能有着 60%+ 的提升。关键的是，在相同硬件上运行时，这两模型所需要的训练时间大致相同。

> 有些读者可能无法理解这段话，或者匆匆跳过了这段话。但是我觉得这里还是挺值得思考的，**为什么在参数量相同，模型性能接近，训练效率更优秀的情况下，训练时间相同是值得注意的呢？**不应该是只需要更短的训练时间来达到相同的模型性能嘛？！
>
> 这是因为，虽然我们仍然需要相同的训练时间（**考虑到训练效率，意味着更多 epoch 得到接近的性能**）才能得到一个性能相近的模型，但是考虑到 MoD 对 forward 操作的性能优化。这等价于我们得到了一个性能接近但 Inference 效率大大提升的**推理模型**。
>
> 个人的理解是：高效的推理在生产端可能是更重要的？（欢迎指正）

![Figure3](./assets/MoD-Figure3.png)

我们测试了两种路由方式：在**每个块前**路由或**每隔一个块**进行路由，使用的容量从序列总量的 12.5% 到 95% 不等。虽然**每隔一个块**的路由方案对于优秀的性能至关重要，但我们发现激进的容量减少效果最佳（当容量减少到总序列的 12.5% 时，观察到渐进的改进，此时有 87.5% 的令牌绕过块处理，性能在此点之后开始下降）。因此，似乎只要有频繁的机会进行完整容量的自注意力和多层感知机计算，网络对显著的容量减少是鲁棒的。

可学习的路由至关重要，如果 MoD 只使用一个随机的路由（对高斯采样得到的路由权重进行 top-k 操作）会导致性能与原始 baseline 和使用可学习路由的 MoD 有很大的 GAP。

~~Figure3 展示的是 MoD 模型的超参调优。在固定 6e18 的 FLOPs budget 下寻找最优的超参进行 isoFLOP 分析。左图中的灰色框（也就是所有 loss 低于大概 3.13 部分的灰色阴影区域）代表着，如果你能在固定 FLOPs 下，模型的 loss 能降至对应区域，那么该模型的性能就好于**当前的 isoFLOP optimal baseline**。结果发现，最优秀的模型是哪些**可以在每一块有选择路由的模型。**并且 TOP-K 为 256，这代表每个 block，只有 256 个 tokens 能进行计算，剩余的 1792 个 tokens 则直接跳过了（残差链接）。右图则是选定对应的模型。可以发现，模型3能和 optimal baseline 有相同的性能，但是由于 MoD 的性质，只需要更少的 FLOPs，所以性能提升了接近 66%。~~

~~我对这张图的理解便是：相同的 FLOPs，MoD 能训练出效果更好的模型；相同的模型效果，MoD 所需要的 FLOPs 更少，性能更高。~~

> **Q**：`isoFLOP` 是什么？
>
> 这个术语可以理解为，在保持浮点数运算操作次数（FLOPs）相同的条件下，比较不同模型和不同算法的效果。所以这里的 `iso` 也许是 `isometric` 的意思。在初次阅读的时候，全网找来找去也没检索到该术语的起源，后面发现它可能是起源于 NIPS2022 的《[Training Compute-Optimal Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2022/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf)》，有趣的是这同样是 DeepMind 的 Paper。所以这应该是 DeepMind 首创的？（欢迎指正）
>
> **Q**：如果我们固定了 budgets，是自动固定 top-k 中的 k 还是手动呢？
>
> 显然，我认为是自动的，固定的 k 的情况下我们能计算出 Layer 固定需求的 FLOPs，所以给定了 budgets 或者 capacity，我认为 top-k 中的 k 能够直接计算得出。
>
> ==难道它同时需要指定 static budget 和 sequence capacity 嘛？static budget 和 sequence capacity 有啥关系呢？==



### 3.2 Auto-regressive Evaluation



### 3.3 Mixture-of-Depths-and-Experts (MoDE)



## 4. Discussion

> Need refinement.

学习型路由机制有时是非因果的;也就是说,在确定给定标记的路由决策时,会使用关于未来的信息。这通常适用于 top-k 路由机制,因为它们无需辅助平衡损失。然而,top-k 路由机制在后训练自回归采样过程中存在困难,因为在那里无法使用关于未来标记标识的信息来确定路由决策。在这项工作中,我们展示了可以在训练期间成功使用 top-k 路由方案,但在后续的自回归采样中不需要它。一个简单的辅助分类器或对路由器的辅助损失就足以学习 top-k 路由决策,以便在自回归采样期间模仿 top-k 决策,同时最小化或不降低性能。

直观上,标记可能会学习绕过某些块,因为在该步骤中所做的预测更容易,因此不需要太多计算。然而,这种策略显然不是网络所学习的全部。如果某个标记在特定块中不参与自注意力,那么后续的标记也将无法关注它。因此,标记是否决定路由不仅影响当前步骤的预测,还会通过因果自注意力影响未来的预测,而网络如何权衡这些影响是由它们对总体语言模型目标的影响所指导的。

这一见解为 MoD 变体打开了大门,MoD 变体可以分离查询、键和值的路由。例如,对于某个自注意力计算,也许某个标记希望成为查询的一部分,但不希望成为键的一部分。我们可以进一步将这一思路扩展到"长期记忆"的领域:也许有些标记非常有价值作为键,而无论它们在出现时是否有用作查询。学习型路由可能是一种强大的机制,用于决定这些标记是什么,也许会将它们引导到一个长期记忆缓冲区中,在未来的自注意力过程中可用。这种长期记忆方法的一个优点是,标记只需在"记忆编码"的那一刻决定它们将来是否应该被检索。这比在未来的每一步都对整个记忆缓冲区执行完整的基于内容的查找更加高效,并可能是大幅增加用于做出预测的上下文长度的一步。

与在有效相同的计算(通常是 MLP)之间路由的 MoE 转换器不同,MoD 转换器展示了在不同类型的计算之间路由的价值。在这项工作中,类型要么是常规转换器块,要么是空计算(在功能上等同于乘以零)。然而,我们可以进一步扩展这一思路,在更多类型的计算之间路由。例如,也许有些标记会被路由到"内存查找"功能,而另一些则会被路由到"工具使用"功能。一般而言,我们部署的路由机制为调整网络可用的计算类型及其相对成本(总 FLOP 数)提供了一个旋钮;如果要引入一种昂贵的计算,那么可以通过将其容量设置为一个较小的值来抵消,从而只路由少量标记到该计算中。

总的来说,MoD 转换器是另一种可用于调整模型每次前向传递的计算量(从而影响推理时间)的工具。实现 MoD 所使用的机制也是通用的,为许多扩展和与其他技术(如 MoE)的集成打开了大门。

## Reference

- [Blog: FLOPS vs FLOPs](https://blog.csdn.net/Caesar6666/article/details/125102569)
