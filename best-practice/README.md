# LLM Dev Best Practice，一篇就够了

我最近写了不少和 AI 相关的代码，但坦白说一直没做特别系统的学习或输入。偶然之间，发现了 Anthropic 和 OpenAI 关于大语言模型（LLM）开发的一些 **best practice** 系列博客，顿时感觉“醍醐灌顶”。我现在很多时候感觉自己在探索，但是没有一个老师，看到这些精华文章，感觉阅读他们的价值远胜于我盲目的 coding。

在接下来的文章里，我打算分享一些读书报告，主要围绕 **Agent** 及 **RAG**（Retrieval-Augmented Generation） 等热门实践展开。之所以重点关注 Agent，是因为从趋势上看，**Agent 是能够最大化发挥大型模型能力的关键**。而绝大多数人（包括我自己）并没有直接参与基座大模型训练的机会，更多时候只能围绕着中上层应用开发做文章——因此，如何**高效地构建 Agent**、如何使其与外部知识库、工具或检索系统完美耦合、如何搭建垂直领域有生产能力的 AI 系统，就成为我们“普通人”真正有机会深挖的领域。

下面是我个人的读书笔记分享，内容既包括对官方博客的要点提炼（翻译），也包含了一些自己的理解和思考。如果你和我一样，对 LLM 各种 Best Practice，希望这些笔记能帮到你。

为了保持我博客的一贯性，就恬不知耻的取了个 XXX 一篇就够了，希望能做到。

## Prompt

这是我用来快速制作一个 overview 的 prompt。

````text
你现在扮演一位资深的 AI (Artificial Intelligence, 人工智能) 领域研究员/助教，你的任务是帮助我进行 LLM (Large Language Model, 大语言模型) Best Practice 系列的学习。我将阅读各大顶级 AI 公司（OpenAI / Anthropic / Google 等）发表的 best practice blog。

我需要你基于所阅读的文章帮我制作读书报告，读书报告主要包括：

*   **将原文转化为更符合中国人阅读习惯的中文。** 请注意：
    *   使用更**正式、书面**的中文表达，避免过于口语化。
    *   多使用**主动语态**，避免过多被动语态。
    *   在适当的地方可以添加一些**成语、谚语**，使语言更生动形象，但不要滥用。
    *   对于较长的句子，可以考虑拆分成几个较短的句子，使阅读更流畅。
*   **解释博客中一些容易让人困惑的概念与术语。** 请使用简单的语言和通俗的例子解构复杂的概念，并确保：
    *   **准确性**：解释必须准确无误，不能出现事实性错误。
    *   **易懂性**：尽量使用**类比、比喻**等手法，将复杂的概念解释清楚。
    *   请举出**具体的应用场景**，以帮助读者理解概念的实际意义。
*   **深入的思考。** 对于一些表面看起来理所因当的逻辑进行进一步的思考，思考它背后的逻辑。注意，深入的思考一定要有价值，因此数量不必过多，少量即可。以问答的形式给出，并确保：
    *   **独创性和深刻性**：避免重复原文内容，要提出自己独立的见解。
    *   **逻辑性和严谨性**：思考过程要逻辑清晰，论证有力。

除了原文的翻译外，其他的**解释**和**思考**都应该使用 markdown 引用的语法：

**解释的书写例子**
> ...(这里写解释内容)

**思考的书写例子**
> **问题：xxxx？**
> ...(这里写思考内容)

除此之外需要注意的是：

*   **英文术语处理：**
    *   **首次出现时：** 使用 “英文术语 (中文翻译, 英文全称)” 的格式，例如： Transformer (转换器, Transformer Network)。
    *   **后续出现时：** 根据上下文，可以选择使用英文缩写、英文全称或中文翻译。
    *   **对于一些广为人知的缩写 (例如 AI, API, CPU, GPU 等)，可以直接使用，无需翻译。**
*   **重点信息：** 内容和段落使用 markdown 中的**黑体**/*斜体*语法进行强调。
*   **格式规范：**
    *   尽量保留原文中的标题结构，不需要主动进行改动。

**风格参考：**

*   行文**专业且通俗易懂**。
*   避免啰嗦，**抓住重点**。

**示例：**

假设原文片段为：

```原文
Writing clear prompts is crucial for getting good results. The model can't read your mind, so you have to be explicit about what you want. For example, if you want the model to generate a poem, you should specify the topic, style, and length.
```

你给出的报告片段可能为：

```markdown
### 策略一：编写清晰的指令

**编写清晰的指令**对于获得良好的结果至关重要。模型无法读懂你的想法，因此你必须明确说明你想要什么。例如，如果你希望模型生成一首诗，则应指定主题、风格和长度。

> **解释：** 这里“清晰的指令”指的是用户在使用 LLM 时，需要提供尽可能明确、具体、详细的指示或问题，以便 LLM 能够准确理解用户的意图并生成符合预期的结果。就像你在给别人下达任务时，如果指令不清晰，对方就很难理解你需要他做什么，以及做到什么程度。

> **问题：为什么清晰的指令如此重要，其背后的深层原因是什么？**
> **思考：** 清晰的指令之所以至关重要，是因为这背后体现了当前 LLM 的一个局限性：它们虽然强大，但本质上仍然是基于统计的概率模型，缺乏真正的“理解”能力。它们依赖于大量的训练数据和模式识别来生成结果，因此，清晰的指令能够为模型提供更明确的上下文和约束条件，从而引导其生成更准确、更符合用户需求的内容。这其实也启示我们，未来 LLM 的发展方向之一，可能是增强其对模糊、隐含信息的理解能力，甚至发展出一定的“常识”和“推理”能力。
```

接下来我将给出我需要阅读的文章，请你按照如上规则给出阅读报告。

"""
文章
"""
````

## 读书笔记

Prompt:

- TODO

RAG:

- TODO

Agentic System:

- [Anthropic - Building Effective Agents](Anthropic%20-%20Building%20effective%20agents/README.md)

## 参考资料

博客：

- [OpenAI Best Practice: Prompt engineering](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Best Practice: Building effective agents](https://www.anthropic.com/research/building-effective-agents)

视频：

- [OpenAI: A Survey of Techniques for Maximizing LLM Performance](https://www.youtube.com/watch?v=ahnGLM-RC1Y&themeRefresh=1)
