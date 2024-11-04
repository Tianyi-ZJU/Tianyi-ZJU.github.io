# Transformer

## Chap1

### Transformer在做什么

首先将每个token（单词）转换为一个高维向量（embedding），方向对应语义，如性别。这里涉及到embedding matrix，记作$W_E$,每一列对应一个token的embedding，每一行对应一个维度。

Transformer的目标是逐步调整这些embeddings，使它们不单单编码单个token的信息，还编码了token之间的关系，融入更丰富的上下文语义。这里运用到attention。

之后通过*Unembedding matrix* $W_U$。将上下文中的最后一个向量（这样做效率最高）作用这个矩阵，得到输出后，利用softmax得到最终的输出概率。

softmax：$P_i=\frac{\exp(x_i)}{\sum _i \exp(xi)}$，其中可以加入参数$T$（温度）,$P_i=\frac{\exp(x_i/T)}{\sum _i \exp(xi/T)}$,这样可以控制输出的分布的平滑程度，$T$越大，会给予概率较小的输出更大的概率，反之亦然。

## Chap2

### Attention

#### 单头注意力

在进入Attention后，每个token的embedding会被调整。首先会引入每个token的位置信息，这个向量记作$\vec{E}$，接下来我们要融入上下文语义。

我们对token进行Query，这里引入矩阵$W_Q$，$\vec{E_i}\stackrel{W_Q}{\longrightarrow}\vec{Q_i}$，这会将向量映射到低纬度空间。

接着引入第二个矩阵键（key）矩阵$W_k$，$\vec{E_i}\stackrel{W_k}{\longrightarrow}\vec{K_i}$，同样也会把向量映射到低纬度空间。概念上讲，key视为用于回答query。

接下来我们要检查$\vec{K}$和 $\vec{Q}$的相似度，这里引入点积，$\vec{Q_i}\cdot\vec{K_j}$，可以想象一张表格，左边是$\vec{K}$，上面是$\vec{Q}$，表中的值就是点积。

|             | $\vec{Q_1}$ | $\vec{Q_2}$ | ……   |
| ----------- | ----------- | ----------- | ---- |
| $\vec{K_1}$ |             |             |      |
| $\vec{K_2}$ |             |             |      |
| ……          |             |             |      |

这里我们可以看到，如果两个向量越相似，点积越大。假如点积的值是一个很大的正值，那么这两个向量就很相似，即“the embedding of "xx" **attend to** the embedding of "yy""；反之则反。这个值的范围是$[-\infty,+\infty]$，我们显然不能直接运用这个值，我们再次应用softmax，得到一个概率分布，我们就可以把每一列看做权重。我们把这个网格叫做注意力模式（attention pattern）。容易发现这个矩阵的大小是上下文的平方。事实上，为了使分布更平滑，在点积的时候，我们会除以$\sqrt{d_k}$，其中$d_k$是向量的维度。

事实上，我们会对这个注意力模式对角线下方的值在softmax之前置为$-\infty$，这样在softmax之后，这些值会变为0。这是为了不让后词影响前词，这个操作叫做masking。也有的注意力模式不会应用这个过程，比如在翻译的时候。

引入第三个矩阵值矩阵$W_V$，$\vec{E_i}\stackrel{W_V}{\longrightarrow}\vec{V_i}$，意义上理解为要调整成给后词的嵌入向量，这个矩阵的大小是嵌入向量维度的平方。我们转换表格：

|             | $\vec{E_1}$ | $\vec{E_2}$ | ……   |
| ----------- | ----------- | ----------- | ---- |
| $\vec{V_1}$ |             |             |      |
| $\vec{V_2}$ |             |             |      |
| ……          |             |             |      |

表格中的值（权重）不变，我们将这个表格的值乘上$\vec{V}$，得到一个新的向量，每一列就得到一个嵌入向量$\Delta \vec{E_i}$（把这一列都加起来)，$\vec{E_i}+\Delta \vec{E_i} = \vec{E_i^{\prime}}$ 就是调整后的向量。

以上整个过程就是**单头注意力机制**，这整个过程可以表示为$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$，其中$Q,K,V$分别是Query，Key，Value的矩阵。

我们注意到$W_V$的参数量很大，事实上我们不会这样做。我们把这个矩阵拆分成两个矩阵的乘积，右矩阵列是上下文数量，行较少（降维），暂称为$Value_{\downarrow}\ matrix$(非通用)，左矩阵列较小，是映射回原来的高维空间，暂称为$Value_{\uparrow}\ matrix$。这种操作的实质是低秩分解("Low rank" tramsformation)。

#### 多头注意力

Transformer中完整的注意力机制由多头注意力组成，即多个注意力模式，每个注意力模式中都有不同的参数矩阵，每个注意力模式都会产生嵌入向量$\Delta \vec{E_i}^{(j)}$，最后的向量调整为$\vec{E_i}+\sum_{j}\Delta \vec{E_i}^{(j)}$，形成一个更精准的嵌入。

实际上所有Head的$Value_{\uparrow}\ matrix$会左右拼接在一起，称为输出矩阵(output matrix)，单个注意力头的值矩阵则指上文提到的右矩阵$Value_{\downarrow}\ matrix$。

做完这一过程后向量还会经过多层感知机，之后不断重复这两个过程。

## Chap 3

这个视频里一直在试图解释 MLP 在做什么，这样有益于理解它的结构，但我觉得稍微有点繁琐了，所以我不想记录有关解释的方面。

### MLP

对于每一个流入 MLP 的向量$\vec{v}$，都会经历以下过程：

```mermaid
graph LR

v["$$v$$"] --Linear--> $$v_1$$ --ReLU--> $$v_2$$ --Linear-->$$v_3$$ --->p(("+"))
v--->p--->$$v_o$$
```

> 仅图示，忽略这里的具体符号
>
> mermaid图在网页上可能渲染不出来

先后进入线性层、激活函数、线性层，得到的向量和原来的相加，得到输出向量。

#### part 1

在第一个线性层，我们把需要处理的向量（记作$\vec{E_i}$）进行升维:

$$
W_{\uparrow}\vec{E_i}+\vec{B_{\uparrow}}
$$

$W_{\uparrow}$的行比列更多以进行升维（视频中是$(4\times 12288)\times 12288$），$\vec{B_{\uparrow}}$是bias偏置。

#### part 2

这里的激活函数选的是ReLU，即小于 0 的值 0，大于等于 0 的不变。Neurons指的是 ReLU 过后的向量各个维度的值。

#### part 3

接下来再经过一个线性层，进行降维

$$
W_{\downarrow}\square + \vec{B_{\downarrow}}
$$

这是一轮过程，接着重复多次这个过程。

### 为什么能记录这么多信息

这里提到一个很有意思的结论：一个 $N$ 维空间只能容下 $N$个正交向量，但倘若我们不要求基向量之间严格正交，比如说他们可以在 $89^{\circ} -91^{\circ}$之间，那么随着空间维度的增大，能容下的向量大约是指数增长的 : $exp(\epsilon  \cdot N)$

这也许能够解释为什么Transformer能在维度有限的空间内容下这么多的“概念”。
