# Federated Transformer
> 标题: Federated Transformer: Multi-Party Vertical Federated Learning on Practical Fuzzily Linked Data<br>
> Oct 2024<br>
> 领域综述<br>
> 文章地址: https://arxiv.org/abs/2410.17986<br>

#### Abstract
**联邦学习 (Federated Learning, FL)** 允许多个参与方在不共享原始数据的情况下协同训练模型. 在 FL 的各种变体中, **垂直联邦学习 (Vertical Federated Learning, VFL)** 在跨组织协作中尤为重要, 每个参与方为共享实例贡献不同的特征. 在这种场景下, **模糊标识符 (fuzzy identifiers)** 通常用于跨参与方链接数据, 从而引入了 **多方模糊 VFL (multi-party fuzzy VFL)**.

---

#### Key Challenges
- **现有方法的局限性**:
  - 当前模型只能处理以下两种情况:
    - **多方 VFL (multi-party VFL)**.
    - **两方模糊 VFL (fuzzy VFL between two parties)**.
  - 将这些模型扩展到 **多方模糊 VFL** 时会导致:
    - 性能下降.
    - 隐私保护成本增加.

---

#### Proposed Solution: Federated Transformer (FeT)
- **概述**:
  - 提出一种新颖框架 **Federated Transformer (FeT)**, 专为具有 **模糊标识符** 的 **多方 VFL** 设计.
  - 使用分布式 **transformer 架构** 对模糊标识符进行编码, 转化为数据表示.

- **关键创新**:
  1. 引入三种新技术以提升性能.
  2. 实现了一个结合以下技术的 **多方隐私框架 (multi-party privacy framework)**:
     - **差分隐私 (Differential Privacy, DP)**.
     - **安全多方计算 (Secure Multi-Party Computation, SMPC)**.
  3. 在保护本地数据表示的同时, 最大限度降低效用成本.

---

#### Results
- **性能 (Performance)**:
  - 在扩展到 **50 个参与方**时, FeT 的准确率比基线模型提高最多 **46%**.
  - 在 **两方模糊 VFL** 设置中, FeT 相较于最先进的 VFL 模型在性能和隐私保护上均表现更优.

- **隐私 (Privacy)**:
  - 增强的隐私框架确保本地表示的安全共享, 并减少效用损失.

---

#### Significance
**FeT** 是一个强大的解决方案, 能够将 **模糊多方 VFL** 扩展到现实应用中, 在隐私、效用和性能之间实现良好平衡.


### 1 Introduction

**Federated Learning (FL)** 是一种学习范式, 允许多个参与方在保护本地数据隐私的情况下协同训练模型. 在 FL 的多种形式中, **垂直联邦学习 (Vertical Federated Learning, VFL)** 在现实应用中尤为流行, 正如一项技术报告所强调的那样. 在 VFL 中, 各参与方持有相同实例集合的不同特征, 通过共同特征 (例如姓名或地址) 作为标识符 (即键) 来跨参与方链接数据集.

现实应用中常需要 **多方模糊 VFL**, 其具备以下两个关键特性:
1. **支持多方协作**:
   - 常见于跨医院协作、传感器网络和金融机构之间的合作.
2. **支持模糊标识符**:
   - 在跨参与方通过地址等模糊标识符进行链接的场景中尤为重要.
   - 例如, 多家共享起点和终点地址的车辆租赁公司可以协作预测旅行时间.

---

#### **Motivation**
多方模糊 VFL 的重要性可以通过一个城市内的旅行成本预测示例来说明:
- **场景**: 出租车、汽车、自行车和公交公司合作预测旅行成本.
- **需求**: 
  - 旅行信息为隐私数据, 无法共享, 因此需要 **VFL**.
  - 路线标识符 (起点和终点 GPS 位置) 只能通过模糊方法进行链接.
- **优势**: 使用模糊标识符的多方模糊 VFL 能显著提升预测准确率.

---

#### **Existing Limitations**
当前的 VFL 方法通常处理以下两种场景之一:
1. **多方 VFL**:
   - 使用 **Private Set Intersection (PSI)** 链接数据集, 但需要存在精确的通用键, 而这在涉及模糊标识符的场景中不现实.
2. **两方模糊 VFL**:
   - 利用跨方键的相似性进行训练, 但扩展到多方模糊 VFL 时存在以下问题:
     - 性能下降.
     - 隐私保护成本高.

---

#### **Challenges**
1. **性能下降**:
   - 随着模糊标识符参与方数量增加, 面临以下问题:
     - 键对数量的二次增长.
     - 模糊标识符的错误链接增加.
     - 模型规模扩大, 导致过拟合风险增加.
2. **隐私保护成本上升**:
   - 多方数据关联会导致:
     - **计算成本显著增加** 或 **准确率损失**.
3. **通信瓶颈**:
   - 标签方 (主方) 与无标签方 (次方) 在每轮训练中需频繁通信, 导致主方通信开销显著增加.

---

#### **Proposed Solution: Federated Transformer (FeT)**
为了解决上述问题, 我们提出了 **Federated Transformer (FeT)**, 在多方模糊 VFL 中提升性能并降低隐私保护成本.

---

##### **1. Tackling Performance Issues**
- **方法**:
  - 将键相似性编码为根据positional encoding averaging对齐的representations, 消除键对的二次计算需求.
  - 设计了 **Trainable Dynamic Masking Module (可训练动态掩码模块)**:
    - 自动过滤错误链接的键对.
    - 在 **MNIST 数据集上的 50 方模糊 VFL** 中, 准确率提升高达 **13%**.

---

##### **2. Mitigating Privacy Protection Costs**
- **方法**:
  - 引入 **SplitAvg**, 一种混合方法, 结合:
    - **基于加密的方法**.
    - **基于噪声的方法**.
  - **优势**:
    - 即使参与方数量增加, 噪声水平仍保持一致.
    - 降低隐私保护的成本.

---

##### **3. Reducing Communication Overhead**
- **方法**:
  - 实施 **Party Dropout 策略**:
    - 在每轮训练中随机排除部分次方.
  - **效果**:
    - 通信成本降低约 **80%**.
    - 提升模型泛化性能.

---

##### **Key Contributions**
1. **设计 Federated Transformer (FeT)**:
   - 在多方模糊 VFL 场景中表现出色.
2. **提出 SplitAvg**:
   - 通过保护本地表示增强 FeT 的隐私保护能力, 并通过理论证明其满足 **差分隐私**.
3. **实验结果**:
   - **FeT 在 50 方 VFL 场景中准确率比基线模型提高最多 46%**.
   - 即使在传统的两方模糊 VFL 场景中, FeT 在性能和隐私保护上仍优于最先进的模型.

### 2 Preliminaries

本节介绍理解我们方法所需的基础概念, 特别是 **Differential Privacy (DP)** 的基本框架和理论.

---

#### **Definition of Differential Privacy**
- **定义**:
  - **Differential Privacy (DP)** 提供了一种严格的数学框架, 用于保护个体隐私.
  - 它通过衡量两个相邻数据库 (即仅差一个记录的数据库) 生成相同输出的概率来量化隐私.
- **定义 1**:
  - 考虑一个随机函数 \( M : R^d \to O \), 和两个相邻数据库 \( D_0, D_1 \sim R^d \), 其仅相差一条记录.
  - 如果对任意可能的输出集合 \( O \subseteq O \), \( M \) 满足以下条件:
    \[
    \Pr[M(D_0) \in O] \leq e^\epsilon \Pr[M(D_1) \in O] + \delta,
    \]
    则称 \( M \) 满足 \( (\epsilon, \delta) \)-differential privacy, 其中 \( \epsilon \geq 0 \), \( \delta \geq 0 \).

ϵ (隐私损失):
ϵ 越小，两个概率越接近，隐私保护越强。
δ (松弛项):
表示隐私保证可能失效的小概率（例如，极端情况）。

1. 如果𝑀满足(ϵ,δ)-差分隐私，意味着无论一个个体的数据是否在数据库中，机制的输出概率差异都受到严格限制，保证了攻击者无法通过𝑀的输出推断出任何特定个体的显著信息。
2. ϵ 和 δ 越小，隐私保证越强，但可能会降低机制的效用（例如准确性或有用性）
---

#### **Mechanisms for Differential Privacy**
- **机制**:
  - 满足 DP 的单个查询被称为一个机制 (**Mechanism**).
  - **高斯机制 (Gaussian Mechanism)**:
    - 通过向函数输出添加高斯噪声实现 DP.
- **定理 1 (Gaussian Mechanism)**:
  - 给定函数 \( f : X \to R^d \), 其全局 \( L_2 \)-敏感度为 \( \Delta_2 \), 表示 \( f \) 在任意两个相邻数据库上的输出在 \( L_2 \)-范数中的最大差异.
  - 对任意 \( \epsilon \geq 0 \) 和 \( \delta \in [0, 1] \), **高斯机制**定义为:
    \[
    M(x) = f(x) + Z, \quad Z \sim N(0, \sigma^2 I),
    \]
    其中 \( Z \) 表示均值为 0, 方差为 \( \sigma^2 \) 的高斯噪声.
  - 若满足以下条件, 则该机制满足 \( (\epsilon, \delta) \)-differential privacy:
    \[
    \Phi\left(\frac{\Delta_2}{2\sigma} - \frac{\epsilon\sigma}{\Delta_2}\right) - e^\epsilon \Phi\left(-\frac{\Delta_2}{2\sigma} - \frac{\epsilon\sigma}{\Delta_2}\right) \leq \delta,
    \]
    其中 \( \Phi(t) \) 是标准正态分布的累计分布函数 (CDF).

---

#### **Composition of Gaussian Mechanisms**
- **多次查询**:
  - 当对同一数据库进行多次查询时, 每次查询独立添加高斯噪声以保持 DP.
- **隐私损失的累积**:
  - 多次查询组合时的隐私损失由 **Moments Accountant** 描述.
- **定理 2 (Moments Accountant)**:
  - 存在常数 \( c_1 \) 和 \( c_2 \), 对于采样概率 \( q = \frac{L}{N} \) 和训练步数 \( T \), 若 \( \epsilon < c_1 q^2 T \), 则 **Differentially Private SGD (DPSGD)** 满足 \( (\epsilon, \delta) \)-differential privacy, 条件是选择:
    \[
    \sigma > c_2 \frac{q \sqrt{T \log(1/\delta)}}{\epsilon}.
    \]


### 3 Related Work

---

#### **1. Performance**

##### **Limitations of Traditional Methods**
- **Two-Party VFL**:
  - 传统的 VFL 方法通常局限于 **两方场景**.
- **Multi-Party VFL**:
  - 现有多方 VFL 方法假设存在 **精确标识符**, 保证各方数据的完美对齐.
  - 这些方法通常采用 **SplitNN Framework**:
    - 每方维护部分模型, 通过表示和梯度传递进行协同训练 (即 **Split Learning**).
  - **问题**:
    - 精确对齐的需求在许多实际场景中不现实, 因为标识符通常不精确.

##### **Emerging Semi-Supervised VFL**
- **目标**:
  - 利用未关联记录, 通过半监督或自监督学习提升模型性能.
- **问题**:
  - 仍假设数据可以通过精确标识符对齐, 而这种假设在现实中往往不可行.
  - 数据关联质量显著影响 VFL 准确性, 探索有效的关联方法仍是关键问题.

##### **FedSim and Fuzzy Keys**
- **方法**:
  - 基于德国记录链接中心 (**German Record Linkage Center, GRLC**) 的真实项目设计:
    - 承认参与方的键通常无法精确对齐.
    - 支持一对多的模糊链接场景, 如 **GPS 地址**.
  - **优势**:
    - 通过软链接和键相似性传输提高训练性能.
  - **局限**:
    - 不具备扩展到多方的可扩展性.
    - 直接传输相似性引入新的隐私问题.

##### **FeT's Contribution**
- **可扩展性**:
  - 设计可扩展架构, 有效应对模糊键在多方场景中的性能挑战.
- **性能**:
  - 相较于 **FedSim**, 在 **多方模糊 VFL** 和 **两方场景** 中均显示出显著的性能改进.

---

#### **2. Privacy**

##### **Privacy Concerns in VFL**
1. **主方对次方的推断**:
   - 主方可能从次方的数据表示中推断信息.
2. **次方对主方的推断**:
   - 次方可能从主方的梯度中推断信息.
3. **外部攻击**:
   - 外部攻击者可能通过部署的模型进行 **Membership Inference Attack**.

##### **Focus of This Paper**
- 本文主要解决 **次方对主方的推断问题**, 即保护representations.
- **其他隐私问题**:
  - 作为开放挑战, 在本文中未解决.

##### **Existing Privacy Methods**
- **Encryption-Based Methods (加密方法)**:
  - 利用计算密集型的加密技术保护中间结果.
  - **问题**:
    - 多方扩展时通信开销显著.
- **Noise-Based Methods (噪声方法)**:
  - 通过扰动或操作本地表示来保护数据.
  - **问题**:
    - 缺乏理论隐私保证.
    - 多方扩展时需引入大量噪声, 导致性能下降.

##### **FeT's Combined Strategy**
- **混合方法**:
  - 结合加密方法和噪声方法:
    - 确保模型在多方扩展中无需过多噪声.
  - **优势**:
    - 提供更高效的隐私保护.
    - 在隐私与性能之间实现更好的平衡.

### 4 Problem Statement

本节正式定义了 **多方模糊垂直联邦学习 (Vertical Federated Learning, VFL)** 的概念.

---

#### **1. Problem Formulation**
我们考虑一个 **监督学习任务**, 其中:
- **参与方**:
  - **主方 (Primary Party, \( P \))**:
    - 持有标签的参与方.
    - 数据记录记为 \( x^P := \{x_i\}_{i=1}^n \), 对应的标签为 \( y := \{y_i\}_{i=1}^n \).
  - **次方 (Secondary Parties, \( S_k \))**:
    - \( k \) 个不持有标签的次方, 每个次方 \( S_k \) 拥有自己的数据集 \( x^{S_k} \).
- **标识符 (Identifiers)**:
  - 各方共享的共同特征, 表示为 \( x_i = [k_i, d_i] \), 其中:
    - \( [\cdot] \): 表示特征拼接.
    - \( k_i \): 标识符, 可能存在不精确性和模糊性, 即使其值在同一范围内.

---

#### **2. Optimization Objective**
联邦学习任务的目标是最小化以下损失函数:
\[
\min_{\theta} \frac{1}{n} \sum_{i=1}^n L(f(\theta; x_i^P, x^{S_1}, \ldots, x^{S_k}); y_i) + \Omega(\theta),
\]
其中:
- \( L(\cdot) \): 损失函数.
- \( f(\cdot) \): 模型函数.
- \( \theta \): 模型参数.
- \( \Omega(\theta) \): 正则化项, 用于防止过拟合.
- \( n \): 主方 \( P \) 的样本数量.

---

#### **3. Threat Model**
本研究重点防御 **特征重建攻击 (Feature Reconstruction Attacks)**:
- **攻击目标**:
  - 针对主方共享的本地representations进行信息推断.
- **假设**:
  1. **诚实但好奇 (Honest-but-Curious)**:
     - 各方遵守协议, 但可能尝试推断其他方的额外信息.
  2. **无共谋**:
     - 假设各方不会互相串通.
- **未涵盖的威胁**:
  - **标签推断攻击 (Label Inference Attacks)**:
    - 攻击目标为标签信息.
  - **后门攻击 (Backdoor Attacks)**:
    - 攻击目标为标签和梯度信息.
  - **说明**: 这些威胁超出本研究范围, 将在未来工作中探索.

---

#### **Key Points**
1. **问题定义**:
   - 联邦学习在多方模糊标识符的场景下最小化全局损失函数.
2. **威胁模型**:
   - 本研究专注于特征隐私保护, 假设环境中无恶意参与者且无串通行为.
3. **未来研究**:
   - 进一步探索其他攻击形式对联邦学习系统的影响.



### 5 Approach

本节解决多方模糊 VFL 中的性能与通信挑战, 提出了一种基于 **Transformer** 的架构 **Federated Transformer (FeT)**. FeT 编码键信息为数据表示, 减少对键相似性的依赖, 并通过引入以下三种技术来提升性能和降低通信成本:

1. **Dynamic Masking (动态掩码)**:
   - 通过可训练的动态掩码模块, 准确排除错误关联的数据记录.
2. **Party Dropout (参与方丢弃)**:
   - 随机失效部分参与方以缓解通信瓶颈和模型过拟合.
3. **Positional Encoding Averaging (位置编码平均)**:
   - 解决参与方之间的位置编码对齐问题, 提升模型表现.

---

#### **5.1 Model Structure**

**架构概述**:
- 每个次方 (Secondary Party) 拥有一个 **Encoder**.
- 主方 (Primary Party) 包含一个 **Encoder** 和一个 **Decoder**.
- 编码器和解码器均基于标准 **Transformer** 模型.
- **关键技术**:
  - 使用多维位置编码 (**Multi-Dimensional Positional Encoding**) 将键信息整合到特征向量中.
  - 次方的编码器输出会被聚合后输入到主方的解码器.

**模块细节**:
1. 隐私机制详见 **Section 6**.
2. 训练过程详见 **Section 5.2**.

接下来具体介绍三种性能优化与通信成本降低技术.

---

#### **Dynamic Masking (动态掩码)**
- **问题**:
  - 邻域大小因参与方和键值不同而差异显著.
  - 包含过多邻居可能导致模型难以提取有效信息, 从而引发过拟合.
- **解决方案**:
  - 引入动态 “Key Padding Mask”:
    - 由键值通过可训练的 **MLP** 模块生成掩码.
    - 过滤掉远离主方键值的数据记录, 减少无关数据对模型的影响.
  - **隐私优势**:
    - 使用键值生成掩码而非相似性, 防止跨方传输相似性数据.

**可视化分析**:
1. 动态掩码专注于主方键值的局部区域:
   - 冷色区域 (远离的次方键值) 的数据记录被赋予小的负掩码值, 在注意力层中权重减小.
2. 不同样本的焦点区域范围与方向各异:
   - 例如, 左图聚焦于底部区域, 中图聚焦于顶部区域, 右图则覆盖全局区域.

---

#### **Party Dropout (参与方丢弃)**
- **问题**:
  - 主方通信带宽开销随着参与方数量线性增长, 成为瓶颈.
  - 多参与方可能导致参数过多, 引发过拟合.
- **解决方案**:
  - 借鉴传统 **Dropout**, 随机将部分参与方的表示设为 0.
  - **效果**:
    - 正则化模型, 降低过拟合风险.
    - 将主方的通信开销减少 **80%**.
    - 提高扩展到大量参与方场景的可扩展性.
  - 为了确保训练和测试阶段表示的尺度一致，Party Dropout 会在 SplitAvg 框架中动态调整。具体来说，当有rd比例的参与方被丢弃时，仅用未被丢弃的参与方数量(1−rd)k进行平均。这样保证了无论rd 的取值如何，最终的平均表示尺度始终保持一致。

---

#### Positional Encoding Averaging (位置编码平均)

**问题背景**:
- 在位置编码 (**Positional Encoding, PE**) 中, 通常期望编码表示之间的距离与标识符之间的距离呈正相关.
- 在 **Federated Transformer (FeT)** 中:
  - 每个参与方有其独立的编码器和位置编码层 (**PE Layer**), 负责将本地标识符编码为表示.
  - 由于每方独立编码, 导致严重的 **位置编码未对齐 (PE Misalignment)** 问题:
    - 每个参与方内部, 标识符与对应的编码表示呈正相关.
    - 跨参与方之间, 标识符与编码表示几乎无相关性.
- 这种相关性缺失会导致数据整合问题, 从而影响模型准确性.
- 跨所有参与方直接共享 PE 层不可行, 因为会暴露隐私信息.

**解决方案**: **位置编码平均 (Positional Encoding Averaging)**

- **方法**:
  - 每隔 \( T_{\text{pe}} \) 个 epoch, 在安全多方计算 (**Secure Multi-Party Computation, MPC**) 框架下:
    1. 对所有参与方的 PE 层进行平均处理.
    2. 将平均后的 PE 层广播到所有参与方, 确保一致性.
    - 类似于横向联邦学习中的 **FedAvg** 方法.

- **隐私性**:
  - 尽管传输模型的隐私性可能受到关注, 但这是横向联邦学习中的一个独立性开放问题, 不影响本文方法的有效性.

**优势**:
- 通过位置编码平均, 解决了跨参与方的 PE 未对齐问题.
- 在确保隐私的同时提升数据整合能力和模型准确性.


### 5.2 Training

本节描述 **Federated Transformer (FeT)** 的训练过程, 其关键步骤如下:

---

#### **Training Overview**
1. **Privacy-Preserving Record Linkage (PPRL)**:
   - 主方 \( P \) 和每个次方之间计算标识符的相似性.
   - 次方提供随机子集用于链接 (算法第 5 行).
   - 对于主方的每条记录, 从次方的这些子集中确定 \( K \) 个最近邻 (第 6 行).

2. **Transformer 数据嵌入维度**:
   - 数据嵌入的维度为 \( B \times L \times H \):
     - \( B \): 批量大小 (**Batch Size**).
     - \( L \): 序列长度 (**Sequence Length**), 对主方 \( L = 1 \), 对次方 \( L = K \).
     - \( H \): 隐藏层大小 (**Hidden Layer Size**).

3. **处理流程**:
   - 使用多维位置编码 (**Multi-Dimensional Positional Encoding**) 将标识符转换为向量, 并与数据表示结合, 输入到自注意力模块中 (第 7, 10 行).
   - 次方的表示在安全多方计算 (**MPC**) 协议下进行平均 (第 12 行).
   - 主方利用注意力模块进行前向传播, 计算最终预测结果 (第 13 行).

4. **反向传播**:
   - 主方向次方发送梯度更新, 优化其本地模型 (第 14-16 行).

5. **隐私保护机制**:
   - **Norm Clipping**:
     - 对梯度进行归一化裁剪 (第 8, 11 行).
   - **分布式高斯噪声**:
     - 在聚合阶段添加噪声保护 (第 12 行).
   - 详见 **Section 6**.

<figure style="display: block; text-align: center;">   <img src="FL/2024_11_23/images/2024-11-25-10-52-25.png" alt="name" style="display: block; margin: auto; width: 100%; height: auto;"></figure>

### 7.2 Performance

---

#### **1. Two-Party Fuzzy VFL**

**实验目标**: 
评估 FeT 在两方模糊 VFL 设置下的性能, 不使用隐私保护机制.

**实验结果**:
- FeT 在所有评估指标上均优于现有的领先两方模糊 VFL 方法.
- **显著改进**:
  - FeT 不涉及相似性数据的传输, 同时提升了隐私保护性能.
  
| **Algorithm** | **house (RMSE)**    | **bike (RMSE)**     | **hdb (RMSE)**      |
|---------------|---------------------|---------------------|---------------------|
| Solo          | 73.27 ± 0.16       | 244.33 ± 0.75       | 33.97 ± 0.61        |
| Top1Sim       | 58.54 ± 0.35       | 256.19 ± 1.39       | 31.56 ± 0.21        |
| FedSim        | 42.12 ± 0.23       | 235.67 ± 0.27       | 27.13 ± 0.06        |
| **FeT**       | **39.75 ± 0.29**   | **232.98 ± 0.62**   | **26.94 ± 0.15**    |

---

#### **2. Effect of Number of Neighbors K**

**实验目标**: 
通过改变邻居数量 \( K \) (从 1 到 100) 评估 FeT 的性能.

**结果与分析**:
1. **性能随 K 增大而提升**:
   - FeT 能有效筛选有用信息, 即使无关数据记录数量增加, 仍能保持性能优势.
2. **对比基线**:
   - 在较大的 \( K \) 值下, FeT 始终优于所有基线方法, 突显其在模糊 VFL 场景中的卓越性能.

---

#### **3. Effect of Number of Parties**

**实验目标**:
在不同参与方数量下评估 FeT 的性能 (使用合成数据).

**实验设置**:
- 特征随机平分到多个参与方, 主方的特征维度通过 **PCA** 降至 4.
- 模拟模糊链接场景, 在每个参与方的键上添加独立高斯噪声 (噪声规模为 0.05).

**实验结果**:
1. **优势随参与方数量增加而显现**:
   - FeT 在大多数参与方数量设置下表现优于基线方法.
2. **基线对比**:
   - **Solo**: 缺乏信息性特征, 导致性能不足.
   - **Top1Sim**: 链接受噪声影响, 性能受限.
   - **FedSim**: 由于次方对主方键值缺乏了解, 导致软链接和训练步骤中的错位, 性能不佳.
3. **特殊情况**:
   - 在 **gisette** 数据集 (\( k = 10 \)) 中, 所有模型的性能略低于 Solo, 可能由于数据集较小导致的过拟合.

---

#### **Figure Analysis**
- **Figure 6**:
  - 展示了不同邻居数量 \( K \) 对 FeT 性能的影响.
  - FeT 在所有 \( K \) 设置下均表现出色, 尤其在较大 \( K \) 时明显优于基线方法.
- **Figure 7**:
  - 不同参与方数量下 FeT 和基线方法的性能对比.
  - FeT 在多数场景中均优于其他方法, 随参与方数量增加, 性能优势进一步显现.
