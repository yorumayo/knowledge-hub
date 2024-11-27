## Entity Alignment Approaches
> 标题<br>
> An Experimental Study of State-of-the-Art Entity Alignment Approaches<br>
> Introduction to Entity Alignment<br>
> 2020, 2023<br>
> 领域综述<br>
> 文章地址: <br>
> https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9174835<br>
> https://link.springer.com/chapter/10.1007/978-981-99-4250-3_1<br>
---
## 1. Intro
### 1.1.  Background
#### 1.1.1. KG
1. 存储形式
   * triples <entity, relation, entity>
   * 每个entity对应一个unique object in the real world, relations描述这些对象之间的联系
   * triples内在互联(inherently interconnected)
   * e.g. https://dbpedia.org/page/Spain
2. Core Compononets
   * Entity Set (E): 图中的所有节点, 对应现实中的唯一对象
   * Relation Set (R): 描述实体间的关系类型
   * Triple Set (T): 表示图中的有向边, T ⊆ E × R × E
   * 单个三元组 (h, r, t) 表示头实体 h 与尾实体 t 通过关系 r 的连接
3. Types of KGS
   * General KG
     * Cover a wide range of information across multiple domains.
     * Examples: DBpedia, YAGO, Google’s Knowledge Vault.
   * Domain-Specific KG
     * Focus on specific domains for more detailed information.
     * Examples: Medical KGs, Scientific KGs.
4. Applications of KG
   * **Keyword Search**: Enables more precise and effective retrieval.
   * **Fact-Checking**: Validates the authenticity of information.
   * **Question Answering**: Provides knowledge-based responses to complex queries.
5. Integration of KG
   * Construction Limitations: Typically built from a single data source, limiting domain coverage.
   * Integration:
     * Combine information from other KGs to address gaps or complement existing data.
     * Example: A general KG might only include basic details about a scientist. A domain-specific KG could provide additional details, such as biographies and publication lists.

#### 1.1.2. EA

1.1.2.1. Definition
- **Purpose**:
  - 实体对齐(EA)的目标是识别不同知识图谱(KGs)中等价的实体, 从而实现不同来源知识的整合.
- **Definition**:
  - **Source Knowledge Graph (KG)**: \( G_1 = (E_1, R_1, T_1) \)
  - **Target Knowledge Graph (KG)**: \( G_2 = (E_2, R_2, T_2) \)
  - **Seed Entity Pairs** (训练集): \( S = \{(u, v) | u \in E_1, v \in E_2, u \leftrightarrow v \} \), 其中 \( u \leftrightarrow v \) 表示等价关系(即 \( u \) 和 \( v \) 指代同一现实世界对象).
  - **目标**: Identify equivalent entity pairs in the test set.

1.1.2.1.* Example

<img src="FL/2024_11_23/EA_1.png" alt="EA" style="display: block; margin: auto; width: 50%; height: auto;">

- **Context**:
  - 对齐英语知识图谱 (\( KG_{EN} \)) 和西班牙语知识图谱 (\( KG_{ES} \)) 中的实体.
- **Given**:
  - **Seed Entity Pair**: “Mexico” (\( KG_{EN} \)) ↔ “Mexico” (\( KG_{ES} \)).
  - **目标**: Given the seed entity pair, EA aims to find the equivalent entity pairs in the test set, e.g., returning Roma(ciudad) in KG(ES) as the corresponding target entity to the source entity Roma(city) in KG(EN).
- **Entity Uniqueness**:
  - Each entity is uniquely identified, 例如 “Roma(film)” 和 “Roma(city)” 是不同的实体.

1.1.2.2. Method

1. **Assumption**:
   - Entities that are equivalent in different KGs share **similar local structures** (e.g., neighbors and relationships).

2. **Representation Learning**:
     - Entities are embedded into a low-dimensional feature space as data points.

3. **Process**:
   - **Calculate Dissimilarity**:
     - The pairwise dissimilarity of entities is measured as the **distance** between their embedded data points in the feature space.
   - **Entity Matching**:
     - Based on the distance, it is determined whether two entities are equivalent or not.


### 1.2 Related Works

虽然EA问题是近年来才被提出的, 但这一问题的更通用版本——识别来自不同数据源的、指向同一现实世界实体的记录——已经被不同领域的研究社区从多个角度进行了深入研究, 并使用了不同的名称, 包括实体解析(Entity Resolution, ER)、实体匹配(Entity Matching)、记录链接(Record Linkage)、去重(Deduplication)、实例/本体匹配(Instance/Ontology Matching)、链接发现(Link Discovery) 以及实体链接/消歧(Entity Linking/Entity Disambiguation).接下来是对其中部分相关工作的描述.

#### 1.2.1 Entity Linking (EL)
- **Definition**:
  - Entity Linking (EL) 或 Entity Disambiguation 的过程是指在自然语言文本中识别实体提及 (Entity Mentions) 并将其链接到给定参考目录中的对应实体, 这个参考目录通常是一个KG. 该过程包括确定文本中的某个具体提及指代的是哪个实体. 例如, 给定单词 “Rome”, 任务是确定它是指意大利的城市, 一部电影, 还是其他实体, 并将其链接到参考目录中的正确实体.
- **Techniques**:
  - Prior studies have used various information sources, such as:
    - **Surrounding words** for context.
    - **Prior probabilities** of target entities.
    - **Disambiguated mentions** for cross-referencing. (已经消歧的实体mentions)
    - **Background knowledge** from sources like Wikipedia.
- **Limitations**:
  - 在需要对齐KGs的场景中, 这些信息大多无法获取, 如Entity Embeddings或实体链接的先验分布. 此外, EL关注的是将自然语言文本映射到KGs, 而本研究则探讨如何在两个KG之间进行实体映射.

---

#### 1.2.2 Entity Resolution (ER)
- **Definition**:
  - Entity resolution, 也被称为 entity matching, deduplication, 或 record linkage, 假设输入数据是关系型数据, 每个数据对象通常包含大量通过多个属性描述的文本信息. 因此, 在 entity resolution 中, 各种相似度或距离函数被用来衡量两个对象之间的相似性. 基于相似度的测量, 可以采用基于规则的或基于机器学习的方法(Rule-based or machine-learning-based)来将两个对象分类为 matching 或 non-matching.
- **Process**:
  1. **Attribute Alignment**:
     - Align attributes of data objects manually or automatically.
  2. **Similarity Calculation**:
     - Use similarity or distance functions (e.g., Jaro-Winkler distance for names, numerical distance for dates).
  3. **Classification**:
     - Combine similarity scores of aligned attributes to classify object pairs as matching or non-matching.
- **Techniques**:
  - Rule-based or machine-learning-based approaches based on similarity scores.

---

#### 1.2.3 Entity Resolution on KGs

## 2. SOTA approaches
> 提出了一个涵盖所有现有方法的广泛 EA 框架, 并将这些方法分为三大类<br>
> 在各种场景下对这些解决方案进行了评估, 考虑了它们的有效性 (efficacy), 效率 (efficiency), 和可扩展性 (scalability).<br>
> 创建了一个新的 EA 数据集, 该数据集反映了对齐过程中实际遇到的困难, 而这些困难在现有文献中大多被忽略<br>

### 2.1 Intro
---

#### 2.1.1. Fair Comparison Within and Across Categories
- **Limitations of Existing Studies**:
  - 大多数最新研究仅比较部分方法.
  - 不同方法遵循不同协议:
    - 一些仅使用 **Knowledge Graph (KG)** 的结构进行对齐.
    - 一些结合了额外信息.
    - 一些执行单次对齐(one-pass alignment).
      - 模型从输入数据中直接生成对齐结果, 而不需要进一步的优化循环或重新训练.常用于简单的规则匹配方法或非迭代模型.
    - 一些采用迭代 (或重新训练) 策略.
  - 文献中的直接比较突出整体有效性, 但缺乏分类内及分类间的公平比较.

- **Our Approach**:
  - 本章纳入了大多数最先进的 **Entity Alignment (EA)** 方法进行全面比较, 包括之前未与其他方法比较的最新方法.
  - 将方法分为三组:
    - 对每组内方法进行详细分析 (intra-group evaluation).
    - 对不同组间方法进行比较 (inter-group evaluation).
  - 提供更全面的性能评估与方法定位.

---

#### 2.1.2. Comprehensive Evaluation on Representative Datasets
- **Dataset Categories**:
  - **Cross-lingual Benchmarks**:
    - 例如: DBP15K.
  - **Mono-lingual Benchmarks**:
    - 例如: DWY100K.

- **Limitations of Existing Datasets**:
  - 数据集中的 **KGs** 密度较高, 与实际场景不符.
  - 仅在一到两个数据集上报告结果, 难以评估在不同场景 (如跨语言/单语言, 高密度/正态分布, 大规模/中等规模) 下的效果.

- **Our Approach**:
  - 对所有重要数据集 (DBP15K, DWY100K, SRPRS) 进行全面实验评估.
  - 数据集共包含九对知识图谱.
  - 评估维度:
    - **Effectiveness (有效性)**.
    - **Efficiency (效率)**.
    - **Robustness (鲁棒性)**.

---

#### 2.1.3. New Dataset for Real-Life Challenges
- **Challenges in Existing Datasets**:
  1. 假设每个源 KG 的实体在目标 KG 中都有对应实体, 但实际中并非如此.
     - 例如: YAGO 4 和 IMDB 中, YAGO 4 的大部分实体 (99%) 无法在 IMDB 中匹配.
  2. 假设不同 KG 的实体共享相同命名规范:
     - 基于字符串相似度的基准方法在这些数据集中可达到完美准确率, 但这一假设在现实场景中无效 (例如 “America” 与 “USA”).
  3. 忽视了同一 KG 中不同实体可能具有相同名称的情况:
     - 例如: “Paris” 在源 KG 和目标 KG 中可能分别指代法国城市和德州城市.

- **Our Contribution**:
  - 引入了一个新的单语言数据集, 更贴近现实中的以下挑战:
    - **Unmatchable Entities (无法匹配的实体)**.
    - **Ambiguous Entity Names (歧义实体名称)**.

---


#### 2.1.4. Main Contributions of This Chapter
1. **Comprehensive Evaluation**:
   - 提供对最先进 EA 方法的全面评估:
     1. 识别现有方法的主要组成部分, 提出一个通用 EA 框架.
     2. 将方法分为三组, 进行分类内及分类间评估, 理解优缺点.
     3. 在多种场景中评估方法, 包括:
        - **Cross-/Mono-lingual Alignment** (跨语言/单语言对齐).
        - **Dense/Normal Data** (高密度/正态分布数据).
        - **Large-/Medium-scale Data** (大规模/中等规模数据).
   - 从 **Effectiveness**, **Efficiency**, 和 **Robustness** 多维度提供见解.

2. **New Dataset**:
   - 创建了一个新的单语言数据集, 反映以下现实挑战:
     - **Unmatchable Entities**.
     - **Ambiguous Entity Names**.
   - 作为更有效的基准, 为评估 EA 系统提供支持.

### 2.2. A General EA Framework

本节提出了一个通用的 **Entity Alignment (EA)** 框架, 用于涵盖最先进的 EA 方法. 通过对当前 EA 方法的深入分析, 我们识别出了以下四个主要组件, 如 Fig 2.1所示:

<figure style="display: block; text-align: center;">
  <img src="FL/2024_11_12/images/2_1.png" alt="A general EA framework" style="display: block; margin: auto; width: 50%; height: auto;">
  <figcaption style="margin-top: 8px; font-size: 14px; color: #555;">Figure 2.1: A general EA framework</figcaption>
</figure>

1. Embedding Learning Module
   - **功能**: 为实体train embeddings.
   - **两种主要方法**: 
     - 基于 **KG Representation** 的模型:
       - 例如: **TransE**.
     - 基于 **Graph Neural Network (GNN)** 的模型:
       - 例如: **Graph Convolutional Network (GCN)**.

2. Alignment Module
   - **功能**: 将上一模块中学习到的实体嵌入在不同 KGs 中对齐, 目标是将这些嵌入映射到统一的空间.
   - **两种常用approach**:
     - **Margin-based Loss (基于边界的损失)**:
       - 确保来自不同 KGs 的 seed entity embeddings are close to each other.
     - **Corpus Fusion (语料融合)**:
       - 在语料级别对齐 KGs, 并直接将不同 KGs 的实体嵌入到同一向量空间.

3. Prediction Module
   - **功能**: 在建立统一的嵌入空间后, 预测test集中每个源实体对应的目标实体.
   - **常用方法**:
     - 基于距离的相似度测量:
       - **Cosine Similarity** (余弦相似度).
       - **Manhattan Distance** (曼哈顿距离).
       - **Euclidean Distance** (欧几里得距离).
     - The target entity with the highest similarity (or lowest distance) is then selected as the counterpart


4. Extra Information Module
   - **功能**: 在基本模块之外, 一些 EA 方法利用额外的信息来提升性能.
   - **常用方法**:
     - **Bootstrapping (自举)**:
       - 使用高置信度的对齐结果作为训练数据, 进行后续迭代对齐.
     - **Multi-type Literal Information (多类型文本信息)**:
       - 结合属性 (attributes), 实体描述 (entity descriptions), 和实体名称 (entity names), 补充 KG 的结构信息.
     - **表示方式**:
       - 这些附加信息以蓝色虚线在 Fig 2.1中表示.

* Example
进一步结合第 1 章中的示例, 我们解释这些模块的具体作用:
  1. **Embedding Learning Module**:
     - 生成 **KG(EN)** 和 **KG(ES)** 中实体的嵌入表示.

  2. **Alignment Module**:
     - 将实体嵌入映射到同一向量空间中, 使得 **KG(EN)** 和 **KG(ES)** 中的实体嵌入可以直接进行比较.

  3. **Prediction Module**:
     - 利用统一的嵌入, 为 **KG(EN)** 中的每个源实体预测 **KG(ES)** 中的等价目标实体.

  4. **Extra Information Module**:
     - 使用多种技术来提升 EA 的性能:
       - **Bootstrapping Strategy (自举策略)**:
         - 将上一轮中检测到的高置信度 EA 对 (例如: (Spain, España)) 添加到训练集中, 用于下一轮学习.
       - **Additional Textual Information (额外文本信息)**:
         - 使用补充的文本信息 (例如实体描述或属性) 来增强实体嵌入的对齐效果.

- Organization of State-of-the-Art Approaches
  - 根据 EA 框架中的各模块对最先进的方法进行了分类, 并将它们呈现在 **Table 2.1** 中.
  - 有关这些方法的更详细信息, 读者可以参考附录.
  - 接下来, 我们将解释这些模块如何在各种最先进的方法中实现.
<figure style="display: block; text-align: center;">   <img src="FL/2024_11_12/images/2024-11-23-23-08-34.png" alt="Table 2.1" style="display: block; margin: auto; width: 50%; height: auto;">   <figcaption style="margin-top: 8px; font-size: 14px; color: #555;">Table 2.1: A summary of EA approaches</figcaption> </figure>

#### 2.2.1 Embedding Learning Module

在本节中, 我们将解释 **Embedding Learning Module** 中使用的技术, 这些技术利用知识图谱 (KG) 的结构为每个实体创建嵌入表示. **Table 2.1** 显示, 此模块中最常用的模型是 **TransE** 和 **Graph Convolutional Network (GCN)**. 我们将简要介绍这些基础模型.

**1. TransE**
- **方法概述**:
  - **TransE** 将关系视为在低维空间中作用于实体嵌入的平移.
  - 给定一个关系三元组 (h, r, t), TransE 假设尾实体 t 的embedded representation 应该接近头实体 h 和关系 r embedded representation 的和, 即 \( h + r \approx t \).
- **特点**:
  - 保留实体的结构信息.
  - 在embedding space中为具有相似neighbors的实体生成相近的表示.

**2. GCN**
- **方法概述**:
  - **GCN** 一种直接处理基于图数据的卷积网络.
  - 它通过编码节点neighborhoods的信息为每个节点创建embeddings.
  1. 输入:
     1. KG每个节点的特征向量.
     2. 图结构的矩阵表示 (如邻接矩阵).
  2. 输出:
     - 新的特征矩阵, 表示节点embeddings.
  3. 模型结构:
     - 典型的 GCN 模型由多个堆叠的 GCN 层组成.
       -> 捕获从 当前处理的目标实体 向外延伸几跳(hops) 的 partial KG structure.

**3. 基于 TransE 的改进方法**
- **MTransE**: 移除了训练中的负三元组.
- **BootEA** 和 **NAEA**: 用limit-based objective function替代了原始的margin-based loss function.
- **MuGNN**: 用logistic loss代替margin-based loss.
- **JAPE**: 设计了新的损失函数.

**4. 基于 GCN 的改进方法**
- **RDGCN**:
  - 考虑到 GCN 限制: 原始 GCN 模型未考虑 KG 中的关系信息.
  - 使用双原图卷积神经网络 (Dual-Primal Graph Convolutional Neural Network, DPGCNN) 来补充关系信息.
- **MuGNN**:
  - 利用基于注意力的 GNN 模型, 为邻居节点分配不同的权重.
- **KECG**:
  - 将图注意力网络 (**Graph Attention Network, GAT**) 和 **TransE** 结合, 同时捕获图内结构信息和图间alignment information.

**5. 新型嵌入模型**
- **RSNs**:
  - 观察到的问题:
    - triple-level learning无法捕获实体之间的long-term relational dependencies.
    - 无法在实体之间传播语义信息.
  - 解决方案:
    - 使用残差学习的循环神经网络 (**Recurrent Neural Networks, RNNs**) 来学习实体之间的long-term relational paths.
- **TransEdge**:
  - 提出了新的能量函数, 用于测量the error of edge translation between entity embeddings.
  - This method models edge embeddings using context compression and projection.

#### 2.2.2 Alignment Module

该模块旨在unify separated KG embeddings

**1. Margin-based Loss Function**
- **方法概述**:
  - 在embedding learning module之后使用margin-based loss function是主流方法.
  - 要求:
    - In Positive Pairs (refer to seed entity pairs): 实体之间的距离应尽可能小.
    - In Negative Pairs (generated by corrupting the positive pairs): 实体之间的距离应尽可能大.
    - positive and negative pairs之间的距离需要维持一定的margin.
  - 将两个分离的 KG 嵌入空间合并为一个统一的向量空间.
  - **Table 2.1** 显示, 大多数基于GNNs的方法依赖a margin-based alignment model to merge the two KG embedding spaces.

**2. Matching Framework**
  - 在 **GM-Align** 中, 使用匹配框架 (**Matching Framework**) 最大化种子实体对的匹配概率, 以实现对齐过程.

**3. Corpus Fusion**
- **方法概述**:
  - 利用seed entity pairs连接两个KG的训练语料.
  - 例如 **BootEA** 和 **NAEA**, 通过交换seed entity pairs中的实体生成新的三元组to align the embeddings in a unified space
- **procedure**:
  - 给定一个实体对 (u, v):
    - **For G1**:
      - \( T^{new}_1 = \{(v, r, t) | (u, r, t) \in T_1\} \cup \{(h, r, v) | (h, r, u) \in T_1\} \).
    - **For G2**:
      - \( T^{new}_2 = \{(u, r, t) | (v, r, t) \in T_2\} \cup \{(h, r, u) | (h, r, v) \in T_2\} \).
- **Overlay Graph (覆盖图)**:
  - 覆盖图通过用边连接seed entity pairs中的实体构建的，其余实体则根据它们在训练语料中的相似性或共现关系进行边的连接。然后，使用覆盖图的邻接矩阵和训练语料来学习实体嵌入。

**4. others**
  - Transition Functions: 早期的工作，将一个 KG 的嵌入向量映射到另一个 KG 的嵌入空间.
  - Additional Information: 使用实体属性等附加信息, 将实体嵌入对齐到统一的空间中.

#### 2.2.3 Prediction Module

**Prediction Module** 主要负责计算源实体与目标实体嵌入之间的相似度分数, 并选择得分最高的目标实体作为对齐结果.

**1. 常用方法**
- **generate a ranked list**:
  - 为每个源实体生成目标实体的排名列表, 排序依据为其嵌入间的距离度量.
  - **常用距离度量**:
    - **Euclidean Distance** (欧几里得距离).
    - **Manhattan Distance** (曼哈顿距离).
    - **Cosine Similarity** (余弦相似度).
  - 列表中排名最高的目标实体被认为是源实体的匹配实体.
  - 相似度分数可通过 \( 1 - similarity \) 转换为距离分数, 反之亦然.

**2. GM-Align 的匹配方法**
- 在 **GM-Align** 中, 匹配的目标实体是具有最高matching probability的实体.

**3. CEA**
- **观察到的问题**:
  - 不同entity alignment decisions之间存在相关性, 如果一个目标实体已经高置信度匹配到某个源实体, 则不太可能再匹配到另一个源实体.
  - 将这种相关性建模为一个stable matching problem, 基于距离度量来解决该问题, 减少了错误匹配的数量, 提升了实体对齐的准确性.

#### 2.2.4 Extra Information Module

本节介绍 **Extra Information Module** 的方法, 这些方法通过利用额外信息提升 **Entity Alignment (EA)** 的性能.

**1. Bootstrapping Strategy (自举策略)**
- **概述**:
  - 又称 **Iterative Training** 或 **Self-Learning Strategy**.
  - 通过迭代标注高置信度的 EA 对, 将其作为下一轮的训练集, 从而逐步改进对齐结果.
- **常见方法**:
  - **ITransE**:
    - 为每个未对齐的源实体寻找最相似的未对齐目标实体, 若相似度分数超过某个阈值, 则视为高置信度配对.
  - **BootEA**, **NAEA**, 和 **TransEdge**:
    - 计算每个源实体与每个目标实体对齐的概率, 仅选择概率分数高于阈值的配对.
    - 使用 **Maximum Likelihood Matching Algorithm** 和 **1-to-1 Mapping Constraint** 生成高置信度的 EA 对.

**2. Multi-type Literal Information (多类型文本信息)**
- **目标**:
  - 提供更全面的视角来改进对齐.
- **Methods**:
  - **JAPE**, **GCN-Align**, 和 **HMAN**:
    - 考虑属性名称的statistical characteristics .
  - **AttrE** 和 **M-Greedy**:
    - 通过编码characters of attribute values生成attribute embeddings.
    - **AttrE**:
      - 使用attribute embeddings将实体embeddings统一到同一空间.
    - **M-Greedy**:
      - 使用attribute embeddings补充实体embeddings.

**3. Entity Names (实体名称)**
- 越来越多方法将 **Entity Names** 用作输入特征来学习实体嵌入或作为单独特征进行对齐.
- **方法**:
  - **GM-Align**, **RDGCN**, 和 **HGCN**:
    - 将实体名称作为输入特征以学习实体embeddings.
  - **CEA**:
    - 同时利用实体名称的语义特征和字符串级特征 as individual features for alignment.
  - **KDCoE** 和 **HMAN (Description-enhanced version)**:
    - encode entity descriptions为向量表示, 将其作为新特征进行对齐.
- **Limitations**
  - **availability**:
    - 某些信息类型 (如实体名称) 在大多数场景中普遍存在. 但其他信息类型 (如实体描述) 在许多知识图谱中可能缺失.
  - **数据集限制**:
    - 由于知识图谱对齐的图结构特性, 大多数现有对齐数据集的文本信息有限, 使得一些方法 (如 **KDCoE**, **M-Greedy**, 和 **AttrE**) 应用受限.

### 2.3 Experiments and Analysis
This section presents an in-depth empirical study

#### **2.3.1 Categorization**
根据主要组件的特点, 我们可以将当前方法分为以下三类:
1. **Group I**: 仅使用 KG 结构进行对齐.
2. **Group II**: 使用 **Iterative Training Strategy** (迭代训练策略) 提升对齐效果.
3. **Group III**: 在 KG 结构之外利用附加信息.

---

##### **Group I: KG Structure-Based Methods**
- **特点**:
  - 仅依赖 KG 的结构信息对齐实体.
- **示例**:
  - 在 **KGEN** 中, 实体 Alfonso Cuarón 连接到实体 Mexico 和其他三个实体, 实体 Spain 连接到实体 Mexico 和另一个实体.
  - 在 **KGES** 中可以观察到相同的结构信息.
  - 已知 **KGEN** 中的 Mexico 与 **KGES** 中的 Mexico 对齐, 通过 KG 结构可以推断:
    - Spain 的目标实体为 España.
    - Alfonso Cuarón 的目标实体为 Alfonso Cuarón.

---

##### **Group II: Iterative Training-Based Methods**
- **特点**:
  - 使用自举策略 (**Bootstrapping Strategy**), 即迭代标注高置信度的实体对齐对作为下一轮的训练集, 从而逐步改进对齐结果.
  - 可归类为 Group I 或 Group III, 具体取决于是否仅使用 KG 结构.
- **示例**:
  - 根据 **Fig. 1.1**, 第一轮中可以轻松对齐:
    - (Spain, España).
    - (Alfonso Cuarón, Alfonso Cuarón).
  - 对于源实体 Madrid, 因目标实体 **Roma(ciudad)** 和 **Madrid** 具有相同的结构信息 (均与种子实体相隔两跳, 度数为 1), 导致无法明确对齐目标实体.
  - **解决方案**:
    - 自举策略在第二轮中将第一轮的高置信度对齐对作为新的种子对:
      - (Spain, España).
      - (Alfonso Cuarón, Alfonso Cuarón).
    - 第二轮中, 源实体 Madrid 的目标实体 Madrid 是唯一满足与种子对 (Mexico, Mexico) 相隔两跳, 且与种子对 (Spain, España) 相隔一跳的实体.

---

##### **Group III: Methods Using Additional Information**
- **特点**:
  - 除了 KG 结构外, 利用语义信息补充结构数据.
- **示例**:
  - 即使结合 KG 结构和自举策略, 对于源实体 **Gravity(film)** 的目标实体仍难以确定:
    - 其结构信息 (连接到 Alfonso Cuarón, 度数为 2) 同时匹配目标实体 **Gravity(película)** 和 **Roma(película)**.
  - **解决方案**:
    - 结合 KG 结构和实体名称中的identifiers信息, 可区分这两个实体, 可以轻松识别 **Gravity(película)** 为 **Gravity(film)** 的目标实体.

---

#### 2.3.2 Experimental Settings

##### 1. Methods to Compare
  - 排除的方法: **KDCoE** 和 **MultiKE**因评估数据集中缺乏实体描述而排除. **AttrE**仅在单语言环境中有效, 不适用于多语言数据集.
  - 会展示**JAPE** 和 **GCN-Align** 的structure-only versions的结果, 即**JAPE-Stru**和**GCN-Align(SE)**.
  - 会比较几种依赖对象名称相似性识别等价实体的方法:
    1. **Lev**:
       - 使用 **Levenshtein Distance**, 一种基于字符串的工具, 计算两个序列之间的不相似度.
    2. **Embed**:
       - 基于两个实体名称的averaged word embeddings, or name embeddings的 **Cosine Similarity**.
       - 使用预训练的 **fastText Embeddings** 作为词嵌入. 对多语言 KG pairs, 则使用 **MUSE Word Embeddings**.

---

##### 2. Implementation Details
- Intel Core i7-4790. NVIDIA GeForce GTX TITAN X. 128 GB RAM. 基于python.
- 使用作者提供的源代码和原论文中的参数配置运行模型.
- 对未包含在原论文中的数据集, 采用与原实验一致的参数设置.
- **DBP15K 数据集**:
  - 除 **MTransE** 和 **ITransE**, 所有方法均在原论文中报告了此数据集的结果.
  - 比较我们的实现结果与原论文报告结果:
    - 若结果差异超出 ±5% 的合理范围, 用星号 \( \ast \) 标记.
    - 理论上不应有显著差异, 因为实现使用相同的源代码和参数.
- **SRPRS 数据集**:
  - 仅 **RSNs** 在原论文中报告了结果.
  - 对所有方法进行实验, 结果见 **Table 2.3**.
- **DWY100K 数据集**:
  - 对所有方法运行实验, 并将 **BootEA**, **MuGNN**, **NAEA**, **KECG**, 和 **TransEdge** 的结果与原论文进行比较.
  - 用 \( \ast \) 标记差异显著的方法.
- **结果展示**:
  - 在每个组内, 使用加粗字体标记最佳结果.
  - 对所有方法中best **Hits@1** 的结果标记为▲, 因为该指标最能反映 EA 方法的有效性.
    > Hits@1 表示 Top-1 命中率: 在为每个源实体生成的目标实体排名列表中, 如果排名第一的目标实体是正确的对齐实体, 则视为命中. 然后计算所有源实体中命中的比例.



<figure style="display: block; text-align: center;">   <img src="FL/2024_11_12/images/2024-11-24-12-13-40.png" alt="name" style="display: block; margin: auto; width: 100%; height: auto;">   <figcaption style="margin-top: 8px; font-size: 14px; color: #555;">Table 2.2: Experimental results on DBP15K</figcaption> </figure>




## 3. Recent Advance of Representation Learning Stage

> 近年来, 大量研究致力于学习更好的KG表示以促进实体对齐. 本章总结了 EA 表示学习阶段的最新进展, 并提供了详细的实证评估, 以揭示当前解决方案的优劣.

### 3.1 Overview
为更好地理解当前先进的表示学习方法, 我们提出了一个通用框架来描述这些方法, 包括六个模块:

1. **Pre-Processing (预处理)**.
2. **Messaging (消息传递)**.
3. **Attention (注意力机制)**.
4. **Aggregation (聚合)**.
5. **Post-Processing (后处理)**.
6. **Loss Function (损失函数)**.

**workflow**:
1. 在 **Pre-Processing** 阶段, 生成初始的实体和关系表示.
2. 通过一个representation learning network获取KG representations, 该网络通常包括以下三个步骤:
   - **Messaging**: 提取邻域元素的特征.
   - **Attention**: estimate每个邻居的权重.
   - **Aggregation**: 根据注意力权重integrates邻域信息.
3. 通过 **Post-Processing**, 获取最终的representations.
4. 在训练阶段, 使用Loss Function优化整个模型.

<figure style="display: block; text-align: center;">   <img src="FL/2024_11_12/images/2024-11-24-15-47-46.png" alt="name" style="display: block; margin: auto; width: 100%; height: auto;">   <figcaption style="margin-top: 8px; font-size: 14px; color: #555;">Table 3.1: Overview and comparison of advanced representation learning</figcaption> </figure>

#### 3.1.1. Pre-Processing
  1. 一些方法利用预训练模型to embed names or descriptions into initial representations.
  2. 还有一些使用基于GNN的网络生成initial structural representations.

#### 3.1.2. Messaging
   - **Linear Transformation**:
      - 最常用，使用可学习矩阵to transform neighboring features.
   - **其他方法**:
      - extracting neighboring features by concatenating multihead messages
      - directly utilizing neighboring representations.

#### 3.1.3. Attention
- **核心任务**: 计算相似度.
- Most of the methods concatenate the representations , 然后乘以learnable attention vector to calculate attention weights.
- some use inner product of entity
representations to compute similarity.

#### 3.1.4. Aggregation
- **主要方法**:
  - 几乎所有方法聚合 **1-hop** 邻域实体或关系信息.
  - 少量方法提出结合multi-hop邻域信息.
  - 一些方法使用随机选取的实体集合 (称为 **Anchor Set**) 以生成position-aware representations.

#### 3.1.5. Post-Processing
- 大部分都采取拼接GNN所有层的输出以增强最终表示.
- 一些方法通过比如Gate Mechanism (门控机制)等策略自适应地组合特征.

#### 3.1.6. Loss Function
- 主流方法使用基于边界的损失函数 (**Margin-based Loss**) 进行训练.
- 一些改进方法:
  - 添加 **TransE** 损失.
  - 使用 **LogSumExp** 和normalization operation.
  - 使用 **Sinkhorn** 算法计算损失.

### 3.2 Models

使用下面公式来描述表示学习的核心过程:

\[
e^l_i = \text{Aggregation}_{\forall j \in N(i)} \big( \text{Attention}(i, j) \cdot \text{Messaging}(i, j) \big),
\]

其中:
- **Messaging**: 提取邻域元素的特征.
- **Attention**: estimate每个邻居的权重.
- **Aggregation**: 将邻域信息与注意力权重整合.

#### 3.2.1 ALiNet
> (Alignment-aware Network)<br>
> 解决KG中neighboring structure的非同构性问题<br>
> It aims to aggregate multi-hop structural information for learning entity representations


#### 3.2.2 MRAEA
> (Multi-View Relation-Aware Entity Alignment)<br>
> 利用多视角关系感知机制来捕获实体之间的复杂关系<br>
> It proposes to utilize the relation information to facilitate the entity representation
learning process


#### 3.2.3 RREA
> (Relation-aware Entity Alignment)通过引入关系感知的图注意力网络<br>
> 采用自注意力机制，将实体信息传播到关系，再将关系信息聚合回实体<br>
> It proposes to use relational reflection transformation to aggregate features for
learning entity representations

#### 3.2.4 RPR-RHGT
> (Relation-Path Reasoning with Relational Heterogeneous Graph Transformer)结合关系路径推理和异构图转换器<br>
> 建模关系路径<br>
> 这项工作为EA引入了一个基于元路径的相似性框架 [2]。它认为在预对齐实体的邻域中经常出现的路径是可靠的

#### 3.2.5 RAGA
> RAGA(Relation-Aware Graph Attention Networks)采用关系感知的图注意力网络，捕获实体和关系之间的交互<br>
> 通过自注意力机制，将实体信息传播到关系，再将关系信息聚合回实体
> 



#### 3.2.6 Dual-AMN
Dual-AMN（Dual Attention Matching Network）提出了一种新颖的图编码器，由简化关系注意力层和代理匹配注意力层组成。该编码器智能地对图内和跨图信息进行建模，同时大大降低了计算复杂度。


#### 3.2.7 ERMC
ERMC（Entity Representation with Multi-Context）通过引入多上下文信息，增强实体表示的丰富性。该方法结合结构、属性和描述信息，提高了实体对齐的效果。


#### 3.2.8 KE-GCN
KE-GCN（Knowledge Embedding with Graph Convolutional Networks）将知识嵌入与图卷积网络相结合，捕获实体的结构和语义信息。该方法通过融合多种信息源，提高了对齐性能。


#### 3.2.9 RePS
RePS（Relation Path Sampling）通过关系路径采样，捕获实体之间的深层次关系。该方法利用关系路径信息，增强了实体嵌入的表达能力，从而提高对齐精度。


#### 3.2.10 SDEA
SDEA（Self-Supervised Dual-Encoder Alignment）采用自监督的双编码器架构，进行实体对齐。该方法通过自监督学习，减少了对人工标注数据的依赖，提高了对齐效率。


### 3.3 Experiments

#### 3.3.1 Experimental Setting

#### 3.3.2 Overall Results and Analysis




## 4. Recent Advance of Alignment Inference Stage

### 4.1 Introduction





## 5. Large-Scale Entity Alignment

> Abstract<br>
> 本章聚焦于大规模EA的概念, 并提出一种新的方法来解决这一任务. 该解决方案能够处理大规模的KG pairs, 并提供高质量的对齐结果.

主要贡献包括:
1. **Seed-Oriented Graph Partition Strategies**:
   - 设计了一组种子导向的图划分策略, 将大规模 KG 对划分为较小的子图对.
2. **Reciprocal Alignment Inference**:
   - 在每个子图对内, 使用现有方法学习统一的实体表示, 并引入一种新的双向对齐推断策略来建模双向对齐交互, 从而提高对齐的准确性.
3. **Variant Strategies for Scalability**:
   - 为进一步提升双向对齐推断的可扩展性, 提出两种变体策略, 显著降低了内存和时间成本, 但略微降低了效果.
4. **Versatility**:
   - 该解决方案可以应用于现有基于表示学习的 EA 模型, 增强其处理大规模 KG 对的能力.
5. **New Dataset**:
   - 创建了一个包含数百万实体的新 EA 数据集.
6. **Comprehensive Experiments**:
   - 通过全面实验验证模型的效率.
   - 在流行的 EA 数据集上与最先进的基线方法进行比较, 展示了模型的有效性和优越性.


### 5.1 Introduction

#### Overview of EA Pipeline
- **Two-stage Pipeline**:
  1. **Representation Learning**:
     - 使用 **KG Embedding Models** (如 **TransE**, **GCN**) 学习实体嵌入表示.
     - 利用种子实体对将不同 KG 的嵌入投射到一个公共嵌入空间.
  2. **Alignment Inference**:
     - 使用统一嵌入空间中的相似度或距离预测对齐结果.
     - 常见方法: 根据相似度度量对目标 KG 的实体排序, 选择排名最高的目标实体作为匹配实体.

<figure style="display: block; text-align: center;">   <img src="FL/2024_11_12/images/2024-11-24-22-00-18.png" alt="name" style="display: block; margin: auto; width: 100%; height: auto;"> </figure>

#### Challenges in Large-Scale EA
- **计算资源需求高**:
  - 当前技术需要大量参数, 且消耗大量计算资源.
  - 例如, 在 **DWY100K** 数据集 (20 万实体) 上, 大多数方法的运行时间超过 **20,000 秒**, 一些方法甚至无法产生对齐结果.
- **大规模 KG**:
  - 真实场景中的 KG 往往包含数千万实体, 当前方法难以扩展, 需要研究大规模 EA.

#### Proposed Solutions
1. **Seed-Oriented Graph Partitioning**:
   - 将大规模 KG 对划分为多个较小的子图对.
   - 目标:
     - 保留 KG 的原始结构.
     - 确保源 KG 和目标 KG 的划分结果匹配, 即等价实体被分配到相同的子图对.
   - **SBP (Seed-oriented Bidirectional Partition)**:
     - 进行双向划分, 聚合源到目标和目标到源的划分结果, 以平衡结构完整性和对齐信号.
     - 提出迭代变体 **I-SBP**, 利用上一轮的高置信度对齐结果改进划分性能.

2. **Reciprocal Alignment Inference**:
   - 建模实体的双向偏好:
     - 传统的直接对齐推断方法 (Direct Alignment Inference) 仅考虑单向相似度, 忽略了反向对齐的影响.
     - 提出双向偏好建模与整合, 通过生成互惠偏好矩阵来提升对齐精度.
   - **Example**:
     - 在 **Fig. 5.1a** 中, 使用直接对齐推断, **[A. Dessner]en** 和 **[B. Dessner]en** 都会与 **[A. Dessner]es** 匹配.
     - 通过建模双向偏好 (如 **Fig. 5.1b** 所示), 可以避免错误匹配并识别正确的等价实体.

3. **Variant Strategies for Efficiency**:
   - 提出两种变体以提高效率:
     - **No-Ranking Aggregation**: 在偏好聚合过程中移除排序过程.
     - **Progressive Blocking**: 将实体分块, 在每块内进行对齐.

#### LIME Framework
- 提出了一种适用于大规模 EA 的框架 **LIME**:
  - 通用模型: 可与任何实体表示学习模型结合使用.
  - 评估:
    - 在大规模数据集 **FB_DBP_2M** (包含数百万实体和数千万事实) 上进行实验验证.
    - 与主流小规模数据集上的最先进方法比较, 展现出良好的性能.


#### Contributions
1. 确定了当前 EA 方法的扩展性问题, 并提出框架 **LIME** 解决大规模实体对齐.
2. 提出 **Seed-Oriented Bidirectional Graph Partitioning** 方法, 将大规模 KG 对划分为较小子图对.
3. 提出 **Reciprocal Alignment Inference**, 建模并整合实体的双向偏好以提升对齐精度.
4. 引入两种变体, 提升扩展性, 同时保持较小的性能损失.
5. **LIME** 通用性强, 可增强现有 EA 方法的扩展能力.
6. 构建了一个包含数百万实体的新数据集, 并通过全面实验验证了模型的有效性.

#### Organization
- **Sect. 5.2**: LIME 框架概述.
- **Sect. 5.3**: 图划分策略.
- **Sect. 5.4**: 双向对齐推断策略.
- **Sect. 5.5**: 对齐推断变体.
- **Sect. 5.6**: 实验设置.
- **Sect. 5.7**: 实验结果.
- **Sect. 5.8**: 相关工作.
- **Sect. 5.9**: 总结.

## 6. Long-Tail Entity Alignment
> Abstract<br>
> 当前的大多数EA方法主要依赖于KG的结构信息. 然而在真实世界的 KG 中, 大多数实体的邻域结构稀疏, 而仅少数实体与其他实体密集连接. 这些稀疏连接的实体被称为 **Long-Tail Entities** (长尾实体), 限制了结构信息在 EA 中的有效性.

为了解决这一问题, 提出了以下创新:
1. **Entity Name Information**:
   - 引入实体名称作为信号源, 使用 **Concatenated Power Mean Word Embeddings** 增强长尾实体的弱结构信息.
2. **Complementary Framework**:
   - 结合结构和名称信号, 根据实体的度数动态调整两种信号的重要性, 提出 **Degree-Aware Co-Attention Network**.
3. **Iterative Training**:
   - 在后对齐阶段, 使用高置信度的对齐结果作为锚点, 通过迭代训练从目标 KG 补充源 KG 的事实信息, 从而改进对齐性能.



## 7. Weakly Supervised Entity Alignment

## 8. Unsupervised Entity Alignment

## 9. Multimodal Entity Alignment