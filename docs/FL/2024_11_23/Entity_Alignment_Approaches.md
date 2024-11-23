# Entity Alignment Approaches
> 标题<br>
> An Experimental Study of State-of-the-Art Entity Alignment Approaches<br>
> Introduction to Entity Alignment<br>
> 2020, 2023<br>
> 领域综述<br>
> 文章地址: <br>
> https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9174835<br>
> https://link.springer.com/chapter/10.1007/978-981-99-4250-3_1<br>
---
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

##### 1.1.2.1. Definition
- **Purpose**:
  - 实体对齐（EA）的目标是识别不同知识图谱（KGs）中等价的实体, 从而实现不同来源知识的整合.
- **Definition**:
  - **Source Knowledge Graph (KG)**: \( G_1 = (E_1, R_1, T_1) \)
  - **Target Knowledge Graph (KG)**: \( G_2 = (E_2, R_2, T_2) \)
  - **Seed Entity Pairs** (训练集): \( S = \{(u, v) | u \in E_1, v \in E_2, u \leftrightarrow v \} \), 其中 \( u \leftrightarrow v \) 表示等价关系（即 \( u \) 和 \( v \) 指代同一现实世界对象）.
  - **目标**: Identify equivalent entity pairs in the test set.

###### 1.1.2.1.* Example
![alt text](FL/2024_11_23/EA_1.png)
<!-- <img src="" alt="EA" style="display: block; margin: auto; width: 50%; height: auto;"> -->

- **Context**:
  - 对齐英语知识图谱 (\( KG_{EN} \)) 和西班牙语知识图谱 (\( KG_{ES} \)) 中的实体.
- **Given**:
  - **Seed Entity Pair**: “Mexico” (\( KG_{EN} \)) ↔ “Mexico” (\( KG_{ES} \)).
  - **目标**: Given the seed entity pair, EA aims to find the equivalent entity pairs in the test set, e.g., returning Roma(ciudad) in KG(ES) as the corresponding target entity to the source entity Roma(city) in KG(EN).
- **Entity Uniqueness**:
  - Each entity is uniquely identified, 例如 “Roma(film)” 和 “Roma(city)” 是不同的实体.

##### 1.1.2.2. Method

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

### 2.2 A General EA Framework

本节提出了一个通用的 **Entity Alignment (EA)** 框架, 用于涵盖最先进的 EA 方法. 通过对当前 EA 方法的深入分析, 我们识别出了以下四个主要组件, 如 Fig 2.1所示:

<figure style="display: block; text-align: center;">
  <img src="FL/2024_11_23/2_1.png" alt="A general EA framework" style="display: block; margin: auto; width: 50%; height: auto;">
  <figcaption style="margin-top: 8px; font-size: 14px; color: #555;">Figure 2.1: A general EA framework</figcaption>
</figure>

#### 2.2.1. Embedding Learning Module
- **功能**: 为实体train embeddings.
- **两种主要方法**: 
  - 基于 **KG Representation** 的模型:
    - 例如: **TransE**.
  - 基于 **Graph Neural Network (GNN)** 的模型:
    - 例如: **Graph Convolutional Network (GCN)**.

#### 2.2.2. Alignment Module
- **功能**: 将上一模块中学习到的实体嵌入在不同 KGs 中对齐, 目标是将这些嵌入映射到统一的空间.
- **两种常用approach**:
  - **Margin-based Loss (基于边界的损失)**:
    - 确保来自不同 KGs 的 seed entity embeddings are close to each other.
  - **Corpus Fusion (语料融合)**:
    - 在语料级别对齐 KGs, 并直接将不同 KGs 的实体嵌入到同一向量空间.

#### 2.2.3. Prediction Module
- **功能**: 在建立统一的嵌入空间后, 预测test集中每个源实体对应的目标实体.
- **常用方法**:
  - 基于距离的相似度测量:
    - **Cosine Similarity** (余弦相似度).
    - **Manhattan Distance** (曼哈顿距离).
    - **Euclidean Distance** (欧几里得距离).
  - The target entity with the highest similarity (or lowest distance) is then selected as the counterpart


#### 2.2.4. Extra Information Module
- **功能**: 在基本模块之外, 一些 EA 方法利用额外的信息来提升性能.
- **常用方法**:
  - **Bootstrapping (自举)**:
    - 使用高置信度的对齐结果作为训练数据, 进行后续迭代对齐.
  - **Multi-type Literal Information (多类型文本信息)**:
    - 结合属性 (attributes), 实体描述 (entity descriptions), 和实体名称 (entity names), 补充 KG 的结构信息.
  - **表示方式**:
    - 这些附加信息以蓝色虚线在 Fig 2.1中表示.

##### 2.2.4.* Example
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

#### Organization of State-of-the-Art Approaches
- 根据 EA 框架中的各模块对最先进的方法进行了分类, 并将它们呈现在 **Table 2.1** 中.
- 有关这些方法的更详细信息, 读者可以参考附录.
- 接下来, 我们将解释这些模块如何在各种最先进的方法中实现.
<figure style="display: block; text-align: center;">   <img src="FL/2024_11_23/2024-11-23-23-08-34.png" alt="Table 2.1" style="display: block; margin: auto; width: 50%; height: auto;">   <figcaption style="margin-top: 8px; font-size: 14px; color: #555;">Table 2.1: A summary of the EA approaches</figcaption> </figure>


## 4. 文章实验
## 5. 总结&未来方向