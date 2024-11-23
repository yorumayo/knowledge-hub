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
## 1.1.  Background
### 1.1.1. KG
1. 存储形式
   * triples <entity, relation, entity>
   * 每个entity对应一个unique object in the real world, relations描述这些对象之间的联系
   * triples内在互联(inherently interconnected)
   * e.g. https://dbpedia.org/page/Spain
2. Core Compononets
   * Entity Set (E): 图中的所有节点，对应现实中的唯一对象
   * Relation Set (R): 描述实体间的关系类型
   * Triple Set (T): 表示图中的有向边，T ⊆ E × R × E
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

### 1.1.2. EA

#### 1.1.2.1. Definition
- **Purpose**:
  - 实体对齐（EA）的目标是识别不同知识图谱（KGs）中等价的实体，从而实现不同来源知识的整合。
- **Definition**:
  - **Source Knowledge Graph (KG)**: \( G_1 = (E_1, R_1, T_1) \)
  - **Target Knowledge Graph (KG)**: \( G_2 = (E_2, R_2, T_2) \)
  - **Seed Entity Pairs** (训练集): \( S = \{(u, v) | u \in E_1, v \in E_2, u \leftrightarrow v \} \)，其中 \( u \leftrightarrow v \) 表示等价关系（即 \( u \) 和 \( v \) 指代同一现实世界对象）。
  - **目标**: Identify equivalent entity pairs in the test set.

##### Example

<img src="FL/2024_11_23/EA_1.png" alt="EA" style="display: block; margin: auto; width: 50%; height: auto;">

- **Context**:
  - 对齐英语知识图谱 (\( KG_{EN} \)) 和西班牙语知识图谱 (\( KG_{ES} \)) 中的实体。
- **Given**:
  - **Seed Entity Pair**: “Mexico” (\( KG_{EN} \)) ↔ “Mexico” (\( KG_{ES} \))。
  - **目标**: Given the seed entity pair, EA aims to find the equivalent entity pairs in the test set, e.g., returning Roma(ciudad) in KG(ES) as the corresponding target entity to the source entity Roma(city) in KG(EN).
- **Entity Uniqueness**:
  - Each entity is uniquely identified, 例如 “Roma(film)” 和 “Roma(city)” 是不同的实体。

#### 1.1.2.2. Method

1. **Assumption**:
   - Entities that are equivalent in different KGs share **similar local structures** (e.g., neighbors and relationships).

2. **Representation Learning**:
     - Entities are embedded into a low-dimensional feature space as data points.

3. **Process**:
   - **Calculate Dissimilarity**:
     - The pairwise dissimilarity of entities is measured as the **distance** between their embedded data points in the feature space.
   - **Entity Matching**:
     - Based on the distance, it is determined whether two entities are equivalent or not.


## 1.2 Related Works

虽然EA问题是近年来才被提出的，但这一问题的更通用版本——识别来自不同数据源的、指向同一现实世界实体的记录——已经被不同领域的研究社区从多个角度进行了深入研究，并使用了不同的名称，包括实体解析(Entity Resolution, ER)、实体匹配(Entity Matching)、记录链接(Record Linkage)、去重(Deduplication)、实例/本体匹配(Instance/Ontology Matching)、链接发现(Link Discovery) 以及实体链接/消歧(Entity Linking/Entity Disambiguation)。接下来是对其中部分相关工作的描述.

### 1.2.1 Entity Linking (EL)
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

### 1.2.2 Entity Resolution (ER)
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

### 1.2.3 Entity Resolution on KGs

# 2. SOTA approaches

## 4. 文章实验
## 5. 总结&未来方向
