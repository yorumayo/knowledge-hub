# T2V-CompBench
> T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation<br>
> https://arxiv.org/abs/2407.14505<br>
> 参考: 

## 摘要

1. **研究背景与目标**
   - 文本到视频（Text-to-video, T2V）生成模型已经取得了显著进展，但其将不同对象、属性、动作和运动组合成一个视频的能力仍未得到探索。
   - 先前的文本到视频基准测试也忽略了评估这一重要能力。

2. **本文贡献**
   - 在本工作中，我们进行了首次系统性的组合文本到视频生成研究。
   - 我们提出了T2V-CompBench，这是首个专为组合文本到视频生成设计的基准测试。
   - T2V-CompBench涵盖了组合性的各个方面，包括一致的属性绑定（consistent attribute binding）、动态属性绑定（dynamic attribute binding）、空间关系（spatial relationships）、运动绑定（motion binding）、动作绑定（action binding）、对象交互（object interactions）以及生成性数字感知（generative numeracy）。

3. **评估指标的设计**
   - 我们进一步精心设计了基于多模态大模型（MLLM-based metrics）、基于检测（detection-based metrics）和基于跟踪（tracking-based metrics）的评估指标，可以更好地反映700个文本提示所涵盖的七个类别中的组合文本到视频生成质量。
   - 提出指标的有效性通过与人工评估的相关性得到了验证。

4. **模型评估与分析**
   - 我们还对多种文本到视频生成模型进行了基准测试，并对不同模型和不同组合类别进行了深入分析。
   - 我们发现，组合文本到视频生成对于当前模型来说非常具有挑战性。
   - 我们希望我们的尝试能够为未来在这一方向上的研究提供启发。

## 1 Introduction

1. **研究背景**
   - 文本到视频（Text-to-video, T2V）生成在近年来取得了显著进展 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]。
   - 然而，根据细粒度文本描述生成准确描述多个对象、属性和运动的复杂动态场景视频仍是一个具有挑战性的任务。
   - 在本工作中，我们旨在对组合文本到视频生成进行系统性研究。

2. **组合文本到图像生成的研究**
   - 组合文本到图像（Text-to-image, T2I）生成的目标是将多个对象、属性及其关系组合到复杂场景中，这已在先前的方法中得到了广泛研究 [14, 15, 16]。
   - 组合文本到图像生成的基准测试 [17] 已被接受为文本到图像基础模型的重要评估维度 [18, 19, 20]。
   - 然而，大多数关于文本到视频生成的研究集中在生成简单文本提示的视频，忽略了组合文本到视频生成的重要性。
   - 现有的视频生成基准测试 [21, 22, 23] 主要评估视频质量、运动质量和文本与视频的对齐度，使用单对象文本提示，对于组合文本到视频生成的基准测试尚未得到系统性和广泛的研究。

3. **T2V-CompBench的提出**
   - 为此，我们提出了T2V-CompBench，这是一个专为组合文本到视频生成设计的综合基准测试。
   - 该基准测试通过包含属性、数量、动作、交互以及时空动态的多个对象，强调了组合性。
   - 我们设计了一个提示集（prompt suite），包含7个类别，每个类别由100个视频生成文本提示组成。
   - 在构建提示时，我们强调了时间动态性，并确保每个提示至少包含一个动作动词。
   - 这七个类别如下，并在图2中进行了示例说明：
     1. **一致的属性绑定（Consistent attribute binding）**：此类别包含至少两个动态对象的提示，每个对象都有独特的属性，这些属性在整个视频中保持一致。
     2. **动态属性绑定（Dynamic attribute binding）**：此类别包含属性随时间变化的问题。
     3. **空间关系（Spatial relationships）**：每个提示中至少提到两个具有特定空间关系的动态对象。
     4. **动作绑定（Action binding）**：此类别的提示包含至少两个对象，每个对象具有不同的动作。
     5. **运动绑定（Motion binding）**：此类别的提示包含具有特定移动方向的对象。
     6. **对象交互（Object interactions）**：此类别测试模型理解和生成动态交互的能力，包括物理交互和社交交互。
     7. **生成性数字感知（Generative numeracy）**：此类别的文本提示包含至少两个对象，数量从一到八不等。

4. **评估组合文本到视频模型的挑战**
   - 组合文本到视频模型的评估也是一大挑战。
   - 常用的指标，例如Inception Score [24]、Fréchet Inception Distance (FID) [25]、Fréchet Video Distance (FVD) [26] 和CLIPScore [27]，无法完全反映文本到视频生成模型的组合性。
   - 评估文本到视频模型的组合性需要对每帧中的对象和属性以及帧之间的动态和运动有细粒度的理解，这比评估文本到图像模型的复杂度高出几个数量级。

5. **解决方案：加入时序动态和设计新评估指标**
   - 为了应对这一挑战，我们在评估中加入了帧间的时间动态性，并设计了不同的指标来评估基准测试中的不同类别。
   - 具体来说，我们设计了基于多模态大语言模型（MLLM-based metrics）的指标，包括图像-LLM（image-LLM）和视频-LLM（video-LLM），用于评估动态属性绑定、一致属性绑定、动作绑定和对象交互。
   - 我们设计了基于检测（detection-based metrics）的指标来评估空间关系和生成性数字感知。
   - 我们提出了基于跟踪（tracking-based metrics）的指标来评估运动绑定。
   - 我们通过计算与人工评估的相关性验证了所提出评估指标的有效性。

6. **模型评估与分析**
   - 我们在T2V-CompBench上对先前的文本到视频生成模型进行了评估，并对当前模型在不同组合类别中的表现进行了系统性研究和分析。

7. **论文贡献**
   - 本文的贡献有三点：
     1. 据我们所知，这是首次提出用于组合文本到视频生成的基准测试，包含七个类别和700个文本提示。
     2. 我们为七个类别提出了综合的评估指标，并通过与人工评估的相关性验证了它们的有效性。
     3. 我们对各种开源T2V模型进行了基准测试，并提供了具有深刻分析的系统研究，这将为未来在这一方向上的研究提供启发。

## 2 Related work

1. **文本到视频生成（Text-to-video generation）**
   - 现有的文本到视频生成模型大致可以分为两类，即基于语言模型的模型（language model-based）[8, 28, 29, 30, 31, 32] 和基于扩散模型的模型（diffusion-model based）[3, 4, 9, 10, 11, 1, 12, 13]。
   - 在本文中，我们评估了20个模型，包括13个官方开源模型：ModelScope [13]、ZeroScope [33]、Latte [34]、Show-1 [35]、VideoCrafter2 [36]、OpenSora 1.1和1.2 [37]、Open-Sora-Plan v1.0.0和v1.1.0 [38]、AnimateDiff [39]、VideoTetris [40]、LVD [41] 和 MagicTime [42]，以及7个商用模型：Pika [43]、Gen-2 [44]、Gen-3 [45]、Dreamina [46]、PixVerse [47]、Dream Machine [48] 和 Kling [49]。

2. **组合文本到图像生成（Compositional text-to-image generation）**
   - 最近的研究深入探讨了文本到图像生成中的组合性 [14, 15, 50, 51, 17, 52, 53, 16, 54, 41, 55, 51, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]。
   - T2I-CompBench [17] 提出了首个综合基准测试，用于评估文本到图像模型中的组合性，重点关注属性绑定、关系和复杂组合。
   - 尽管这些评估仅适用于图像领域，但视频生成需要更深入地考虑时空动态。
   - 并行工作VideoTetris [40] 提出了一种时空组合扩散框架，使得组合文本到视频生成成为可能。
   - 我们的工作开创了组合文本到视频生成的基准测试的发展。

3. **文本到视频生成的基准测试（Benchmarks for text-to-video generation）**
   - 现有工作使用如UCF-101 [66] 和 Kinetics-400/600/700 [67, 68, 69] 这样的数据集评估文本到视频模型的FVD分数，这些数据集仅限于人类行为的特定主题。
   - 最近的研究设计了文本提示和评估指标，用于评估开放领域的视频质量和文本与视频的对齐性。
   - Make-a-Video-Eval [4] 包含300个提示，但只考虑了空间外观。
   - FETV [23] 基于主要内容、可控属性和提示复杂性对提示进行了分类。
   - VBench [21] 和 EvalCrafter [22] 提出了综合基准测试，从多个角度评估T2V模型，但大多数提示集中在单对象而不是多个对象的组合。
   - 虽然VBench将空间关系作为子类别，但其提示（如“花瓶左边的钟”）并未反映视频中的动态。
   - 在文本到视频生成中缺乏对组合性的全面定义。我们提出了首个用于组合文本到视频生成的基准测试。

4. **文本到视频生成的评估指标（Evaluation metrics for text-to-video generation）**
   - 先前的方法从视频质量和视频与文本的对齐度角度评估视频生成器。
   - 对于视频质量，通常使用的指标如Inception Score (IS) [24] 和 Fréchet Video Distance (FVD) [26] 被用于评估视频质量的多样性和真实性。
   - 对于文本与视频的对齐，CLIPScore [27] 被提出用于使用预训练的CLIP模型 [70] 测量文本提示与图像的相似性。
   - 然而，这些指标并不适用于组合性中的复杂提示。
   - 我们提出了针对我们基准测试的评估指标，并通过广泛的人类相关性研究验证了其有效性。

## 3 Benchmark Construction

### 3.1 Prompt Generation

1. **提示生成**
   - 我们为组合文本到视频生成定义了七个提示类别。每个类别包含100个提示，这些提示是通过向GPT-4 [71] 提供对象类别列表、提示结构和特定提示类别的其他信息生成的。
   - 根据提示类别，提供的信息可能包括属性、空间关系、运动方向、动作、交互和数量。
   - 尽管并非所有提示类别都设计用于评估动作和运动，我们确保所有基准中的提示至少包含一个动作动词，以防止T2V模型生成静态视频。
   - GPT-4 返回的内容包括提示和解析的提示元信息，以便于评估。
   - 生成文本提示和元信息的提示模板详见附录A。所有生成的提示均由人工验证，不合适的提示会被筛除。
   - 整个过程如图2所示。

### 3.2 Prompt Categories

1. **一致的属性绑定（Consistent attribute binding）**
   - 一致的属性绑定要求属性在生成的视频帧中始终与相应的对象保持一致。
   - 为了构建提示，我们定义了四种属性类型，包括颜色（color）、形状（shape）、纹理（texture）和人类属性（human attributes）。
   - GPT-4 被要求生成包含至少两个对象、两个属性（每个属性与一个对象相关）和一个动作动词的句子。
   - 约30%的提示包含颜色属性，20%包含形状属性，20%包含纹理属性，30%包含两种不同类型的属性来描述两个对象。
   - 所有提示中，80%在现实世界中常见，20%则需要想象。

2. **动态属性绑定（Dynamic attribute binding）**
   - 该类别涉及属性随时间变化的问题。例如：“绿色鳄梨变黑，同时旁边的番茄变为深红色”。
   - 我们提示GPT-4生成诸如颜色变化或状态变化的动态属性。80%的提示描述了现实世界中常见的属性变化，而20%是非常见和人工的。

3. **空间关系（Spatial relationships）**
   - 该类别要求模型生成包含正确空间关系的至少两个对象的视频。我们定义了三维空间中两个对象之间的六种空间关系：“在...左侧”、“在...右侧”、“在...上方”、“在...下方”、“在...前面”和“在...后面”。
   - 大约35%的提示包括左/右关系，35%包括上/下关系，剩余30%包括前面/后面关系。
   - 对于包括左或右的空间关系，我们通过反转关系构造了一些对比提示。

4. **动作绑定（Action binding）**
   - 该类别测试模型在存在多个对象和多个动作描述时，将动作绑定到相应对象的能力。
   - 我们提示GPT-4生成包含两个对象的文本提示，每个对象与一个动作动词相关联。
   - 此类别包括80%常见提示和20%非常见提示。非常见提示可以进一步分为对象的非常见共存和非常见的动作-对象对。

5. **运动绑定（Motion binding）**
   - 该类别的提示包含一个或两个具有特定运动方向的对象，旨在测试模型将特定运动方向与对象绑定的能力。
   - 我们定义了四种运动方向：“向左”、“向右”、“向上”和“向下”。提示中的每个对象沿其中一个方向运动。
   - 提示中60%的运动方向是水平的，40%是垂直的。

6. **对象交互（Object interactions）**
   - 该类别测试模型理解和生成动态交互的能力，包括引起运动变化或状态变化的物理交互以及社交交互。
   - 物理交互提示占50%，其中30%涉及状态变化，20%涉及运动变化。
   - 另外50%为社交交互，其中30%是常见的交互，20%是描绘拟人化动物进行社交交互的非常见交互。

7. **生成性数字感知（Generative numeracy）**
   - 此类别的文本提示包括一到两类对象，数量从一到八不等。
   - 大约60%的提示描述单个对象类别，另外40%描述两个对象类别。

### 3.3 Prompt Suite Statistics

1. **提示集统计**
   - 提示集统计与词云可视化和提示统计如图3所示。
   - T2V-CompBench 因其对多个对象和时间动态的关注而脱颖而出：
     1. 与先前主要关注单对象提示的基准测试相比，T2V-CompBench 的提示平均涉及两个以上的对象（名词），每个提示平均包含约3.2个名词。
     2. T2V-CompBench 考虑了时间动态性，几乎所有提示都包含动作动词，平均每个提示包含1.1个动作动词。


## 4 Evaluation Metrics

1. **挑战**
   - 我们观察到组合文本到图像生成的评估指标 [17] 不能直接用于评估组合文本到视频生成，这是由于视频中大量帧和复杂的时空动态。
   - 大多数视频生成模型生成2-5秒内的视频。为了公平比较，我们均匀抽取6帧用于基于MLLM的评估，16帧用于基于检测的评估，并以每秒8帧（FPS）的帧率采样视频用于基于跟踪的评估。

### 4.1 基于MLLM的评估指标（MLLM-based evaluation metrics）

- 多模态大语言模型在理解图像和视频的复杂内容方面展示了巨大的能力 [72, 73, 71, 74]. 受其在视频理解中有效性的启发，我们利用MLLMs作为组合文本到视频生成的评估工具。

1. **Video LLM-based metrics for consistent attribute binding, action binding, and object interactions.**
   - 为了处理视频中的复杂时空信息，我们研究了视频LLM模型，如Image Grid [75] 和 PLLaVA [74]，后者将LLaVA [73] 从单图像输入扩展到多帧输入。
   - 实验发现，Image Grid在组合类别中的表现优于PLLaVA。具体来说，Image Grid [75] 从视频中均匀采样6帧，形成图像网格作为LLaVA的输入。
   - 此外，我们通过链式思维机制（chain-of-thought mechanism）提升视频LLM的能力并避免幻觉问题，即首先要求MLLM描述视频内容，然后要求其对文本-视频对齐进行评分。
   - (1) **一致属性绑定**：我们使用GPT-4解析提示为不同的解耦短语（例如，“蓝色汽车驶过白色栅栏在晴天”解析为“蓝色汽车”和“白色栅栏”），然后要求视频LLM提供每个解耦提示与图像网格的匹配分数。每个解耦短语的得分取平均作为最终得分。
   - (2) **动作绑定**：我们使用GPT-4提取对象及其动作。例如，给定提示“狗在田间奔跑，同时猫爬上树”，我们提取短语“狗”、“狗在田间奔跑”、“猫”和“猫爬上树”。然后要求视频LLM检查每个对象-动作组合与视频的对齐，以获得最终得分。
   - (3) **对象交互**：我们提示视频LLM对视频-文本对齐进行评分。

1. **Image LLM-based metrics for dynamic attribute binding**
   - 像“绿色叶子枯萎变成棕色”这样的动态属性绑定类别具有挑战性，因为它需要对帧间动态变化的深刻理解。
   - 我们发现当前视频LLM在此类别中的表现不理想，因此设计了基于Image LLM（如LLaVA [73]）的逐帧评估指标。
   - 我们使用GPT-4解析初始状态（“绿色叶子”）和最终状态（“棕色叶子”），并提示LLaVA [73] 评估每帧与两个状态的对齐情况。
   - 基于16帧和2个状态的得分，我们定义了一个评分函数，鼓励第一帧与初始状态对齐，最后一帧与最终状态对齐，中间帧处于两者之间。我们将此指标称为D-LLaVA。

### 4.2 基于检测的评估指标（Detection-based Evaluation Metrics for Spatial Relationships and Numeracy）

1. **2D空间关系**
   - 大多数先前的视觉语言模型在空间关系和数字感知相关的理解方面面临困难。因此，我们引入对象检测模型GroundingDINO（G-Dino）[77] 来检测每帧中的对象，过滤掉具有高交并比（IoU）的重复边界框，并基于对象检测结果定义基于规则的指标。
   - 对于包括“左”、“右”、“上方”和“下方”的二维空间关系，我们为每帧定义了类似于T2I-CompBench [17] 的基于规则的指标。
   - 具体来说，对于每对对象，记其中心坐标为 \((x_1, y_1)\) 和 \((x_2, y_2)\)。当 \(x_1 < x_2\) 且 \(|x_1 - x_2| > |y_1 - y_2|\) 时，认为第一个对象在第二个对象的左侧。对于其他空间关系，规则类似。
   - 如果一帧中有多对对象，我们基于其IoU和置信度选择最可能的一个。每帧得分为 \((1 - \text{IoU})\)，如果有对象对满足关系，或为0，如果没有对象对满足关系。视频级得分是所有帧得分的平均值。

2. **3D空间关系**
   - 三维空间关系（“前面”、“后面”）不能通过二维边界框位置来识别。通过GroundingDINO [77] 检测到的二维对象边界框，我们进一步利用Segment Anything [78] 预测指定对象的掩膜，然后利用Depth Anything [79] 预测深度图。
   - 对象的深度定义为掩膜内像素的平均深度值。我们基于两个对象之间的IoU和相对深度定义帧级得分，视频级得分为所有帧级得分的平均值。

3. **生成性数字感知（Generative numeracy）**
   - 为了评估生成性数字感知，我们对每个对象类别检测到的对象数量进行计数。如果检测到的数量与文本提示中的数量匹配，我们为指定的对象类别赋予得分1，否则赋予得分0。帧级得分是所有对象类别得分的平均值，视频级得分是所有帧的平均值。

### 4.3 基于跟踪的评估指标（Tracking-based Evaluation Metrics for Motion Binding）

1. **运动绑定评估**
   - 运动绑定的评估指标应该识别视频中对象的运动方向。在许多视频中，对象的运动与摄像机的运动相互交织，导致难以确定对象的实际运动方向。
   - 在视频中，对象的实际运动方向是前景对象和背景之间的相对运动方向。因此，我们引入了一种基于跟踪的方法，分别确定背景和前景对象的运动方向。
   - 特别地，我们使用GroundingSAM [80] 获取前景对象和背景的掩膜，并采用DOT [81] 跟踪视频中的前景对象点和背景点。
   - 分别计算前景对象点和背景点的平均运动向量，对象运动向量和背景运动向量之间的差异即为对象的实际运动方向。
   - 最终得分反映实际运动方向是否与文本提示中的运动描述一致。

## 5 Experiments

### 5.1 Experimental Setup

1. **评估的模型**
   - 我们在T2V-CompBench上评估了13个开源文本到视频模型和7个商用模型的性能。
   - 开源模型包括ModelScope [13]、ZeroScope [33]、Latte [34]、Show-1 [35]、VideoCrafter2 [36]、Open-Sora 1.1和1.2 [37]、Open-Sora-Plan v1.0.0和v1.1.0 [38]、AnimateDiff [39]、VideoTetris [40]、MagicTime [42] 和 LVD [41]。
   - 商用模型包括Pika [43]、Runway Gen-2 [44]、Runway Gen-3 [45]、Dreamina [46]、PixVerse [47]、Luma Dream Machine [48] 和 Kuaishou Kling [49]。
   - 在这些模型中，LVD [41] 专门设计用于利用LLM引导的布局规划，适用于具有多个对象的视频，而其他模型是文本到视频的基础模型。
   - VideoTetris [40] 被提出用于组合多个对象和对象数量的动态变化。
   - 此外，对于动态属性绑定，我们评估了MagicTime [42]，它从AnimateDiff [39] 训练而来，专为变形延时视频生成而设计。

2. **实现细节**
   - 我们遵循T2V模型的官方实现，更多细节请参考附录B。

### 5.2 Evaluation Metrics

1. **传统评估指标**
   - 我们将提出的评估指标与先前研究中广泛使用的三个指标进行了比较：
     1. **CLIPScore [27]**（简称为CLIP）：计算CLIP文本和图像嵌入之间的余弦相似度。
     2. **BLIP-CLIP [16]**（简称为B-CLIP）：应用BLIP [82] 生成图像的描述，然后计算两个提示之间的CLIP文本-文本余弦相似度。
     3. **BLIP-VQA [17]**（简称为B-VQA）：应用BLIP [82] 提出问题，以评估文本-图像对齐度。视频级评分通过所有帧的平均值计算。

2. **提出的评估指标**
   - 如第4节中所介绍，基于图像LLM的指标**D-LLaVA**用于评估动态属性绑定，基于检测的指标**G-Dino**用于评估空间关系和生成性数字感知，基于跟踪的指标**DOT**用于评估运动绑定。
   - 更多关于我们提出的评估指标的细节，请参考附录C。此外，我们测试了所有类别的基于视频LLM的指标（ImageGrid-LLaVA、PLLaVA）和基于图像LLM的指标（LLaVA）。
   - 在下一小节中，我们通过评估指标与人工评分之间的相关性确定每个类别的最佳指标。

### 5.3 Human Evaluation Correlation Analysis

1. **人工评估**
   - 在每个类别的人工评估中，我们从100个提示中随机选择15个，并使用不同的视频生成模型生成共90个视频。此外，我们在动态属性绑定类别中加入了10个真实视频，在对象交互类别中加入了16个真实视频。用于人工评估的视频总数为656个。
   - 我们使用Amazon Mechanical Turk，并要求三位标注人员为每个视频的文本-视频对齐评分。每个文本-视频对的得分取三人评分的平均值，并使用Kendall's τ和Spearman's ρ计算人工评分与自动评估分数之间的相关性。更多人工评估细节请参考附录D。

2. **评估指标之间的比较**
   - 人工相关性结果如表1所示。结果验证了我们提出的评估指标的有效性（以粗体显示），其中**ImageGrid-LLaVA**用于一致属性绑定、动作绑定和对象交互，**LLaVA**用于动态属性绑定，**G-Dino**用于空间关系和生成性数字感知，**DOT**用于运动绑定。
   - CLIP和B-CLIP表现出相对较低的人工相关性，表明它们无法捕捉复杂视频和文本提示中的细粒度属性和动态变化。**LLaVA**、**ImageGrid-LLaVA**和**PLLaVA**在动作绑定中表现良好，但在捕捉属性或状态的动态变化（如动态属性绑定）以及需要理解空间关系、运动方向和数字感知的任务中表现不足。
   - 相比之下，我们提出的**D-LLaVA**显著增强了捕捉视频中动态属性变化的能力，导致其在动态属性绑定中的人工评估相关性更高。此外，基于检测和跟踪的指标在捕捉组合性的空间和计数方面表现出改进。**ImageGrid-LLaVA**在一致属性绑定和对象交互方面优于**LLaVA**，因为它考虑了时间序列而不仅仅是静态帧。

### 5.4 Quantitative Evaluation

1. **定量评估结果**
   - 模型在T2V-CompBench上的表现如表2所示。与不同模型的比较中，我们观察到以下几点：
     1. 商用模型在七个组合类别中的表现优于开源模型。
     2. Show-1 [35]、VideoCrafter2 [36]、VideoTetris [40]、Open-Sora 1.1和1.2 [37] 以及 Open-Sora-Plan v1.1.0 [38] 整体表现较好。
     3. Open-Sora-Plan v1.1.0 在七个组合类别中相比之前版本Open-Sora-Plan v1.0.0有显著改进，并在一致属性绑定、动作绑定和生成性数字感知方面领先。
     4. Show-1 [35] 在对象交互方面表现出色，而VideoTetris [40] 在动态属性绑定方面表现优异。
     5. LVD [41] 在空间关系和运动绑定方面表现良好，这得益于其利用LLM引导布局规划的设计。

### 5.5 Qualitative Evaluation

1. **质性评估结果**
   - 七个组合类别的具有挑战性案例如图4和图5所示，挑战难度从上到下逐渐降低。图4显示了开源模型的表现，而图5则展示了商用模型的表现。
   - 我们观察到以下几点：
     1. 最具挑战性的是动态属性绑定和生成性数字感知（图4和图5的第1-2行），这些类别需要对时间动态的细粒度理解或准确计数。在这些类别中，模型往往忽略部分提示。在动态属性绑定中，T2V模型往往专注于提示中的关键词，忽略属性或状态的动态变化，倾向于生成一致结果而未反映所需的转换。在数字感知中，T2V模型在数量小于三时处理较好，但当数量超过三时表现不佳。
     2. 第二难的类别包括空间关系、运动绑定和动作绑定（图4和图5的第3-5行）。对于空间关系，模型经常混淆“左”和“右”等定位术语。在运动绑定中，随着“向左航行”或“向右飞行”等移动方向的描述，问题变得更严重。对于动作绑定，模型未能正确将某些动作与对象关联。例如，给定提示“狗在田间奔跑，同时猫爬上树”，模型往往生成两个动物都在奔跑而非执行各自的动作，或者可能完全忽略一个对象。
     3. 最后具有挑战性的类别是对象交互和一致属性绑定（图4和图5的第6-7行）。在对象交互中，模型通常生成几乎静态的视频，忽略整个交互过程。在一致属性绑定中，模型有时会将属性与特定对象混淆，或者忽略某些对象。

## 6 Conclusion and Discussions

1. **系统性研究**
   - 我们进行了首次关于文本到视频生成中组合性的系统性研究。
   - 我们提出了T2V-CompBench，这是一个用于组合文本到视频生成的综合基准测试，包含7个类别的700个提示。
   - 我们进一步为这7个类别设计了一套评估指标。

2. **模型评估与分析**
   - 最终，我们对9个文本到视频生成模型进行了基准测试，并对当前文本到视频模型的组合性进行了有见地的分析。
   - 组合文本到视频生成对于当前模型来说非常具有挑战性，我们希望我们的工作能够激励未来的研究改进文本到视频模型的组合性。

3. **研究的局限性**
   - 我们工作的一个局限性是缺乏一个统一的评估指标来覆盖所有类别。我们认为这个局限性指出了多模态大语言模型（LLMs）或视频理解模型面临的新挑战。
   - 社区应意识到视频生成模型可能被用于生成误导人们的假视频的潜在负面社会影响。
