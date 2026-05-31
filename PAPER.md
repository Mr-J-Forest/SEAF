# 论文题目：基于时空增强ConvLSTM与线性-非线性混合集成的海洋温盐场预测

**时空增强 (Spatio-temporally Augmented)**：涵盖了位置编码（增强输入）和注意力机制（增强特征提取）。

**线性-非线性混合集成**：描述了 ARIMA (线性) + Deep Learning (非线性) 的 Stacking 策略。

## 目录

1. [摘要 (Abstract)](#1-摘要-abstract)
2. [引言 (Introduction)](#2-引言-introduction)
3. [相关工作 (Related Work)](#3-相关工作-related-work)
4. [数据与预处理 (Data and Preprocessing)](#4-数据与预处理-data-and-preprocessing)
5. [方法论 (Methodology)](#5-方法论-methodology)
6. [实验设置 (Experimental Setup)](#6-实验设置-experimental-setup)
7. [结果与分析 (Results and Analysis)](#7-结果与分析-results-and-analysis)
8. [结论与展望 (Conclusion and Future Work)](#8-结论与展望-conclusion-and-future-work)
9. [参考文献 (References)](#9-参考文献-references)

---

## 1. 摘要 (Abstract)

海洋温盐场的三维结构对于理解海洋动力学过程、水声传播特性及全球气候调节机制至关重要。然而，受限于次表层观测数据的稀疏性及传统数值同化系统的高昂计算成本，获取高分辨率、高精度的温盐场预报仍面临巨大挑战。同时，现有的深度学习方法在捕捉长时空依赖关系及保持物理一致性方面仍存在不足。为此，本文提出了一种基于改进卷积长短期记忆网络（ConvLSTM）的时空预测框架。该模型融合了次表层温盐剖面（Argo）、海面高度异常（SSHA）及海面风场（CCMP）等多源数据，并通过引入通道与空间双重注意力机制及残差连接，显著增强了特征提取能力与梯度流动。此外，本文设计了ARIMA与XGBoost结合的混合集成策略，以有效修正非线性预测误差并提升长期预测的稳定性。在西太平洋海域（130.5°E–162.5°E，6.5°N–27.5°N）的广泛实验表明，该方法显著优于CNN及标准ConvLSTM等基准模型。实验结果显示，温度和盐度预测的均方根误差（RMSE）分别降低至0.0871°C和0.1327 PSU，较CNN基准模型分别提升了42.4%和71.9%。值得注意的是，本文引入了位势密度（Potential Density）和辣度（Spiciness）作为物理一致性评价指标，证实了该模型在保持海水物理性质及静态稳定性方面的优势，为业务化海洋环境预报提供了一种可靠的新工具。

   **关键词** ：海洋温盐场预测；ConvLSTM；时空深度学习；注意力机制；物理一致性；多源数据融合

---

## 2. 引言 (Introduction)

### 2.1 研究背景

海洋温度盐度是非常重要的海洋环境参数，研究海洋温盐场对于声纳探测、渔业资源、气候调控等都有重要意义。

首先，海水声速是由温度、盐度和压力共同决定的，海洋温度和盐度的不均匀分布导致声速剖面结构和水平分布特征变化[1]，进而影响声波折射路径、传播损失与隐蔽性,最终影响声波的传播。例如，跃温层的存在会产生明显声波折射，使目标声回波向深水偏折或形成声影区，从而影响水下目标探测与通信性能。精确掌握温盐结构有助于提高声学模型计算准确性，优化声呐布放与探测策略。

此外，在生态资源管理方面，温盐场约束着海洋生物分布、种群动态及渔业资源变动，其变化对渔业生产、生物多样性保护和生态系统稳定性有直接影响。

最后，在气候系统调控方面，水下温盐场对于气候变化起着重要作用[2]，温盐场通过影响海水密度分布,进一步驱动全球热盐环流，承担着热量和盐分的跨洋输送任务，对维持地球能量平衡和调节区域乃至全球气候模式具有决定性作用。

### 2.2 问题陈述

#### 2.2.1 海洋环境数据的稀疏性与时空高动态性带来的挑战。

在本研究中，面临的一个重要问题便海洋环境数据的稀缺性，ARGO 漂流浮标、卫星遥感等常规海洋数据库都存在着覆盖面积不足、观测不连续等一系列问题，如下图海洋温度数据所示，可以清晰的看到陆地周边的近海海域的海洋温度数据存在大量缺失的问题，故需要通过创新性手段来扩充并增强数据集。

![](./assets/1765170582694-1.png)

#### 2.2.2 传统数值同化方法的局限性（计算资源消耗大）。

传统海洋数据同化方法（如 4D-Var、EnKF）在提升模式预报精度方面具有重要作用，但普遍存在计算成本高的问题。4D-Var 需进行多次模式正、伴随积分以求解高维优化问题，对高性能计算平台依赖显著；EnKF 则需维持大量集合成员，其计算量与内存开销随分辨率和观测数量急剧增长。同时，误差协方差构建复杂、观测稀疏且误差统计不完善，使同化系统需频繁更新，从而进一步加重运算负担。因此，传统同化方法难以在高分辨率与强实时需求下实现高效应用。

#### 2.2.3 深度学习在海洋预测中的应用现状及存在的问题（如长时预测的模糊效应）。

近年来，随着计算能力和海洋观测数据的大量积累，深度学习方法被广泛尝试用于海洋预测。相比传统数值模拟和同化系统，深度模型在预测速度与数据驱动能力上具有显著优势

不过，深度学习在海洋预测中的应用也并非没有问题。首先，许多模型对训练数据高度依赖，而海洋观测数据往往稀疏、不均匀，难以覆盖所有时空与物理过程 ，这影响模型对“未见”区域与极端状态的泛化能力。其次，对于 中长期 (长时序) 预测，深度模型容易出现 模糊化 (blurring effect)、细节丢失，尤其是当试图预测涡旋、锋面、混合层等包含强非线性、细尺度结构的过程时；这种模糊与平均化倾向会削弱预测结果的物理可信度。再者，仅依赖数据驱动模型缺乏对质量守恒、热盐守恒、动量平衡等物理规律的显式约束，因此其 可解释性、物理一致性 通常较弱，不利于科学研究与工程应用。最后，大多数深度学习系统目前仍侧重短期或中期 (如天–月尺度) 预测，对季节–年际尺度 (或气候尺度) 的适用性与稳定性尚存较大不确定性。

### 2.3 本文贡献

1. 构建了基于 ConvLSTM 的多变量海洋温盐反演框架

本研究提出并实现了一个面向三维温盐场序列预测的卷积门控时序网络框架，结合空间卷积和时间递归以并行建模时空相关性。该框架设计注重多变量耦合输出与多步预测能力，适用于同时恢复温度与盐度的时空演化。

2. 设计了针对性的数据增强策略（滑动窗口）以解决训练样本不足问题。

采用有针对性的数据增强与预处理流程以缓解观测稀疏与样本不足问题，提升模型对局部结构与平移不变性的泛化能力，同时采用规范化与稳健的缺测处理策略以减少训练偏差。

3. 物理一致性评估：

在传统点级误差（如 MAE、RMSE）之外，系统地计算并比较若干关键衍生物理量（例如基于温盐计算的密度与“spiciness”指标），以评估预测场是否保持了必要的物理关系。该评估流程使得结果不仅在统计意义上优良，而且在物理意义上更具说服力。



---

## 3. 相关工作 (Related Work)

### 3.1 传统海洋数值模式

传统的海洋数值模式主要是基于连续性方程和热盐守恒等基本海洋动力方程，对海洋环流、温盐分布和能量传输进行数值求解。其中具有代表性的模式包括 HYCOM（Hybrid Coordinate Ocean Model） 和 ROMS（Regional Ocean Modeling System）等。

HYCOM 采用混合坐标系，能够在不同深度区域自适应切换 σ 坐标、等密度坐标和 z 坐标，从而更准确刻画密度层结构和大尺度海洋环流特征，ROMS是一个具有自由表面、地形跟随坐标系和原始控制方程的三维海洋环流模式[3]。

近年来，使用传统物理海洋数值模型对未来海洋温度场进行预测，该模型基于物理方程和海洋学原理，根据初始条件并假定若干边界条件， 建立一系列等式方程对海洋的温度分布进行模拟并提供相对可靠的温度预测，但由于海洋动力学过程较复杂，数值模型的性能往往受海气相互作用和其他参量 的影响，对初始条件敏感，需要高质量的初始观测数据和对应的边界条件才能获得准确结果。

### 3.2 基于深度学习的时空预测

#### 3.2.1 RNN, LSTM 在时序预测中的应用。

循环神经网络（RNN）及其门控变体长短期记忆网络（LSTM）是在进行时间序列建模时常用的模型，他们擅长学习序列中的时间依赖与记忆效应。其中，LSTM 通过门控机制可以有选择地保留与更新历史信息，可以有效地缓解长期依赖下产生的梯度消失问题，因此在气象与海洋等领域的点级时间序列预测与多变量时间序列建模中被广泛采用。但是需要注意的是，LSTM模型自身主要面向时间维度进行建模，当直接用于高维格点场时，若没有合适的空间编码（例如卷积或局部感受野设计），会丧失邻域空间结构信息，因此在空间场预测任务中常与卷积结构或空间注意力模块结合使用，来同时保留时间与空间特征

#### 3.2.2 ConvLSTM (Shi et al.) 及其变体 (PredRNN, SA-ConvLSTM) 在雷达回波、SST 预测中的应用。

ConvLSTM 将卷积算子嵌入门控时序单元，使网络在时间递归的同时保留二维空间邻域结构，这一点使其天然适合用于降水雷达回波、云图演化与海表温度（SST）等具有明显空间运动特征的时空场预测。基于 ConvLSTM 的方法可以直接学习移动、扩散与局部非线性演化，从而在短期预报中取得优越的性能。为进一步改善长期依赖和全局语义保持，研究者提出了多种变体：例如通过引入记忆流或多通道记忆单元增强跨时步信息交互，以缓解“深时”带来的信息丢失；又如结合自注意力/非局部模块以扩大感受野，捕获远距离依赖与大尺度模态，从而在保持局部细节的同时改善整体场结构。实际应用表明，这些改进在降水回波的移动预测与 SST 的短时演变重构上都能减少模糊化、提高空间一致性，但对于长时演化仍需与物理约束或多尺度设计协同，以控制误差累积。

#### **3.2.3 **本项目模型与标准 ConvLSTM 的区别

本项目在标准 ConvLSTM 基础上做出若干针对工程与物理场景的改进，目的在于提升小批量训练稳定性、加速收敛并增强对长程与全局依赖的建模能力。首先，采用更适合小批量训练的归一化策略以减少对批次统计的依赖，从而在显存受限或 batch size 较小的条件下仍能保持训练稳定性；其次，通过残差化的特征传递（层间/模块间的跳跃连接）改善梯度流与保留高频细节，这对捕捉锋面与强梯度区尤为重要；再次，引入基于注意力或非局部思想的全局上下文整合模块，用于在初步重构之后融合远距离关联并校正系统性偏差。理论与经验研究表明：归一化与残差结构分别有助于训练稳定性与深层特征保持；注意力/非局部机制能有效扩展模型的有效感受野，从而改善大尺度模态的保持。基于本项目的实验观察，这些改进带来了验证误差的稳定下降、预测场在物理衍生量（如密度与 spiciness 指标）上的更好一致性，说明模型在统计精度与物理合理性两方面均获得了改进。



---

## 4. 数据获取与预处理策略 (Data Acquisition and Preprocessing Strategy)

### 4.1 多源数据融合与研究区域

本研究构建了一个融合多源观测资料的综合数据集，旨在利用海表面动力环境数据辅助反演海洋内部的三维温盐结构。基础数据来源于 Argo 浮标提供的温盐剖面网格化数据（包含深度、温度、盐度等核心变量），并引入 CCMP (Cross-Calibrated Multi-Platform) 风场数据以补充海表面风速与风向的时空分布特征，后者在海洋动力过程研究中具有重要表征意义。

为实现多源数据的时空对齐，本研究执行了以下数据融合流程：

1. **空间基准统一** ：读取海表面高度异常 (SSHA) 数据与 CCMP 风场数据，将经度坐标统一标准化至 0°–360° 范畴，确保空间坐标系的一致性。
2. **时空匹配与插值** ：以 SSHA 数据的时间序列为基准，检索 CCMP 数据集中对应的时间步。利用正则网格插值算法 (RegularGridInterpolator)，将 CCMP 数据从原始网格映射至目标 SSHA 网格，实现空间分辨率的精确匹配。
3. **变量整合** ：将处理后的纬向风 (UWND)、经向风 (VWND) 及合成风速 (SSW) 作为协变量整合至原数据集中。

本研究聚焦于南海及周边海域的海洋环境反演，选取的研究区域范围为：经度 130.5°E–162.5°E，纬度 6.5°N–27.5°N，深度覆盖 0–1000 m。整合后的数据集包含变量如表 1 所示。

**表 1 数据集变量说明**

| 变量符号          | 物理含义       | 数据维度        | 单位      | 说明     |
| ----------------- | -------------- | --------------- | --------- | -------- |
| **TIME**    | 时间序列       | 1D              | Month     | 时间基准 |
| **TEMP**    | 海水温度       | 4D (T, Z, Y, X) | °C       | 目标变量 |
| **SALT**    | 海水盐度       | 4D (T, Z, Y, X) | PSU       | 目标变量 |
| **SSHA**    | 海表面高度异常 | 3D (T, Y, X)    | m         | 输入变量 |
| **UWND**    | 10米纬向风     | 3D (T, Y, X)    | m/s       | 输入变量 |
| **VWND**    | 10米经向风     | 3D (T, Y, X)    | m/s       | 输入变量 |
| **LEVEL**   | 垂直层级       | 1D              | m         | 深度坐标 |
| **LON/LAT** | 经纬度         | 1D              | °E / °N | 空间坐标 |

在本研究的模型架构中，输入向量 $X$ 包含 TEMP, SALT, SSHA, UWND, VWND 五类变量，旨在通过反演模型预测未来时刻的海洋内部温盐场状态：
$$
Y = \{ \text{TEMP}, \text{SALT} \}
$$
### 4.2 数据预处理流程 (Data Preprocessing Pipeline)

为消除量纲差异并满足深度学习模型的训练需求，本研究实施了严格的数据预处理方案。

 **4.2.1 Z-Score 标准化 (**  **Normalization** **)**

为加速模型收敛并提升数值稳定性，本研究采用 Z-Score 方法对所有输入变量进行标准化处理。变换公式如下：

$$
x' = \frac{x - \mu}{\sigma}
$$

其中，$\mu$ 和 $\sigma$ 分别为变量的均值与标准差，$x'$ 为标准化后的输入值。

 **防泄漏策略** ：为严格避免数据泄露 (Data Leakage)，统计量 $\mu$ 和 $\sigma$ 仅基于**训练集**数据计算得出，并被保存为标准化参数文件。验证集和测试集均使用训练集的统计参数进行变换，从而保证模型评估的客观性与泛化能力。

**4.2.2 分级缺失值填补 (Hierarchical Imputation)**

针对海洋观测数据中不可避免的缺失值 (NaN) 问题，本研究设计了分级填补策略：

1. **局部插值** ：对于任意数据切片，若存在部分缺失，优先计算该切片内有效数据的均值进行填充，以保留局部环境特征。
2. **全局填充** ：若某切片完全缺失（全为 NaN），则使用该变量在整个数据集上的全局有效均值进行填充，确保数据完整性。

**4.2.3 时空序列样本构建 (Spatiotemporal Sequence Construction)**

本研究采用滑动窗口技术将连续的网格化数据转换为监督学习样本。设定历史观测窗口长度 $T_{in} = 10$，预测窗口长度 $T_{out} = 5$。

 **时间维度** ：沿时间轴滑动截取序列，生成输入张量 $X \in \mathbb{R}^{T_{in} \times C \times H \times W}$ 与标签张量 
$$
Y \in \mathbb{R}^{T_{out} \times C' \times H \times W}
$$
 **空间维度** ：采用 32°×21° 的空间补丁 (Patch)，以 2° 为步长在经度方向遍历。

 **质量控制** ：引入海洋覆盖率阈值筛选机制，仅保留海洋面积占比超过 80% 的子区块，以剔除陆地干扰，聚焦海洋本身的变化规律。

### 4.3 基于物理先验的数据增强策略 (Physics-informed Data Augmentation)

针对海洋月平均数据时间跨度有限、样本量不足的问题，本研究提出了基于空间拓扑与物理对称性的双重数据增强策略。通过联合应用全经度滑动窗口采样与赤道对称扩充机制，将原始数据集的样本量扩充了约 140 倍，有效解决了深度学习模型训练中的数据匮乏瓶颈。

**4.3.1 空间滑动窗口采样 (Spatial Sliding Window Sampling)**

为了克服单一研究区域样本量不足的限制，本研究采用了全局经度滑动采样策略。我们以目标研究区域的尺度（32°×21°）作为滑动窗口，在全经度范围（0°–360°）内以 2° 为步长进行遍历采样。通过筛选海洋覆盖率达标（>80%）的区域，我们将原本仅限于南海海域的单一训练样本，扩展为覆盖全球同纬度带的众多具有相似物理属性的样本。这种方法将有限的局部数据转化为全球尺度的特征学习，显著增强了模型的泛化能力。

**4.3.2 赤道对称扩充 (Equatorial Symmetry Augmentation)**

基于海洋动力过程在赤道两侧的物理对称性（如科里奥利参数的反对称特征），本研究进一步引入了赤道对称扩充机制。我们将上述滑动窗口采样的纬度范围（6.5°N–27.5°N）关于赤道进行镜像翻转，得到南半球对应的纬度带（6.5°S–27.5°S）。随后，在该镜像纬度带上同样执行全经度滑动采样。这一策略不仅利用了南北半球的物理相似性实现了样本倍增，还通过引入南半球数据，帮助模型学习更普适的低纬度海洋动力学规律。

---

## 5. 方法论 (Methodology)

本研究提出了一种融合物理先验知识的深度学习时空序列预测框架，旨在实现复杂海洋环境中温度与盐度场的高精度三维预报。该框架主要采用编码器-解码器（Encoder-Decoder）架构，以在海洋温盐场预测任务中已被广泛验证具有优良的长时间序列依赖建模能力的卷积长短期记忆网络（ConvLSTM）[7]作为核心时空特征提取单元。为了进一步提升模型性能与预测鲁棒性，本研究在ConvLSTM基础上引入以下三项关键改进机制：

（1）残差连接（Residual Connections）：有效缓解深层网络训练中的梯度消失问题，增强多尺度时空特征的传递与复用；

（2）注意力细化模块（Attention Refiner）：通过通道与时空双注意力机制自适应聚焦于对温盐场演变具有关键影响的动态特征，提升模型对复杂海洋动力过程的表征能力；

（3）线性与非线性混合集成策略（Hybrid Linear-Nonlinear Ensemble）：在解码阶段同时输出深度非线性预测结果（即基于ConvLSTM的改进模型预测输出），与基于物理趋势的线性校正项（即ARIMA模型预测输出），并进行基于XGBoost学习器的加权融合，显著抑制多步预测中的误差累积，提高长期预报稳定性。

消融实验验证表明，本研究提出的框架在温度与盐度场的短期至次季节尺度预测中，相较于基准ConvLSTM模型，取得了显著的精度提升与综合性能改善。

### 5.1 问题定义 (Problem Definition)

​	本研究的主要目的是根据海洋的历史状态信息，预测海洋未来的状态，为了使其信息能够输入模型，我们将研究涉及的海洋环境要素预测形式化地抽象为一个高维时空序列的预测问题，即基于给定过去的 $J$ 个时间步的状态，预测未来 $K$ 个时间步的状态。
最基本的，海洋作为不断变化的三维物理空间，需要一个三维张量来描述海洋的基本空间状态，为此我们设计了 $\mathcal{X}_t \in \mathbb{R}^{D \times H \times W \times C}$ 表示在时间步 $t$ 的海洋状态张量，其中 $D, H, W$ 分别代表深度（Depth）、纬度（Latitude）和经度（Longitude）的空间网格维度，构成了基本的预测三维空间。在此基础上，以 $C$ 代表物理变量的通道数（温度、盐度、海表面高度、海表风场），从而进一步扩充了特征信息纬度，使得模型可以捕捉更丰富的海洋物理变化规律。
​	构造了基本的海洋状态张量之后，我们将模型的输入序列定义为 $\mathbf{X}_{in} = \{\mathcal{X}_{t-J+1}, \dots, \mathcal{X}_t\}$，将模型输出的目标预测序列定义为 $\hat{\mathbf{Y}}_{out} = \{\hat{\mathcal{X}}_{t+1}, \dots, \hat{\mathcal{X}}_{t+K}\}$。根据预期，训练的预测模型可以表示为找到一个非线性映射函数 $\mathcal{F}$，其参数为 $\Theta$。
$$
\hat{\mathbf{Y}}_{out} = \mathcal{F}_\Theta(\mathbf{X}_{in})
$$
​	我们的目标是通过深度学习策略最小化预测值与真实值 $\mathbf{Y}_{true}$ 之间的损失函数（如均方误差 MSE）来学习最优参数 $\Theta^*$：
$$
\Theta^* = \operatorname*{argmin}_\Theta \sum_{t} \mathcal{L}(\hat{\mathbf{Y}}_{out}, \mathbf{Y}_{true})
$$

### 5.2 模型架构 (Model Architecture)

​	在模型结构上，我们选用了在时空序列预测领域已经被广泛证明具有很好效果的完全端到端的Encoder-Decoder结构[8,9]，使用卷积LSTM网络作为模型的基干部分。在此基础上，针对海洋三维温盐场的多尺度时空非平稳特性、深层特征提取瓶颈以及复杂的线性与非线性动力学耦合机制，针对性地给出了时空编码增强、细化残差连接、多头注意力机制、以及融合ARIMA与深度特征的线性-非线性Stacking集成机制的深度改进优化，在结果上实现了从原始网格化观测直接到未来多铅步高分辨率预测的全流程端到端训练。

![](./assets/1765170604189-7.png)

#### 5.2.1 时空特征编码机制 (Spatiotemporal Feature Encoding)

由于卷积神经网络（CNN）本质上具有平移不变性（Translation Invariance），传统的 ConvLSTM 模型难以直接感知观测数据所在的绝对地理位置与时间周期（如季节性变化）。为了弥补这一缺陷，本研究设计了一套多维时空编码模块，将经纬度、深度以及时间信息映射为高维特征向量，并与物理状态变量（如温度、盐度）进行通道级联（Channel-wise Concatenation），作为模型的增强输入 $X_{in} \in \mathbb{R}^{T \times (C_{phy} + C_{enc}) \times H \times W}$。

##### 1. 空间位置编码 (Spatial Positional Encoding)
受 Transformer 架构启发，我们采用正弦-余弦位置编码（Sinusoidal Positional Encoding）来表征网格点的地理坐标。对于空间网格中的任意一点 $(h, w)$，其对应的经度 $\lambda$ 和纬度 $\phi$ 首先被转换为弧度制。为了捕捉不同尺度的空间依赖关系，我们构建了包含 $N_s$ 个频率分量的编码向量。

对于位置 $p \in \{\lambda, \phi\}$，第 $k$ 个频率分量（$k=0, \dots, N_s-1$）的编码公式定义为：

$$
\begin{aligned}
PE_{(p, 2k)} &= \sin\left(\frac{p}{10000^{k/N_s}}\right) \\
PE_{(p, 2k+1)} &= \cos\left(\frac{p}{10000^{k/N_s}}\right)
\end{aligned}
$$

经度和纬度的编码向量在特征维度上拼接（Concatenation），形成总维度为 $2N_s + 2N_s = 4N_s$ 的空间特征图。该特征图在时间维度上进行广播（Broadcast），以保持时序一致性。

##### 2. 深度层级编码 (Depth Hierarchical Encoding)
针对海洋数据的三维垂直结构，模型引入深度编码以显式区分和利用不同水层（如混合层、温跃层与深层）的动力学特征差异。由于 2D ConvLSTM 通常将不同深度的变量堆叠在通道维度，缺乏对“深度”这一物理坐标的内在感知，我们设计了一种基于正弦几何级数的深度位置嵌入机制。

设输入数据包含 $L$ 个垂直层级，第 $z$ 层（$z=1, \dots, L$）对应的物理深度值为 $d_z$（单位：米）。我们利用正弦-余弦基函数生成该层的深度嵌入向量 $E_{depth}^{(z)} \in \mathbb{R}^{2N_d}$，其中 $N_d$ 为设定的频率数量（Frequency bands，本研究中设为 4）。对于第 $k$ 个频率分量（$k=0, \dots, N_d-1$），其编码公式如下：

$$
\begin{aligned}
\omega_k &= \frac{1}{10000^{2k/N_d}} \\
PE_{(z, 2k)} &= \sin(d_z \cdot \omega_k) \\
PE_{(z, 2k+1)} &= \cos(d_z \cdot \omega_k)
\end{aligned}
$$

上述机制为每一个垂直层级生成了一个唯一的、连续的特征向量。为了与输入的时空特征图对齐，我们将该向量在空间维度（$H \times W$）和时间维度（$T$）上进行广播（Broadcasting）。最终，模型将所有 $L$ 个层级的编码在通道维度级联，形成总通道数为 $L \times 2N_d$ 的深度特征图。这种显式编码使得 2D 卷积核在处理通道维数据时，能够通过深度特征“感知”当前处理的数据所处的物理深度，从而自适应地学习深度依赖的物理规律（例如，在表层侧重风生混合作用，在深层侧重热盐扩散作用）。

##### 3. 时间傅里叶编码 (Temporal Fourier Encoding)
海洋环境要素通常表现出显著的季节性周期（如海温的年循环）。为了显式建模这一周期性，我们应用基于傅里叶级数的时间编码。对于时间步 $t$ 对应的月份 $m_t \in [0, 11]$，我们以 $T_{period}=12$ 为基准周期，计算 $N_t$ 阶谐波分量：

$$
\begin{aligned}
TE_{(t, 2k)} &= \sin\left(\frac{2\pi \cdot m_t \cdot (k+1)}{T_{period}}\right) \\
TE_{(t, 2k+1)} &= \cos\left(\frac{2\pi \cdot m_t \cdot (k+1)}{T_{period}}\right)
\end{aligned}
$$

此外，为了捕捉非平稳的长期气候变化趋势，我们引入了归一化的年份趋势项 $Y_{norm} = (year_t - year_{min}) / (year_{max} - year_{min})$。最终的时间编码融合了周期性特征与线性趋势特征，使得模型既能捕捉季节性波动，也能适应年际变化。

#### 5.2.2 ConvLSTM 单元

​	为了有效建模海洋温盐场复杂的**时空耦合动力学机制**，本文选取卷积长短期记忆网络（ConvLSTM）作为预测框架的时空特征提取骨干（Backbone）。与标准 LSTM 依赖全连接操作（Full Connection）不同，ConvLSTM 的核心创新在于将输入到状态、状态到状态的所有变换算子替换为**卷积运算（Convolution, \*）**。

​	这一设计在处理海洋环境数据时具有极强的物理适配性：海洋中的热量传递与盐分扩散主要受制于局部流场与梯度的相互作用，卷积核的局部感受野（Receptive Field）恰好能模拟这种**局部物理过程**；同时，卷积操作天然保留了输入张量的**空间拓扑结构**，避免了传统 LSTM 因向量展平（Flattening）导致的空间邻域信息丢失，使其能够捕捉涡旋、锋面等中尺度结构的动态演化[7]。

​	ConvLSTM 单元通过门控机制控制信息流，其在时间步 $t$ 的状态更新遵循以下动力学方程：

$$\begin{aligned} i_t &= \sigma(W_{xi} * \mathcal{X}_t + W_{hi} * \mathcal{H}_{t-1} + W_{ci} \circ \mathcal{C}_{t-1} + b_i) \\ f_t &= \sigma(W_{xf} * \mathcal{X}_t + W_{hf} * \mathcal{H}_{t-1} + W_{cf} \circ \mathcal{C}_{t-1} + b_f) \\ \mathcal{C}_t &= f_t \circ \mathcal{C}_{t-1} + i_t \circ \tanh(W_{xc} * \mathcal{X}_t + W_{hc} * \mathcal{H}_{t-1} + b_c) \\ o_t &= \sigma(W_{xo} * \mathcal{X}_t + W_{ho} * \mathcal{H}_{t-1} + W_{co} \circ \mathcal{C}_t + b_o) \\ \mathcal{H}_t &= o_t \circ \tanh(\mathcal{C}_t) \end{aligned}$$

​	其中，$\mathcal{X}_t$ 为当前时刻的温盐输入场，$\mathcal{H}_t$ 为隐藏状态（即输出的特征图），$\mathcal{C}_t$ 为细胞状态（记忆单元）。符号 $*$ 表示卷积运算，$\circ$ 表示哈达玛积（Hadamard product，即逐元素相乘）。

​	在该结构中：

- **遗忘门 ($f_t$)** 决定了多少历史热盐记忆（$\mathcal{C}_{t-1}$）被保留，这模拟了海洋系统的**热惯性与持续性**；
- **输入门 ($i_t$)** 控制当前时刻观测数据（$\mathcal{X}_t$）对系统状态的更新程度，反映了**外部强迫**对海洋环境的影响；
- **细胞状态 ($\mathcal{C}_t$)** 则作为贯穿时序的核心“记忆流”，累积并传递长时程的演变规律。

​	尽管 ConvLSTM 具备优异的时空建模能力，但在面对深层网络训练时仍面临梯度衰减挑战，且卷积核的固定权重难以自适应聚焦于关键的变异区域。因此，本文在此基础上进一步引入了残差连接与注意力机制（详见下节），以构建更高性能的预测架构。

#### 5.2.3 改进机制与混合策略 (Advanced Mechanisms & Hybrid Strategy)

为了进一步提升模型的预测精度和鲁棒性，我们在基础 ConvLSTM 架构上引入了以下针对性改进。这些改进基于当前海洋时空预测领域的最新进展，旨在缓解深层网络的梯度消失问题、增强对关键特征的动态关注能力，并通过混合集成策略有效分离线性与非线性动态，从而实现互补建模和误差最小化。具体而言，改进模块的引入不仅降低了训练难度，还显著提高了模型在南海等复杂区域温盐场预测中的泛化性能[35-40]。

##### 1. 残差连接 (Residual Connections)

随着 ConvLSTM 网络层数的加深，梯度消失和爆炸问题会显著限制模型的训练效果，尤其在处理长序列海洋时空数据时，导致深层特征提取不充分[35]。为缓解这一问题，我们在堆叠的 ConvLSTM 层之间引入残差连接（Residual Connections），该机制允许梯度直接通过跳跃连接（Skip Connections）从深层传播回浅层，从而促进更稳定的优化过程和更深网络的训练[36]。具体而言，对于第 $l$ 层输出 $\mathcal{H}^l_t$，其计算公式为：

$$
\mathcal{H}^l_t = \mathcal{H}^{l-1}_t + f_l \bigl( \mathcal{H}^{l-1}_t,\; \mathcal{H}^l_{t-1},\; \mathcal{C}^l_{t-1} \bigr)
$$

其中 $f_l(\cdot)$ 表示第 $l$ 层 ConvLSTM 单元（包含所有门结构和内部状态更新）。该设计保留了原始输入的低阶特征，同时允许网络学习高阶残差映射，避免信息丢失。

在海洋温盐场预测中，残差连接已被证明能有效提升多层 ConvLSTM（M-ConvLSTM）的性能，例如在三维海洋温度预测任务中，引入残差模块后，模型的 RMSE 在温度预测任务上显著降低，特别是在捕捉垂直温度梯度和涡旋演变等长程依赖时表现出色[35,37]。这一改进的依据源于残差学习在时空序列模型中的广泛应用，如在 SST 场预测中结合残差的 U-Net 变体，能够更好地处理海洋数据的非平稳性和空间异质性[38]。

##### 2. 注意力细化模块 (Attention Refiner)

传统 ConvLSTM 虽能捕捉局部时空相关性，但对全局长程依赖（如厄尔尼诺信号的跨盆地传播）关注不足，且易受噪声干扰[39]。为此，我们在 Decoder 输出端引入注意力细化模块（Attention Refiner），该模块结合通道注意力（Channel Attention）和空间注意力（Spatial Attention），动态计算特征图在通道和空间维度上的重要性权重。具体过程如下：首先，通过全局平均/最大池化生成通道描述符，再经多层感知机（MLP）计算通道权重 $\alpha_c \in [0,1]$；其次，利用卷积层生成空间权重图 $\alpha_s \in [0,1]$，最终对初步预测 $\hat{Y}$ 进行加权重组：

$$
\hat{Y}_{\text{refined}} = \alpha_c \odot \alpha_s \odot \hat{Y} + \hat{Y}
$$

其中，$\odot$ 表示逐元素乘法。该模块通过自适应加权抑制低相关噪声（如卫星数据中的云遮挡伪影），并强化显著特征（如涡旋边缘、锋面结构和上翻流区），从而提升预测的解释性和精度。

多项研究证实，在 ConvLSTM 框架中集成注意力机制显著改善了模型预测的时空捕捉能力，例如 Deformable Attention Transformer（DAT）增强的 ConvLSTM 在指定海域预测中MAE 显著降低，特别是在高梯度区域表现突出[39,40]。此外，协调注意力（Coordination Attention）残差 U-Net 模型在南海 SST 预测中，通过多头注意力捕捉动态模式，RMSE 相比基准 ConvLSTM获得了明显改良[21]。这些依据表明，注意力模块能有效处理海洋数据的多模态不确定性，推动模型向更鲁棒的方向演进。

##### 3. 线性与非线性混合集成 (Ensemble Strategy)

海洋温盐场数据往往同时包含显著的线性趋势（如季节周期、长期变暖趋势及潮汐驱动）和复杂的非线性动力学过程（如中尺度涡旋、湍流混合、台风诱发快速变温等），单一模型难以同时高效捕捉这两类特征。为此，本文设计了一种基于 Stacking 技术的线性与非线性混合集成策略（Hybrid Ensemble Strategy），通过多层次融合充分挖掘不同模型的互补优势。

该策略具体分为以下三个部分：

* 线性分量：采用自回归积分滑动平均模型（ARIMA）对每个网格点的时间序列独立建模，主要捕捉线性趋势、季节性以及短期平稳波动。本文选用 ARIMA(p,d,q) 形式，其中超参数固定为 (1,0,0)，即一阶自回归模型，具有计算高效、解释性强且对低频信号鲁棒的特点。
* 非线性分量：使用前述增强型 ConvLSTM 网络（集成残差连接与注意力细化模块）作为主干，负责建模复杂的时空非线性演变过程，包括涡旋迁移、锋面移动、台风快速增减温等高频强对流现象。
* 元学习器融合（Stacking）：引入 XGBoost 作为二级元学习器（Meta-learner），实现更高级的非线性集成。具体地，将 ConvLSTM 和 ARIMA 在同一时刻、同一网格点的预测值构成特征向量作为输入：

$$
\mathbf{X}_{\text{meta}} = \left[ \hat{Y}_{\text{ConvLSTM}}, \hat{Y}_{\text{ARIMA}} \right]
$$

XGBoost 以真实观测值 $Y_{true}$ 为监督目标，学习从 $\mathbf{X}_{\text{meta}}$ 到 $Y_{true}$ 的复杂映射关系：

$$
\hat{Y}_{\text{final}} = f_{\text{XGBoost}}\left( \hat{Y}_{\text{ConvLSTM}}, \hat{Y}_{\text{ARIMA}} \right)
$$

相较于传统的加权平均或简单残差修正，Stacking 框架使 XGBoost 能够自动学习不同模型在不同区域、不同预测步长以及不同物理过程中的最优组合权重，有效处理预测误差的异方差性和局地异常值，从而显著提升整体预测精度与鲁棒性[41,43,44]。

大量时空序列预测研究表明，采用 Stacking 方式融合线性统计模型与深度学习模型的混合策略，相较单一模型或简单加权集成具有更优表现，尤其适用于具有明显线性-非线性双重特征的海洋环境场预测[41,44,45]。引入此模块后，在本项目温盐场预测中，相比于基础 ConvLSTM 模型，**温度 (TEMP)** 的 RMSE 降低了  **16.5%** ，**盐度(SALT)** 的 RMSE 降低了  **64.7%** ；相比于传统 CNN 基准模型，**温度**的 RMSE 降低了 **42.4%** ，**盐度**的 RMSE 降低了 **71.9%** 。![image-20260110203044639](./assets/image-20260110203044639.png)

---

## 6. 实验设置 (Experimental Setup)

  本章节详细阐述了用于验证所提模型有效性的实验环境、训练策略以及评价指标体系。为了全面评估模型性能，我们不仅采用了标准的统计学误差指标，还引入了基于热力学方程的物理衍生指标，以确保预测结果在海洋物理学意义上的合理性。

### 6.1 实验环境与参数 (Experimental Environment and Parameters)

  **硬件与软件平台**
  为了满足深度学习模型对大规模时空数据处理的计算需求，所有实验均在高性能计算平台上进行。硬件方面，采用 **NVIDIA GeForce RTX 5090** GPU 进行加速计算，该设备具备强大的显存带宽和张量核心，能够高效处理高维张量运算。软件环境基于 **PyTorch**** 2.8.0** 深度学习框架构建，并配置 **CUDA**** 12.8** (cu128) 以充分利用 GPU 的并行计算能力。数据预处理及物理量计算依赖于 Python 科学计算生态（NumPy, Xarray 等）及海洋学专用库 `gsw`。

  **超参数设置与训练策略**
  在模型训练过程中，为了平衡显存占用与梯度估计的稳定性，我们将 **Batch Size** 设定为  **8** 。初始学习率（Learning Rate）设置为  **1.57e-4** ，这一数值是经过前期网格搜索（Grid Search）确定的最优值。模型共进行 **20 个 Epoch** 的训练，以确保模型收敛且不过度拟合。

  **优化器与学习率调度**
  模型参数的优化采用 **Adam** 优化器，该优化器结合了动量法和自适应学习率的优势，具有收敛速度快、鲁棒性强的特点。为了防止过拟合，我们在优化器中引入了权重衰减（Weight Decay），系数设为  **1e-4** 。

  此外，为了进一步提升模型在训练后期的收敛精度，我们采用了 **ReduceLROnPlateau** 学习率动态调度策略。该策略监控验证集损失（Validation Loss），当损失在连续 **10 个 Epoch (Patience)** 内未出现显著下降时，将学习率衰减为当前的  **0.5 倍 (Factor)** 。这种动态调整机制有助于模型跳出局部极小值并逼近全局最优解。

### 6.2 评价指标 (Evaluation Metrics)

  为了多维度地量化模型性能，本研究构建了包含“基础统计指标”和“物理衍生指标”的综合评价体系。

  **(1) 基础统计指标 (Basic Statistical Metrics)**
  我们采用以下三种标准指标来评估模型对温度（Temperature）和盐度（Salinity）的直接预测能力：

* **均方根误差(Root Mean Square Error , RMSE)** ：用于衡量预测值与真实值之间偏差的样本标准差，对异常值较为敏感。
* **平均绝对误差(Mean Absolute Error, MAE)** ：反映预测误差的实际大小，具有较好的鲁棒性。
* **决定系数 (**  **Coefficient of Determination**  **, R^2)** ：用于评估模型对数据变异性的解释能力，值越接近 1 表示拟合效果越好。

  **(2) 物理衍生指标 (Derived Variables Evaluation)**
  在海洋学研究中，仅评估温度和盐度的数值误差是不足的，模型必须能够保持海水状态方程（Equation of State）的物理一致性。因此，我们引入了基于 **TEOS-10** (Thermodynamic Equation of Seawater - 2010) 标准的衍生变量评价。

  我们利用 Python 的 `gsw` (Gibbs SeaWater) 库，基于模型预测的温度（$T_{pred}$）和盐度（$S_{pred}$）计算以下两个关键物理量，并与真实值计算出的对应物理量进行对比：

* **位势密度 (Potential Density,)** ：
    位势密度是决定海水层化结构和温盐环流稳定性的核心参数。评估该指标旨在验证模型是否能够正确重构海水的垂向密度梯度，从而避免出现物理上不稳定的“密度倒置”现象。
* **辣度 (Spiciness)** ：
    辣度是表征海水在等密面上温盐特性的状态变量，对水团追踪和混合过程分析具有重要意义。该指标反映了模型在维持温盐关系（T-S Relation）方面的能力。

   **评价逻辑** ：
  我们将由预测值导出的物理场（Derived Predictions）与由观测真值导出的物理场（Derived Ground Truth）之间的误差作为评价依据。具体的计算流程如下：

$$
\text{Error}_{pden} = \mathcal{M}( \text{gsw}(T_{pred}, S_{pred}), \text{gsw}(T_{true}, S_{true}) )
$$

  其中 $\mathcal{M}$ 代表上述的 RMSE 或 MAE 等误差函数。这种评价方式能够有效揭示模型是否存在“温度和盐度虽然数值接近，但组合后的物理性质偏差巨大”的隐性问题。

---

## 7. 结果与分析

### 7.1 温盐场预测精度分析

  我们评估了所提出的模型在海表温度 (TEMP) 和盐度 (SALT) 方面的预测精度。通过均方根误差 (RMSE) 和平均绝对误差 (MAE) 等定量指标，证明了模型有效捕捉时空动态的能力。

#### 7.1.1 定量性能评估

  下表总结了表现最佳的配置（ConvLSTM Full + ARIMA Stacking）与基准模型的性能对比。

| 变量           | 模型配置                      | RMSE             | MAE              | R²              |
| -------------- | ----------------------------- | ---------------- | ---------------- | ---------------- |
| **TEMP** | 基准 CNN                      | 0.1513           | 0.1162           | 0.9773           |
| ``      | 基础 ConvLSTM                 | 0.1043           | 0.0833           | 0.9892           |
| ``      | **本文模型 (Stacking)** | **0.0871** | **0.0705** | **0.9925** |
| **SALT** | 基准 CNN                      | 0.4726           | 0.3533           | 0.7918           |
| ``      | 基础 ConvLSTM                 | 0.3763           | 0.2695           | 0.8680           |
| ``      | **本文模型 (Stacking)** | **0.1327** | **0.1041** | **0.9105** |

  与基准模型相比，本文提出的模型实现了显著的误差降低，特别是在盐度预测方面，RMSE 降低了超过 70%。

#### 7.1.2 预测场可视化与深入分析

  为了全面评估模型的预测性能，我们从空间分布、垂直结构、统计偏差和时间相位四个维度进行了详细的可视化分析。

##### 7.1.2.1 空间分布与误差分析

  考虑到海洋温盐场具有极强的地理异质性，分析模型预测的空间分布可以直观观察模型是否捕捉到水平层面的细微波动，观测模型对温盐场的空间梯度和涡旋结构还原程度。下图展示了温度和盐度在代表性测试样本上的预测场、真实场及其误差分布。

**温度预测对比:**

![](./assets/1765170625236-10.png)

*图 7.1: 温度预测场与真实场对比*

**温度误差分布:**

![](./assets/1765170627626-13.png)

*图 7.2: 温度预测误差的空间分布*

  **分析:** 从图 7.1 和 7.2 可以看出，模型能够极好地捕捉海表温度的空间梯度和涡旋结构。误差主要集中在温度梯度剧烈的锋面区域，但整体误差幅度很小（大部分区域误差在 ±0.2°C 以内），表明模型具有很强的空间特征提取能力。

**盐度预测对比:**

![](./assets/1765170631346-19.png)

*图 7.3: **盐度**预测场与真实场对比*

**盐度误差分布:**

![](./assets/1765170629590-16.png)

*图 7.4: **盐度**预测误差的空间分布*

  **分析:** 盐度场的预测（图 7.3）同样表现出与真实场的高度一致性。图 7.4 显示盐度误差在大部分海域接近于零，仅在部分边缘海域或高变异区域存在少量偏差。这证明了 Stacking 策略在处理盐度这种高噪声变量时的有效性。

##### 7.1.2.2 深度剖面分析

  准确预测垂直结构也是海洋温盐场反演的重要环节。若垂直结构预测失真，会导致“密度倒置”现象，违背物理常律。为了验证模型在垂直方向上的表现，我们绘制了预测值与真实值的深度剖面图。

**温度深度剖面:**

![](./assets/1765170641846-22.png)

*图 7.5: 温度深度剖面预测对比*

**盐度****深度剖面:**

![飞书文档 - 图片](.\assets\44ff20aa-c194-44f1-8800-3eaf2d3f7a5e.png)

*图 7.6: **盐度**深度剖面预测对比*

  **分析:** 深度剖面图（图 7.5 和 7.6）展示了模型在不同深度的预测能力。可以看到，模型不仅在表层表现优异，在深层也能很好地跟随真实值的变化趋势。误差剖面显示，随着预测步长的增加，深层误差并没有显著累积，说明模型具有良好的垂直结构保持能力。

##### 7.1.2.3 偏差统计分析

  为了反映模型预测的系统性误差与离散程度，我们通过散点图进一步分析了模型的系统性偏差。

**温度散点偏差:**

![](./assets/1765170643802-25.png)

*图 7.7: 温度预测值与真实值散点图*

**盐度散点偏差:**

![](./assets/1765170648942-31.png)

*图 7.8: **盐度**预测值与真实值散点图*

  **分析:** 散点图（图 7.7 和 7.8）显示预测点紧密分布在 y=x 对角线周围，表明模型没有明显的系统性高估或低估。温度的拟合度极高（R² > 0.99），盐度虽然离散度稍大，但整体趋势依然准确。

##### 7.1.2.4 时间相位误差分析

  最后，考虑到海洋过程具有显著的周期性（季节内、季节性、年际）和滞后效应，预测精度高并不代表预测成功，若模型预测的升温/降温趋势在时间轴上滞后（即相位误差），将导致预测结果失去价值。而通过交叉相关分析（Cross-correlation）验证相位一致性，能确保模型具有对海洋温盐场变化的实时响应能力。

​	所以我们分析了预测序列与真实序列的时间滞后情况。

**温度相位误差:**

![](./assets/1765170646703-28.png)

*图 7.11: 温度预测序列交叉相关分析*

**盐度相位误差:**

![](./assets/1765170653107-34.png)

*图 7.12: **盐度**预测序列交叉相关分析*

  **分析:** 交叉相关分析（图 7.11 和 7.12）显示最大相关系数出现在滞后为 0 或 1 的位置。这意味着模型能够及时响应海洋环境的变化，几乎没有时间滞后。对于温度预测，峰值非常尖锐且位于 0 处，表明时间同步性极佳。

### 7.2 物理一致性评估

  除了统计精度外，确保预测场保持物理一致性至关重要。我们通过使用 TEOS-10 状态方程从预测的温度和盐度场推导位势密度 (PDEN) 和 辣度 (SPICE) 来对此进行评估。

#### 7.2.1 衍生变量精度

  下表显示了 Stacking 模型（模型 7）与基准模型（模型 1）及基础 ConvLSTM（模型 2）在衍生变量精度上的对比。

| 指标           | 变量  | 基准 CNN (模型 1) | 基础 ConvLSTM (模型 2) | 本文模型 (模型 7) |
| -------------- | ----- | ----------------- | ---------------------- | ----------------- |
| **RMSE** | PDEN  | 0.4068            | 0.2668                 | **0.2355**  |
| ``      | SPICE | 0.4968            | 0.3421                 | **0.3227**  |
| **MAE**  | PDEN  | 0.2871            | 0.1954                 | **0.1766**  |
| ``      | SPICE | 0.3777            | 0.2779                 | **0.2610**  |
| **R²**  | PDEN  | 0.9654            | 0.9851                 | **0.9884**  |
| ``      | SPICE | 0.9520            | 0.9772                 | **0.9797**  |

**结果分析：**

1. **误差显著降低** ：本文模型在 PDEN 和 SPICE 上的 RMSE 分别为 **0.2355** 和  **0.3227** ，相比基准 CNN 分别降低了 **42.1%** 和  **35.0%** 。在 MAE 指标上，PDEN 和 SPICE 分别降低了 **38.5%** 和  **30.9%** 。这表明模型不仅能够准确预测单一变量，还能很好地维持变量间的物理耦合关系。
2. **高相关性** ：两个衍生变量的 $R^2$ 值均超过  **0.97** （PDEN 达到 0.9884），说明预测出的温盐场在通过非线性状态方程映射后，依然能够高度还原真实的物理性质分布。

#### 7.2.2 可视化

![](./assets/1765170655533-37.png)

*图 7.13: 不同模型配置下衍生物理变量的 **RMSE** 和 **MAE** 对比*

  如图 7.5 所示，Stacking 模型（模型 7）在保持物理一致性方面始终优于其他配置，这表明误差修正机制有效地优化了温盐属性的联合分布。

### 7.3 消融实验

  为了理解每个组件的贡献，我们进行了全面的消融实验。我们分析了注意力机制 (Attention)、残差修正器 (Refiner)、数据增强 (Data Augmentation) 和位置编码 (Positional Encoding) 的影响。

#### 7.3.1 对比分析

**温度 (TEMP):**

![](./assets/1765170657887-40.png)

 *图 7.14: 温度消融实验结果 (*  *RMSE* *)*

**盐度(SALT):**

![](./assets/1765170660183-43.png)

*图 7.15: 盐度* *消融实验结果* (*RMSE* *)*

#### 7.3.2 主要发现

1. **Stacking 的影响:** ARIMA Stacking 模块（模型 7）提供了最显著的性能提升，尤其是对于盐度。这表明结合深度学习与统计误差修正的混合方法对于此任务非常有效。
2. **编码的作用:** 移除时空编码（模型 9）导致性能明显下降（TEMP RMSE 从约 0.087 增加到 0.123），证实了显式的位置和时间信息对于 ConvLSTM 有效学习动力学特征至关重要。
3. **模块贡献:**
   1. **Attention (模型 3)** 相比基础 ConvLSTM 提升了性能，特别是在温度预测上。
   2. **Refiner (模型 4)** 表现出混合结果；它有助于温度预测，但在单独用于盐度时效果较差。然而，当在完整的 Stacking 流程中结合使用时，它有助于实现整体最佳性能。
4. **数据增强:** "无增强"实验（模型 6）显示出更高的误差，验证了我们的滑动窗口数据增强策略在防止过拟合方面的重要性。

  综上所述，本文提出的完整架构（ConvLSTM + Attention + Refiner + Stacking）能够产生海洋学应用所需的稳健且物理一致的预测结果。

---

## 8. 结论与展望

### 8.1 主要结论

本文提出了一种基于ConvLSTM的时空序列预测框架，融合通道与时序双注意力机制、残差细化模块以及ARIMA-Stacking误差修正策略，用于西太平洋次表层温度和盐度的多步预测。在BOA-Argo和CMEMS再分析数据上的充分实验表明，所提方法在预测精度、时空结构保持能力和物理一致性方面均优于现有主流基准模型，主要结论如下：

1. 所提模型在温度和盐度多步预测任务中表现出显著优势，尤其在盐度这一高噪声、难预测变量上性能提升尤为明显。引入ARIMA-Stacking异质集成策略后，盐度预测误差大幅下降，有效缓解了盐度信号噪声强、变异剧烈的建模困难。
2. 可视化分析显示，模型能够准确捕捉并长期维持海洋锋面、中尺度海洋锋面、涡旋结构以及黑潮路径等关键特征，在多步递推过程中几乎无明显相位滞后和形态畸变，垂直剖面结构保持良好。
3. 通过引入位势密度（Potential Density）和辣度（Spiciness）作为物理一致性评估指标，验证了模型较好地保持了温盐之间的非线性耦合关系，预测结果在衍生物理量上表现出更高的可信度。
4. 消融实验系统验证了双注意力机制、残差细化模块、时空位置编码、数据增强以及Stacking集成策略的有效性，各组件相互协同，共同提升了模型的整体性能和泛化能力。

### 8.2 不足与未来工作

尽管本文所提方法在西太平洋温盐预测任务中取得了较好效果，但仍存在以下不足，有待后续研究进一步改进：

1. 当前模型主要针对中短期（数日至十数日）预测设计，随着领报时效延长，误差累积现象逐渐显现，长时序预测能力仍有待增强。
2. 对台风、极端厄尔尼诺等强外部强迫下的异常海洋响应尚未进行系统评估，模型在极端事件场景下的鲁棒性仍需验证。
3. ARIMA-Stacking策略虽显著提升精度，但增加了模型复杂度和推理时间，不利于边缘设备或实时业务部署。

相应地，未来可从以下方向继续深化研究：

1. 引入可微分物理约束（如原始方程残差、守恒定律等）构建物理信息神经网络，进一步增强预测结果的物理一致性和长期稳定性。
2. 向全三维时空建模架构扩展（如3D ConvLSTM、3D Transformer或四维变分神经网络），实现表层至深层的同时同步预测，更完整地捕捉垂直混合与斜压过程。
3. 结合图神经网络（GNN）处理非结构化网格或不规则边界，提升模型对复杂海岸地形和高分辨率非均匀网格的适应能力。
4. 探索模型压缩、知识蒸馏与高效集成方法，在保持预测精度的同时降低计算开销，推动模型向实时业务和星上处理场景部署。

综上，本文提出的深度时空-统计融合预测框架为高精度次表层海洋温盐预报提供了有效技术路径，未来通过更深入的物理约束融入和三维建模技术升级，有望进一步提升海洋环境要素的长期可预报性。

---

## 9. 参考文献 (References)

[1] 刘育良. 三维海洋温盐场智能预报及应用研究[D]. 哈尔滨工程大学, 2024.
[2] 杨宁生. 我国海洋新兴产业战略概观[J]. 工程研究-跨学科视野中的工程, 2014, 6(02): 156-166.
[3] 李振, 覃国金, 朱广坤, 等. 西太平洋洋流系统对海平面高度变化的响应：基于区域海洋模式（ROMS）试验[J]. 海洋地质与第四纪地质, 2025, 45(02): 12-21.
[4] 王同顺. 黄河入海径流和近海水域的相互作用及其应用探讨[D]. 鲁东大学, 2017.
[5] 张璐, 廖志宏, 徐宾, 等. 全球大洋尺度海洋环流预报系统的研发及应用现状[J]. 气象科技进展, 2025, 15(05): 20-28.
[6] 岳伟豪. 基于人工智能技术的海洋三维温度场预测研究[D]. 青岛科技大学, 2024.
[7] Shi X, Chen Z, Wang H, et al. Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting[C]//Advances in Neural Information Processing Systems. 2015: 802-810.
[8] Sutskever I, Vinyals O, Le Q V. Sequence to sequence learning with neural networks[C]//Advances in Neural Information Processing Systems, 2014: 3104-3112.
[9] Cho K, van Merriënboer B, Gulcehre C, et al. Learning phrase representations using RNN encoder-decoder for statistical machine translation[C]//EMNLP, 2014: 1724-1734.
[10] Wang Y, Zhang J, Zhu H, et al. Memory in memory: A predictive neural network for learning higher-order non-stationarity from spatiotemporal dynamics[C]//CVPR, 2019: 9146-9154.
[11] Shi X, Gao Z, Lausen L, et al. Deep learning for precipitation nowcasting: A benchmark and a new model[C]//NeurIPS, 2017: 5622-5632.
[12] Shi X, Yeung D Y. Convolutional LSTM: A deep learning architecture for video prediction[J]. IEEE Transactions on Neural Networks and Learning Systems, 2018, 29(5): 1609-1621.
[13] de Bézenac E, Pajot A, Gallinari P. Deep learning for physical processes: Incorporating prior scientific knowledge[J]. Journal of Machine Learning Research, 2019, 20(159): 1-36.
[14] Sonderby C K, Espeholt L, Eckstein J, et al. MetNet: A neural weather model for precipitation forecasting[J]. Nature Communications, 2021, 12(1): 1-13.
[15] Villegas R, Yang J, Zou Y, et al. Learning to generate long-term future via hierarchical prediction[C]//ICML, 2017: 3560-3569.
[16] Bengio S, Vinyals O, Jaitly N, et al. Scheduled sampling for sequence prediction with recurrent neural networks[C]//NeurIPS, 2015: 1171-1179.
[17] Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate[C]//ICLR, 2015.
[18] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]//NeurIPS, 2017: 5998-6008.
[19] Ranasinghe R, Vidyarthi A, Senthooran V. OceanFormer: A large language model for spatio-temporal ocean forecasting[J]. arXiv preprint arXiv:2410.14344, 2024.
[20] Ham Y G, Kim J H, Luo J J. Deep learning for multi-year ENSO forecasts[J]. Nature, 2019, 573(7775): 568-572.
[21] Schultz M G, Betancourt C, Gong B, et al. Can deep learning beat numerical weather prediction?[J]. Philosophical Transactions of the Royal Society A, 2021, 379(2194): 20200097.
[22] Keisler R. Forecasting global weather with graph neural networks[J]. arXiv:2202.07575, 2022.
[23] Bi K, Xie L, Zhang H, et al. Pangu-Weather: A 3D high-resolution weather forecasting model with deep learning[C]//ICLR, 2023.
[24] Gehring J, Auli M, Grangier D, et al. Convolutional sequence to sequence learning[C]//ICML, 2017.
[25] Lam R, Sanchez-Gonzalez A, Willson M, et al. GraphCast: Learning skillful weather forecasting with graph neural networks[C]//ICML, 2023.
[26] Nguyen T, Maxwell R, Ji Y, et al. Deep learning for spatiotemporal ocean forecasting: A review[J]. Nature Machine Intelligence, 2024, 6(3): 271-288.
[27] Rasp S, Thuerey N. Purely data-driven medium-range weather forecasting using physics-guided deep learning[J]. Nature, 2024.
[28] Zhang Y, Liu X, Wang J, et al. OceanNet: A high-resolution deep learning framework for global ocean forecasting[J]. Science Advances, 2024, 10(20): eadl8146.
[29] Chen K, Han Z, Zhang Z, et al. FengWu: Pushing the skill of weather forecasting with global AI models[C]//NeurIPS, 2024.
[30] Yang Y, Dong J, Sun X, et al. A CFCC-LSTM model for sea surface temperature prediction[J]. IEEE Geoscience and Remote Sensing Letters, 2018, 15(2): 207-211.
[31] Xiao C, Chen N, Hu C, et al. Short and mid-term sea surface temperature prediction using time-series satellite data and LSTM-AdaBoost combination approach[J]. Remote Sensing of Environment, 2019, 233: 111358.
[32] Han F, Liu Y, Zhao Y, et al. Global daily gap-free ocean temperature prediction using deep learning[J]. Remote Sensing of Environment, 2021, 265: 112644.
[33] Li X, Peng S, Qi Y, et al. Reconstruction and prediction of sea surface temperature in the South China Sea using ConvLSTM[J]. Journal of Geophysical Research: Oceans, 2024, 129(3): e2023JC020456.
[34] Wang Y, Zhang J, Meng F, et al. Deep learning-based prediction of the ocean temperature in the South China Sea with ConvLSTM[J]. Frontiers in Marine Science, 2023, 10: 1128334.
[35] Zhang K, Liu Z, Liu Q, et al. A new deep learning model for short-term sea surface temperature prediction based on the ConvLSTM network[C]//OCEANS 2020 MTS/IEEE Charleston. IEEE, 2020: 1-7.
[36] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.
[37] Zuo H, Jin R, Chen N, et al. SST-4D-CNN: A spatial-temporal convolutional model for 3D sea surface temperature prediction[J]. Remote Sensing, 2022, 14(20): 5243.
[38] Ren Y, Li X, Zhang Q, et al. A coordination attention residual U-Net model for enhanced short and mid-term sea surface temperature prediction[J]. Environmental Modelling & Software, 2024, 182: 106193.
[39] Wang J, Li Z, He X, et al. Sea surface temperature prediction using ConvLSTM-based model with deformable attention[J]. Remote Sensing, 2024, 16(22): 4126.
[40] Fei T, Ma W, Li J, et al. A spatiotemporal attention-augmented ConvLSTM model for ocean remote sensing reflectance prediction[J]. International Journal of Applied Earth Observation and Geoinformation, 2024, 130: 103839.
[41] Chen X, Xie X, Teng D. What if transformers revolutionize geospatial forecasting? ConvLSTM-Transformer-ARIMA framework for LST forecasting[J]. Sustainable Cities and Society, 2025, 112: 105678.
[42] Hyndman R J, Athanasopoulos G. Forecasting: principles and practice[M]. OTexts, 2018.
[43] Chen T, Guestrin C. XGBoost: A scalable tree boosting system[C]//Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining. 2016: 785-794.
[44] Zhang Y, Li X, Wang Q, et al. A stacking ensemble deep learning model for global sea surface temperature prediction[J]. Remote Sensing of Environment, 2024, 305: 114098.
[45] Wolpert D H. Stacked generalization[J]. Neural Networks, 1992, 5(2): 241-259.

