# Faceform Wrap 的 Wrapping 算法还原与借鉴评估

> 整理日期：2026-10-10
>
> 范围：Faceform Wrap（原 R3DS Wrap / Russian3D Scanner）经典 **Wrapping** 节点的算法还原；本机安装（Wrap 2025.10.8）二进制符号佐证；wrapping 节点家族概览；对 `parametric_deformation` 项目"快速创建同拓扑捏脸"的借鉴方案。
>
> 证据来源：① 官方 Wrap3 历史文档存档（web.archive.org，docs.russian3dscanner.com → 现 docs.faceform.com）；② 对本机 `Wrap.exe`（35.7 MB）做字符串/符号（Itanium ABI mangled name）分析；③ 公开学术文献。官方从未发表描述该算法的论文，本文标注了哪些是官方文档原话、哪些是二进制证据、哪些是合理推断。

---

## 0. 结论先行

1. **Wrapping 节点 = "控制点变形图（embedded deformation graph）+ 由粗到细的多级非刚性 ICP"**。官方文档原话：Wrapping 采用 coarse-to-fine 方式，basemesh 被一组控制节点（control nodes）驱动变形；每一级在 basemesh 上按一定密度采样控制节点，"The algorithm tries to find such position of control nodes so that the basemesh fits the target geometry as close as possible"（寻找使 basemesh 尽量贴合目标的控制节点位置），求出解后加密采样进入下一级，如此迭代。
2. 二进制证据表明其优化内核是：**数据项 = point-to-plane 为主、point-to-point 为辅的 ICP 残差**（`FittingPointToPointPointToPlaneFunctor`、`ICPRoutine::FindClosestPointsFunctor`、`FindClosestPointsFunctorWithFilteringHeuristics`），**正则项 = 变形图平滑/ARAP 类残差**（`ARAPWithMirrorResidual`、`SmoothRotationResidual`、`VertexAnchorResidual`），**求解器 = 高斯牛顿/LM + CHOLMOD 稀疏求解**（`WNewton`、"Count iterations of GN"、sparseLM-1.3-improved + `splm_Axb_CHOLMOD`），**最近点查询 = Embree BVH**，**残差导数 = 自研自动微分**（`Block/BlockTypeAutoDiff`、内部库代号 "NumGrind"）。节点界面上可调的 stiffness/smoothness、control points weight、Dp 等参数均有 Initial/Final 两档，在迭代中插值——即教科书式的**正则权重退火（coarse-to-fine annealing）**。
3. 该算法完全可借鉴且**用 NumPy/SciPy 就能落地**：变形图把变量从数万顶点压到数百~数千控制节点，稀疏 GN 每级几十次迭代即可收敛；对应关系用 `scipy.spatial.cKDTree` + 法线夹角阈值 + 顶点掩码过滤即可达到可用质量。这正是"把任意拓扑的脸拟合到固定拓扑 basemesh、差值即 morph"的通用方案，直接服务于本项目模板包的 `bs/linear` 数据生产与 `detach_shape_keys` 物化。
4. 需要注意的域差问题：Wrapping 的默认工作对象是"真人扫描 → 真人 basemesh"。VRoid 动漫脸互相包裹可行；动漫脸 ↔ 真人扫描直接包裹效果差，需要地标/语义点引导或 3DMM（GNM）初始化（与项目已有的 GNM 拟合研究可以拼成一条管线）。

---

## 1. Wrapping 节点官方算法描述（存档原文）

来源：Wrap3 文档 "Wrapping" 节点页（russian3dscanner.com，经 Wayback Machine 存档；现 Faceform 文档 docs.faceform.com 同构）。官方 "Wrapping Under the Hood" 一节的要点（大意转述 + 关键句原文引用）：

- Wrapping 是**迭代过程**，采用 **coarse-to-fine** 策略；
- basemesh 被**一组控制节点（control nodes，界面中显示为白点）**驱动变形；
- 流程分若干步（steps）。每一步在 basemesh 上按特定密度采样一组控制节点，**密度逐级增大**；
- "The algorithm tries to find such position of control nodes so that the basemesh fits the target geometry as close as possible."（算法寻找使 basemesh 尽可能贴合目标几何的控制节点位置）
- 求得当前级的解后进入下一级：加密采样控制节点、重复拟合；
- 用户可放置**控制点对（control points）**约束对应关系；**floating geometry**（头发、眼等不参与变形、最后贴回的部件）与 **fixed/free polygons**（不变形/可滑动区域）参与计算。

节点参数（Wrap3/Faceform 各版本名称略有出入，按本机字符串与旧文档综合）：

| 参数 | 作用 | 对应数学量 |
|---|---|---|
| Subdivisions（fitting steps） | 粗到细分级数 | 多分辨率层级数 |
| Sampling Initial / Final | 首级/末级控制节点采样密度 | 变形图节点间距 |
| Smoothness（或 Stiffness）Initial / Final | 首级/末级刚度 | 平滑正则权重 α 的退火区间 |
| Control Points Weight Initial / Final | 控制点约束权重（首/末） | 地标数据项权重 |
| ICP Iterations / Optimization Iterations | 每级 ICP 外迭代与优化内迭代 | GN 外层 / 内层迭代数 |
| Dp Initial / Final | 优化步长（置信域） | 步长/置信半径退火 |
| Normal Threshold / icpFittingNormalThreshold | 最近点对应的最大法线偏差 | 对应关系鲁棒过滤 |
| Max Relative Slope | 对应点的最大相对坡度 | 拒绝陡坡处的坏对应 |
| Inner Gap / Outer Gap | 包裹后沿法线向内/外抬升的距离 | 沿目标法线的偏移项 |
| Vertex Mask | 参与拟合的顶点掩码 | 掩码数据项 |
| Use Free Polygons / Visualize Points | 自由多边形、可视化 | — |

所有 Initial/Final 参数在迭代进程中线性（或按调度）插值——即"先粗后细、先软后硬"的退火。

---

## 2. 本机二进制证据（Wrap 2025.10.8）

对 `Wrap.exe` 提取可打印字符串与 C++ mangled 符号，与 Wrapping 直接相关的关键证据（节选，均为原文）：

数据项 / 对应关系（ICP）：

```text
N5Block17BlockTypeAutoDiffIN8Wrapping38FittingPointToPointPointToPlaneFunctorEEE
N5Block17BlockTypeAutoDiffIN8Wrapping36FittingPointToPointOnTriangleFunctorEEE
N10ICPRoutine24FindClosestPointsFunctorE
N8Wrapping47FindClosestPointsFunctorWithFilteringHeuristicsE
N5Block17BlockTypeAutoDiffIN19EmbeddedDeformation26FittingPointToPlaneFunctorEEE
icpFittingNormalThreshold
ICP Fitting Weight Initial / ICP Fitting Weight Final
ICP Fitting Trust Threshold Initial / Final
robustifierThresholdICPConstraintInitial / Final
Vertex Mask ICP
ICP Aligment...
```

正则 / 变形模型（ARAP、变形图、对称）：

```text
N5Block13BlockAutoDiffIN21SymmetryEstablishment12Optimization28ARAPMiddleWithMirrorResidualELi6ELi9EEE
N5Block13BlockAutoDiffIN21SymmetryEstablishment12Optimization22ARAPWithMirrorResidualELi6ELi12EEE
...22SmoothRotationResidualELi3ELi12EEE
...20VertexAnchorResidualELi3ELi3EEE
...20PointOnPlaneResidual / 20PointsMirrorResidual
N19OpticalFlowWrapping28EmbeddedDeformation2DeformerE / ...3DeformerE
N23DeformationGraphRoutine12NodesCreatorE
N21EmbeddedGraphDeformer8Abstract7ProblemIfx...
FN26WrappingAcceleratedVersion20EmbeddedGraphStorageEN4Wrap7WObjectE...   # FastWrapping = 加速版，同样是嵌入图存储
ARAP Mesh Modeling. Iteration:
```

求解器 / 加速 / 自动微分：

```text
Count iterations of GN:
N7WNewton7ProblemIfN15CartoonWrapping9Iterative8Jacobian17JtJvProductHolderIfxEEEE
N15CartoonWrapping9Iterative27BlockDiagonalPreconditionerIfEE        # 块对角预条件（JtJ 稀疏迭代解）
CHOLMOD / splm_Axb_CHOLMOD()
../../Kirill/OptimizationFramework01/sparselm-1.3-improved/splm_cholmod.c   # 基于开源 sparseLM 改造 + SuiteSparse CHOLMOD
embree.dll                                                            # BVH 光线/最近点加速
参数名：smoothnessInitial/Final、controlPointsWeightInitial/Final、
      globalControlPointsWeightInitial/Final、semanticPointsWeightInitial/Final、nICPIterations
进度串：Wrapping (subdivision %1 of %2) / Wrapping done. Elapsed:
```

推断（标注为推断）：Wrapping 的变量是**控制节点的刚体变换**（每个节点旋转 R_k + 平移 t_k，共 6~12 个标量——残差签名 `ELi6ELi12`/`ELi6ELi9` 与 6/12 维残差块吻合），顶点位置由控制节点插值（embedded deformation，Sumner & Popović 2004 式）；每级内层做"找对应 → 解稀疏正规方程"的 ICP 外迭代；ARAP 残差族用于对称建立与刚度正则。

---

## 3. 节点家族概览（同一套内核的不同封装）

从符号看，产品线是"一个优化框架 + 多种数据项/变形模型"的封装：

| 节点 | 数据来源 | 变形模型 | 佐证符号 |
|---|---|---|---|
| Wrapping | 手动控制点 + 最近点 | 变形图（粗到细） | `FittingPointToPointPointToPlaneFunctor` |
| FastWrapping | 同上（加速版） | 嵌入图（预建存储） | `WrappingAcceleratedVersion::EmbeddedGraphStorage` |
| FacialWrapping | 自动检测（人脸地标/语义点） | 变形图 | `semanticPointsWeight*`、`GLFacialDetection` |
| BlendWrapping / FitFaceToScreenPoints | 屏幕点/正面照 | **blendshape 空间拟合** | `LocalBlendshapeSpace`、`BlendWrappingByScreenPoints` |
| OpticalFlowWrapping(2) | 一张或多张照片 | 嵌入变形图 + 深度匹配特征 | `EmbeddedDeformation2/3Deformer`、`DeepMatchingFeatureMatcherEstimator` |
| ImageFacialWrapping（ImageBasedWrapping） | 单张照片 | 局部 blendshape 空间 + 光照/区域估计 | `IlluminationEstimator`、`PatchedSphericalHarmonicsEstimator`、`RegionEstimator` |
| CartoonWrapping | 卡通模型 | 嵌入图 + GN（块对角预条件） | `CartoonWrapping::Iterative::Problem/Preconditioner` |

### 3.1 FacialWrapping 的检测机制（本机证据，2026-10-11 补充）

> **产品档位注意**：官方文档将 FacialWrapping、FacialAnnotation、ImageFacialWrapping、SaveFacialDetection、NeuralFaceRegression、TemporalSmoothing、Script 等标注为 **"Only in Wrap4D"**——Wrap 与 Wrap4D 是两个产品（同一份二进制，`Wrap3FactoryObject` 构造时传入 `ProductInfo + LicenseState`，节点面板按许可过滤），基础版 Wrap 的 Tab 节点搜索里搜不到这些节点。本机为 Wrap 2025.10.8 试用模式（`AppData/Local/Faceform/Wrap/trial.ini`）。基础版可用的是 Wrapping / FastWrapping（手动控制点）；其余节点归属需查官方节点列表（docs.faceform.com/Wrap/Nodes，逐节点有档位标注）。

FacialWrapping = "Wrapping + 自动语义点检测"的封装：不再手动点控制点，节点在目标脸上自动检测一组**语义特征点**（眼、外唇/内唇等），与模板 basemesh 上预定义的同名语义点配对，作为分区加权控制点约束喂进同一套 coarse-to-fine 拟合。二进制与盘上文件证据：

```text
检测生命周期：FacialDetection / Load/Edit/Save Detection（检测可编辑可保存，人工修正后复用）
             FacialDetectionMatching / FacialDetectionTracking
             DetectionToMask / DetectionToScreenPoints / UndistortFacialDetection
语义点权重：  semanticPointsWeight{,Initial,Final} × 分区 {Lips, Eyes, Other}   # 检测点按部位分区加权
             detectionWeight、detectionEyes/OuterLips/InnerLips{Initial,Final}
拓扑校验：    "Facial detection has inner lips but no outer lips given to connect."
神经网络：    ONNX 运行时（"ONNX Profile (*.onnx)" 文件对话框字符串）
             PointDetector、NeuralFaceRegression、NeuralBlendRigToGeom
```

`Gallery/PointDetectorData/` 是 PointDetector 的模板侧数据：Basemesh.obj + 约 38 个表情 basemesh（Smile、JawLeft、EyeClosedLeft、LipsFunnel 等 blendshape 式全集）+ `AlignmentPoints.txt`（8 个命名对齐点，每点 = 顶点索引 + UV 坐标的 JSON）+ `FreePolygons.txt`（自由多边形顶点列表）。即 PointDetector 以神经网络把观测回归到这组模板表达空间上定位语义点（推断：类似 3DMM/landmark 回归），几何拟合本体仍是经典 ICP。`neutralDetectionPath + blendshapeDetectionPaths` 与 `Gallery/Expressions/`（Neutral、Dimple、EyebrowsUpJawOpen 等成对的 .drc/.jpg/.wrap）表明同一套检测还能跨表情序列跟踪，用于自动产出 blendshape 目标。

对本项目的直接可借鉴点：① 语义点"命名 + 模板侧预定义（顶点索引/UV）+ 检测后可人工编辑再保存"的数据形态，与项目 semantic_map 方向一致，格式可直接吸收；② 检测点**分区加权（唇/眼/其他，各有 Initial/Final 退火）**——唇眼是包歪重灾区，独立权重是 Wrap 实测经验的结晶；③ "检测 → 人工修正 → 复用"的工作流比全自动更符合捏脸数据生产的精度要求。

**检测模型是什么（本机证据，2026-10-11）**：全部为**自训练模型，无 Google/MediaPipe 痕迹**（二进制中无 mediapipe/tensorflow/keras 字符串）。检测栈是多层的：

```text
FacialLandmarksDetection::RegressionLocalizerWithFlatLandmarksRegressorOpenCV
  → 两级 ONNX 回归器（粗定位模型 + 地标回归模型，"flat landmarks"），经 OpenCV dnn readNetFromONNX 推理
  → 模型为外部 predictor 文件（"Incorrect ONNX input shape. Please check the predictor file."），
    本机基础版 Wrap 安装中不含任何 .onnx 模型文件（属 Wrap4D 资产或按需加载）
ActiveAppearanceModel2D / LocalPatchedActiveAppearanceModel2D (LPAAM2D)
  → 经典局部分块主动外观模型 + 高斯牛顿在 blendshape 空间拟合（自家训练的统计外观模型）
DeepMatchingWrapper / DeepMatchingFeatureMatcherEstimator、DeepFlowCorrespondencesSearcher
  → Weinzaepfel ICCV 2013 的 DeepMatching/DeepFlow——名字带 deep 但是经典算法，非深度学习
NeuralBlendRigToGeom / NeuralBlendRigLatent{OnnxEncoder} / PredictBlendRigControls /
ImageToBlendRigControls / AudioToFace{Normalizer+Encoder+Diffusion}
  → 自家 ONNX：图像/音频 → BlendRig 控制量的回归与扩散模型（AudioToFace 用了 diffusion，2025 年代技术）
推理运行时：onnxruntime.dll (12MB) + OpenCV dnn 双通道
```

**体积构成（安装 1.1 GB）**：大头与神经网络无关——数学内核 MKL+OpenBLAAS 共 ~217 MB（mkl_core 71 + mkl_avx2 47 + mkl_def 36 + mkl_tbb_thread 21 + libopenblas 42）；核心程序三份拷贝 ~92 MB（Wrap.exe 36 + WrapServer 29 + WrapCmd 27）；Embree 30 MB；ICU 29 MB；视频编解码全家桶 ~55 MB（x265/aom/SVT-AV1/avcodec）；Qt5 ~45 MB；图像/体数据 I/O ~37 MB；onnxruntime 12 MB（仅运行时）。总计 181 个 DLL，Linux 下编译交叉打包（构建路径字符串 `/home/radiatus/...`），连带 GTK/GLib 栈。检测用的神经网络模型本身只有几 MB 量级，且不在基础版安装内——**1.1 GB 是单体桌面应用的全依赖捆绑，不是模型**。

### 3.2 BlendRig 神经模型家族与开源平替（2026-10-11）

`NeuralBlendRigToGeom`、`PredictBlendRigControls`、`ImageToBlendRigControls`（以及 `AudioToFace` 扩散模型）构成"控制量层"：**BlendRig = Faceform basemesh 上的一组控制量（类似 ARKit 52 式参数），三个模型分别做 图像→控制量、时序/音频→控制量、控制量→网格**。全部闭源：公开网络检索这三个名字零结果（本文档的符号分析是唯一的公开信息来源），权重也不在基础版安装内。即使拿到权重，它们绑定 Faceform 自家 basemesh 的 BlendRig 定义，对 VRoid 拓扑无用。开源平替按功能拆：

| 功能 | 开源平替 | 域 | 备注 |
|---|---|---|---|
| 图像→表情控制量 | **MediaPipe Face Landmarker**（468 地标 + **52 ARKit blendshape 系数**，Apache-2.0） | 真人 | 实时、单图/视频皆可；ARKit 52 已是事实标准 |
| 图像→3DMM 参数 | DECA / EMOCA / SMIRK / INFERNO（FLAME 系） | 真人 | 单图→FLAME 形状/表情参数；SMIRK（CVPR 2024）表情最准 |
| 音频→表情序列 | **NVIDIA Audio2Face-3D**（2025-09 开源，HF: nvidia/Audio2Face-3D-v3） | 真人 | 输出 blendshape 曲线；学术开源基线：FaceFormer、CodeTalker（音频→FLAME 顶点） |
| 控制量→网格 | FLAME / BFM / FaceVerse 线性解码器 | 真人 | 本质是 3DMM 本身；**不需要神经解码器**——本项目模板包里 BS 数据 + NumPy 求值就是这一步 |
| （动漫域） | 无成熟开源动漫 3DMM | 动漫 | 实用路线：MediaPipe ARKit52 → VRM 标准表情映射（VRM 预设表情含 ARKit 兼容集）；动漫域训练数据可从 VRM 库批量造 |

对本项目的映射：`ImageToBlendRigControls` ≈ MediaPipe blendshape 输出 + ARKit52→VRM 映射；`NeuralBlendRigToGeom` ≈ 项目自己的模板求值管线（P1，无需神经网络）；`PredictBlendRigControls`/AudioToFace ≈ 项目的口型同步研究（`docs/audio-driven-lipsync-survey.md`）。

**"MediaPipe 很老了吗"（2026-10-11 补充）**：Face Mesh 架构确实源于 2019，blendshape 输出头是 2023 年加的，但 Google AI Edge 仍在维护，实时 blendshape 输出场景下开源界尚无替代者（OpenSeeFace 更老，2019–2022；Apple Vision/ARKit 是 iOS 私有）。学术侧更新的代际是**输出 FLAME 参数**的精度路线：SMIRK（CVPR 2024，极端/不对称表情）→ FIELDS（2025，对 FLAME 表情参数直接监督，SMIRK 的直接后继）、OFER（CVPR 2025，遮挡鲁棒）、SPARK（实时自监督个性化）；总览列表见 FLAME-Universe 仓库。但对动漫域，"更新"不改变核心瓶颈——所有这些方法都训在真人+FLAME 上，落到 VRoid 仍要解决域差映射；而 ARKit52 是与 VRM 表情集最好对接的参数化，这是选 MediaPipe 而非学术 SOTA 的真正理由。

### 3.3 MetaHuman 的"模型"与竞品格局（2026-10-11）

**MetaHuman 不是一篇论文式的统计 3DMM**，其技术来自 Epic 收购的两家公司：3Lateral（塞尔维亚，4D 扫描装配/Rigging）与 Cubic Motion（英国，面部捕捉求解）。架构是"**数据驱动的装配体**"而非"身份系数模型"：

- **DNA 文件格式**：完整描述一个角色的装配——拓扑、关节层级、blendshape 增量、以及 RigLogic（控制量 → 关节变换 + corrective blendshape 激活值的映射规则），全部是数据而非硬编码；
- **RigLogic 运行时**：读 DNA，输入控制曲线（FACS 式，含 ARKit 兼容子集，数百个 corrective blendshape），求值出表情——角色无关的统一运行时驱动任意 MetaHuman；
- **MetaHuman Creator**：云端参数化界面，本质是对一组**预装配的模板脸**做形变融合（没有公开的 β 身份系数）；
- **MetaHuman Animator**：iPhone 深度/单目视频 → 控制曲线的求解器，用自家 4D 扫描数据训练（闭源模型）。
- **开放程度**：DNA 格式与 [MetaHuman-DNA-Calibration](https://github.com/EpicGames/MetaHuman-DNA-Calibration)（C++/Python API + Maya 的 DNA Viewer + Ada/Taro 样例 DNA）开源；2025-06 MetaHuman 1.0 出 Early Access，**许可改为任意引擎可用、可商用再分发内容**。

竞品按架构分三类：① **形变融合装配体**（同 MetaHuman 思路）：Daz Genesis 8/9（morph 市场）、Reallusion CC5+Headshot（照片→morph，集成 Audio2Face）；② **统计 3DMM 商品化**：FaceGen（照片→统计脸，2000 年代老牌）、Meshcapade ME（SMPL/FLAME 学术谱系的商业体）、Avatar SDK MetaPerson、（Ready Player Me 据报道 2025-2026 年间停运，来自竞品页面说法待核实）；③ **开源学术 3DMM**：FLAME（研究标准，Meshcapade 分发许可）、BFM、FaceVerse/FaceScape（清华，中文脸，开放）。动漫域没有统计 3DMM 商品：VRoid Studio 的滑杆体系就是"装配体"路线，与本项目模板包设计同构。

对本项目的三层启示：① MetaHuman 的 DNA+RigLogic 与本项目模板包格式（manifest+npz+driver.py）是同构的——"参数是具体的 morph/控制曲线，不是 β 系数"，验证了装配体路线的可扩展性；② DNA 开源+许可放宽后，MetaHuman 头部可作为**真人域模板包候选**（固定拓扑、现成装配、可读 DNA 提取 morph 数据）；③ MetaHuman 的大量模板脸本身就是 industry-grade wrap 产物——与本文 §5 的 wrap 管线互为印证。

回到同源框架的另外两个参考点：

- **BlendWrapping 思路**：不是自由变形，而是"在已有 blendshape 基上解系数"拟合目标——与本项目"参数/BS 驱动"的定位同构；
- **ImageFacialWrapping** 的照片驱动管线（检测 → 特征匹配 → 局部 blendshape 空间 + 球谐光照估计联合优化）与 GNM 系 3DMM 拟合（见 `docs/gnm-paper-algorithm-and-reconstruction-survey.md`、`docs/face-perspective-distortion-fitting-research.md`）是同一生态位，说明"照片 → 参数化脸"工业界最终也收敛到 3DMM/blendshape 拟合 + 正则化优化。

---

## 4. 对应的学术文献（可复现路径）

Wrapping 节点可由以下三篇拼出等价物：

1. **Amberg, Romdhani, Vetter. "Optimal Step Nonrigid ICP Algorithms for Surface Registration", CVPR 2007** —— 模板网格逐级（由粗到细）做非刚性 ICP：能量 = 数据项 + α·平滑项，α 逐级下降；与 Wrapping 的 Subdivisions + Smoothness Initial/Final 结构一一对应。最贴近的实现蓝本。
2. **Sumner, Popović. "Deformation Transfer for Triangle Meshes", SIGGRAPH 2004** —— 变形图（deformation graph / embedded deformation）：以稀疏控制节点刚体变换驱动全网格，正是"控制节点"的数学化（FastWrapping/OpticalFlow 系明确用了 `EmbeddedDeformation`）。
3. **Sorkine, Alexa. "As-Rigid-As-Possible Surface Modeling", SGP 2007** —— ARAP 能量（局部旋转刚性 + 全局位置耦合），符号中的 `ARAP*Residual` 直接对应；也可用其替代变形图平滑项。

> **概念澄清：ARAP 不是有限元/物理仿真。** ARAP 源自 2D 形状操纵（Igarashi et al. 2005，Photoshop Puppet Warp 同源），是几何处理里的静态能量最小化：给定部分控制点/把手的新位置，求"最保持局部刚性"的一个形状。它与弹性力学的血缘在于能量形式——每单元 area·‖∇φ − R‖² 正是共旋（co-rotational）弹性应变能去掉惯性/外力/时间积分后的静力学特例（Chao et al. 2010, "As-Rigid-As-Possible Volumetric Meshes" 严格论证了体网格 ARAP 等价于一种特殊弹性材料的静态解）。与 FEM 弹性仿真的区别：后者已知外力/密度/阻尼求随时间演化的位移场（会振动、有惯性拖尾），ARAP 无质量、无时间、一次求解、结果确定。在 Wrap 中 ARAP 仅作**正则项**（stiffness 权重）：数据项负责贴合目标，ARAP 项负责防拉伸/防翻转。
>
> **与布料模拟的关系（准静态视角）**：从约束优化看两者机制同族——能量/约束 + 迭代；布料"迭代到收敛"的平衡态在数学上就是对应弹性系统的准静态解。本质差异是**动力学 vs 静力学**与**产品语义**：布料的"运动"是产品（惯性、时间步、接触响应不可去，每帧是新的近似解），包裹的"形状"是产品（要求确定性、可复现地收敛到能量极小）。工程映射：把布料求解器去掉惯性项、把 XPBD 式逐约束投影换成收敛到极小的牛顿/稀疏解、把"最近点弹簧力"换成每轮重估的 ICP 最小二乘对应（含法线阈值/mask/鲁棒核过滤），就得到 Wrapping 的内核。面网格 ARAP 与布料本构的对应：惩罚拉伸+剪切、旋转自由（与布料折叠自由一致），但**弯曲刚度为零**——包裹里的抗弯由变形图平滑项（控制节点变换的空间一致性）充当。

辅助组件都有现成开源对应：point-to-plane ICP（Chen & Medioni 1992 / Rusinkiewicz 2019 综述）、鲁棒核（Geman-McClure/Huber，对应 `robustifierThreshold*`）、Embree BVH ↔ `scipy.spatial.cKDTree`（VRoid 头规模足够）、sparseLM + CHOLMOD ↔ `scipy.sparse.linalg`（或 `scikit-sparse` 的 CHOLMOD 绑定）。

---

## 5. 对 parametric_deformation 的借鉴方案

### 5.1 定位

目标："快速创建同拓扑捏脸"——把**任意拓扑**的脸模型（扫描、别的 VRM、别的模板）变成**本项目固定拓扑**（如 VRoid 头）上的 morph 数据。Wrapping 证明了一条已工业验证的路线：

> 目标网格（任意拓扑） --wrap--> 模板 basemesh（固定拓扑） --差分--> `ΔV = V_wrapped - V_base` --打包--> 模板包 `bs/linear` 的一个新 BS/morph（滑杆 0→1）。

与现有架构的接口：

- 输出天然落进模板包数据格式（`bs/linear/*.bs.json` + npz delta，见 `docs/parametric-deformation-framework.md` §2.1）；
- `TemplateAdapter.detach_shape_keys()` 已有"把差值物化成 shape key"的机制，wrap 结果可直接走这条路；
- 模板包 `manifest.json` 的 `actions` 可加一个 `wrap_target` 动作（输入 OBJ + 可选控制点 JSON，输出 morph）；
- 项目的 **semantic_map（语义分区）正好充当 Wrapping 的 Vertex Mask / 语义点**：头发、眼睛、口腔内膜剔除出拟合，眉/唇沿线放控制点——这是手装 Wrap 时最花人工的两步，我们有现成语义数据。

### 5.2 推荐管线（NumPy/SciPy 可实现，P1"内部计算自持"兼容）

```text
0) 预处理
   - semantic_map 生成模板侧拟合掩码；对目标侧剔除漂浮部件（连通域/掩码）
   - 刚体粗对齐：手动/自动地标 Procrustes，或质心+PCA+ICP（rigid ICP）

1) 建变形图（一次性，按模板缓存）
   - basemesh 顶体素降采样取 k 个控制节点（首级 ~100 → 末级 ~2000）
   - 每顶点保留最近 4 节点权重 w_ik（归一化）
   - v_i = base_i + Σ_k w_ik ( R_k (base_i - c_k) + t_k )

2) 分级 L = 0..L_max（对应 Subdivisions / Sampling Initial→Final）
   a. 对应：cKDTree(目标顶点) 查询 v_i 最近点 y_i
      过滤：法线夹角 > θ_normal 剔除；|n_i·(y_i - v_i)|/‖y_i - v_i‖ > slope 剔除；
            掩码外剔除；鲁棒核（Huber/GM）降权 —— 对应 FindClosestPointsFunctorWithFilteringHeuristics
   b. 能量：Σ_i ρ( λ_plane ((v_i - y_i)·n_y)^2 + λ_pp ‖v_i - y_i‖^2 )      # point-to-plane 为主
          + α_L Σ_(k,l)∈edge ‖vec(R_k) - vec(R_l)‖² + ‖t_k - t_l‖²          # 平滑/ARAP 正则
          + w_cp Σ_(i,j)∈控制点 ‖v_i - q_j‖²                                # 手动/语义点
          + (可选) 镜像对称残差                                              # ARAPWithMirrorResidual
      α_L: smoothnessInitial → smoothnessFinal 线性退火
   c. 求解：Gauss-Newton/LM，稀疏正规方程 J^T W J Δ = -J^T W r
      （scipy.sparse.linalg.spsolve；变量 ~ 12k 标量量级，毫秒-秒级/迭代）
   d. 采样密度 ×2 进入下一级

3) 输出
   - V_wrapped = 变形图求值（+ 最后一级可选全顶点精修）
   - 沿法线施加 inner/outer gap、Laplacian 轻平滑去高频噪声
   - ΔV 存 npz → bs/linear 条目 → 滑杆/shape key
```

### 5.3 工程注意点

- **域差（最重要）**：动漫脸 ↔ 动漫脸（VRM→VRM）可直接包；动漫 ↔ 真人扫描默认参数会失败，建议：(a) 目标先用 GNM/3DMM 拟合出中性脸再 wrap（项目已有 GNM 研究，天然衔接）；或 (b) 必须人工控制点 + 只拟合掩码区。
- **point-to-plane 优先**：收敛速度与贴合度都显著好于纯 point-to-point；Wrapping 两者都用（`PointToPointPointToPlane`），plane 权重为主。
- **正则选型（NRICP 是框架，分野在正则项）**：Amberg 的"相邻节点变换一致性"（线性、全仿射）每步是精确线性解、参数少、快，但不约束剪切、易出尖锐棱（trimesh docstring 原话 "can generate sharp edges"）；Sumner 四面体变形图变量少、保形好、参数好调，但步数偏多；ARAP 显式估计局部旋转、无剪切漂移、细节朝向保持最好，但非线性、需 local/global 迭代、代码量大。建议 Amberg 起步，出现斜切/锐边/体积流失再换变形图或加 ARAP 正则——Wrap 内部正是 NRICP 骨架 + ARAP 残差补强的组合。
- **对应过滤决定成败**：法线阈值 + 坡度阈值 + 掩码 + gap 缺一不可，否则鼻孔/耳内/唇缝会把网格拽穿——Wrap 把这些做成一级参数不是偶然。
- **对称残差便宜且有效**：半脸拟合时加镜像 ARAP 项（二进制里 `ARAPWithMirrorResidual` 的用法），还能自动建立对称顶点索引供后续捏脸滑杆用。
- **质量自检**：对每个顶点记录最终距离/法线偏差直方图，超阈值区域标红输出——代替 Wrap 里人眼看 "Correspondences" 可视化。
- **性能参考**：VRoid 头 1~5 万顶点、控制节点 ≤2k 时，纯 NumPy 每级几十次 GN 迭代在秒级；Embree/cKDTree 不是瓶颈。无需 GPU 起步。

### 5.4 开源参考实现（2026-10-11 已逐一验证）

**第一梯队：pip 可装、可直接落地本项目**

| 项目 | 内容 | 语言/许可 | 备注 |
|---|---|---|---|
| **trimesh** `trimesh.registration` | **`nricp_amberg`**（Amberg 2007 直实现：顶点变换 + Laplacian 平滑正则 + 粗到细 steps + 地标 + 距离阈值过滤）与 **`nricp_sumner`**（Sumner 2004 的"对应计算部分"= 非刚性配准，四面体嵌入变形图，保形更好、参数更好调） | Python（NumPy/SciPy），MIT | 就是 §5.2 管线的现成版；目标可是 mesh/点云/裸顶点；`steps=(ws,wl,wn,max_iter)` 正对应 Wrap 的 stiffness/控制点/法线权重退火；`return_records` 便于调参；官方示例 `examples/nricp.py`（扫描→模板） |
| **Open3D** | `TriangleMesh.deform_as_rigid_as_possible(constraint_ids, constraint_pos, max_iter, energy_weight)`（Sorkine ARAP 带把手）+ `RaycastingScene.compute_closest_points`（返回最近点/三角形 id/法线，Embree 级） | Python/C++，MIT | 自拼 wrap 循环的两大件：ARAP 求解器 + 高质量对应查询 |
| **libigl Python 绑定**（`pip install libigl`） | `igl.arap_precomputation`/`arap_solve`/`ARAPData` 等完整 ARAP API；另有 `slim`/`scaf`（局部单射映射） | Python/C++，MPL-2.0 | 想自控残差与调度时的求解引擎 |
| **MeshMonk**（jsnyde0/meshmonk） | 刚性/非刚性/金字塔配准的 Python-first 重写（C++20 内核 + 绑定）：`pip install 'meshmonk[io]'`，`meshmonk.pyramid_register(floating, target)`，含 CLI | C++/Python，Apache-2.0 | 原仓库（electronicforce）已移除；MATLAB 版在 KU Leuven MIRC GitLab；论文 White et al. 2019 |

**第二梯队：论文参考实现（读/对照用）**

| 项目 | 内容 | 许可/状态 |
|---|---|---|
| moChen0607/Deformation-Transfer-for-Triangle-Meshes | Sumner 2004 完整复现：非刚性配准（middle surface 变形图）+ 对应点估计 + 变形转移，Python+SciPy | **无 license 标注**，只读参考；dense LU 解，约 2 万顶点 20 min，非性能标杆 |
| charlienash/nricp | Amberg 2007 的 MATLAB 向量化重写，代码短，可当算法说明书读 | 无 license，2017 年后未更新 |
| Rubikplayer/flame-fitting | FLAME 模板→扫描的非刚性拟合完整管线（多尺度 + 地标 + 正则），3DMM 语境的工业参考，与本项目 GNM 研究互通 | MIT（代码）；FLAME 模型数据另有许可 |
| Sumner 原作者页（MIT CSAIL，存档可访问） | 论文页面，历史上提供 MATLAB 代码 | — |
| prashantdomadiya/Guided-Deformation-Transfer | SIGGRAPH 2019 poster 附带的 Blender 插件版变形转移 | 许可未标注 |

**Blender 端工具**：**ShapeKeyWrap**（MykytaPetrenko/ShapeKeyWrap，GPL-3.0，2026-09 仍在维护）——形状键跨拓扑转移，可作快速基线/可视化对照；GPL 代码注意不要混入本插件的宽松许可部分。**pytorch3d** 官方教程 "deform source mesh to target"（平滑+法线+地标损失的可微拟合）供日后 GPU/可微需求。

**推荐路径**：先用 trimesh `nricp_amberg`（快）或 `nricp_sumner`（保形）打通"外部脸 → VRoid basemesh → ΔV → bs/linear"原型；语义掩码（去头发/眼睛）通过给参与顶点加权/剔除实现；不够用时换 Open3D ARAP + RaycastingScene 自拼循环，补 point-to-plane 与镜像对称项；两者都验证后再考虑把求解器换成 libigl/MeshMonk。

---

## 6. 近期研究还在做吗？（2022–2025 走向）

经典框架（非刚性 ICP / 变形迁移）已有约 20 年历史，但它没有停止演进，而是发生了分工：**工业生产线固化在经典优化上，学术研究迁移到"学习"赛道**。

- **研究仍活跃**：CGF 2025 综述 *Non-Rigid 3D Shape Correspondences*（Zhuravlev et al.）专门覆盖 2022–2025 年工作；同期有 NFR（arXiv 2025）、Deep Functional Maps（NeurIPS 2023）、GraphSCNet（CVPR 2023）、多模态自监督匹配（CVPR 2023）、配准式 3D 形状匹配（CVPR 2025）等。
- **学习类方法的主攻方向**：(1) 学习到的特征描述子替代手工几何描述子（深度网络给顶点/embedding，找对应比 ICP 最近点稳得多，尤其大变形、拓扑差异大时）；(2) Functional Maps 谱方法神经化（在低维谱系数上做对应，速度快且对噪声鲁棒）；(3) 隐式/神经表示（NeRF、SDF、Gaussian Splatting 头像的非刚性跟踪）；(4) 端到端可微——把模板拟合整个塞进网络训练循环。
- **经典优化没有被替代的原因**：学习类方法需要与目标域匹配的训练数据（真人扫描数据集大量存在，动漫 VRM 没有可用数据集）；经典非刚性 ICP 无需数据、单对网格即可解、结果可控可解释、正则权重可人工干预——生产工具（Wrap 系列）至今以它为骨干，AI 只用于其外围（FacialWrapping 的地标检测、图像特征匹配用深度网络，几何拟合本体仍是 ICP+变形图）。
- **对本项目的含义**：我们要处理的恰是"动漫域、无训练数据、单对网格、需要人工可控"的场景，正是学习类方法的弱项、经典方法的强项。可借鉴的近期进展是**把学习到的描述子当 ICP 对应关系的"外挂"**（如用预训练形状匹配网络替代/校准 cKDTree 最近点），作为经典管线的可选增强，而非推翻重来。

## 7. 参考资料

- Wrap3 官方文档 Wrapping 节点页（Wayback 存档）：`web.archive.org/web/2018/http://www.russian3dscanner.com/docs/Wrap3/Nodes/Wrapping/Wrapping.html`
- Faceform 现行文档：Wrapping / FacialWrapping / FastWrapping — docs.faceform.com
- Faceform 论坛（normal threshold 等参数讨论）：forum.faceform.com
- Amberg et al., Optimal Step Nonrigid ICP, CVPR 2007
- Sumner & Popović, Deformation Transfer for Triangle Meshes, SIGGRAPH 2004
- Sorkine & Alexa, As-Rigid-As-Possible Surface Modeling, SGP 2007 — igl.ethz.ch/projects/as-rigid-as-possible/
- 本机 `Wrap.exe`（2025.10.8）字符串/符号分析（本文第 2 节）
- trimesh 非刚性配准 API：trimesh.org/trimesh.registration.html（`nricp_amberg`/`nricp_sumner`，源码 trimesh/registration.py）
- MeshMonk（Python-first 重写）：github.com/jsnyde0/meshmonk；MATLAB 版：gitlab.kuleuven.be/mirc/meshmonk
- Sumner & Popović 论文页（存档）：people.csail.mit.edu/sumner/deformationtransfer/
- Sumner 2004 Python 复现：github.com/moChen0607/Deformation-Transfer-for-Triangle-Meshes
- FLAME 非刚性拟合管线：github.com/Rubikplayer/flame-fitting
- Blender 形状键转移：github.com/MykytaPetrenko/ShapeKeyWrap；SIGGRAPH 2019 Guided Deformation Transfer：github.com/prashantdomadiya/Guided-Deformation-Transfer
- Open3D ARAP 变形：open3d.org 文档 TriangleMesh.deform_as_rigid_as_possible；libigl Python 绑定：github.com/libigl/libigl-python-bindings
