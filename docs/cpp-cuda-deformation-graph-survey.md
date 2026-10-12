# 变形图包裹（Wrapping）算法 C++/CUDA 自研选型调研

> 整理日期：2026-10-12
>
> 背景：最终成果为自研 C++/CUDA 包裹算法（对标 Faceform Wrap 的 Wrapping 节点：把任意拓扑网格贴合到固定拓扑模板，差值作 morph），不调用 trimesh 等慢速 Python 函数。本文整理：算法模块分解、各模块开源参考实现（含本机 Wrap.exe 符号佐证的工程判断）、hwdong/deformation_graph 精读、GPU 化路线。
>
> 前置阅读：`docs/wrap-wrapping-algorithm-research.md`（算法还原 + trimesh 基准实测）。

---

## 0. 结论先行

1. **算法栈可分解为 5 个模块**：建图绑定（FPS 采样 + kNN 权重）、最近点查询（BVH）、能量块（数据/平滑/anchor/landmark/对称残差）、稀疏求解（GN + 稀疏线性解）、外循环调度（粗到细退火/掩码/鲁棒核）。每个模块都有成熟开源参考，无需从零发明。
2. **GPU 化的关键判断**：残差评估、Jacobian·v、最近点查询天然数据并行（一次 kernel）；唯一难点是稀疏求解。Wrap 用 CPU CHOLMOD 直接解；GPU 上有两条已验证路线——**块对角预条件 PCG / matrix-free JtJ**（surfelwarp 实测可用，节点 6/12 维块天然适合 GPU）和 **cuDSS 直接解**（变量 1~3 万规模，直接解毫秒级）。
3. **许可面干净**：DynamicFusion 家族（含 CUDA 实现）几乎全 BSD-3；cuBQL/libigl(MPL)/Open3D/MIT 均可用。hwdong 仓库无 license（只读思路）；CGAL 系 GPL 慎碰。
4. **推荐路径**：先用 Eigen+CHOLMOD 写 C++ CPU 版打通并和 trimesh 数值对齐（golden reference），再按"最近点查询 → 残差 kernel → 求解器"顺序 GPU 化。

---

## 1. 算法模块分解（含 Wrap 符号对照）

| 模块 | Wrap 证据（本机符号分析） | CPU 方案 | GPU 方案 | 参考源码 |
|---|---|---|---|---|
| 建图：节点采样 | `DeformationGraphRoutine::NodesCreator`、控制节点逐级加密 | FPS（欧氏或测地）+ 网格/体素采样 | GPU FPS（并行 min-dist，~50 行 CUDA） | hwdong（测地版）、SurfelWarp warp field |
| 绑定：顶点-节点权重 | 节点密度 Initial/Final | k=4 最近节点，w=(1−d/d_max)² 归一化 | kNN（暴力即可：nV×k 距离矩阵） | 同上 |
| 最近点查询 | `embree.dll`、`FindClosestPointsFunctor*` | Embree / nanoflann | **cuBQL**（NVIDIA 官方）或 ToruNiina/lbvh、OptiX | 见 §3 |
| 能量块 | `BlockTypeAutoDiff`（自研 NumGrind AD）、p2p+p2plane、ARAP/对称残差 | 手写解析 Jacobian（残差是仿射的线性组合，完全可解析）或 C++ AD 库 | residual kernel + J·v / Jᵀ·v（matrix-free，atomic 累加） | surfelwarp `warp_solver` |
| 稀疏求解 | `splm_cholmod.c`（sparseLM 改造+CHOLMOD）、`WNewton`（GN）、`BlockDiagonalPreconditioner` | Eigen SparseQR/SimplicialLDLT 或 SuiteSparse CHOLMOD | (a) 块对角预条件 PCG；(b) cuDSS 直接解；(c) AmgX | surfelwarp、cuDSS 文档 |
| 外循环调度 | Subdivisions、Stiffness/控制点权重 Initial→Final、法线/坡度阈值、Vertex Mask | 纯调度逻辑，CPU 写即可 | 不需要 GPU 化 | Wrap3 文档 + 本项目基准脚本 |

规模估算（VRoid 头 5 万顶点）：节点 1~2k → 变量 12k~24k 标量；正规方程规模 24k²，非零元每行几十——**直接解和迭代解都在毫秒级可及**。

---

## 2. hwdong/deformation_graph 精读（368 行，已全文读）

仓库：`hwdong/deformation_graph`（C++，95★，**无 license 文件**，Eigen + 自带 Peyré fast-marching 测地代码）。

**覆盖范围：只有建图，没有求解器。**

- `DeformationGraph` 结构：`GNode_gs`（节点位置）、`node_nodes`（节点邻接+距离）、`v_nodes`（顶点绑定+距离）、存/读文件；
- `gen_DeformationGraph`：fast-marching **测地 FPS** 选种子节点（节点数可指定）→ 对每个种子重跑一次全网格测地 → 以 `distmax × 2.1` 为半径截断 → 取测地最近的 ≤4 个节点作邻接、≤6 个节点作顶点绑定；
- **没有**：能量函数、优化求解、变形求值、权重公式（Sumner 的 w=(1−d/d_max)²）。

评价：

- 价值 = 建图思路参考（尤其测地绑定对薄结构——手指、耳廓——优于欧氏 kNN，避免"隔着空隙绑定"）；代码是研究原型（控制台刷屏、O(k·n) 慢、中文注释），且无 license，**不能抄代码，只能抄思路**。
- 与 Sumner 2007 论文差异：论文绑定用欧氏距离 + (1−d/d_max)² 权重；绝大多数实现走欧氏 kNN。测地绑定是锦上添花，首版不必做。

---

## 3. 候选源码清单

| # | 项目 | 内容 | 语言/GPU | 许可 | 对自研的价值 |
|---|---|---|---|---|---|
| 1 | trimesh `registration.py` | nricp_amberg（顶点变换版）+ nricp_sumner（tet 变形图版）完整实现 | Python | MIT | **数值对照基准（golden reference）**；两个算法的矩阵装配细节可直接翻译成 C++/CUDA kernel |
| 2 | hwdong/deformation_graph | 变形图**建图**（测地 FPS） | C++ | 无 license | 只读思路（§2） |
| 3 | mihaibujanca/dynamicfusion | Newcombe DynamicFusion 完整管线：Kinfu 刚体 → ED 图非刚性（含节点在线插入/删除） | C++ CPU | **BSD-3** (414★) | ED 图工程化最完整的 CPU 参考：图数据结构、能量装配、种子插入逻辑 |
| 4 | **weigao95/surfelwarp** | SurfelWarp（CVPR 2018）非体积动态重建：**GPU warp field** | C++ **CUDA** | **BSD-3** (292★，作者是论文合作者) | **架构首选**：`core/warp_solver` 全套——残差/Jacobian kernel（.cu）、matrix-free JtJ（`ApplyJtJMatrixFreeAtomic.cu`）、块状 GN、自带 `GradientCheck`/`SanityCheck`（对照 Eigen 数值） |
| 5 | dgrzech/sobfu | Sobfu（CVPR 2019）概率非刚性重建：ED 图 + GPU 稀疏求解 | C++ CUDA | **BSD-3** (164★，2026-08 仍维护) | 第二条 CUDA 路线：pre-DBOW/后端求解的另一种组织方式 |
| 6 | swarth100/dynfu | DynamicFusion 教学级实现 | C++ | BSD-3 (43★) | 读物 |
| 7 | libigl `arap.h` / `arap_dof.h` | ARAP 求解器 + "自由度子空间"把手变形（降维加速） | C++ header-only | MPL-2.0 | CPU 求解器参考；arap_dof 的"把变形约束进低维子空间"思路与节点变量化同源 |
| 8 | Open3D `deform_as_rigid_as_possible` | Sorkine ARAP 带把手，Eigen 稀疏实现 | C++/Python | MIT | 短小可读的 ARAP C++ 参考 |
| 9 | **NVIDIA/cuBQL** | 官方 CUDA BVH build + query（最近点/相交） | CUDA | Apache-2.0 (92★) | **GPU 最近点查询首选**（官方维护，支持 wide BVH） |
| 10 | ToruNiina/lbvh | header-only 并行 LBVH | CUDA | MIT-ish (266★) | 备选：代码极简，适合内嵌 |
| 11 | Embree / nanoflann | CPU 最近点/射线查询 | C++ | Apache-2.0 | CPU 阶段用（Wrap 本尊就是 Embree） |
| 12 | MeshMonk（jsnyde0 重写） | 完整非刚性配准库（刚性/非刚性/金字塔） | C++20/Python | Apache-2.0 | 配准外循环与对称正则的参考 |
| 13 | （避雷）CGAL Mesh Deformation、dolphin-li/DynamicFusion | — | — | GPL / 无 license | 不碰 |

---

## 4. GPU 化工程判断

**天然并行（一次 kernel，照搬 surfelwarp 结构）：**

- 最近点查询：模板 nV 顶点 × 目标 BVH（cuBQL/lbvh），每轮重查；法线/坡度/掩码过滤同 kernel 内完成；
- 残差与 Jacobian·v：数据项（p2p+p2plane）、平滑项（边上一致性）、anchor、landmark 都是节点变换的简单函数；matrix-free J·v / Jᵀ·v 用 atomic 累加，不装配完整正规方程；
- 绑定权重：初始化时一次算好（nV×4 索引+权重），存 SoA。

**求解器（唯一难点），三条路线：**

1. **块对角预条件 PCG + matrix-free JtJ**（surfelwarp 路线，= Wrap `BlockDiagonalPreconditioner` 同款）：每个节点的 6/12 维对角块可以直接求逆存显存（1~2k 节点 × 144 元素，微不足道），PCG 迭代全部走 kernel。缺点：写起来最费劲，收敛监控要细；
2. **cuDSS 直接解**：24k 变量的稀疏 LDLT 在 GPU 上毫秒级，代码量最小。缺点：每轮 GN 都要重新装配 JᵀWJ（CSR 装配要 atomic 或两遍扫描）；
3. **AmgX**：代数多重网格预条件，适合更大规模，本项目规模偏小、收益不大。

建议：**先用路线 2 把端到端跑通，性能不够再换路线 1**（数值上两者应一致到迭代容差）。

**精度**：surfelwarp 全程 float 可行（几何归一化到单位尺度）；数值对不齐时用 double 修正步。每一步 GPU 结果与 CPU golden reference（trimesh 或 Eigen 版）对齐——surfelwarp 自带的 `GradientCheck`/`SanityCheck*Direct` 就是这套做法，直接抄。

---

## 5. 推荐自研路线

1. **阶段 1（CPU 原型，1-2 周）**：Eigen + CHOLMOD/SimplicialLDLT，实现 ED 图版 NRICP（建图→绑定→最近点→残差→GN，多级退火）。对照 trimesh `nricp_sumner` 数值（同输入差值 <1e-6 量级）与 TOSCA 基准指标（corr ~1% diag @ 25 地标）。这版同时是永久 golden reference。
2. **阶段 2（GPU 数据面）**：cuBQL/lbvh 最近点 kernel + 残差/J·v kernel，求解暂留 CPU（数据传回）。
3. **阶段 3（GPU 求解）**：cuDSS 直接解 → 块对角 PCG。
4. **阶段 4（业务层）**：多级调度、对称残差、语义掩码、landmark 注入、质量自检输出（对应误差直方图）。

验收指标沿用本仓库 TOSCA 基准（`.test_tmp/nricp_bench/`）：victoria0→victoria2 跨拓扑 + 25 地标，corr_mean ≤ 2.1（1% diag），单对耗时目标 CPU < 1s / GPU < 100ms。
