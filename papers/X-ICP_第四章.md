# IV 系统概览

> 对照原文：Tuna et al., *X-ICP*, T-RO 2024, Section IV。
>
> 本地查看公式：用浏览器打开同目录 `X-ICP_第四章.html`，或用 VS Code / Cursor 预览本文件。

为在 LiDAR 退化环境中可靠地做点云配准与位姿估计，本文提出 **X-ICP**：一种由可定位性引导的约束点云配准方法。框架总览见图 2。检测与抑制 LiDAR 退化的两个组件分别记为 **Loc.-Module** 与 **Opt.-Module**。

演示中，二者嵌入 ANYbotics 开发的 scan-to-map ICP 系统 [61]，该系统基于开源点云配准库 libpointmatcher 的修改版 [62]。建图流水线约 **5 Hz**，使用常见的 point-to-plane ICP 代价。本文贡献适用于任意基于迭代优化的 point-to-plane 配准框架，与具体 scan-to-map 实现无关。

如 III-A 所述，scan-to-map 的鲁棒性与精度依赖初值质量，因此 X-ICP 配上鲁棒状态估计器时，效果优于纯航位推算。

**图 2：** 所提可定位性感知点云配准框架。位姿先验用于变换并去畸变输入点云，再与已有点云地图一起送入迭代 ICP 优化环。环内无漂移位姿由可定位性检测（第 V 节）与感知优化（第 VI 节）计算。

---

## IV-A 可定位性检测模块概览

Loc.-Module 的目标是近似 ICP 优化 Hessian \(\boldsymbol{A}'\) 的**零空间**。为此引入覆盖全部 6 个自由度的可定位性向量 \(\boldsymbol{\eta}\)。该向量标出：哪些特征向量应视为落在 Hessian 的零空间里。

分析直接在特征空间中进行（用特征向量），因此环境的退化方向不必与机器人坐标系或地图坐标系对齐。

可定位性向量定义为：

**式 (4)**

$$
\boldsymbol{\eta} = \{\boldsymbol{\eta}_{t},\ \boldsymbol{\eta}_{r}\} \in \mathbb{R}^{6}
$$

$$
\boldsymbol{\eta}
=
\begin{bmatrix}
\eta_{{}_{\mathtt{L}}\boldsymbol{v}_{t_1}} \\
\eta_{{}_{\mathtt{L}}\boldsymbol{v}_{t_2}} \\
\eta_{{}_{\mathtt{L}}\boldsymbol{v}_{t_3}} \\
\eta_{{}_{\mathtt{L}}\boldsymbol{v}_{r_1}} \\
\eta_{{}_{\mathtt{L}}\boldsymbol{v}_{r_2}} \\
\eta_{{}_{\mathtt{L}}\boldsymbol{v}_{r_3}}
\end{bmatrix}
$$

其中

$$
{}_{\mathtt{L}}\boldsymbol{v}_{t_j} = \boldsymbol{V}_{t}(\ldots,\, j),\quad j\in\{1,2,3\}
$$

是 Hessian \(\boldsymbol{A}'\) 中对应平移 \(\boldsymbol{t}\)、并表示在 LiDAR 系下的特征向量；

$$
{}_{\mathtt{L}}\boldsymbol{v}_{r_j} = \boldsymbol{V}_{r}(\ldots,\, j),\quad j\in\{1,2,3\}
$$

只对应旋转 \(\boldsymbol{r}\)。\(\boldsymbol{V}_{t}\)、\(\boldsymbol{V}_{r}\) 的求法见 V-A。

原文式 (4) 旋转第三项印成了 \(\boldsymbol{v}_{t_3}\)，按上下文应为 \(\boldsymbol{v}_{r_3}\)。

\(\boldsymbol{\eta}\) 用类别变量给出每个特征向量的可定位性：

$$
\eta_j \in \{\textit{none},\ \textit{partial},\ \textit{full}\},\quad j\in\{1,\ldots,6\}
$$

分别对应：

| 符号 | 含义 |
|------|------|
| `none` | 不可定位 non-localizable |
| `partial` | 部分可定位 partially-localizable |
| `full` | 可定位 localizable |

各类别后续动作见 V-C。

**直观理解：** 第三章里 \(\boldsymbol{A}'=\sum \boldsymbol{A}_i\boldsymbol{A}_i^{\top}\) 是信息矩阵。某方向几乎没有 \(\boldsymbol{A}_i\) 指向它，该方向就接近 \(\boldsymbol{A}'\) 的零空间，Loc.-Module 就给它打上 `none` 或 `partial`。

---

## IV-B 可定位性感知优化模块概览

Opt.-Module 使用 Loc.-Module 输出的离散类别 \(\boldsymbol{\eta}\)，构造并求解约束优化，以得到问题 (3) 的最优状态 \(\boldsymbol{x}^*\)，细节在第 VI 节。这里用基于拉格朗日乘子的约束优化，在已观测到的可定位性下求尽可能好的解。

三种结果：

| \(\eta_j\) | 对该方向的位姿更新 |
|------------|---------------------|
| `none` | 保持配准初值不变 |
| `partial` | 受控更新 |
| `full` | 不加约束，正常 ICP 更新 |

完整 X-ICP 框架（见图 2）在存在 LiDAR 退化时仍能可靠估计机器人位姿。主要贡献在第 V、VI 节展开。

**图 3：** 可定位性检测模块概览。A — 信息分析：某一特征向量 \(\boldsymbol{v}_j\) 方向上的直方图，区分强贡献与弱贡献，红色区域为待滤除的信息对。B — 滤波：度量直方图中各向量与 \(\boldsymbol{v}_j\) 的对齐程度。C — 分类：用决策树为每个优化特征向量指定可定位性类别。
