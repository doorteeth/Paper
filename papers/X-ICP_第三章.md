# III 问题建模与预备知识

> 对照原文：Tuna et al., *X-ICP: Localizability-Aware LiDAR Registration for Robust Localization in Extreme Environments*, T-RO 2024, Section III。
>
> 本地查看公式：用 VS Code / Cursor 打开后按 `Ctrl+Shift+V` 预览；或直接用浏览器打开同目录下的 `X-ICP_第三章.html`。

本节概述点云配准过程，并针对 LiDAR 退化环境给出问题表述。向量与矩阵用粗体，矩阵用大写字母。

---

## III-A 点云配准

点云配准定义为：求刚体变换 \(\boldsymbol{T}_{\mathtt{ML}} \in SE(3)\)，使 LiDAR 坐标系（记为 \(\mathtt{L}\)）中含 \(N_p\) 个点的读入点云 \({}_{\mathtt{L}}\boldsymbol{P} \in \mathbb{R}^{3 \times N_p}\)，与地图坐标系（记为 \(\mathtt{M}\)）中含 \(N_q\) 个点的参考点云 \({}_{\mathtt{M}}\boldsymbol{Q} \in \mathbb{R}^{3 \times N_q}\) 最佳对齐。

刚体变换写为

$$
\boldsymbol{T}_{\mathtt{ML}} = \big[ \boldsymbol{R}_{\mathtt{ML}} \ \big|\  {}_{\mathtt{M}}\boldsymbol{t}_{\mathtt{ML}} \big]
$$

其中旋转矩阵 \(\boldsymbol{R} \in SO(3)\)，平移向量 \(\boldsymbol{t} \in \mathbb{R}^{3}\)，且

$$
\boldsymbol{t} = [t_x,\ t_y,\ t_z]^{\top}
$$

对读入点云中每个点 \({}_{\mathtt{L}}\boldsymbol{p} \in \mathbb{R}^{3}\)，通过对应搜索（常用 k-d 树）在参考点云中找最近点 \({}_{\mathtt{M}}\boldsymbol{q} \in \mathbb{R}^{3}\)。数据关联过程定义为

$$
\mathcal{M} = \mathrm{matching}\big( {}_{\mathtt{L}}\boldsymbol{P},\ {}_{\mathtt{M}}\boldsymbol{Q},\ \boldsymbol{T}_{\mathtt{LM},\mathrm{init}} \big)
$$

匹配结果是点对集合

$$
\mathcal{M} = \Big\{ \big( {}_{\mathtt{M}}\boldsymbol{p},\ \{{}_{\mathtt{M}}\boldsymbol{q},\ {}_{\mathtt{M}}\boldsymbol{n}\} \big) \Big\}
$$

其中 \({}_{\mathtt{M}}\boldsymbol{p}\) 与 \({}_{\mathtt{M}}\boldsymbol{q}\) 是匹配点对，\({}_{\mathtt{M}}\boldsymbol{n} \in \mathbb{R}^{3}\) 是点 \(\boldsymbol{q}\) 处的单位表面法向，即

$$
\| {}_{\mathtt{M}}\boldsymbol{n} \| = 1
$$

匹配点数满足 \(N \le N_p\)，后文以此为问题规模。

初值变换 \(\boldsymbol{T}_{\mathtt{LM},\mathrm{init}}\) 作为初始猜测，把扫描点云变到参考坐标系，以改善匹配过程与优化收敛。该初值的精度对最小化收敛至关重要 [11]，但其影响与质量分析不在本文范围内。

点云对齐已有多种误差函数；本文使用 point-to-plane 代价 [7]。带该代价的 ICP 最小化问题为：

**式 (1)**

$$
\min_{\boldsymbol{R},\ \boldsymbol{t}} \sum_{i=1}^{N}
\left\|
\big( (\boldsymbol{R}\boldsymbol{p}_{i} + \boldsymbol{t}) - \boldsymbol{q}_{i} \big) \cdot \boldsymbol{n}_{i}
\right\|_{2}
$$

可用 SVD [59]、LU 分解、Gauss-Newton、Levenberg-Marquardt 等求解器。本文侧重对任意矩阵都存在的直接线性代数求解器，例如 SVD。

按 Pomerleau 等 [60] 的推导，把式 (1) 改写成二次代价。先定义每个对应的 Jacobian 列向量

$$
\boldsymbol{A}
=
\begin{bmatrix}
\boldsymbol{p}_{i} \times \boldsymbol{n}_{i} \\
\boldsymbol{n}_{i}
\end{bmatrix}
$$

于是 Hessian \(\boldsymbol{A}'\) 与线性项 \(\boldsymbol{b}'\) 分别为

$$
\boldsymbol{A}'
=
\sum_{i=1}^{N}
\begin{bmatrix}
\boldsymbol{p}_{i} \times \boldsymbol{n}_{i} \\
\boldsymbol{n}_{i}
\end{bmatrix}
\big[
(\boldsymbol{p}_{i} \times \boldsymbol{n}_{i})^{\top}
\quad
\boldsymbol{n}_{i}^{\top}
\big]
$$

$$
\boldsymbol{b}'
=
\sum_{i=1}^{N}
\begin{bmatrix}
\boldsymbol{p}_{i} \times \boldsymbol{n}_{i} \\
\boldsymbol{n}_{i}
\end{bmatrix}
\boldsymbol{n}_{i}^{\top} (\boldsymbol{q}_{i} - \boldsymbol{p}_{i})
$$

式 (1) 等价于下面的二次代价优化（原文式 (2)）：

**式 (2)**

$$
\min_{\boldsymbol{x} \in \mathbb{R}^{6}}
\ \boldsymbol{x}^{\top} \boldsymbol{A}' \boldsymbol{x}
- 2 \boldsymbol{x}^{\top} \boldsymbol{b}'
+ \mathrm{Const.}
$$

其中优化变量为

$$
\boldsymbol{x}
=
\begin{bmatrix}
\boldsymbol{r} \\
\boldsymbol{t}
\end{bmatrix}
\in \mathbb{R}^{6}
$$

\(\boldsymbol{r} \in \mathfrak{so}(3)\) 是旋转向量（\(SO(3)\) 的李代数），\(\boldsymbol{t} \in \mathbb{R}^{3}\)。\(\boldsymbol{A}' \in \mathbb{R}^{6 \times 6}\) 是该优化问题的 Hessian，\(\boldsymbol{b}' \in \mathbb{R}^{6}\) 编码两片点云之间的约束。Hessian 是优化的二阶矩矩阵，刻画 Jacobian 的局部行为。

式 (2) 可进一步写成最小二乘：

**式 (3)**

$$
\min_{\boldsymbol{x} \in \mathbb{R}^{6}}
\left\|
\boldsymbol{A}' \boldsymbol{x} - \boldsymbol{b}'
\right\|_{2}
$$

当 \(\boldsymbol{A}'\) 为（半）正定时，该最小化容易求解。这个 \(6 \times 6\) 线性方程组的解，就是当前线性化下最优的平移 \(\boldsymbol{t}\) 与旋转 \(\boldsymbol{r}\)。由于非线性，ICP 会重复上述步骤直至收敛。

### 原文式 (2) 的结构对照

原文把 \(\boldsymbol{A}\) 直接写进求和，结构如下：

$$
\boldsymbol{A}
=
\begin{bmatrix}
\boldsymbol{p}_{i} \times \boldsymbol{n}_{i} \\
\boldsymbol{n}_{i}
\end{bmatrix},
\qquad
\boldsymbol{A}^{\top}
=
\big[
(\boldsymbol{p}_{i} \times \boldsymbol{n}_{i})^{\top}
\quad
\boldsymbol{n}_{i}^{\top}
\big]
$$

因此

$$
\boldsymbol{A}' = \sum_{i=1}^{N} \boldsymbol{A}\boldsymbol{A}^{\top},
\qquad
\boldsymbol{b}' = \sum_{i=1}^{N} \boldsymbol{A}\,\boldsymbol{n}_{i}^{\top}(\boldsymbol{q}_{i}-\boldsymbol{p}_{i})
$$

物理含义：

- 平移部分由法向 \(\boldsymbol{n}_{i}\) 约束；
- 旋转部分由力矩项 \(\boldsymbol{p}_{i} \times \boldsymbol{n}_{i}\) 约束。

后文 Loc.-Module 正是分别分析这两块。

---

## III-B 在退化环境中运行

实际应用中，若缺少具有几何信息的结构，上述点云配准会因 LiDAR 退化而失败。配准步骤得到的解 \(\boldsymbol{T}_{\mathtt{ML}}\) 变为欠约束，即 6 自由度变换中有一个或多个维度从点对应中（几乎）不可观测。

因此本文的核心问题是：在存在环境退化时，既求最优变换 \(\boldsymbol{T}_{\mathtt{ML}}\)，又找出那些难以估计的方向。以往点云配准研究往往忽略这一情形；本文给出专门方案，使系统在极端场景中仍能有效工作。
