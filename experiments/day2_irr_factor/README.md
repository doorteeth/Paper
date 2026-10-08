# Day 2 — 信息缩减比该写成什么样的因子

Day 1 看的是 Hessian 特征值。这一步把特征值变成可定位性分数，再放进一个能手算后验的线性因子图，用来选定论文里的那一个公式。

```bash
cd experiments/day2_irr_factor
python3 run_irr_factor.py
```

依赖：`numpy`、`matplotlib`。图和数字在 `out/`。

## 分数

各向同性 IMU 先验下，沿激光信息矩阵的特征向量，

\[
\gamma = \frac{\lambda}{\lambda_{\mathrm{imu}}+\lambda}
= 1 - \frac{v^{\top}P^{+}v}{v^{\top}P^{-}v}.
\]

反解就是该方向上激光真正贡献的信息：\(\lambda = \lambda_{\mathrm{imu}}\,\gamma/(1-\gamma)\)。因子用这个 \(\lambda\)，不要用 \(\mathrm{sigmoid}(\gamma)\times\lambda_{\mathrm{nominal}}\)。

## 你要看懂的三件事

1. `out/corridor_gamma.png`  
   端面消失时，隧道轴的 \(\gamma\) 从 0.86 连续降到 0。硬阈值 0.3 一直到端面几乎没了才翻转。sigmoid 乘上“健康轴”的名义信息，在整个灰色地带仍比真实 \(\lambda\) 大一到两个数量级。IRR 反解和真实 \(\lambda\) 重合。

2. `out/factor_sweep.png`  
   整段隧道同一个 \(\gamma\)，60 次试验。\(\gamma=0.35\)（刚过硬阈值）时，隧道轴 ATE：硬切换 1.66，sigmoid 1.65，IRR 1.05。NEES：硬切换 179，sigmoid 131，IRR 在整条扫描上保持 0.95–1.07。健康点 \(\gamma=0.995\)（真实信息和名义信息一致）五条曲线重合，说明实现没有把健康场景改坏。

3. `out/segment_path.png`  
   只有中间一段 \(\gamma=0.35\)，两端健康，末端有回环。ATE：硬切换 0.77，只在前端做对融合再发布紧因子 0.44，IRR 0.34。窗口之后的平均绝对误差：0.27、0.15、0.06。回环修得动退化轴，是因为那段因子是软的。前端把均值融对了但协方差仍按健康扫描发布，修不干净。

## 这一步对论文公式的约束

- 贡献写成：用 \(\gamma\) 反解该方向的信息，写入里程计因子。sigmoid 留作消融，不作为方法。
- 硬切换在 \(\gamma\) 刚超过 0.3 时和固定协方差是同一条曲线。部分退化不是 0/1 能分开的。
- 均匀退化、两端又被钉死时，整段因子乘同一个权重不会改变轨迹，只改变 NEES。实验必须有一段局部退化，再加回环或第二个传感器，否则“写进后端”在 ATE 里看不见。

## 还没有做的

这里的 \(\gamma\) 用的是生成测量的真实 \(\lambda\)，不是从点云协方差估出来的。测量无偏。所以这是公式选择，不是论文里的系统实验。
