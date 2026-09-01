# Day 1 — Hessian 谱

不要读论文。在这个目录跑脚本，对照终端输出和 `out/enter_corridor_lmin.png`。

```bash
cd experiments/day1_hessian_spectrum
python3 compute_spectrum.py
```

依赖：`numpy`、`matplotlib`。

## 你要看懂的四件事

1. **房间**：平移三个特征值同一量级 → 可观。
2. **走廊 yaw=0**：最弱平移轴 ≈ `[0, 1, 0]`，沿隧道。
3. **走廊 yaw=45**：最弱轴跟着隧道转，不是死钉在雷达的 x/y 上（X-ICP 为什么在特征空间检测）。
4. **进走廊曲线**：端面消失，\(\lambda_{\min}(H_{tt})\) 塌下去 → 这就是退化。

公式就是 X-ICP 式 (2)：\(a_i=[p\times n;\ n]\)，\(H=\sum a_i a_i^\top\)，平移/旋转块分开做特征分解。

## 今天结束的标准

能用自己的话解释：为什么走廊里 \(v_{t,\min}\) 指向前方，以及为什么机器人转 45° 后这个向量会变、但仍然沿隧道。

不要改 FAST-LIO，不要开始 Day 2。
