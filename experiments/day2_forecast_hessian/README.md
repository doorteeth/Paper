# Day 2 — 预测 Hessian + 有界局部地图

不要读论文。跑脚本，对照终端里的 Verdict 和 `out/forecast_lmin.png`。

```bash
cd experiments/day2_forecast_hessian
python3 forecast_hessian.py
```

依赖：`numpy`、`matplotlib`。

## 场景（写死在脚本里）

- 走廊沿世界 \(+Y\)，**门口/入口墙在 \(y=0\)**（法向 \(+Y\)），没有远处端面。
- 车从 \(y=2.5\) 以 \(2\,\mathrm{m/s}\) 往 \(+Y\) 开，雷达 360°，量程 \(15\,\mathrm{m}\)。
- FIFO 局部地图半径 \(8\,\mathrm{m}\)（**短于量程**）。
- 对应规则：当前扫描里的点，**只有它还在局部地图里**才进入 \(H_t=\sum aa^\top\)。当前扫描看不见的墙，历史点进不了 \(H_t\)。

预测：匀速 \(\hat T_{t+1\mathrm{s}}\)，在该位姿上用 FIFO 再积一次 \(\hat H\)，弱轴 \(V_D\) 用来「保护」\(|n\cdot V_D|\ge\cos 60^\circ\) 且仍在当前扫描里的点（这里就是门口）。

## 你要看到的三件事

1. **时间差：** 某段 \(t\)，\(\lambda_{\min}(H_t)\) FIFO 还大，但 \(\lambda_{\min}(\hat H_{t+1s})\) 已经塌。
2. **看得见但地图没了：** 当前扫描仍有门口点，FIFO 已删门口 \(\Rightarrow\) \(H_t\) 塌；轴沿世界 \(+Y\)。
3. **留下则不塌：** 同一拍按 \(V_D\) 把门口留在地图里，\(\lambda_{\min}(H_t)\) 不塌。阴影区 = 扫描有门口、FIFO 没有。

若 Verdict 里 `pass` 不是 `True`，这条 claim 在本玩具里不成立，先改几何，不要写论文。

## 这还不是

- 不是 Day 3（FIFO vs 共视点数 vs \(V_D\) 三条弹出规则的误差对比）。
- 不是 Thorne ICRA 2025 的当前帧 \(\max\lambda_{\min}\) 子图；本脚本没有容量 \(N\) 下的当前贪心。
- 不是 FAST-LIO。
