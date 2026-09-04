# Day 2 — 预测 Hessian（地图球 ≥ 量程）

上一版用 `R_map < R_sensor` 人为造出「看得见、地图没有」。那和常见 SLAM 相反。本版改成：

- 量程 \(R_{\mathrm{sensor}}=15\,\mathrm{m}\)
- 局部地图球 \(R_{\mathrm{map}}=20\,\mathrm{m}\)（**不小于**量程）
- 弹出：只留最近 `WINDOW=4 s` 的关键帧（8 m 车程），不是更小的半径

对应仍是当前扫描 ∩ 局部地图。看不见的墙进不了 \(H_t\)。

```bash
cd experiments/day2_forecast_hessian
python3 forecast_hessian.py
```

## 两种维护（一次跑完）

| policy | 地图里有什么 | 像不像常见 SLAM |
|--------|----------------|-----------------|
| `union` | 窗口内每一帧的**完整扫描**取并集。后面 360° 再扫到门口，窗口里仍有一份拷贝 | 像。关键帧存整帧 |
| `owned` | 点只挂在**第一次看见它的那一帧**上，再观测不刷新。那一帧滑出窗口，点就没了 | 不像默认实现；是 claim 额外需要的假设 |

看终端 summary：`union` 的 `pass` 才代表「改成真实大小关系后 claim 仍成立」。`owned` 过只能说明「不重插入时」成立。

图：`out/forecast_lmin_union.png`、`out/forecast_lmin_owned.png`。

## 三条判定（每个 policy 各算一次）

1. \(\hat H_{t+1s}\) 比当前窗口 \(H_t\) 先塌
2. 扫描里还有门口、窗口地图没有 \(\Rightarrow H_t\) 塌，轴沿 \(+Y\)
3. 按预测 \(V_D\) 把门口留在匹配地图里，同一拍 \(H_t\) 不塌

## 这还不是 Day 3 / Thorne / FAST-LIO
