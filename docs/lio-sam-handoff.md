# 交接：退化感知里程计因子

这份文档给下一个对话用。当前仓库已经做完合成走廊上的公式和消融。下一个环境有 LIO-SAM 容器，任务是把同一种因子写进 LIO-SAM，并在 Newer College 2021 的 Cloister 上跑通、再比较。

不要重做 Day 1–5，除非要核对某张图。不要改 FAST-LIO。不要先下整套数据集。

## 命题

目标会议是 RA-L 或 IROS。命题只有一句：

部分退化时，激光里程计因子沿特征方向写入配准 Hessian 的特征值 \(\lambda\)。硬阈值、\(\mathrm{sigmoid}(\gamma)\times\lambda_{\mathrm{nominal}}\)、以及“前端已经按 \(\lambda\) 融合、因子仍发布健康扫描的名义协方差”都留作消融。

可定位性分数是诊断量，不是最终写进因子的数。检测退化这件事 X-ICP 已经做过。这里的增量是因子里的信息用 \(\lambda\)，使回环或 IMU 还能修正这段。

## 公式

各向同性 IMU 信息 \(\lambda_{\mathrm{imu}}\) 下，沿激光信息矩阵的特征向量：

\[
\gamma=\frac{\lambda}{\lambda_{\mathrm{imu}}+\lambda}.
\]

反解 \(\lambda=\lambda_{\mathrm{imu}}\,\gamma/(1-\gamma)\)。因子里写 \(\lambda\)，也就是配准收敛处 Hessian 的特征值。

四种对照，均值可以相同，变的是该方向的信息：

| 名称 | 该方向写入的激光信息 |
|---|---|
| 固定协方差 | \(\lambda_{\mathrm{nominal}}\)（健康轴，或 LIO-SAM 现在的常数协方差） |
| 硬切换 | \(\gamma\ge 0.3\) 时用 \(\lambda_{\mathrm{nominal}}\)，否则该方向不加激光约束 |
| sigmoid | \(w(\gamma)\,\lambda_{\mathrm{nominal}}\)，\(w=1/(1+\exp(-(\gamma-0.3)/0.05))\) |
| Hessian / IRR | 收敛 Hessian 的 \(\lambda\) |

\(\gamma=0.5\) 时 sigmoid 的权重约 0.98，已经饱和，所以它会和硬切换、固定协方差重合。这是预期，不是实现错误。

合成实验里，IMU 和激光被融成一条边，因为图里没有单独的 IMU 因子。LIO-SAM 的图里已经有 IMU 预积分。激光 between factor 只放激光的信息，IMU 因子保持原样。再把 IMU 融进激光因子会把 IMU 算两次。

均匀退化、两端又被钉死时，整段因子乘同一个权重不改变轨迹，只改变 NEES。序列里必须有一段局部退化，并且还有回环或 IMU，轨迹差异才看得见。

## 合成实验已经说明的事

代码、图和原始数字在 `experiments/day1_hessian_spectrum` 到 `experiments/day5_estimated_normals`。每个 Day 的 `README.md` 是该步的检查记录。脚本退出码 0 表示当时的断言通过。

Day 1 只看 Hessian 谱。走廊里最弱平移轴沿隧道；传感器偏航后，弱轴跟着隧道转，不钉在雷达坐标轴上。端面消失时最小特征值连续下降。

Day 2 在线性高斯链上选定公式。\(\gamma=0.35\)、中间一段退化、末端有回环：窗口之后的平均绝对误差，硬切换 0.27，前端软融合但因子仍紧 0.15，IRR 0.06。IRR 的 NEES 在扫描上保持 0.95–1.07。\(\gamma=0.995\) 时五种写法重合。

Day 3 从走廊点云估计 \(\gamma\)，偏航 35°。完整端面 / 两个端面点 / 纯走廊的 \(\gamma\) 为 0.992 / 0.500 / 0.000，弱轴与隧道方向点积为 1。50 次试验，窗口之后：硬切换 0.253，前端紧因子 0.173，IRR 0.068。IRR 的 NEES 为 0.95。

Day 4 从 IMU 初值做点到平面高斯–牛顿，对应仍是已知平面。无端面时 ICP 停在 IMU 初值上（平均绝对误差都是 0.561）。两个端面点 \(\gamma=0.500\)。50 次试验，窗口之后：硬切换 0.216，前端紧因子 0.157，ICP Hessian 0.075。NEES：79、42、0.98。全程完整端面时三者的沿隧道误差都是 0.113。

Day 5 去掉平面方程。地图和扫描都是噪声点云，法向由半径 0.45 m 内的主成分估计，对应是最近邻。小于最强信息 0.2% 的方向不更新位姿，信息仍写入因子。IMU 信息按 36 点端面的解析信息固定为 45000，使该档 \(\gamma\) 落在 0.5 附近；这个数在试验前固定。40 次试验：无端面 \(\gamma=0.040\)，配准与 IMU 的差为 0；36 点端面 \(\gamma=0.510\)；完整端面 \(\gamma=0.921\)，单步隧道误差从 0.00379 降到 0.00122。名义信息 \(4.46\times 10^6\)，约为 36 点端面真实信息的 95 倍。窗口之后：硬切换 0.00353，前端紧因子 0.00266，收敛 Hessian 0.00171。NEES：20.1、11.0、1.11。全程完整端面时三者约为 0.0026–0.0027。

这四层的方向一致：\(\gamma\) 刚过 0.3 时硬切换不动作，和固定协方差是同一条曲线；把 \(\lambda\) 写进因子后，窗口之后的误差更低，NEES 回到 1 附近；几何健康时几种写法接近。

合成结论停在一条受控走廊上。只有一种隧道、一种噪声、一个事先放在 \(\gamma\approx 0.5\) 的端面。对照是固定协方差、硬阈值、sigmoid 和前端紧因子，没有放进已发表的退化感知因子。地图是同一次表面采样，不是带漂移的累积地图。旋转退化、长时漂移和公开激光序列都还没有测。

## 下一个实验

宿主是 LIO-SAM。只改激光关键帧 between factor 的噪声。扫描匹配的位姿均值、IMU 预积分、回环检测保持原样。第一版四种噪声用同一条均值，这样轨迹差异只来自协方差。

数据只用 Newer College 多相机扩展（Zhang 等，arXiv:2112.08854）里的 **Collection 2 / Cloister**。回廊每一条边沿走廊方向约束弱，转角和院子里更强，轨迹走两圈。这和合成实验里“中间灰色、两端健康、末端有回环”是同一种结构。278 秒，429 米。

不要下 Collection 1（Quad-Easy / Medium / Hard、Stairs），不要下 Collection 3（Maths Institute），不要下 Collection 2 的 Park（1567 秒）。下载入口是申请表：https://ori-drs.github.io/newer-college-dataset/download/ 。序列说明在 https://ori-drs.github.io/newer-college-dataset/multi-cam/ 。同时取出该序列旁边的真值 csv。列是秒、纳秒、\(x,y,z\)、四元数 \(x,y,z,w\)。多相机数据的真值在 **Base** 系，10 Hz。雷达在 Base 上方约 9 cm。用 evo 做 SE(3) 对齐时，这个固定杆臂会被吸收。

传感器是 Ouster OS0-128，10 Hz，1024 列，加雷达内 IMU（ICM-20948，100 Hz）。话题：

| 话题 | 用途 |
|---|---|
| `/os_cloud_node/points` | 点云，LIO-SAM 的 `pointCloudTopic` |
| `/os_cloud_node/imu` | 第一遍就用这只 IMU |
| `/alphasense_driver_ros/imu` | Alphasense 上的 Bosch BMI085，200 Hz，6 轴。先不要用 |

四个相机话题不用。bag 太大时滤掉：

```bash
rosbag filter cloister.bag cloister_lidar_imu.bag \
  "topic == '/os_cloud_node/points' or topic == '/os_cloud_node/imu'"
```

官方 LIO-SAM 要求带姿态输出的 9 轴 IMU。这只 Ouster IMU 和 BMI085 都没有那种姿态。第一遍仍用雷达内 IMU，外参先用单位阵，`useImuHeadingInitialization: false`。地图倒了或左右反了，再按 LIO-SAM 仓库 Issue 94 改 Ouster 的坐标，不要改因子公式。外参稳定之后，如果要换 200 Hz 的 Alphasense IMU，外参用 `ori-drs/halo_description` 里 Kalibr 给出的 `imu_link` 到 `os_imu`：平移 \((-0.05072,-0.02064,-0.04077)\) m，rpy \((3.13503,-0.00927,-0.00483)\)。那是 IMU 到 Ouster IMU，不是直接到点云坐标系，还要再乘 Ouster 自己的 IMU 到雷达。

`params.yaml` 第一遍：

```yaml
pointCloudTopic: "/os_cloud_node/points"
imuTopic: "/os_cloud_node/imu"
sensor: ouster
N_SCAN: 128
Horizon_SCAN: 1024
downsampleRate: 4
lidarMinRange: 1.0
useImuHeadingInitialization: false
useGpsElevation: false
loopClosureEnableFlag: true
extrinsicTrans: [0.0, 0.0, 0.0]
extrinsicRot: [1, 0, 0, 0, 1, 0, 0, 0, 1]
extrinsicRPY: [1, 0, 0, 0, 1, 0, 0, 0, 1]
```

128 线先按 `rosbag play -r 1` 播。`downsampleRate: 4` 只为先跑通。容器名以该环境里 `docker ps` 为准。官方镜像名是 `liosam-kinetic-xenial`。播放时用消息头里的时间戳，不必先开 `/clock`。

跑通的标志：`/lio_sam/mapping/cloud_registered` 上两条回廊连成两圈，转角没有把地图撕开。然后把轨迹和真值 csv 做 SE(3) 对齐。这一步仍是 LIO-SAM 原协方差，还不是创新点实验。

## 因子该改哪里

改激光关键帧之间的 `BetweenFactor` 噪声，位置在 LIO-SAM 的 `mapOptmization.cpp`，关键帧保存并加入因子图的函数里。扫描匹配在 `scan2MapOptimization()`。在收敛的那一次高斯–牛顿上，用点到平面残差的雅可比累加 6×6 信息矩阵，做特征分解。每个特征方向：

\[
\gamma_i=\frac{\lambda_i}{\lambda_i+\lambda_{\mathrm{imu},i}}.
\]

\(\lambda_{\mathrm{imu},i}\) 取该关键帧 IMU 预积分信息在同一方向上的分量，并写进日志。不要为了让 \(\gamma\) 好看再去调这个数。

激光因子在该方向上的信息按上一节的四张表选取。近零特征值不要求逆爆掉：该方向的方差用一个很大的上限，位姿均值保持扫描匹配的结果。特征向量要变到 GTSAM 该 between factor 使用的坐标系，再写成噪声模型。IMU 预积分因子和回环因子不动。

每个关键帧记下时间、\(\gamma\)、最小特征值和对应特征向量。Cloister 的回廊段应出现低于转角和院子的 \(\gamma\)。若整段 \(\gamma\) 都接近 1，说明这条序列的退化太弱，换 Hilti SLAM Challenge 2021 的 Basement，不要把阈值改到能“检出”退化。

## 真实数据上的判据

四种噪声各跑同一条 bag。报告 SE(3) 对齐后的 ATE，以及退化段之后的局部误差。能算出 NEES 时一并报告。

- 回廊段 \(\gamma\) 低于 0.3 时，硬切换会丢掉该方向，它和 Hessian 因子的误差应接近。
- \(\gamma\) 在 0.3 到大约 0.6 时，Hessian 因子的误差应低于硬切换和 sigmoid，NEES 留在 1 附近。硬切换、sigmoid、固定协方差应彼此接近。
- 院子或转角等 \(\gamma\) 接近 1 的片段，四种轨迹误差应接近。
- 只改协方差之后原程序跑不通，先修外参、时间戳和降采样，不要改 \(\gamma\) 的定义。

## 不要做的事

- 不要把四种方法做成四套不同的扫描匹配均值，否则看不出协方差的作用。
- 不要用 KITTI 或 Newer College 的 Park 作为第一条真实序列。几何太完整时，因子权重几乎不改变轨迹。
- 不要用 X-ICP 论文里的 Seemühle、Rümlang、Opfikon。那是 ANYmal 内部日志。
- 合成侧还缺一条 \(\gamma\) 从接近 0 扫到接近 1 的曲线，Day 5 只有 \(\gamma\approx 0.5\) 一档。那是 `experiments/day5_estimated_normals` 上改变端面点数即可做的事，不阻塞 Cloister。真实序列跑通之前不要展开新的合成场景。
