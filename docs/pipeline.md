# 项目流程图

> 维护规则:阶段结构/产物契约/回环路径发生变化时,所在阶段的收尾任务必须同步更新本图。
> 任何人看不懂这张图 = 图已过期,必须修。

## 流水线总览(通俗版)

```mermaid
flowchart TD
    subgraph IN["原始输入(只读,不改动)"]
        DATA["标定照片<br/>训练:若干摆拍位置 × 每处多张连拍(张数可不齐)<br/>测试:独立拍摄若干张<br/>(同一台相机、相同拍摄参数)<br/>当前数据:80位置×8张 + 65张"]
        CFG["配置文件<br/>棋盘格规格 + 算法选项"]
    end

    ING["① 照片入库体检<br/>逐张给模糊度/曝光/棋盘可见性打分,<br/>存疑只标记不剔除,原图另存副本供人工审"]
    COR["② 找棋盘角点<br/>每个角点定位到亚像素,并把方法调到当前数据下的最优"]
    NOI["③ 测噪声底<br/>同一位置8张连拍相减,实测噪声水平:<br/>这批数据天生最准能到多少"]
    CAL["④ 求解相机内参<br/>4种相机模型 × 多种优化方法,都试一遍"]
    FSH["⑤ 算理论精度极限<br/>标定前就预测:这批数据最好能做到什么程度,<br/>并检查摆拍位置是否有缺陷"]
    EVA["⑥ 防作弊考试<br/>用没见过的新题考试、换着分组复测、逐一抽查,<br/>用数据挑出最合适的相机模型,防止死记硬背"]
    ANA["⑦ 误差归因<br/>实际误差 vs 理论极限:差在哪个方向、哪一环、<br/>数据潜力用尽没有、还有没有优化空间"]
    REP["⑧ 汇总证据出报告<br/>每一步的结论、图表、数据全部留痕可追溯"]

    DATA --> ING
    CFG --> ING
    ING -->|"照片清单+体检结论"| COR
    COR -->|"全部角点坐标"| NOI
    COR -->|"全部角点坐标"| CAL
    NOI -->|"噪声水平"| FSH
    CAL -->|"初步标定结果"| FSH
    FSH -->|"理论精度极限"| ANA
    CAL -->|"内参+逐点残差"| EVA
    EVA -->|"各实验结果"| ANA
    ANA --> REP

    ANA -. "某几张照片拖后腿 → 剔除后重新标定" .-> CAL
    ANA -. "角点横向/纵向偏差异常 → 换角点方法重提" .-> COR
    ANA -. "实测好于理论极限(不可能) → 噪声或角点有系统偏差,回头查" .-> NOI
    EVA -. "模型死记硬背(过拟合) → 换更简单的模型" .-> CAL
    ANA -. "摆拍位置覆盖不足 → 建议补拍(人工决定)" .-> DATA

    style IN fill:#f5f5f5,stroke:#999
    style REP fill:#e8f5e9,stroke:#4caf50
```

**实线** = 数据流(每一步结果都存成文件,带来源指纹,可独立查看/重跑)
**虚线** = 发现问题后的回头路(只重跑受影响的环节,不用从头再来)

## 阶段契约速查(技术对照)

| 阶段 | 通俗名 | CLI 子命令 | 输入(只读) | 输出产物(`output/`) | 模块 | 需求 |
|---|---|---|---|---|---|---|
| ① | 照片入库体检 | `calib ingest` | input_data/, config/ | manifest.json(每张带全部体检数字与判定来源), suspect_originals/(存疑原图副本), suspect_diff_vs_*.txt(跨轮存疑对比表) | io/ | IO-01~03 |
| ② | 找棋盘角点 | `calib corners` | manifest, 图像, 检测配置 | corners_train/test_×classic/sb.npz(4 份角点), detection_summary.json(检测质量摘要), marked_copies/(全部照片的角点标记副本,classic/sb 两套), mosaic/(总览拼图 + 存疑拼图 + 逐格索引) | corners/ | COR-01~03 |
| ③ | 测噪声底 | `calib noise` | corners.npz, 图像 | noise.json | noise/ | NOI-01~02 |
| ④ | 求解相机内参 | `calib calibrate` | corners.npz, 模型/求解器配置 | baseline_<检测器>_<模型>/calibration.json + residuals.npz(逐帧残差), comparison.json(2×2 对比摘要) | models/, solve/ | MOD-01~02, SOL-01~02 |
| ⑤ | 算理论精度极限 | `calib fisher` | corners.npz, noise.json, 标定结果 | crlb.json | noise/ | NOI-03 |
| ⑥ | 防作弊考试 | `calib evaluate` | corners.npz, 实验配置 | experiments/*.json, 汇总表 | evaluation/ | EVA-01~05 |
| ⑦ | 误差归因 | `calib analyze` | 任意上游产物 | analysis/*.json | analysis/ | ANA-01~04 |
| ⑧ | 汇总证据出报告 | `calib report` / `calib run-all` | 全部产物 | report.html/md, figures/ | visualization/, report/ | REP-01~03, ITER-01~04 |

## 核心设计原则

1. **文件是唯一真相源**:每一步的结果都落盘存证,环节之间只通过文件传递,任何环节都可以单独重跑、单独检查
2. **来源可追溯**:每个结果文件自带"用什么配置、从哪些文件算出来的"指纹,报告里任何数字都能顺藤摸瓜
3. **迭代不从头来**:上游调整后,只重跑受影响的下游环节
4. **回头路有路标**:⑦⑧ 的诊断结论会自动给出"回哪一步、做什么"的建议
