# Phase 1: 数据基座 + 最小完整标定流程 - Research

**Researched:** 2026-10-16
**Domain:** 相机标定数据接入 / 棋盘角点检测 / OpenCV 基线标定(Python 3.12 + OpenCV 4.x)
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### 废帧体检与迭代回环
- **D-01:第一版只标记不剔**——体检不自动剔除任何照片,只产出存疑清单(每张照片:各项指标值 + 存疑来源)供人工审阅
- **D-02:判定基准两者结合**——同姿态连拍内相对比较为主(充分利用 8 连拍结构,整批数据整体偏糊也能挑出异常),全局归一阈值兜底单张姿态;两种来源分别记进 manifest,审清单时能看出"跟同组比出来的存疑"还是"踩了全局线"
- **D-03:剔除永远是逻辑剔除**——input_data 只读;标定命令支持人工指定剔除清单,清单走文件(manifest 标记或 config 内清单),不走命令行参数
- **D-04:人机迭代回环(用户总计划)**——程序按当前规则判废出清单 → 人工查看误杀/漏杀 → 人工提出调整方案 → 程序执行重跑 → 再审……循环到人工认可:角点识别方案定版 + 保留图像清单定版,认为已达到当前角点提取的极限。**每轮的废帧清单与角点配置必须留痕且跨轮可比**(这轮 vs 上轮剔了哪些、为什么),否则人工审会变成凭记忆
- **D-05:收手标准 = 人工认可**——Phase 1 迭代到用户看角点叠加图和废帧清单满意为止;"当前方案最优"的数据证明(合成退化实验)留给 Phase 5(COR-02)

#### 产物保真(通用规则,用户明确要求记录)
- **D-06:实验过程尽可能保留更多信息**——存疑照片单独文件夹存**原图副本**(拼图缩略图看不清细节);角点识别后**全部**照片存带角点标记的副本(不只是存疑/抽样);汇总拼图与表格只是索引,不是替代;manifest.json 存全部数字

#### 角点检测
- **D-07:两个检测器都全量跑、各存一份**——经典 findChessboardCorners + cornerSubPix 与新款 findChessboardCornersSB 都跑,角点坐标各存一份 npz;既给迭代回环提供"哪张照片哪个方法更稳"的直接对比,也给 Phase 5 寻优垫底数据
- **D-08:亚像素窗口从实测方格像素边长推导(比例化,不写死像素值)**——粒度(每帧一个值 vs 逐角点局部自适应)由研究阶段用数据定。头号陷阱:窗口超过方格边长会产生**系统性定位偏差且在重投影误差里看不见**(见 research/PITFALLS.md)
- **D-09:同姿态连拍角点排序规范化断言**(无镜像翻转污染)是成功标准,具体校验方法(如连拍内同名角点位置连续性检查)由规划阶段定

#### 基线标定
- **D-10:两类模型都跑并排**——普通针孔+畸变(cv2.calibrateCamera)与鱼眼 KB4(cv2.fisheye.calibrate)都跑,与两份角点组合成 **2×2 = 4 个基线结果并排对比**;一次标定是秒级,成本忽略。立刻知道这批广角数据更像哪类模型,给 Phase 2 模型层优先级提供证据。注意:KB4 对非鱼眼广角镜头可能收敛不稳,若发生属正常发现,记录即可

#### CLI 与配置
- **D-11:八命令全骨架**——体检/角点/噪声/标定/理论极限/考试/归因/报告八个子命令名一次立全(docs/pipeline.md 阶段契约速查表),用不到的空壳提示"后续阶段实现";后续阶段只填实现不动结构,`calib --help` 即全流程通俗地图
- **D-12:参数只从 config 读**——所有算法参数默认值集中 calibration.yaml(棋盘参数在 board.yaml),不做命令行参数覆盖;参数源唯一(文件即唯一真相源),每次运行产物记录所用配置
- **D-13:全部 CLI 帮助与输出用通俗中文**(项目硬性规则,术语首现加白话解释)

### Claude's Discretion
- 体检指标的具体算法(模糊度/曝光的公式选取)——研究阶段选标准做法并保证天然比例化
- manifest.json / corners npz 的内部字段结构
- 产物目录布局(遵循 docs/pipeline.md 契约表,落 output/ 不进 git)
- 带标记副本的图片格式与压缩(可读性优先,磁盘可控)
- 测试选型(本阶段重点:数据模型对张数不齐/单张姿态的韧性,不崩不错)
- 拼图布局、抽样规则

### Deferred Ideas (OUT OF SCOPE)
None — 讨论未超出阶段范围(角点寻优的量化证明本就在 Phase 5 规划内,不是新增)。
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| IO-01 | 按姿态 ID 数据模型加载 train(每姿态目录,重复张数 ≥1、允许不齐)与 test(独立采集),目录契约只读;manifest 记录姿态/重复索引/重复数分布 | 本文 §架构模式「姿态数据模型」:目录名即姿态 ID、文件名排序即时序;解析规则 + 张数不齐/单张姿态的退化处理 + 对应测试设计 |
| IO-02 | 入库预检——模糊度/曝光/棋盘可见性筛查,废帧剔除留痕;预检指标比例化(按棋盘占比等),不依赖绝对像素阈值 | 本文 §架构模式「体检指标」:拉普拉斯方差(棋盘框内)+ 曝光裁剪比例 + 双检测器可见性探针;D-02 的同姿态相对比较为主 + 全局归一兜底,两来源分别记录 |
| IO-03 | 配置解析与校验(board.yaml + calibration.yaml);棋盘参数零硬编码,square_size 与内参解耦(改错只影响距离类指标并断言提醒) | 本文 §代码示例「配置解析」:pyyaml safe_load + 字段校验;square_size 只进物点坐标,不进任何像素级逻辑(窗口推导用实测方格像素边长,不用 square_size) |
| COR-01 | 角点检测与亚像素精化,候选方法/参数集可配置;窗口扫描范围从实测方格像素边长推导(比例化),不写死像素值 | 本文 §架构模式「双检测器」「窗口推导」「排序规范化」:SB 与经典并跑各存一份;winSize = floor(sq_px/2) − 1 且断言 2w+1 < sq_px;D-09 断言算法 |
| (最小标定命令) | 流水线阶段④最小子集:真实 train 数据出基线内参 + 逐帧重投影误差落盘 | 本文 §架构模式「2×2 基线」:calibrateCamera(默认 5 畸变系数)与 fisheye.calibrate(flags=0)× 两份角点;KB4 收敛失败的 try/except 处理(对应 D-10) |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

以下为 CLAUDE.md 中与本阶段直接相关的硬性指令,规划必须逐条满足:

1. **技术栈锁定**:Python 3.12 + opencv-python 4.14.x(**锁 <5.0**)+ numpy 2.5.x + scipy 1.18.x(Phase 1 实际只需 opencv/numpy/pyyaml/typer;scipy 在依赖里一并锁好,Phase 2 用)
2. **文档用语(硬性)**:所有文档、报告、代码注释、CLI 帮助用通俗中文;专业术语首现就地一句白话解释;避免"切片/闭包/runner"类直译行话
3. **数据契约(硬性)**:input_data 全部图像来自同一相机相同参数环境;`input_data/train|test` 只读
4. **目录契约**:`config/` 存 board.yaml 与算法配置;运行产物写 `output/`(不进 git)
5. **模块布局**:src/calib/ 下 io / corners / solve / visualization 等,与流水线阶段一一对应
6. **GSD 工作流**:代码改动走 `/gsd-execute-phase` 等入口,不绕过
7. **不用 autodiff 框架 / 不用 Ceres / 不用 numba**:Phase 1 无自写求解器,天然满足
8. **uv 管理依赖**:首日 `uv lock` + 冒烟测试三件套(STATE.md 已有待办)

---

## Summary

本阶段是把「照片 → 角点 → 基线内参」这条最小闭环在真实数据(80 姿态 × 8 连拍 + 65 张测试,共 705 张 2560×1440 JPEG,451MB)上跑通。技术上有三个关键点,研究都已给出可执行的配方:(1) **数据模型**——目录名即姿态 ID、文件名排序即时间序,张数全部数据驱动,不硬编码 8;(2) **角点双检测器**——经典 findChessboardCorners + cornerSubPix 与新款 findChessboardCornersSB 并跑各存一份,两者角点排序约定不同,必须经过统一的「排序规范化」层(用行/列方向的确定性几何规则)再入盘,并用连拍内同名角点位置连续性做断言;(3) **亚像素窗口比例化**——从每张图实测的方格像素边长(相邻角点间距中位数)推导 winSize = floor(sq_px/2) − 1,并断言 2w+1 < 方格边长,否则会产生在重投影误差里完全看不见的系统性偏差(本项目头号陷阱)。

体检(IO-02)采用「拉普拉斯方差(棋盘框内)+ 曝光裁剪比例 + 双检测器可见性探针」三个标准指标,判定以同姿态连拍内相对比较为主(D-02),全局归一阈值兜底,两个来源分别写进 manifest。基线标定按 D-10 跑 2×2 = 4 个组合(cv2.calibrateCamera 默认 5 畸变系数 ≈ 项目模型 M2;cv2.fisheye.calibrate KB4),每个组合输出内参 + 逐帧重投影误差;KB4 对非鱼眼广角数据可能收敛不稳或报 Ill-conditioned,属预期发现,try/except 记录即可。

环境侧:uv 0.11.32 与 Python 3.12.13 已就位,PyPI 清华镜像可达但代理限速(opencv 轮子约 90MB,实测下载+解包需 10 分钟级)——首日 `uv lock` + 安装要预留这段时间,属于 STATE.md 已记待办。12 个依赖包全部通过 slopcheck 合法性检查,版本经 PyPI 官方接口逐一核实。

**Primary recommendation:** 按「io/dataset + precheck + config → corners/detect + window + canonical → solve/baseline → visualization/overlay + mosaic → cli/main(8 命令骨架)」建模块;所有算法参数进 calibration.yaml;产物 schema 按 ARCHITECTURE.md 长表约定落地 manifest.json 与 corners npz;测试以「合成棋盘图(已知真值角点)」为核心夹具,配真实数据小子集冒烟。

## Architectural Responsibility Map

本项目是单机离线命令行流水线,不涉及浏览器/服务端分层;按流水线阶段划分职责归属:

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| 姿态数据模型 / 目录扫描 / manifest 读写 | io/ | util/lineage(血缘字段) | IO-01 归 io;血缘指纹被所有产物复用 |
| 体检指标(模糊/曝光/可见性) | io/precheck | corners/detect(可见性探针复用检测器) | 体检属入库阶段;棋盘可见性天然要调检测器 |
| 配置解析校验 | io/config | — | board.yaml + calibration.yaml 唯一入口 |
| 角点检测 + 亚像素精化 | corners/detect | — | COR-01 核心;双检测器封装在同一接口后 |
| 方格像素边长估计 + 窗口推导 | corners/window | — | 比例化窗口是独立可测的纯函数 |
| 角点排序规范化 + 连拍断言 | corners/canonical | — | D-09 成功标准的承载者 |
| 基线标定(2×2) | solve/baseline | — | 阶段④最小子集,只调 OpenCV 现成函数 |
| 角点叠加副本 / 存疑原图副本 / 拼图 | visualization/ | io/precheck(存疑名单来源) | D-06 产物保真的落点 |
| CLI 8 命令骨架 + 通俗中文帮助 | cli/ | — | D-11/D-13;4 实现 + 4 占位 |
| 产物血缘(配置哈希/上游哈希/时间戳) | util/lineage | 各阶段调用 | ITER-01 从第一阶段就内建,后补成本高 |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| opencv-python | 4.14.0.94(锁 <5.0) | 角点检测(findChessboardCorners/SB + cornerSubPix)、标定(calibrateCamera / fisheye.calibrate)、图像读写与叠加绘制 | 4.x 最后一个稳定大版本;5.0.0(2026-08)发布仅数周,calib3d 正是大版本行为易变模块 [VERIFIED: PyPI opencv-python/4.14.0.94 JSON + CLAUDE.md 栈表] |
| numpy | 2.5.4 | 全部数组运算、npz 产物、指标计算 | 当前稳定线(2026-10-10 发布);requires_python >=3.12,有 cp312 轮子 [VERIFIED: PyPI numpy/2.5.4 JSON] |
| scipy | 1.18.1 | Phase 1 不直接用;进锁文件,Phase 2 least_squares 用 | 与 numpy 2.5.x 成对验证过的组合 [VERIFIED: PyPI scipy/1.18.1 JSON] |
| typer | ≥0.15 | CLI 8 子命令骨架 | 类型注解直接生成帮助/校验,贴合 D-11「命令即地图」[VERIFIED: PyPI typer 最新版 JSON + slopcheck OK] |
| pyyaml | ≥6.0 | board.yaml / calibration.yaml 读取校验 | 只读配置,safe_load 足够 [VERIFIED: PyPI + slopcheck OK] |
| matplotlib | 3.10.x | 拼图与汇总图(架构决定 visualization 层 matplotlib-only) | 项目既有架构决策 [CITED: CLAUDE.md 栈表] |
| pytest | (最新 8.x) | 单元/集成测试 | 项目开发工具既定 [VERIFIED: slopcheck OK] |
| uv | 0.11.2(本地 0.11.32) | 环境与锁文件 | [VERIFIED: 本机 uv --version] |

### Supporting(Phase 1 不装,Phase 2+ 用)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandas | 2.3.x | 结果表格/CSV | Phase 4 评估实验表格 |
| joblib | ≥1.4 | 实验网格并行 | Phase 4 |
| jinja2 | ≥3.1 | 报告组装 | Phase 6 |
| numdifftools | dev-only | 数值雅可比对照(测试用) | Phase 2 模型测试 |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| findChessboardCornersSB + 经典双跑 | 只用其一 | 双跑是 D-07 锁定决策;成本翻倍但单次全量仍是分钟级,信息增益大 |
| 拉普拉斯方差(自算 3 行) | cv2.estimateChessboardSharpness | OpenCV 自带按角点边缘陡度算清晰度,天然聚焦棋盘;可用性待首日冒烟确认,作为增强项而非依赖 [ASSUMED: 待本地验证] |
| typer | click | 功能等价(typer 封装 click);typer 类型注解写法更省 |

**Installation(首日,Wave 0):**
```bash
uv init --lib .   # 或手写 pyproject.toml;生成 src/ 布局
uv add "opencv-python>=4.14,<5.0" "numpy==2.5.*" "scipy==1.18.*" typer "pyyaml>=6" "matplotlib>=3.10"
uv add --dev pytest ruff
uv lock && uv run python -c "import cv2, numpy, scipy; print(cv2.__version__, numpy.__version__, scipy.__version__)"
```
注意:经代理下载 opencv 轮子(约 90MB)实测需 10 分钟级,`uv lock`/安装预留足时间,失败重试即可续传缓存 [VERIFIED: 本地实测]。

## Package Legitimacy Audit

slopcheck 0.6.1 本地可用;对 12 个包执行了 scan,全部 [OK] [VERIFIED: 本地 slopcheck scan 输出]。版本存在性经 PyPI 官方 JSON 接口逐一核实。

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| numpy | PyPI | 20+ 年 | 极高 | github.com/numpy/numpy | OK | Approved |
| scipy | PyPI | 24+ 年 | 极高 | github.com/scipy/scipy | OK | Approved |
| opencv-python | PyPI | 12+ 年 | 极高 | github.com/opencv/opencv-python | OK | Approved |
| typer | PyPI | 6+ 年 | 高 | github.com/fastapi/typer | OK | Approved |
| pyyaml | PyPI | 20+ 年 | 极高 | github.com/yaml/pyyaml | OK | Approved |
| matplotlib | PyPI | 21+ 年 | 极高 | github.com/matplotlib/matplotlib | OK | Approved |
| pandas | PyPI | 17+ 年 | 极高 | github.com/pandas-dev/pandas | OK | Approved |
| jinja2 | PyPI | 17+ 年 | 极高 | github.com/pallets/jinja | OK | Approved |
| joblib | PyPI | 15+ 年 | 极高 | github.com/joblib/joblib | OK | Approved |
| pytest | PyPI | 24+ 年 | 极高 | github.com/pytest-dev/pytest | OK | Approved |
| numdifftools | PyPI | 13+ 年 | 低(dev-only) | github.com/pbrod/numdifftools | OK | Approved(dev-only;注册元数据较新,仅测试用) |
| ruff | PyPI | 4+ 年 | 极高 | github.com/astral-sh/ruff | OK | Approved |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```mermaid
flowchart LR
    subgraph IN["input_data/(只读)"]
        TR["train/ 80 姿态目录<br/>× 每目录 8 张连拍"]
        TE["test/ 65 张平铺"]
    end
    CFG["config/<br/>board.yaml + calibration.yaml"]
    ING["① calib ingest<br/>io/dataset + io/precheck<br/>姿态数据模型 + 模糊/曝光/可见性体检"]
    MAN["output/ingest/manifest.json<br/>+ 存疑照片原图副本/"]
    COR["② calib corners<br/>corners/detect + window + canonical"]
    CNP["output/corners/<br/>corners_classic.npz + corners_sb.npz<br/>+ 全部照片角点标记副本/ + 拼图"]
    CAL["④(最小)calib calibrate<br/>solve/baseline"]
    OUT["output/calibration/<br/>baseline_{classic|sb}_{pinhole|fisheye}<br/>calibration.json × 4 + residuals.npz × 4"]

    TR --> ING
    TE --> ING
    CFG --> ING
    ING -->|只标记不剔除(D-01)| MAN
    MAN -->|全部照片照常进入(D-01/D-03)| COR
    CFG --> COR
    COR -->|"双检测器各存一份(D-07)"| CNP
    CNP -->|"剔除清单过滤(D-03,config 文件)"| CAL
    CFG --> CAL
    CAL --> OUT

    HUM["人工审阅<br/>角点叠加图 + 废帧清单"]
    OUT --> HUM
    CNP --> HUM
    HUM -.."调整剔除清单/体检阈值/角点配置<br/>(改 config,留痕跨轮可比 D-04)".-> CFG
    HUM -.."重跑受影响环节".-> ING

    style IN fill:#f5f5f5,stroke:#999
    style HUM fill:#fff8e1,stroke:#f9a825
    style OUT fill:#e8f5e9,stroke:#4caf50
```

要点:体检**不改变数据流**(存疑照片照常进角点提取,只是被标记);人工回环只通过改 config 再重跑实现,每轮产物按运行时间戳分目录留痕、可 diff。

### Recommended Project Structure

```
src/calib/
├── __init__.py
├── cli/
│   └── main.py          # typer 应用:8 子命令(ingest/corners 实现;noise/fisher/evaluate/analyze/report/run-all 占位,中文提示"后续阶段实现")
├── io/
│   ├── dataset.py       # 姿态数据模型 + input_data 扫描(IO-01)
│   ├── precheck.py      # 模糊/曝光/棋盘可见性(IO-02)
│   ├── config.py        # board.yaml + calibration.yaml 解析校验(IO-03)
│   └── manifest.py      # manifest.json 写盘 + 血缘
├── corners/
│   ├── detect.py        # 经典/SB 双检测器统一封装(COR-01)
│   ├── window.py        # 方格像素边长估计 + 亚像素窗口推导(D-08)
│   └── canonical.py     # 角点排序规范化 + 连拍一致性断言(D-09)
├── solve/
│   └── baseline.py      # 2×2 基线标定(D-10)
├── visualization/
│   ├── overlay.py       # 全部照片带角点标记副本 + 存疑原图副本(D-06)
│   └── mosaic.py        # 汇总拼图(索引,非替代)
└── util/
    └── lineage.py       # 配置哈希/上游哈希/时间戳(ITER-01 地基)

output/                  # 不进 git
├── ingest/run_<时间戳>/manifest.json + suspect_originals/
├── corners/run_<时间戳>/{corners_classic.npz, corners_sb.npz} + marked_copies/ + mosaic/
└── calibration/run_<时间戳>/baseline_*.json + residuals.npz

config/
├── board.yaml           # 已存在(11×8、25mm)
└── calibration.yaml     # 新建:全部算法参数默认值 + 剔除清单(D-03/D-12)

tests/
├── conftest.py          # 合成棋盘图生成夹具(已知真值角点)
├── test_io_dataset.py   # IO-01 + 张数不齐/单张姿态韧性
├── test_io_precheck.py  # IO-02
├── test_io_config.py    # IO-03
├── test_corners.py      # COR-01 + 窗口推导 + 排序规范化
├── test_solve_baseline.py # 最小标定(合成数据回归)
└── test_smoke_realdata.py # 真实数据小子集冒烟(标 integration)
```

### Pattern 1: 姿态数据模型(目录名即姿态 ID)

**What:** input_data 的解析规则,张数全部数据驱动(REU-01 弹性契约从第一天生效)。
**When to use:** io/dataset.py 唯一入口。

实测数据布局(已核对)[VERIFIED: 本地 ls/stat 核对]:
- `input_data/train/` = 80 个目录,目录名形如 `20260717_172523`(YYYYMMDD_HHMMSS,姿态时间戳)
- 每目录**恰好** 8 个 jpg,文件名形如 `20260717_172523_559573.jpg`(目录名 + 下划线 + 微秒后缀)→ **文件名字典序 = 拍摄时间序**
- `input_data/test/` = 65 张平铺 jpg,同名规则(20260721–22 拍摄)
- 全部 2560×1440 JPEG,共 451MB

解析规则:
- `pose_id` = train 子目录名;`repeat_idx` = 目录内按文件名排序后的序号(0 起);`n_repeats` = 该目录实际张数(数据驱动,**不写死 8**)
- test 平铺图:`split=test`,`pose_id` = 文件名去扩展名(每张独立),`repeat_idx=0`
- 文件名不符合时间戳模式时:警告并回退到自然排序(韧性,不崩)
- 空目录 / 非图片文件:警告跳过,计入 manifest 汇总;单张姿态(退化为单帧)合法存续
- 校验:同目录内文件数 ≥1 即可;分辨率不一致 → 硬报错(单相机数据契约)

### Pattern 2: 体检指标(比例化,D-02 双来源)

**What:** 三个标准指标 + 判定逻辑。全部先算原始值再归一,判定以同姿态内相对比较为主、全局归一线兜底,两个来源分别记录。

| 指标 | 算法 | 比例化方式 |
|------|------|-----------|
| 模糊度 | 拉普拉斯方差(Pech-Pacheco 法):`cv2.Laplacian(gray_roi, cv2.CV_64F).var()`,**只在棋盘检测框内计算**(背景无关化) | 归一值 = 原始值 / sq_px²(边缘锐度能量随方格像素边长平方缩放);同姿态内相对比较天然免归一(8 连拍棋盘大小位置几乎相同) |
| 曝光 | 灰度直方图裁剪比例:高光裁剪 = ≥250 的像素占比;阴影裁剪 = ≤5 的像素占比(百分比天然比例化) | 直接用比例;同姿态内相对比较为主 |
| 棋盘可见性 | 双检测器探针:经典(`CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + FAST_CHECK 预筛`)与 SB 各跑一次,分别记录成败 + 检测框面积占比 + 角点数是否 88 | 成败是布尔;面积占比天然比例化 |

判定逻辑(D-02):
```
每个姿态内(≥2 张时):某指标做稳健 z 分数(中位数绝对差 MAD),|z| > 阈值(默认 3.5)
    → suspect_sources += ["pose_relative:blur" 之类]
全局兜底(含单张姿态):归一值超出分位线(默认整批 1% / 99%,首轮跑完看分布再定)
    → suspect_sources += ["global:exposure" 之类]
verdict = "suspect" 当且仅当 suspect_sources 非空;只标记,不剔除(D-01)
```
阈值默认值全部进 calibration.yaml 并带注释 [ASSUMED: 初值来自标准做法,首轮真实数据分布出来后由迭代回环校准——这正是 D-04 回环的对象]。

**Why board-local Laplacian:** 整图拉普拉斯方差依赖背景内容与分辨率,换数据即失效;棋盘框内计算把度量锚定在"标定真正在乎的区域",同时部分免除比例化负担。

### Pattern 3: 双检测器 + 亚像素窗口推导(COR-01 / D-07 / D-08)

**经典路线:** `findChessboardCorners`(自适应阈值+归一化,必要时 FAST_CHECK 预筛)→ `cornerSubPix` 精化。
**SB 路线:** `findChessboardCornersSB`,**其输出已是亚像素精度,不允许再跑 cornerSubPix**(双重精化用不同的角点模型,互相污染)。
两条路线各存一份 npz(D-07)。

**方格像素边长估计(每帧):**
```python
# corners: (88, 2) 检测角点,网格 11×8 行主序
g = corners.reshape(8, 11, 2)
d_h = np.linalg.norm(g[:, 1:] - g[:, :-1], axis=2)  # 行内相邻 (8,10)
d_v = np.linalg.norm(g[1:] - g[:-1], axis=2)        # 列内相邻 (7,11)
sq_px = min(np.median(d_h), np.median(d_v))          # 保守取小:透视下远端方格更小
```

**窗口推导(D-08 核心公式):**
```
winSize = max(3, floor(sq_px / 2) - 1)
断言:2 * winSize + 1 < sq_px   # 恒成立:偶数 sq → 2w+1 = sq-1;奇数 sq → sq-2
```
- cornerSubPix 的 winSize 是**半窗口**:实际搜索区是 (2w+1)×(2w+1);本数据方格约 15–25px [CITED: PROJECT.md],例:sq=20 → w=9(搜索区 19×19 < 20);sq=15 → w=6(13×13)
- **粒度建议(研究结论,响应 D-08 留给研究阶段的决定):v1 用每帧一个值**(行/列中位数取小,稳健);同时统计"违反 2w+1<局部方格边长的角点比例",若在真实数据上超过 ~5%,config 切到逐角点局部模式(该角点 4 邻域间距中位数)。两种粒度都实现,config 键 `subpix.window_granularity: frame|corner`,默认 frame——因为棋盘集中画面中部、强畸变边缘无角点,帧级大概率够用;数据说了算
- zeroZone 用默认 (-1,-1);criteria `(EPS + MAX_ITER, 30, 1e-2)`,均进 config

### Pattern 4: 角点排序规范化 + 连拍断言(D-09)

**What:** 两个检测器对同一棋盘可能给出不同原点/方向的行主序角点序列(SB 与经典排序不一致是已知行为 [CITED: github.com/opencv/opencv/issues/22083]),必须在入盘前统一。

**规范化规则(纯几何、确定性、逐帧独立,不依赖跨帧参照):**
```
11×8 ≠ 8×11 → 转置/90° 旋转天然排除;剩 4 种变体:恒等 / 旋转180° / 水平翻转 / 垂直翻转
e_row = mean(相邻列角点差向量)   # 行主方向
e_col = mean(相邻行角点差向量)   # 列主方向
选择唯一使 (e_row.x > 0) 且 (e_col.y > 0) 的变体
```
水平翻转控制 e_row.x 符号、垂直翻转控制 e_col.y 符号,两个约束独立可满足,四种变体中恰有一个同时满足 → 序列被规范化为「行方向朝画面右、列方向朝画面下」的唯一形态。

**连拍一致性断言(D-09 成功标准):**
```
每姿态:规范化后,对每个角点编号 p ∈ 0..87,
spread(p) = max_{b1,b2∈连拍} ||uv_b1(p) − uv_b2(p)||
通过条件:median(spread) < 0.05 × sq_px 且 max(spread) < 0.3 × sq_px
```
连拍几乎同姿态(PROJECT.md:同姿态 8 张几乎相同),预期 spread 在亚像素级;若某帧真实发生镜像/翻转污染,同编号角点会跳到棋盘对侧(距离 ~ 棋盘尺寸),断言立刻爆掉。容差初值 [ASSUMED: 0.05/0.3 × sq_px,首轮真实数据校准];断言失败 → 记入检测质量摘要 + 标记该帧存疑(供人工审),不崩溃。

### Pattern 5: 2×2 基线标定(D-10)与产物 schema

**输入:** corners npz × 2 {classic, sb} × 剔除清单过滤(D-03)后的全部 train 帧(640 视图 × 88 点 ≈ 5.6 万点,calibrateCamera 秒级到分钟级,可接受)。
**四个组合:**
1. `baseline_classic_pinhole`: `cv2.calibrateCamera` 默认 flags → 5 畸变系数 (k1,k2,p1,p2,k3),对应项目模型 M2
2. `baseline_classic_fisheye`: `cv2.fisheye.calibrate` flags=0(KB4)
3. `baseline_sb_pinhole`
4. `baseline_sb_fisheye`

**fisheye 关键细节** [VERIFIED: OpenCV 4.x 官方文档 docs.opencv.org/4.x/d9/d0c/group__fisheye.html,本会话抓取]:
- KB4 模型:θ_d = θ(1+k1θ²+k2θ⁴+k3θ⁶+k4θ⁸),u = fx(x′+αy′)+cx(α 是 skew,倾斜因子)
- 默认 **flags=0 时 α(skew)是被估计的**;`CALIB_FIX_SKEW` 是选择加入才固定(修正了项目早期研究笔记里"默认固定 skew"的说法)
- 默认终止条件 TermCriteria(COUNT+EPS, 100, DBL_EPSILON);**无 USE_INTRINSIC_GUESS 时** cx,cy 置图像中心,焦距最小二乘初始化——文档示例即 `f = max(图像宽,高)/π`,本数据 2560/π ≈ 814.9px(修正早期笔记中 1128.4 的笔误:那是把 2560+1440 错当直径用的算错值)
- **Python 输入形状**:每视图 imagePoints (88,1,2)、objectPoints (88,1,3),float64;形状不对会报错甚至崩溃
- `CALIB_CHECK_COND` 会在病态视图抛 "Ill-conditioned matrix" —— 对非鱼眼广角数据收敛不稳是 D-10 预期内的正常发现:整体 try/except,失败组合记录为发现(哪个检测器 × KB4 失败、哪些视图病态),其余三个组合照常出结果;可尝试去掉 CHECK_COND 或 CALIB_FIX_K3/K4 重跑一次并对比,均记录进产物

**产物 schema(研究建议,属 Claude's Discretion 范围):**

`output/calibration/run_<ts>/baseline_<detector>_<model>/calibration.json`:
```json
{
  "lineage": {"created": "...", "calib_version": "...",
              "config_sha256": {"board.yaml": "...", "calibration.yaml": "..."},
              "corners_file": "../corners/corners_classic.npz", "corners_sha256": "...",
              "excluded_images": ["20260717_172523_559573"], "round_id": "run_<ts>"},
  "detector": "classic", "model": "pinhole|fisheye",
  "image_size": [2560, 1440], "n_views": 640, "n_points_per_view": 88,
  "rms_overall": 0.0,
  "K": [[fx,0,cx],[0,fy,cy],[0,0,1]], "D": [k1,k2,p1,p2,k3],
  "per_view_rms": {"<image_id>": 0.0},
  "flags_used": 0, "converged": true, "failure_note": null
}
```
`residuals.npz`:image_ids (N,)、residuals (N,88,2)(NaN 掩缺失)、per_view_rms (N,)。

`output/corners/run_<ts>/corners_<detector>.npz`:
```
image_ids: (N,)        # 与 manifest 对齐的 image_id
uv:        (N,88,2) float64, NaN = 未检出
sq_px:     (N,)        # 每帧实测方格像素边长(下游窗口复用/体检归一复用)
win_size:  (N,)        # 每帧实际使用的半窗口
detected:  (N,) bool
meta(json 旁车文件):lineage + 检测器 flags + criteria + 规范化规则版本
```
选带掩码的稠密布局而非长表:705×88×2 float64 ≈ 1MB,下游 calibrate 每视图切片即用;ARCHITECTURE.md 的长表是等价替择,稠密对本项目所有下游更顺手。

`manifest.json` 每张照片一条(骨架见上,含全部原始指标值 + 归一值 + suspect_sources 数组 + 双检测器成败),外加汇总:姿态数、重复数分布、双检测器成功率(分 train/test)、存疑计数按来源分桶。

### Pattern 6: 迭代回环留痕与跨轮可比(D-04)

- 每轮运行落独立目录 `output/<stage>/run_<时间戳>/`,lineage 记 `round_id` 与上一轮产物哈希链
- 剔除清单是**人工编辑 config**(`calibration.yaml` 的 `exclude_images:` 列表)——git 可 diff、回轮可复现,manifest 只记程序自动判定(D-03 允许两处,研究建议 config 清单:人工调整方案的留痕语义更贴近回环)
- 跨轮 diff 呈现:每轮 ingest/corners 结束后生成对比表(上轮 verdict → 本轮 verdict,按 image_id 对齐;新增存疑/解除存疑/持续存疑三桶 + 计数),文本表落盘 + CLI 打印

### Anti-Patterns to Avoid

- **体检即剔除**:任何"自动跳过存疑照片"的捷径——违反 D-01;存疑照片照常进角点提取,只是带标记
- **SB 输出再过 cornerSubPix**:双重亚像素精化,两个角点模型互相污染——经典路线才做 subPix
- **窗口写死像素**(如到处用 11×11):违反 D-08,且是头号陷阱的直接成因
- **按图像(而非姿态)做任何统计/切分**:同姿态 8 张几乎相同,信息泄漏(PITFALLS #2)
- **直接把 detector 原始序列入盘**:SB 与经典排序不同,下游同名角点不可比——必须过规范化层
- **命令行参数覆盖算法参数**:违反 D-12,参数源必须唯一(config 文件)
- **只存汇总不存逐帧**:违反 D-06;拼图和表格只是索引
- **标定结果不带血缘**:ITER-01 从第一天就要配置哈希 + 上游哈希,后补成本高

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| 棋盘角点检测 | 自写角点检测器 | cv2.findChessboardCorners / findChessboardCornersSB | 20 年工业级打磨;本项目 Phase 5 的"寻优"是在现有方法/参数间用数据选,不是发明检测器 |
| 亚像素精化 | 自写梯度质心迭代 | cv2.cornerSubPix(SB 内置) | 收敛准则/零区/窗口语义都有讲究,自写必踩坑 |
| 相机标定求解 | 自写 LM 优化 | cv2.calibrateCamera / cv2.fisheye.calibrate | Phase 1 明确"直接调 OpenCV 现成函数";自写求解属 Phase 2 且仍用 scipy |
| 角点叠加绘制 | 自写画圆画线 | cv2.drawChessboardCorners | 一行出标准叠加图(格子着色+编号) |
| 重投影残差计算 | — | cv2.projectPoints(rvecs/tvecs 回代) | 与标定同一投影实现,天然可比 |
| YAML 配置解析 | 手写分割 | pyyaml safe_load | safe_load 顺带挡住 YAML 反序列化攻击面 |
| CLI 骨架 | argparse 手拼 | typer | 类型注解自动生成中文帮助(D-11/D-13) |
| 拼图 | 引入图像库 | numpy 切片拼贴 + cv2.resize + matplotlib imshow | 零新依赖 |
| 模糊度指标 | 引入 no-reference IQA 库 | 拉普拉斯方差(3 行) | IQA 库(SSIM/Brisque 类)面向自然图像质量,不对标定角点锐度;拉普拉斯方差是标定社区标准做法 |

**Key insight:** 本阶段所有"难"都在**组合与校验**(双检测器、规范化、窗口推导、留痕),不在算法发明——全部数值算法都有现成标准件,把工程精力花在数据契约与断言上。

## Common Pitfalls

### Pitfall 1: 亚像素窗口 ≥ 方格边长 → 系统性偏差且不可见
**What goes wrong:** cornerSubPix 搜索窗跨过相邻角点/方格对边,质心被拉偏;偏差是系统性的、同号的,重投影误差**完全看不见**(标定把偏差吸收进内参/外参)。
**Why:** 窗口语义是半窗口,(2w+1)² 搜索区一旦 ≥ 方格像素边长就混入对边梯度。
**How to avoid:** Pattern 3 的推导公式 + 每帧断言 2w+1 < sq_px;win_size 随 sq_px 一起入盘供审查。
**Warning signs:** 换窗口大小后内参明显漂移而 RMS 几乎不变。
[CITED: .planning/research/PITFALLS.md #1——本阶段头号陷阱]

### Pitfall 2: 连拍当独立视角
**What goes wrong:** 任何按图像(而非姿态)的切分/统计都因 8 连拍近重复而泄漏或失真。
**How to avoid:** 数据模型从第一天就把 pose_id 作为一等公民(manifest、npz、residuals 全带);Phase 1 虽不做 CV 切分,schema 必须为 Phase 4 备好。
[CITED: PITFALLS #2、PROJECT.md 数据结构推论]

### Pitfall 3: SB 与经典角点排序不一致
**What goes wrong:** 直接对比两份角点/混用下游,同名编号指向物理上不同角点,断言与统计全错。
**How to avoid:** Pattern 4 规范化层,对两份输出分别做;连拍断言兜底。
[CITED: github.com/opencv/opencv/issues/22083]

### Pitfall 4: SB 结果再跑 cornerSubPix
**What goes wrong:** SB 内部已做亚像素(且用与 cornerSubPix 不同的角点模型),再精化互相污染,精度反降。
**How to avoid:** detect.py 里 subPix 只挂在经典路线分支;代码注释显式说明。

### Pitfall 5: fisheye 输入形状/dtype
**What goes wrong:** 传 (88,2) 而非 (88,1,2)、float32/64 不对、每视图数组列表形状不齐 → 报错难懂甚至段错误。
**How to avoid:** solve/baseline.py 里做显式 reshape/astype 断言后再调用。
[CITED: OpenCV fisheye 官方文档 + PITFALLS 集成陷阱表]

### Pitfall 6: fisheye 默认行为误读
**What goes wrong:** 以为 flags=0 固定 skew(实际是估计 α);以为初始焦距 ~1128(实际文档示例 max(w,h)/π ≈ 814.9);KB4 对非鱼眼广角收敛不稳/Ill-conditioned 直接崩掉整轮。
**How to avoid:** flags 显式写进产物;整组合 try/except,失败即记录为发现(D-10 预期内)。
[VERIFIED: OpenCV 4.x 官方文档;收敛不稳 CITED: D-10 决策]

### Pitfall 7: cv2.imread 失败返回 None 不抛错
**What goes wrong:** 坏路径/坏文件静默变 None,下游 cvtColor 才炸,报错位置离病因很远。
**How to avoid:** io 层封装 imread,显式 None 检查 + 带文件名的中文错误。

### Pitfall 8: imageSize 参数顺序
**What goes wrong:** calibrateCamera 的 imageSize 是 (宽,高) = gray.shape[::-1],直觉写 shape 会反。
**How to avoid:** 统一 `image_size = gray.shape[::-1]`,测试断言 2560×1440。
[CITED: PITFALLS 集成陷阱表]

### Pitfall 9: 断言/阈值写死像素
**What goes wrong:** 连拍一致性容差、拼图缩放等写死像素值,换数据/换分辨率全失效(违反比例化原则)。
**How to avoid:** 全部以 sq_px 的比例表达(如 0.05×sq_px),或图像尺寸的比例表达。

### Pitfall 10: np.load 默认允许 pickle
**What goes wrong:** npz 读写链路默认 allow_pickle 有反序列化攻击面。
**How to avoid:** 封装 artifact 读写,np.load(..., allow_pickle=False) 强制。
[CITED: PITFALLS 安全节]

### Pitfall 11: 大批量副本写盘阻塞回环
**What goes wrong:** 705 张全量标记副本(D-06)串行写 JPEG q=95 会拖慢每轮迭代。
**How to avoid:** q=90(可读性足够,磁盘约省一半)、cv2.setNumThreads 在 CLI 层统一设置;秒级到分钟级可接受,不必并行化。

## Code Examples

以下示例为推荐写法骨架(中文注释,供任务编写时直接参照;参数名以本地 OpenCV 4.14 冒烟为准,标 [ASSUMED] 处首日确认)。

### 姿态数据模型(IO-01)
```python
# src/calib/io/dataset.py
from dataclasses import dataclass
from pathlib import Path
import re

_TS = re.compile(r"^\d{8}_\d{6}(_\d+)?$")  # 20260717_172523_559573 模式

@dataclass(frozen=True)
class ImageRecord:
    image_id: str      # 文件名去扩展名,全局唯一
    path: Path         # 相对 input_data 的只读路径
    split: str         # "train" | "test"
    pose_id: str       # train: 目录名;test: 即 image_id(每张独立)
    repeat_idx: int    # 目录内按文件名排序的序号
    n_repeats: int     # 该姿态实际张数(数据驱动,不写死 8)

def scan_train(train_dir: Path) -> list[ImageRecord]:
    records = []
    for pose_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        jpgs = sorted(p for p in pose_dir.iterdir() if p.suffix == ".jpg")
        if not jpgs:
            continue  # 空目录:警告并跳过,不崩(韧性)
        for idx, f in enumerate(jpgs):
            if not _TS.match(f.stem):
                ...  # 警告:文件名不符模式,按自然序继续(回退,不崩)
            records.append(ImageRecord(f.stem, f, "train", pose_dir.name, idx, len(jpgs)))
    return records
```

### 方格像素边长 + 窗口推导(COR-01 / D-08)
```python
# src/calib/corners/window.py
import numpy as np

COLS, ROWS = 11, 8  # 来自 board.yaml,不做模块级魔法数字:由调用方传入

def estimate_sq_px(corners: np.ndarray, cols: int, rows: int) -> float:
    g = corners.reshape(rows, cols, 2)
    d_h = np.linalg.norm(g[:, 1:] - g[:, :-1], axis=2)
    d_v = np.linalg.norm(g[1:, :] - g[:-1, :], axis=2)
    return float(min(np.median(d_h), np.median(d_v)))  # 保守取小

def derive_window(sq_px: float, min_win: int = 3) -> int:
    w = max(min_win, int(sq_px // 2) - 1)
    assert 2 * w + 1 < sq_px, f"窗口 {2*w+1} 未小于方格边长 {sq_px:.1f}(头号陷阱)"
    return w
```

### 双检测器 + 亚像素(COR-01 / D-07)
```python
# src/calib/corners/detect.py 骨架
import cv2, numpy as np

def detect_all(gray: np.ndarray, pattern: tuple[int, int], cfg) -> dict:
    out = {}
    # 经典路线:检测 → cornerSubPix(窗口来自实测方格边长)
    ok, c = cv2.findChessboardCorners(gray, pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ok:
        sq = estimate_sq_px(c, *pattern)
        win = derive_window(sq)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                cfg.subpix.max_iter, cfg.subpix.epsilon)
        cv2.cornerSubPix(gray, c, (win, win), (-1, -1), crit)
        out["classic"] = c.reshape(-1, 2)
    # SB 路线:自带亚像素,绝不再过 cornerSubPix(陷阱 4)
    ok_sb, c_sb = cv2.findChessboardCornersSB(gray, pattern, cfg.sb_flags)  # [ASSUMED: flags 常量名首日冒烟确认]
    if ok_sb:
        out["sb"] = c_sb.reshape(-1, 2)
    return out  # 两份各自走规范化层后入盘
```

### 排序规范化 + 连拍断言(D-09)
```python
# src/calib/canonical.py
import numpy as np

def canonicalize(corners: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """行主序 (rows*cols,2) → 行方向朝右、列方向朝下的唯一形态。"""
    g = corners.reshape(rows, cols, 2)
    if np.mean(g[:, 1:, 0] - g[:, :-1, 0]) < 0:      # 行方向朝左 → 水平翻转
        g = g[:, ::-1]
    if np.mean(g[1:, :, 1] - g[:-1, :, 1]) < 0:      # 列方向朝上 → 垂直翻转
        g = g[::-1]
    return g.reshape(-1, 2)

def burst_consistency(uvs: np.ndarray, sq_px: float,
                      med_tol: float = 0.05, max_tol: float = 0.3) -> dict:
    """uvs: (B, 88, 2) 同姿态 B 张连拍(已规范化)。断言同名角点位置连续。"""
    spread = np.linalg.norm(uvs.max(axis=0) - uvs.min(axis=0), axis=1)  # (88,)
    return {"ok": spread.median() < med_tol * sq_px and spread.max() < max_tol * sq_px,
            "spread_median_px": float(np.median(spread)),
            "spread_max_px": float(spread.max()),
            "threshold_px": [med_tol * sq_px, max_tol * sq_px]}
```

### 2×2 基线标定(D-10)
```python
# src/calib/solve/baseline.py 骨架
import cv2, numpy as np

def calibrate_pinhole(objp: np.ndarray, corners_per_view: list[np.ndarray],
                      image_size: tuple[int, int]) -> dict:
    # objp: (88,3) float64,Z=0 平面,单位 mm(square_size 只进这里)
    rms, K, D, rvecs, tvecs, _, _, per_view = cv2.calibrateCameraExtended(
        [objp.astype(np.float64)] * len(corners_per_view),
        [c.reshape(-1, 1, 2).astype(np.float64) for c in corners_per_view],
        image_size, None, None)   # [ASSUMED: Extended 绑定首日确认;不可用则 calibrateCamera + 自算 per_view]
    return {"rms": rms, "K": K, "D": D.ravel(), "rvecs": rvecs, "tvecs": tvecs,
            "per_view_rms": per_view.ravel()}

def calibrate_fisheye(objp: np.ndarray, corners_per_view: list[np.ndarray],
                      image_size: tuple[int, int]) -> dict:
    K = np.eye(3)                       # 交由 fisheye 内部初始化(cx,cy=图像中心,f≈max(w,h)/π)
    D = np.zeros(4)
    flags = 0                           # 默认估计 skew;显式记录进产物(陷阱 6)
    try:
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            [objp.reshape(-1, 1, 3).astype(np.float64)] * len(corners_per_view),
            [c.reshape(-1, 1, 2).astype(np.float64) for c in corners_per_view],
            image_size, K, D,
            flags, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6))
        return {"rms": rms, "K": K, "D": D.ravel(), "converged": True}
    except cv2.error as e:              # Ill-conditioned / 不收敛 = D-10 预期发现,记录不崩
        return {"converged": False, "failure_note": str(e)}
```

### 模糊度指标(棋盘框内,IO-02)
```python
# io/precheck.py 节选
def blur_score(gray_roi: np.ndarray) -> float:
    """拉普拉斯方差:值越小越糊(拉普拉斯 = 二阶差分,锐利边缘响应大)。"""
    return float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())

def exposure_clip(gray: np.ndarray) -> tuple[float, float]:
    """(高光裁剪比例, 阴影裁剪比例):JPEG 8 位,0–255。"""
    total = gray.size
    return (float((gray >= 250).sum() / total), float((gray <= 5).sum() / total))
```

### 配置解析(IO-03 / D-12)
```python
# io/config.py 节选
import yaml
from dataclasses import dataclass

def load_board(path) -> tuple[int, int, float]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)                # safe_load:不执行任意对象反序列化
    b = raw["board"]
    assert b["board_type"] == "chessboard"
    cols, rows, sq = int(b["cols"]), int(b["rows"]), float(b[f"square_size_{b['unit']}"])
    assert 3 <= cols and 3 <= rows and sq > 0   # 校验失败 = 中文错误信息
    return cols, rows, sq                       # square_size 只用于物点坐标,绝不进像素级逻辑
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 只用经典 findChessboardCorners + cornerSubPix | SB 检测器(基于鞍点检测的算法,对畸变/透视/光照更稳,自带亚像素)进入主模块多年 | OpenCV 4.3+ | 双检测器并跑是当前社区共识做法;SB 对广角强畸变数据的成功率通常更高 [CITED: OpenCV 文档 + github #22083] |
| 固定像素窗口 subPix | 按棋盘尺度比例化窗口 | — (本项目方法论) | D-08 锁定;换数据/换分辨率零改码 |
| OpenCV 4.x | OpenCV 5.0.0(2026-08-19 发布) | 数周前 | **锁 <5.0**(CLAUDE.md 硬约束);MAINT-01 待 5.0.x 攒 2–3 个补丁版再审 [VERIFIED: opencv.org/releases] |
| 整图模糊指标 | 棋盘框内指标 + 同姿态相对比较 | — (本项目设计) | 换背景/换构图不失效 |

**Deprecated/outdated:**
- 项目早期研究笔记中"fisheye 默认固定 skew""初始焦距 1128.4px"两处说法已在本次研究中修正(见 Pitfall 6)
- `cv2.CALIB_CB_FAST_CHECK` 单独使用会降低检出率——只作为预筛加速,失败时必须回退完整检测 [CITED: PITFALLS]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | cv2.estimateChessboardSharpness 在 4.14 主模块可用(可作棋盘局部清晰度的增强指标) | 体检指标 / Alternatives | 低:拉普拉斯方差方案不依赖它;首日冒烟一行代码确认 |
| A2 | findChessboardCornersSB 的 flags 常量集(ACCURACY/EXHAUSTIVE/LARGER/CLUSTERING 等)与默认行为如文档所述 | 代码示例 | 低:首日 `python -c "help(cv2.findChessboardCornersSB)"` 确认;flags 全进 config 可调 |
| A3 | 体检阈值默认值(z>3.5、裁剪比例、全局分位线 1%/99%) | Pattern 2 | 低:本来就要在迭代回环里用真实数据校准(D-04) |
| A4 | 连拍一致性容差 0.05/0.3 × sq_px 初值 | Pattern 4 | 低:首轮真实数据分布出来即知;过松/过紧都会在第一轮回环显形 |
| A5 | KB4 在本批广角(非鱼眼)数据上收敛不稳的概率中等偏高 | Pattern 5 / Pitfall 6 | 无:D-10 已声明失败属正常发现并记录;try/except 兜底 |
| A6 | cv2.calibrateCameraExtended 的 Python 绑定在 4.14 可用 | Pattern 5 | 低:不可用则 calibrateCamera + projectPoints 自算 per-view RMS(几行) |
| A7 | 拼图抽样:每姿态 1 张 + 全部存疑 + 指标最差 N 张,拼图 ≤ ~200 格可读 | Claude's Discretion 建议 | 无:布局是明示的自由裁量区 |
| A8 | 方格像素边长在本数据 15–25px 量级(窗口 w≈6–11) | Pattern 3 | 低:PROJECT.md 既有估计;首帧实测即定,推导公式对任意值成立 |

## Open Questions

1. **经典检测器在本数据的成功率**
   - What we know: 棋盘占比小、集中中部、广角桶形畸变——经典路线的已知弱区;SB 通常更稳
   - What's unclear: 具体成功率(可能 70% 也可能 99%)
   - Recommendation: 不需要预研——ingest 的双检测器探针第一轮就给出分 train/test 的成功率对比,直接进回环
2. **逐帧 vs 逐角点窗口粒度**
   - What we know: 棋盘集中中部,边缘强畸变区无角点 → 帧级中位数大概率够
   - Recommendation: v1 帧级 + 违反率统计(config 可切逐角点),用数据裁决(已进 Pattern 3)
3. **KB4 收敛性**
   - Recommendation: 按 A5 处理,失败即记录;2×2 对比本身就是 D-10 要的证据
4. **test 集 65 张的棋盘可见性质量**
   - Recommendation: ingest 覆盖 test 并分 split 出报告,第一轮即知

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| uv | 环境/锁文件 | ✓ | 0.11.32(本地实测) | — |
| Python 3.12 | 全部 | ✓(uv 托管) | 3.12.13(本地实测) | — |
| 系统 python3 | 备选 | ✗ 不合用 | 3.10(numpy 2.5/scipy 1.18 要求 ≥3.12) | 一律走 `uv run` |
| PyPI(清华镜像) | 依赖安装 | ✓(慢) | opencv 轮子 ~90MB 实测 10 分钟级(代理限速) | 重试续传;失败换镜像源 |
| opencv-python 4.14.0.94 | 检测/标定 | ✓ 可安装(缓存已在) | 4.14.0.94(PyPI JSON 核实) | — |
| numpy 2.5.4 / scipy 1.18.1 | 数值/锁文件 | ✓ 可安装 | PyPI JSON 核实,cp312 轮子在 | — |
| input_data/ | 全阶段 | ✓ 只读已就位 | 705 张 / 451MB(实测) | — |
| 磁盘(产物) | D-06 副本 | ✓ | 预估 ~1–1.5GB(705 张标记副本 q90 + 存疑原图 + 拼图) | — |

**Missing dependencies with no fallback:** none
**Missing dependencies with fallback:** 系统 python3(不合用)→ 全部命令走 `uv run`

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x(Wave 0 安装,greenfield 无既有配置) |
| Config file | `pyproject.toml` 内 `[tool.pytest.ini_options]`(Wave 0 建;markers: integration) |
| Quick run command | `uv run pytest tests/ -x -q -m "not integration"` |
| Full suite command | `uv run pytest tests/ -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| IO-01 | 姿态数据模型:80×8 布局解析 / 张数不齐 / 单张姿态 / 空目录 / 文件名模式回退 | unit(临时目录夹具) | `uv run pytest tests/test_io_dataset.py -q` | ❌ Wave 0 |
| IO-02 | 体检:清晰 vs 模糊 vs 过曝 vs 欠曝合成图的指标方向正确;双来源判定与 manifest 记录 | unit | `uv run pytest tests/test_io_precheck.py -q` | ❌ Wave 0 |
| IO-03 | 配置:合法加载 / 缺字段 / 非法值中文报错 / square_size 不出现在像素逻辑 | unit | `uv run pytest tests/test_io_config.py -q` | ❌ Wave 0 |
| COR-01 | 合成棋盘图检测(双检测器)定位 < 0.5px;窗口推导恒满足 2w+1<sq;规范化把 4 种翻转变体映到同一序列;连拍断言对翻转污染的帧报 fail | unit | `uv run pytest tests/test_corners.py -q` | ❌ Wave 0 |
| (最小标定) | 合成数据标定回归:已知真值 K/D,RMS < 0.5px,参数误差在容差;fisheye 形状断言与 try/except 路径 | unit | `uv run pytest tests/test_solve_baseline.py -q` | ❌ Wave 0 |
| 全链路 | 真实数据小子集冒烟(3 姿态 × 8 + 5 张 test:ingest→corners→calibrate 2×2 全跑通、产物 schema 齐、血缘字段齐) | integration | `uv run pytest tests/test_smoke_realdata.py -q -m integration` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/ -x -q -m "not integration"`(秒级)
- **Per wave merge:** `uv run pytest tests/ -q`(含真实数据冒烟,分钟级)
- **Phase gate:** 全套绿 + 在完整 705 张真实数据上人工过一遍角点叠加图与废帧清单(D-05 收手标准)

### Wave 0 Gaps
- [ ] `pyproject.toml` + `uv lock` + 依赖/开发依赖安装(含 opencv/numpy/scipy 冒烟打印版本——STATE.md 首日待办)
- [ ] `tests/conftest.py` — **合成棋盘图生成夹具**:numpy 画棋盘 + 已知内参 projectPoints 生成带已知真值角点的图(可加模糊/噪声/过曝变体)——它是 COR-01、最小标定、体检三类测试的共同地基
- [ ] `tests/test_io_dataset.py` — 覆盖 IO-01 韧性三退化路径(张数不齐截断语义 / 单张姿态 / 模式回退)
- [ ] `tests/test_io_precheck.py`、`tests/test_io_config.py`、`tests/test_corners.py`、`tests/test_solve_baseline.py`、`tests/test_smoke_realdata.py`
- [ ] pytest markers 注册(integration)

## Security Domain

`security_enforcement: true`(.planning/config.json,ASVS Level 1,block_on: high)。本项目为离线单机 CLI,无网络服务/认证/会话;适用面窄但非零。

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | 无用户系统(本地工具) |
| V3 Session Management | no | 无会话 |
| V4 Access Control | no | 无多用户;文件权限由 OS 管 |
| V5 Input Validation | yes | pyyaml **safe_load**(禁 yaml.load);配置字段类型/范围校验 + 中文报错;文件名模式校验;图像读取 None 检查 |
| V6 Cryptography | no | 无加密需求(血缘哈希用 hashlib sha256,标准库) |
| V12 File Integrity(参照) | yes(轻) | 血缘指纹(配置/上游 sha256)本身就是完整性机制 |

### Known Threat Patterns for Python 数据流水线

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| YAML 反序列化执行任意代码(恶意 config) | Tampering / Elevation | `yaml.safe_load` only;代码审查断言无 `yaml.load(` |
| npz pickle 反序列化 | Tampering | 封装 np.load(..., allow_pickle=False);测试断言 |
| 路径穿越(构造的目录/文件名跳出 output/) | Tampering | 产物路径统一由 output 根拼接;拒绝 `..` 与绝对路径成分;input_data 只读(写入前断言路径前缀) |
| 资源耗尽(超大图/海量文件) | DoS | 读图后断言分辨率一致性(2560×1440);文件数上限警告 |
| OpenCV 线程失控(占满核) | DoS(自伤) | CLI 入口统一 cv2.setNumThreads(配置可调) |

[CITED: .planning/research/PITFALLS.md 安全节]

## Sources

### Primary (HIGH confidence)
- OpenCV 4.x 官方文档 fisheye 页(docs.opencv.org/4.x)— 本会话全文抓取:KB4 公式、flags=0 语义(skew 估计)、默认终止条件、初始焦距 max(w,h)/π、Python 签名与数组形状
- PyPI 官方 JSON 接口 — numpy/2.5.4、scipy/1.18.1、opencv-python/4.14.0.94 的版本、requires_python、cp312 轮子存在性逐一核实
- 本地数据核对(ls/stat/find)— input_data 布局、命名模式、张数、体积、分辨率;uv/Python 本地版本
- 本地 slopcheck 0.6.1 scan — 12 包全部 OK
- 项目内部研究:.planning/research/PITFALLS.md(陷阱 #1/#2/#10、集成陷阱表、安全节)、ARCHITECTURE.md(产物长表约定、模块结构)、docs/pipeline.md(八阶段契约)

### Secondary (MEDIUM confidence)
- github.com/opencv/opencv/issues/22083 — SB 与经典角点排序不一致(文档级证据,未本地复现)
- CLAUDE.md 栈表版本兼容性断言(numpy 2.5 + scipy 1.18 成对验证)——PyPI 侧已核实版本存在性,配对行为属文档断言

### Tertiary (LOW confidence / 待首日冒烟)
- SB flags 常量集与 estimateChessboardSharpness / calibrateCameraExtended 的 4.14 绑定可用性(见 Assumptions A1/A2/A6;验证脚本一行,已列入 Wave 0)
- 拼图抽样规则 A7(明示自由裁量,不构成事实断言)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — 版本经 PyPI 官方接口核实,slopcheck 全过,本地可安装已证实(缓存就位)
- Architecture: HIGH — 完全落在既有契约(pipeline.md 八阶段 + ARCHITECTURE.md 产物约定)之内,无新发明
- Pitfalls: HIGH — 项目内部研究(PITFALLS.md)交叉官方文档;两处早期笔记错误已在本次研究修正
- API 细节(SB flags 等少量绑定):MEDIUM — 首日冒烟一行确认(已列 Wave 0),不影响架构

**Research date:** 2026-10-16
**Valid until:** 2026-11-15(栈锁定 <5.0,变化慢;OpenCV 4.x 冻结线内无预期变动)
