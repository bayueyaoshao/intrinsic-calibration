# Phase 2: 模型层 M1–M4 + 双求解后端 + 产物血缘 - Research

**Researched:** 2026-10-17
**Domain:** 相机投影模型(针孔 Brown-Conrady / 鱼眼 KB4)的解析投影与雅可比、双求解后端(OpenCV 原生 / scipy 稀疏最小二乘)、产物血缘
**Confidence:** HIGH(核心结论全部在项目环境 Python 3.12.13 + OpenCV 4.14.0 + numpy 2.5.4 + scipy 1.18.1 里用可复现实验直接验证;验证脚本结论已并入下文,无需重复实验)

---

## Summary

本阶段的技术核心是"把四个相机模型的投影公式和手写导数写对,并与 OpenCV 对齐到浮点极限"。本研究在项目虚拟环境里逐项做了实证:**M1/M2/M3 的针孔 Brown-Conrady 投影(含 M3 有理分母的嵌套方式)与 `cv2.projectPoints`、M4 的 KB4 投影与 `cv2.fisheye.projectPoints`,在 float64 下最大偏差均为 ~5e-13 像素**(机器精度量级);完整的手写解析雅可比(畸变核导数 → 链式到外参扰动)对中心差分的最大偏差 6.7e-10。公式、OpenCV 参数布局映射(M1/M2/M3 的 flags 与畸变向量切片、M4 的 D 布局与 skew 处理)全部实测确认,可直接进测试。

求解侧的实证修正了两个错误预期:(1)**640 视图的一次 OpenCV `calibrateCamera` 求解约 105~115 秒**(M1 104s / M2 107s / M3 115s;鱼眼 27~32s),不是"秒级";scipy 精化(linear 核)13~20 秒即收敛,鲁棒核(soft_l1/huber)在 trf+lsmr 下**形式收敛极慢**(300 次迭代仍 status=0,但残差 RMS 已到噪声水平)——全矩阵试车总耗时约 15~25 分钟,仍是"分钟级",但计划里的单步耗时与"收敛"的判定语义要按实测调整。(2)640 视图的稠密雅可比等价 3.47 GB,**稀疏结构不是优化项而是硬前提**(scipy 必须用 trf + tr_solver='lsmr' + jac_sparsity;method='lm' 只支持稠密,不可用)。向量化组装后单次雅可比 0.15 秒、残差 0.02 秒,求解完全可行。

两个意外发现影响设计:(a)**`cv2.fisheye.projectPoints` 完全忽略 K[0][1](skew 倾斜因子)**——阶段一鱼眼基线估出的 skew=0.1199 在投影里是死参数,因此 M4 应定义成无 skew 的 8 参数模型,OpenCVSolver 对 M4 加 `CALIB_FIX_SKEW`(flags=10)保证模型两侧一致;(b)体检存疑帧共 89 张(train 67 + test 22),而标定只吃 train 640 张——D-04 要做"89 vs 616"全量对比,需要给 test 帧做"只估位姿"的回代评分(用 models.project 做 6 参数小求解,兼得单一数学源与玩具模型兼容)。

**Primary recommendation:** 按"先 models(投影+雅可比+parity 测试锁死)→ 再 _types/ObservationSet → 再双求解器 → 再试车矩阵与存疑复核"的顺序落计划;ScipySolver 用"内外参联合单向量 + 解析稀疏雅可比 + trf/lsmr + x_scale='jac'",鲁棒核默认作对照路径并配"实用收敛"判定;M4 统一无 skew;全部新产物走既有 `make_lineage`。

---

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions(锁定决策,照抄自 02-CONTEXT.md)

**真实数据试车范围(用户拍板)**
- **D-01:全矩阵一次跑齐**——4 个模型 M1–M4 × 求解路径(纯 OpenCV / OpenCV 初始化→scipy 精化)全部在真实数据上试车,不只跑路线图底线的 M1+M2。单次求解秒级、总耗时分钟级;并排多存信息,阶段三四直接引用,不回头补跑
- **D-02:角点输入用阶段一定版的最优组合**——SB 为主 + classic 兜底那份角点(corners npz 的 sb 路线,705 张全覆盖 = 664 SB + 41 兜底,逐帧带 source 来源字段;标定用其中 train 640 张)。classic 全套那份不进阶段二试车
- **D-03(由 D-02 推论):混合角点定义的代价不在本阶段量化**——SB 与 classic 角点间约 0.135px 系统差,如需评估留阶段四五用 classic 角点对照

**存疑帧复核(阶段一 UAT 遗留收编)**
- **D-04:纳入阶段二**——scipy 精化结果出来后,对比 89 张存疑帧 vs 616 张正常帧的逐帧重投影残差分布(阶段一 baseline 已有 per_view_rms 逐帧数据结构先例)。结论两种走向:存疑组明显更差 → 体检指标与真实误差相关,后续剔除清单有依据;两组无差别 → "看着糊"≠"算得差",不剔除。结论落盘成带血缘产物,并回写阶段一 UAT 的 Gap 闭环记录

**上游已锁定决策(重申,下游直接遵守)**
- **D-06:models/ 是投影/雅可比唯一数学源;一致性比对 + 数值雅可比测试先于任何求解代码**(项目固有决策,roadmap 成功标准 1/2 的前提)
- **D-07:模型清单固定**——M1 针孔+k1k2p1p2、M2 +k3(即阶段一 pinhole 基线)、M3 有理 k1–k6+p1p2、M4 鱼眼 KB4(即阶段一 fisheye 基线);M1/M2/M3 与 cv2.projectPoints 比对、M4 与 cv2.fisheye.projectPoints 比对
- **D-08:ScipySolver 残差仅由 models.project 构成**(代码审查 + 测试锁定);两后端返回同一 CalibrationResult
- **D-09:注册机制零侵入**——测试中新增一个玩具模型,仅一处注册即可被两个求解器使用
- **D-10:numdifftools 只作测试基准**,绝不进求解/理论计算路径(CLAUDE.md 技术栈约定;需加入 dev 依赖组)

### Claude's Discretion(研究阶段定夺)

- ScipySolver 的变量组织(内外参联合优化 vs 分块交替)——研究阶段定;注意 640 视图 × 6 外参 ≈ 3840 个外参变量,least_squares 的稀疏结构(jac_sparsity / 分块组装)怎么用值得研究
- ObservationSet / CalibrationResult 具体字段——沿用阶段一失败契约:converged=false 时结构键一个不少、数值字段一律 null、中文 failure_note
- 一致性比对的浮点容差具体值
- `calib calibrate` 命令如何演进到模型注册机制(阶段一 2×2 基线产物保持向后可读)
- 测试文件组织、中文措辞细节

### Deferred Ideas(OUT OF SCOPE,完全不做)

- 混合角点定义(SB + classic 兜底,~0.135px 系统差)的代价量化——如需要,阶段四五用 classic 角点对照(D-03)
- 优化器/鲁棒核/初始化策略的系统性对比实验(SOL-02)——阶段四
- M3 有理模型是否过拟合的判定——阶段四 held-out 评估
- stdDeviationsIntrinsics 与自有 FIM 的交叉验证——阶段三(STATE 研究标记位)

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | 描述(来自 REQUIREMENTS.md) | 研究支撑(哪条发现使它可实现) |
|----|------------------------------|----------------------------------|
| MOD-01 | CameraModel 抽象(投影/反投影/雅可比/参数布局/初值策略)+ 注册机制;models 共享同一投影与雅可比实现(单一数学源) | 下文"CameraModel 接口(推荐)"给出完整签名;注册 = 模块级字典 + 装饰器,玩具模型零侵入验证方案已定 |
| MOD-02 | 四类模型实现 M1–M4 + 与 OpenCV 投影一致的比对测试 + 数值雅可比校验测试 | 四个模型的公式、参数布局映射、容差矩阵全部本地实测(见"代码模板"与"一致性比对容差矩阵") |
| SOL-01 | 求解器统一接口 + 双后端(OpenCV 初始化 → scipy least_squares 精化,含鲁棒核),后端可互换、结果同构(CalibrationResult) | OpenCV flags 映射实测;scipy 稀疏求解在真实规模收敛实测(15 次迭代/13 秒,linear);鲁棒核行为与实用收敛判定已定 |
| ITER-01 | 阶段间仅通过带血缘的落盘产物通信,输入输出契约显式;每阶段可独立运行/重跑/检查 | 既有 make_lineage 直接复用;新产物 upstream 链 = 角点 npz + 配置 + (scipy 时)OpenCV 初值产物;CLI 零参数模式(D-12)延续 |

</phase_requirements>

---

## Architectural Responsibility Map

本项目是离线流水线,不存在浏览器/CDN 等网络分层;按下表的项目内部分层映射(每个能力只有一个属主,防止数学被抄写两份)。

| 能力 | 属主层 | 协作层 | 理由 |
|------|--------|--------|------|
| 投影/反投影/解析雅可比公式 | `models/`(叶子模块,只依赖 numpy) | — | 单一数学源是项目根约束(D-06):solve、noise(Fisher)、analysis 全部只准 import 这里 |
| 内参↔OpenCV 参数布局互译(flags、K/D 切片) | `models/` 各具体模型(自带 OpenCV 适配器) | `solve/opencv.py` 调用 | "参数叫什么、在 cv2 数组里排哪"是模型自身知识;玩具模型不实现适配器 → OpenCVSolver 走通用初始化路径(D-09) |
| 残差/雅可比的分块组装与优化调用 | `solve/`(residuals + scipy_solver) | 只 import models | 残差仅由 models.project 构成(D-08),测试锁定 |
| OpenCV 原生求解调用与初值产出 | `solve/opencv.py` | models 适配器 | OpenCV 是黑盒初始化器;结果解包进统一 CalibrationResult |
| 观测集装载(角点 npz + manifest 对齐/剔除) | `solve/observation.py`(或规划者定的等价位置) | io/artifacts 既有读取函数 | 行序与 manifest 对齐是 D-04 join 的前提 |
| 血缘指纹 | `util/lineage.py`(既有,复用) | 全部写盘方 | 已在阶段一验证的模式 |
| 存疑帧对比与图 | `solve/`(数据计算)+ `visualization/`(图) | ingest manifest(只读) | 结论落盘带血缘;图只是 matplotlib PNG |
| 试车矩阵编排 | `cli/main.py` calibrate 命令(薄) | 配置(D-12,CLI 零参数) | 编排是"列表循环",逻辑在库里 |

---

## Standard Stack

### Core(已安装并锁定,本阶段零新增运行时依赖)

| 库 | 版本(实测) | 用途 | 依据 |
|----|--------------|------|------|
| opencv-python | 4.14.0(锁 <5.0) | 求解后端之一(calibrateCameraExtended / fisheye.calibrate)、parity 比对基准、Rodrigues | [VERIFIED: 本地实测] |
| numpy | 2.5.4 | 全部数组数学、稀疏雅可比组装、npz 产物 | [VERIFIED: 本地实测] |
| scipy | 1.18.1 | `optimize.least_squares`(trf + lsmr + 稀疏雅可比 + 鲁棒核)、`sparse` 结构 | [VERIFIED: 本地实测] |
| matplotlib | 3.11.2 | D-04 存疑对比图(visualization 层唯一画图方) | [VERIFIED: 本地实测] |

### Supporting(本阶段唯一新增,dev 依赖组)

| 库 | 版本 | 用途 | 何时用 |
|----|------|------|--------|
| numdifftools | 0.11.1 | 数值雅可比测试基准(D-10:只进测试,绝不进求解/理论路径) | `tests/test_models_jacobian*.py` |

**安装:**

```bash
# 加进 pyproject.toml 的 [dependency-groups] dev 组(与 pytest/ruff 并列),然后:
uv sync
```

**版本验证:** `uv pip install --dry-run numdifftools` 解析出 0.11.1(2026-10-17 实测可用);[VERIFIED: uv registry]

### Alternatives Considered

| 备选 | 结论 | 何时 reconsider |
|------|------|-----------------|
| 手写 LM/分块交替优化 | **否**——scipy trf+lsmr 在真实规模 15 次迭代收敛(实测),自研优化器违反 Don't Hand-Roll | 永不(除非 scipy 1.x 行为回归) |
| scipy method='lm' | **不可用**——lm 只支持稠密雅可比,640 视图稠密 = 3.47 GB [VERIFIED: 本地实测规模推算] | 视图数 < ~40 的小实验也不必 |
| 用 cv2.projectPoints 自带雅可比当主基准 | **否**——它对 float32 输入给出完全错误的雅可比(实测误差 60),float64 下才正确(误差 2e-7);只作 M1–M3 的辅助交叉验证 | 永不作为唯一基准 |
| joblib 并行试车矩阵 | 本阶段**不加**——全矩阵串行 ~15-25 分钟可接受;并行属阶段四实验网格的优化 | Phase 4 |

## Package Legitimacy Audit

> 本阶段只新增 numdifftools(dev-only,CLAUDE.md 技术栈表已预先认可)。slopcheck 已安装并运行。

| Package | Registry | 年龄 | 下载量 | 源码仓库 | slopcheck | 处置 |
|---------|----------|------|--------|----------|-----------|------|
| numdifftools | PyPI | 项目本身自 2011 年维护;当前发布 0.11.1(镜像显示最近上传 65 天前) | 长期广泛使用 | github.com/pbrod/numdifftools | [OK] | Approved(仅 dev 组) |

**因 slopcheck [SLOP] 移除的包:** 无
**标记可疑 [SUS] 的包:** 无
其余依赖(opencv-python/numpy/scipy/matplotlib)为阶段一已锁定安装,非本阶段新增,不在审计范围。

---

## Architecture Patterns

### System Architecture Diagram(本阶段数据流)

```
config/board.yaml + calibration.yaml(模型/求解器矩阵配置)
        │
        ▼
corners npz(sb 路线, run_20261017_104817:640 train 全检出) ──┐
ingest manifest.json(存疑标记, D-04 join 用)                  │
        │                                                      │
        ▼                                                      ▼
  solve/observation.py:ObservationSet 装载(行序=manifest 对齐, 剔除清单生效)
        │
        ├──▶ models/ ──① parity 测试 + 数值雅可比测试(先于一切求解代码, D-06)
        │      ▲ project / project_jacobian / unproject / param_names / default_params
        │      │ + 每模型的 OpenCV 适配器(flags / K↔params / D 切片)
        │      │
        │      ├──▶ solve/opencv.py   OpenCVSolver ── calibrateCameraExtended(M1-M3, float32)
        │      │            │      └── fisheye.calibrate(M4, float64, flags=10)
        │      │            ▼  CalibrationResult(初值, 含逐视图 rvecs/tvecs)
        │      │            │
        │      │            ▼
        │      └──▶ solve/scipy_solver.py ScipySolver ── 装箱 θ=[内参 P | 每视图 6 外参]
        │                 │   残差 = models.project(唯一来源, D-08)
        │                 │   雅可比 = 解析稀疏(model.project_jacobian + 外参链式)
        │                 │   trf + lsmr + jac_sparsity + x_scale='jac' + 可配鲁棒核
        │                 ▼  CalibrationResult(精化)
        │
        ▼  全矩阵 4 模型 × {opencv, opencv→scipy} (+ 鲁棒核对照)
output/calibration/run_<时间戳>/<model>__<solver>/{result.json, extrinsics.npz, residuals.npz}
        │                    (每个都带 make_lineage 血缘: config 哈希 + 上游 npz/初值产物哈希)
        ├──▶ comparison.json(矩阵汇总)
        │
        ▼
D-04 存疑帧复核:精化结果逐帧残差(train 用标定位姿直接回代;test 65 张先做 6 参数位姿小求解再回代)
        │        × manifest 的 suspect 标记(67 train + 22 test = 89 张 vs 616 张)
        ▼
suspect_review/{suspect_vs_normal.json + .png}(带血缘) ──▶ 回写 01-HUMAN-UAT.md Gap 记录
```

读者可以顺着箭头从角点文件走到最终对比产物;唯一的数学源头 models/ 被测试与求解两侧同时引用,这就是"理论与实际严格可比"的结构保证。

### Recommended Project Structure(models/ 与 solve/ 新文件)

```
src/calib/models/
├── __init__.py        # 注册表:register_model 装饰器 + create_model(model_id) 查表
├── base.py            # CameraModel 抽象 + 共享导数工具(外参链式块、旋转扰动)
├── pinhole_brown.py   # M1/M2/M3 三个类(同一畸变核,参数个数不同)
├── kb4.py             # M4
└── (OpenCV 适配器可作各模型的类方法/嵌套类, 不单独开文件)
src/calib/solve/
├── observation.py     # ObservationSet 数据类 + 从 corners npz/manifest 装载
├── result.py          # CalibrationResult 数据类 + 落盘/读盘(result.json + npz)
├── opencv.py          # OpenCVSolver(flags 表、float32 装包、解包进统一结果)
├── scipy_solver.py    # ScipySolver(装箱/残差/稀疏雅可比/least_squares 调用)
├── residuals.py       # 残差与稀疏雅可比组装(仅 import models;D-08 测试锁定)
├── scoring.py         # 逐帧回代评分(含 held-out 视图只估位姿的小求解, D-04 用)
└── baseline.py        # 既有,保留不动(阶段一 2×2 基线可复现)
tests/
├── test_models_parity.py      # M1-M4 vs OpenCV 投影, 容差矩阵
├── test_models_jacobian.py    # 解析 vs numdifftools(+ M1-M3 辅助 vs cv2 雅可比)
├── test_models_registry.py    # 玩具模型一处注册、双求解器可用(D-09)
├── test_solve_observation.py  # 装载/对齐/剔除契约
├── test_solve_solvers.py      # 合成真值恢复、双后端同构、失败契约
└── test_solve_scipy_residuals.py  # 残差仅用 models.project 的锁定测试(D-08)
```

(文件名可微调,职责边界不变;`solve/types.py` 合并 observation/result 也是合理变体,规划者定。)

### Pattern 1: CameraModel 抽象 + 注册表(单一数学源的结构化)

**推荐接口**(在 ARCHITECTURE.md 草案上补了 `default_params` 与 OpenCV 适配器,均有本阶段实测依据):

```python
class CameraModel(ABC):
    model_id: str                    # "m1".."m4"(短名, 与全部规划文档一致)
    description: str                 # 中文名+一句话白话(报告/CLI 汇总直接用)
    param_names: tuple[str, ...]     # 参数向量布局, 如 ("fx","fy","cx","cy","k1","k2","p1","p2")

    @abstractmethod
    def project(self, pts_cam: np.ndarray, params: np.ndarray) -> np.ndarray:
        """(N,3) 相机坐标系点 + (P,) 参数 → (N,2) 像素。float64 进出。"""

    @abstractmethod
    def project_jacobian(self, pts_cam: np.ndarray, params: np.ndarray) -> np.ndarray:
        """d 像素 / d 内参, (N,2,P), 解析式。"""

    @abstractmethod
    def unproject(self, uv: np.ndarray, params: np.ndarray) -> np.ndarray:
        """像素 → 归一化射线/相机坐标系方向 (N,3)。牛顿迭代(两族模型各 ~5 行)。"""

    @abstractmethod
    def default_params(self, image_size: tuple[int, int]) -> np.ndarray:
        """按图像尺寸给初值:针孔族 fx=fy≈0.75·宽、cx/cy=中心、畸变清零;
        KB4 用 OpenCV 鱼眼惯例 fx=max(宽,高)/π。"""

    # 可选(D-09 关键):OpenCV 适配器。玩具模型不实现 → OpenCVSolver 走通用初始化
    def opencv_adapter(self): ...    # 提供 flags、params→(K,D)、(K,D)→params、float32 装包规则
```

**注册表:**

```python
# models/__init__.py
_REGISTRY: dict[str, type[CameraModel]] = {}
def register_model(cls):            # 装饰器:一处注册
    _REGISTRY[cls.model_id] = cls
    return cls
def create_model(model_id: str) -> CameraModel:
    ...  # 查表, 未注册给中文报错
```

零侵入的判定标准:求解器代码只出现 `CameraModel` 协议与 `create_model`,**不出现任何具体模型 import**——测试里注册一个玩具模型即可证明(成功标准 5)。

**OpenCV 适配器各模型实测配置**(全部 [VERIFIED: 本地实测 cv2 4.14.0]):

| 模型 | params(param_names) | cv2 调用 | flags | 返回 D 布局 → 切片 |
|------|----------------------|----------|-------|--------------------|
| M1 | fx,fy,cx,cy,k1,k2,p1,p2(8) | calibrateCameraExtended | `CALIB_FIX_K3`(=128) | D 长度 5 [k1,k2,p1,p2,k3=0] → 取前 4 |
| M2 | +k3(9) | 同上 | 0 | D 长度 5 → 全取 |
| M3 | +k4,k5,k6(12) | 同上 | `CALIB_RATIONAL_MODEL`(=16384) | D 长度 **14**(尾 6 个是 thin-prism/tilted 恒 0)→ 取前 8 |
| M4 | fx,fy,cx,cy,k1..k4(8,**无 skew**) | fisheye.calibrate | `RECOMPUTE_EXTRINSIC \| CALIB_FIX_SKEW`(= 2\|8 = 10) | D 长度 4 → 全取 |

**M4 的 skew 决定(实测依据):** `cv2.fisheye.projectPoints` 完全忽略 K[0][1]——skew=0.5 的夸张值下投影逐位不变(max|Δ|=0.0);而阶段一 fisheye 基线(未加 FIX_SKEW)估出 skew=0.1199,是个投影永远用不上的死参数。M4 定义为 8 参数(α=0),OpenCVSolver 加 FIX_SKEW 使模型两侧严格一致;合成数据上 flags=2 与 flags=10 拟合质量相同(rms 均 0.0275)。[VERIFIED: 本地实测]

**OpenCVSolver 的通用初始化路径(D-09 兜底):** 模型无适配器时,OpenCVSolver 仍可用 OpenCV 原语出初值:`default_params(image_size)` 组 K、畸变清零 → 逐视图 `cv2.solvePnP` 估位姿(实测接受 4/5/8 长度畸变向量;鱼眼另有 `cv2.fisheye.solvePnP`,返回 (retval, rvec, tvec),要求 (N,1,x) 形状)→ 拼成"初值版 CalibrationResult"。这让玩具模型也能被两个求解器使用,而无需动求解器一行代码。

### Pattern 2: 双求解器与统一 CalibrationResult

**ObservationSet(推荐字段,最小完备):**

```python
@dataclass(frozen=True)
class ObservationSet:
    image_ids: tuple[str, ...]      # (V,) join 键(D-04 对 manifest、产物落盘都用)
    pose_ids: tuple[str, ...]       # (V,) 阶段四按姿态切分的连接键,现在就留
    object_points: np.ndarray       # (P,3) float64;复用 baseline.build_object_points(唯一来源)
    image_points: np.ndarray        # (V,P,2) float64;本阶段全部 88 点检出(有限值断言)
    image_size: tuple[int, int]     # (宽, 高)
    corner_source: tuple[str, ...]  # (V,) "sb"/"classic_fallback" 留痕(40 张兜底帧可单独查)
```

装载 = 读 sb 路线 corners npz(npz 行序与 manifest 逐行对齐,阶段一约定)→ 剔除清单(exclude_images)与未检出帧过滤(沿用阶段一规则,≥2 视图保护)→ 有限值断言。**行序契约:ObservationSet 顺序 = CalibrationResult 里 rvecs/tvecs/per_view_rms 顺序 = 落盘 image_ids 顺序**,全链一处约定。

**CalibrationResult(推荐字段,失败契约照阶段一):**

```
converged: bool                     # False 时下列数值字段一律 None、结构键一个不少
failure_note: str | None            # 中文
model_id / solver: str              # "opencv" | "scipy"(精化路径记 "scipy")
solver_meta: dict                   # OpenCV: flags_used;scipy: method/loss/f_scale/status/nfev/njev/cost/optimality
params: dict[str, float]            # 按 param_names 逐名给出(报告/下游按名取,不猜位置)
rvecs: (V,3) / tvecs: (V,3) float64
rms_overall: float | None
per_view_rms: (V,) | None           # 统一用 models.project 回代(两后端同一把尺, 见 Pitfall 9)
image_ids / image_size / n_views / n_points_per_view
init_from: str | None               # scipy 精化指向 OpenCV 初值产物路径(血缘的一部分)
lineage: dict                       # make_lineage 产物
```

落盘三件套:`result.json`(全部标量+params+lineage+逐帧 per_view_rms 字典)、`extrinsics.npz`(rvecs/tvecs,给阶段三线性化点与阶段四复评用)、`residuals.npz`(逐帧 (V,88,2) 残差,models.project 算出)。目录名 `<model>__<solver>`(如 `m2__opencv`、`m2__scipy_softl1`),阶段一 baseline 目录原样保留。

**ScipySolver 变量组织(Claude's Discretion 的研究结论):**

- **联合单向量** θ = [内参 (P,) | 视图 0 的 (ω,t) 6 个 | 视图 1 的 6 个 | …],总长 M2 时 9+3840=3849。**不做分块交替**(自研优化器味道浓、收敛性质更差、收益为零——实测联合优化 15 次雅可比迭代即收敛)。[VERIFIED: 本地实测]
- 残差向量排序约定:按视图、按点、每点 (u,v) 两行交错(与雅可比行严格对齐,写进 residuals.py 文档字符串)。
- 求解器配置(实测可用组合):`least_squares(fun, θ0, jac=解析稀疏 callable, jac_sparsity=0/1 结构, method='trf', tr_solver='lsmr', x_scale='jac', loss=cfg, f_scale=cfg, max_nfev≥300, ftol/xtol/gtol 从配置)`。[VERIFIED: 本地实测 scipy 1.18.1]
- **初值装箱**:从 OpenCVSolver 的 CalibrationResult 读 params + 逐视图 rvecs/tvecs(视图顺序一致,直接拼)。无初值产物时:`default_params` + 逐视图 solvePnP 估位姿(通用路径)。
- **鲁棒核协议(实测修正)**:linear 核 15 次迭代正式收敛(status=3,13 秒);soft_l1/huber 即使从近收敛初值出发,300 次迭代仍 status=0(代价平台、RMS 已到噪声水平、fx 误差 0.1px 量级)。协议定为:**鲁棒核是对照路径 + 离群点侦察兵**(与项目级研究 PITFALLS #9 两段式一致),`converged` 的判定用"实用收敛":status>0,或 status==0 且末段代价相对改进 < 1e-6 且 RMS 不劣于 linear 路径——判定逻辑写成一个纯函数并单测,判定依据全部进 solver_meta,不静默。
- f_scale 建议默认"auto":从初值残差的鲁棒尺度推(f_scale = 3 × 1.4826 × MAD,MAD 即中位数绝对差,用中位数衡量的波动大小),配置可写死数值覆盖。本数据量级:鱼眼 RMS≈0.083px → f_scale≈0.25。

### Pattern 3: 血缘与独立重跑(ITER-01)

零新机制:新产物一律 `make_lineage(config_files={board.yaml, calibration.yaml}, upstream={角点 npz(以及 scipy 时的 OpenCV 初值 result.json)}, round_id)`。独立重跑 = 新进程 `calib calibrate`:找最新 corners 轮次、读配置、跑矩阵、写新 run_<时间戳> 目录——与阶段一 `calibrate` 完全同构,血缘里 upstream 哈希保证"算的就是那份输入"。集成测试直接复跑一次断言产物血缘完整、结果可复现(OpenCV 单线程确定性 + scipy 确定性 → 数值应逐位一致或 <1e-12)。

### Pattern 4: `calib calibrate` 命令演进(Claude's Discretion)

- **D-12 决定了形态**:CLI 保持零参数,矩阵进 calibration.yaml(示例见下),命令只做"读配置 → 装载 → 跑矩阵 → 落盘 → 中文汇总表"。
- **向后兼容的准确含义**:阶段一 baseline.py 模块与其产物目录原样保留可复现(它有自己的测试);新代码不 import 它、不改它。comparison.json 的格式随新矩阵改变——落在新的 run 目录里,不覆盖旧产物,"向后可读"即旧 JSON 仍可打开对账。
- 配置扩展草案(沿用 io/config.py 的冻结 dataclass + 中文校验模式):

```yaml
calibrate:
  corner_route: "sb"          # D-02:SB 为主 + classic 兜底那份
  models: [m1, m2, m3, m4]    # 全矩阵(D-01)
  solvers: [opencv, scipy]    # scipy = OpenCV 初值 → scipy 精化
  scipy:
    method: trf
    losses: [linear, soft_l1] # linear 主线 + 鲁棒核对照(成功标准 3 要含鲁棒核)
    f_scale: auto              # auto = 3×1.4826×MAD(初值残差);也可写死数值
    max_nfev: 300
    ftol: 1.0e-10
    xtol: 1.0e-12
  fisheye:                    # 既有段保留;M4 的 flags 默认演进为 10(RECOMPUTE|FIX_SKEW)
    flags: 10
    max_iter: 100
    epsilon: 1.0e-6
```

(model_id 命名取短名 m1–m4 与全部规划文档一致;注册表里带中文 description。)

### Pattern 5: D-04 存疑帧复核的实现载体

- **数据**:精化结果(推荐用 M4 或最优先模型的多路径之并;规划时定,建议全部收敛组合都算,汇总取最优组合看结论是否稳健)逐帧残差;manifest 的 `precheck.verdict` + `suspect_sources` 按 image_id join。
- **覆盖 705 张**(89 存疑 = train 67 + test 22;616 正常 = train 573 + test 43):train 帧残差直接来自标定结果(位姿已知);**test 帧先做"只估位姿"的小求解**——每视图 6 参数、内参固定、残差同样只用 models.project(`solve/scoring.py`,对任何注册模型成立,玩具模型也不例外),65 张 × 毫秒级 = 可忽略开销。这也顺手给阶段四 held-out 评分打了地基。
- **产物**:`output/calibration/run_*/suspect_review/suspect_vs_normal.json`(逐帧表:image_id/pose_id/split/suspect/来源记号/per_view_rms + 两组分位数与中位数对比 + 中文结论)+ `suspect_vs_normal.png`(两组残差分布叠加图,matplotlib,visualization 层出图)。血缘 upstream = 标定 result.json + manifest.json。
- **结论两种走向的措辞模板**照 CONTEXT D-04 原文;另回写 `01-HUMAN-UAT.md` 的 Gap 节(文档编辑任务,一项)。

### Anti-Patterns to Avoid

- **在 solve/ 里抄一遍畸变公式**——哪怕一行。唯一数学源靠 import 结构 + parity 测试双保险(D-06/D-08)。
- **把 OpenCV 返回的 per_view 直接当跨后端可比指标**——OpenCV 求解吃 float32 角点(绑定层限制,实测拒收 float64),其内部 RMS 与你 float64 回代的 RMS 差在 1e-4px 量级;跨后端对比一律用 models.project 回代重算。
- **稠密雅可比**——640 视图稠密等价 3.47 GB;任何"先稠密跑通再优化"的路径在真实规模直接不可行。
- **求解器 import 具体模型**——破坏 D-09;求解器只见 CameraModel 协议 + create_model。
- **用 float32 调 cv2.projectPoints 再比对/取雅可比**——投影差 ~1e-4px、雅可比完全错误(实测 60);OpenCV 侧凡可比、可取雅可比的调用一律 float64。

---

## Don't Hand-Roll

| 问题 | 不要自建 | 用现成的 | 为什么 |
|------|----------|----------|--------|
| 非线性最小二乘优化器(含鲁棒核) | 自写 LM/高斯牛顿/分块交替 | scipy `least_squares`(trf+lsmr+loss) | 实测 15 次迭代收敛;鲁棒核、置信区间原料(res.jac)都是现成的 |
| 稀疏矩阵结构与求解 | 手写稀疏索引/lsmr | scipy.sparse + tr_solver='lsmr' | jac_sparsity 由 scipy 自动分组/校验;lsmr 是其标配内部求解器 |
| 旋转的指数映射(Rodrigues) | 自写 Rodrigues 正反变换 | cv2.Rodrigues(或经实测验证的三行公式) | rvec 语义与 OpenCV 全兼容;扰动导数 ∂Xc/∂ω=−[Xc]× 已实测匹配 |
| 物点网格 | 再写一遍 | baseline.build_object_points | square_size 只进一处的解耦断言已存在(阶段一测试) |
| 数值雅可比(生产路径) | 用 numdifftools 参与求解 | 只作测试基准(D-10) | 解析雅可比已验证到 6.7e-10;数值既慢又破坏"单一数学源"叙事 |
| 血缘指纹 | 新格式 | util/lineage.make_lineage | 阶段一已验证,产物间对账靠它 |

**Key insight:** 本阶段最容易"手痒自建"的是优化器和稀疏结构;实测数据表明 scipy 原生路径在真实规模 13~20 秒收敛,自建没有任何收益空间,只有引入数值 bug 的空间。

---

## Runtime State Inventory

本阶段是**新增模块**而非重命名/迁移,五类逐项回答:

| 类别 | 发现 | 需要的动作 |
|------|------|------------|
| 落盘数据 | 阶段一 output/ 产物(baseline 标定、角点)全部只读复用 | 无(slopcheck 侧产出的系统级 numdifftools 安装是 ~/.local 的副作用,与项目 venv 无关,不必清理;项目内用 uv 加 dev 依赖) |
| 活服务配置 | 无任何外部服务(离线工具) | 无 |
| 操作系统注册状态 | 无 | 无 |
| 秘密/环境变量 | 无涉及 | 无 |
| 构建产物 | `calib calibrate` 输出目录格式演进(comparison.json 结构变化) | 新旧目录按时间戳隔离,不迁移;baseline.py 保留即可复现旧结果 |

---

## Common Pitfalls

### Pitfall 1: OpenCV 绑定层的 dtype 陷阱(三处,全部实测)

**What goes wrong:**
1. `calibrateCameraExtended` 拒收 float64(Unsupported format)——必须 float32 进、float64 出(阶段一已发现,本研究再次确认);
2. `cv2.projectPoints` 接受 float64,**但用 float32 输入时它返回的雅可比是完全错的**(对中心差分误差 60 vs float64 的 2e-7);
3. `fisheye` 系函数要求 (N,1,x) 形状,传 (N,x) 报难懂断言(阶段一 `_prepare_view` 已防,新代码沿用)。

**How to avoid:** OpenCV 适配器里把 dtype/形状规则写成唯一入口(装包/解包函数各一),parity 测试与求解器共用;凡取 cv2 雅可比或做比对,强制 float64。

**Warning signs:** parity 差 ~1e-4px 而不是 ~1e-13px(说明走了 float32);雅可比数值校验爆炸。

### Pitfall 2: OpenCV 畸变向量长度因 flags 而变

**What goes wrong:** M3(RATIONAL)返回 14 个系数(尾部 6 个是未启用的 thin-prism/tilted 恒 0),直接当 8 个用会错位;M1(FIX_K3)返回 5 个、k3=0,取前 4 才是 M1。

**How to avoid:** 解包在模型适配器内做切片(D[:8]/D[:4]/D[:5]),配"装包→解包往返一致性"单测(给定 params → cv2 布局 → 解回 params,逐位相等)。

### Pitfall 3: 鲁棒核的形式收敛假阴性

**What goes wrong:** soft_l1/huber 在 trf+lsmr 下即使初值很好,status 也常停在 0(max_nfev 用尽),但 RMS 已达噪声水平——若把 status==0 一律当失败,会误杀本该"通过成功标准 3"的运行;反过来若无判据地当成功,又会掩盖真不收敛。

**How to avoid:** "实用收敛"纯函数判定(status>0 或 末段代价改进<1e-6 且 RMS 不劣于 linear 对照),判定依据全部进 solver_meta;中文 failure_note 如实写"迭代预算用尽但代价已平台"。

**Warning signs:** status=0 且 cost 仍在明显下降(真不收敛,提高 max_nfev 或换初值)。

### Pitfall 4: KB4 在光轴附近的除零

**What goes wrong:** scale=θd/r 在 r→0 时是 0/0(解析极限 = 1/z);真实数据可能存在非常靠近光轴的角点,未加保护会出 NaN 并毒化整个求解。

**How to avoid:** `np.where(r < 1e-12, 1.0/z, theta_d/r)`(极限值实测推导);parity/雅可比测试显式包含近轴点(r≈1e-6)。

### Pitfall 5: 雅可比组装的 Python 循环

**What goes wrong:** 逐点 Python 循环组装(640×88×30 次追加)单次雅可比 ~1.2 秒,上百次迭代直接不可用。

**How to avoid:** 视图内全 numpy、按视图循环、COO 三元组向量化拼接(本研究实测 0.153 秒/次,快 8 倍;残差 0.021 秒/次)。

### Pitfall 6: 残差/雅可比行序不一致

**What goes wrong:** 残差按 (视图,点,u/v) 排、雅可比按别的顺序排——least_squares 得到错误梯度,收敛奇慢或不动,且无报错。

**How to avoid:** 残差与雅可比共用同一"行索引计算"函数/常量;测试:对随机 θ,数值雅可比(2-point)与解析稀疏雅可比在抽样行上逐项一致(本研究验证脚本即此模式)。

### Pitfall 7: M4 skew 的两侧不一致

**What goes wrong:** 阶段一 fisheye 配置估 skew(0.1199),而 projectPoints 忽略 K[0][1](实测)——若 M4 带 skew 参数而投影不用它,参数向量与投影数学脱节。

**How to avoid:** M4 定死 8 参数(α=0),OpenCVSolver M4 用 flags=10(FIX_SKEW);配置默认值同步改,产物里记 flags_used 对账。

### Pitfall 8: 混淆"成功标准 3"的比较对象

**What goes wrong:** "scipy 精化不劣于纯 OpenCV"是**同模型**内比较(M2_scipy vs M2_opencv),不是 M1 vs M4;本数据 M2 本身 RMS≈1.24px(镜头强鱼眼),M4≈0.083px——若误读成"精化后 M1/M2 要达到 M4 水平"会错误判失败。

**How to avoid:** 汇总表按模型分组呈现精化前后对比;阶段文档里写明白话注释。M1/M2/M3 在非鱼眼广角上精化后 RMS 仍高,属预期内的模型失配发现,按失败契约/发现记录,不中断。

### Pitfall 9: 跨后端指标不同尺

**What goes wrong:** OpenCV 内部 RMS(基于 float32 输入)与 models.project 回代 RMS 差 ~1e-4px;如果"不劣于"判定一边用 OpenCV 的数一边用回代的数,判定本身被精度噪声污染(量级恰好和 1e-4 相当时)。

**How to avoid:** CalibrationResult 的 rms_overall/per_view_rsi 一律 models.project 回代(float64、同一实现),OpenCV 返回 RMS 只进 solver_meta 作参考。

### Pitfall 10: 装载时行序漂移

**What goes wrong:** ObservationSet 若按检出掩码过滤后不保留 manifest 顺序信息,D-04 的 join 与阶段四切分会错行。

**How to avoid:** image_ids 是一等字段、全程携带;装载函数断言与 npz 的 image_ids 逐项相等(阶段一 corners npz 行序=manifest 序,直接成立)。

### Pitfall 11: 把 parity 测试写成"同参数同输出"的弱断言

**What goes wrong:** 只测一组典型参数、只测画面中心点——有理分母嵌套、近轴、大入射角的错误全漏掉。

**How to avoid:** 容差矩阵 + 参数采样策略(见 Validation Architecture):典型/边界/随机种子≥3,点覆盖近轴到归一化半径 ~1.2、KB4 到 θ≈1.3 rad。

### Pitfall 12: 试车耗时低估

**What goes wrong:** 按"单次求解秒级"排计划(阶段一 600 视图量级未记录耗时,易误推);实测 640 视图 calibrateCamera ~105~115 秒/次。

**How to avoid:** 全矩阵预算按 15~25 分钟排;OpenCV 调用前 `cv2.setNumThreads(cfg.cv2_threads)`(既有 D-12 约定);若某组合异常拖长(>10 分钟)视为发现记录,不静默等待。

---

## Code Examples(全部公式经本地数值验证)

### M1/M2/M3 投影(针孔 Brown-Conrady,含 M3 有理分母;float64 下与 cv2.projectPoints 偏差 ≤4.5e-13px)

```python
# 来源:本研究对 cv2 4.14.0 的实测对齐(公式与 OpenCV 官方文档一致,嵌套方式经数值判定)
def project_brown(pts_cam, fx, fy, cx, cy, dist):
    """dist: 4/5/8 个系数,布局 [k1,k2,p1,p2,(k3),(k4,k5,k6)]。
    关键嵌套:切向项加在有理分母之外(先除分母,再加切向)。"""
    x = pts_cam[:, 0] / pts_cam[:, 2]
    y = pts_cam[:, 1] / pts_cam[:, 2]
    r2 = x * x + y * y
    if len(dist) == 8:  # M3
        num = 1 + dist[0]*r2 + dist[1]*r2**2 + dist[4]*r2**3
        den = 1 + dist[5]*r2 + dist[6]*r2**2 + dist[7]*r2**3
        radial = num / den
    else:               # M1(4)/M2(5)
        k3 = dist[4] if len(dist) == 5 else 0.0
        radial = 1 + dist[0]*r2 + dist[1]*r2**2 + k3*r2**3
    xd = x*radial + 2*dist[2]*x*y + dist[3]*(r2 + 2*x*x)   # 切向在分母外
    yd = y*radial + dist[2]*(r2 + 2*y*y) + 2*dist[3]*x*y
    return np.stack([fx*xd + cx, fy*yd + cy], axis=1)
```

### M4 投影(KB4 等距;float64 下与 cv2.fisheye.projectPoints 偏差 ≤4.5e-13px)

```python
def project_kb4(pts_cam, fx, fy, cx, cy, k):
    """θ=入射角;θd=畸变后角;近轴保护:极限 scale→1/z。"""
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    r = np.hypot(x, y)
    theta = np.arctan2(r, z)
    t2 = theta * theta
    theta_d = theta * (1 + k[0]*t2 + k[1]*t2**2 + k[2]*t2**3 + k[3]*t2**4)
    scale = np.where(r < 1e-12, 1.0 / z, theta_d / r)
    return np.stack([fx * x * scale + cx, fy * y * scale + cy], axis=1)
```

### 解析雅可比的三个组成块(M2 代表;对中心差分最大偏差 6.7e-10)

```python
# 来源:本研究验证脚本(块 1/2/3 组合后逐列对中心差分验证)
# 块 1:畸变核对 (xn,yn) 与对 k 的导数(M2;M3 的 radial 用商法则扩展,见下)
#   drad/d(r2) = k1 + 2*k2*r2 + 3*k3*r2**2          (非有理)
#   有理时:rad=num/den, ∂rad/∂k1=r2/den, ∂rad/∂k4=-num*r2/den², ∂rad/∂(r2)=(num'−rad·den')/den
#     其中 num'=k1+2k2·r2+3k3·r4, den'=k4+2k5·r2+3k6·r4
# 块 2:归一化坐标对相机坐标点导数
#   ∂xn/∂Xc = (1/z, 0, -xn/z);∂yn/∂Xc = (0, 1/z, -yn/z)
# 块 3:相机坐标点对外参导数(左扰动;与 cv2.Rodrigues 语义一致,实测匹配)
#   ∂Xc/∂ω = -[Xc]×(反对称矩阵),∂Xc/∂t = I
# 链式:d_uv/d_外参 = d_uv/d_Xc · ∂Xc/∂(ω,t);d_uv/d_Xc = fx·(∂xd/∂(xn,yn)·∂(xn,yn)/∂Xc) 等
```

### 稀疏雅可比组装(向量化模式;0.15 秒/次 @ 640 视图)

```python
# 结构事实(实测):每残差行仅碰 P 个内参 + 本视图 6 外参 = 15 列(M2),
# nnz = 2·V·P·15 ≈ 169 万;稠密等价 3.47 GB → 必须 scipy.sparse。
# 组装:视图内全 numpy 算 (P,2,15) 块 → cols 模板预先广播 → csr_matrix((vals.ravel(), (rows, cols.ravel())))
# least_squares 调用(实测可用):
least_squares(residuals, theta0, jac=jacobian_sparse, jac_sparsity=sparsity01,
              method="trf", tr_solver="lsmr", x_scale="jac",
              loss="soft_l1", f_scale=0.15, max_nfev=300)
```

### parity 测试骨架(容差矩阵)

```python
# 典型参数(本数据:fx≈1872/2560px 宽)、边界参数(k4~k6≠0、p1p2≠0、大 θ)、
# 点集:随机 1000 点 + 近轴点(r≈1e-6)+ 大半径(归一化 r≈1.2;KB4 θ≈1.3rad),种子≥3
ours = model.project(pts_cam, params)
cv_ref = cv2.projectPoints(pts_cam.reshape(-1,1,3), np.zeros(3), np.zeros(3),
                           K_from(params), D_from(params))[0].reshape(-1, 2)  # 一律 float64
assert np.abs(ours - cv_ref).max() < 1e-9   # 实测余量 3 个数量级以上(实测值 ~5e-13)
```

### 数值雅可比测试(numdifftools 基准)

```python
import numdifftools as nd          # dev 依赖,只进测试(D-10)
J_ana = model.project_jacobian(pts, params)          # (N,2,P)
J_num = nd.Jacobian(lambda p: model.project(pts, p).ravel())(params)  # (2N, P)
# 判定:逐参数列相对误差 < 1e-6(中心差分实测地板 ~7e-10,余量充足)
for j in range(P):
    assert np.linalg.norm(J_ana.reshape(-1,P)[:,j] - J_num[:,j]) / max(np.linalg.norm(J_num[:,j]), 1e-12) < 1e-6
```

---

## State of the Art

| 旧做法 | 现状 | 变化时点 | 对本阶段影响 |
|--------|------|----------|--------------|
| OpenCV 5.0 | 2026-08 发布,项目锁 4.14.0 | 5.0.0 2026-08-19 | 无(锁定;适配器把 OpenCV 调用隔离在一处,未来迁移审计面小) |
| scipy lm 为默认心智 | 640 视图规模 lm 不可用(稠密限定) | 长期 | 必须 trf+lsmr+jac_sparsity(实测) |
| numdifftools | 0.11.1 当前发布 | 2026(镜像近期) | dev 组直装即可 |

**过时认知清单:** "单次标定秒级"(实测 ~2 分钟/次 OpenCV、13~115 秒 scipy);"fisheye projectPoints 支持 skew"(4.14.0 实测忽略 K[0][1]);"OpenCV 返回的 per_view 可直接跨后端比"(float32 输入污染)。

---

## Assumptions Log

| # | 假设 | 所在章节 | 若错误的后果 |
|---|------|----------|--------------|
| A1 | 真实数据 640 视图的 OpenCV 求解耗时与合成数据同量级(~100~120 秒/次;合成实测 M1 104s/M2 107s/M3 115s/鱼眼 27~32s) | Summary / Pitfall 12 | 试车矩阵总时长偏离 15~25 分钟估计(±2×);不影响正确性 |
| A2 | f_scale=auto(3×1.4826×MAD)在本数据上落在 0.1~0.5 区间,鲁棒核能正常工作 | Pattern 2 | 鲁棒核对照路径 status=0 更普遍;实用收敛判定兜底,结论仍成立 |
| A3 | D-04 对比图用 matplotlib 直方图/累积分布叠加即可(无交互需求) | Pattern 5 | 换图型,工作量不变 |
| A4 | scipy/OpenCV 在固定输入下确定性可复现(单线程 cv2_threads=0;scipy 无随机源) | Pattern 3 | 独立重跑测试改为容差断言(1e-9)而非逐位相等 |
| A5 | model_id 短名 m1–m4(规划文档一致);目录名 `<model>__<solver>` | Pattern 2/4 | 纯命名,改名零成本 |

> A1~A5 全部低风险;无需用户确认即可开工,规划时如对命名有偏好可顺手定夺。

---

## Open Questions

1. **D-04 用哪个组合的残差做主对比**
   - 已知:矩阵会有 4 模型 × 2+ 路径多个收敛组合;鱼眼 M4 拟合最好(RMS 0.083px vs 针孔 1.24px)
   - 不明确:主结论用"最优组合"还是"全部收敛组合各算一遍看稳健性"
   - 建议:全部收敛组合都算(便宜),汇总表以最优组合为主结论、并排展示结论一致性;规划时一句话定死
2. **scipy 精化的损失对照集**(linear + soft_l1 是否加 huber)
   - 已知:成功标准 3 只要求"含鲁棒核"且 M1/M2 收敛;soft_l1 已够
   - 建议:默认 {linear, soft_l1},huber 留给阶段四 SOL-02;配置可扩展
3. **成功标准 1 的"数值雅可比测试对每个模型每个参数通过"的容差表达**
   - 已知:中心差分地板 7e-10,实测余量大
   - 建议:列相对误差 <1e-6(numdifftools 理查森外推更准,余量更大);规划进 VALIDATION.md 时按此写
4. **unproject 的用途边界**
   - 已知:MOD-01 要求接口存在;本阶段求解/评分都用不到(位姿估计走投影残差)
   - 建议:实现(两族各 ~5 行牛顿迭代)+ 基本正确性测试(往返投影),不深究边界(畸变外推风险属阶段五)

---

## Environment Availability

| 依赖 | 被谁需要 | 可用 | 版本 | 兜底 |
|------|----------|------|------|------|
| Python | 全部 | ✓ | 3.12.13(uv venv) | — |
| opencv-python | parity/OpenCVSolver | ✓ | 4.14.0(锁 <5.0) | — |
| numpy / scipy | models/solve | ✓ | 2.5.4 / 1.18.1 | — |
| matplotlib | D-04 图 | ✓ | 3.11.2 | — |
| numdifftools | 雅可比测试 | ✓(待加 dev 组) | 0.11.1(registry 已解析) | 中心差分手写兜底(测试内 ~10 行) |
| 真实数据角点(sb 路线) | 试车矩阵 | ✓ | output/corners/run_20261017_104817(640 train 全检出 = 600 sb + 40 classic_fallback;65 test) | — |
| ingest manifest(D-04 join) | 存疑复核 | ✓ | output/ingest/run_20261016_085253(705 张:train 573 ok + 67 suspect;test 43 ok + 22 suspect) | — |

**缺失且无兜底:** 无。

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest ≥8(pyproject 已配 testpaths=["tests"], integration 标记已定义) |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest -m "not integration"`(现有 106 测试 ~秒级) |
| Full suite command | `uv run pytest`(含真实数据 integration) |

### Phase Requirements → Test Map(六个验证维度)

| 维度 | 覆盖需求 | 测试类型 | 文件 | 命令(<30s,除 integration) | 已存在? |
|------|----------|----------|------|------------------------------|----------|
| ① parity 容差矩阵 | MOD-02 | unit | tests/test_models_parity.py | `uv run pytest tests/test_models_parity.py -x` | ❌ Wave 0(M1-M3 vs projectPoints / M4 vs fisheye.projectPoints;float64 atol 1e-9px;float32 路径 sanity atol 1e-3 锁 dtype 行为;参数采样:典型/边界/随机≥3 种子,点含近轴与大半径) |
| ② 雅可比数值校验 | MOD-02 | unit | tests/test_models_jacobian.py | `uv run pytest tests/test_models_jacobian.py -x` | ❌ Wave 0(numdifftools 逐模型逐参数,列相对误差<1e-6;M1-M3 另加 cv2 雅可比 float64 辅助交叉验证;含近轴点) |
| ③ 双后端收敛对比(合成真值) | SOL-01 | unit | tests/test_solve_solvers.py | `uv run pytest tests/test_solve_solvers.py -x` | ❌ Wave 0(合成棋盘真值:OpenCVSolver 恢复内参<1%;scipy 精化不劣于初值;两后端 CalibrationResult 同构断言;失败契约:键齐全/数值 null/中文 note) |
| ④ scipy 残差单一来源锁定 | SOL-01/D-08 | unit | tests/test_solve_scipy_residuals.py | `uv run pytest tests/test_solve_scipy_residuals.py -x` | ❌ Wave 0(残差函数仅引用 models.project:patch model.project 加已知偏移,断言残差逐项同移;另静态断言模块 import 图) |
| ⑤ 注册零侵入 | MOD-01/D-09 | unit | tests/test_models_registry.py | `uv run pytest tests/test_models_registry.py -x` | ❌ Wave 0(玩具模型类定义在测试文件内,一处注册,双求解器(OpenCV 通用初始化路径 + scipy)都能出结果) |
| ⑥ 真实数据端到端 + 血缘可重跑 | SOL-01/ITER-01 | integration | tests/test_calibrate_matrix.py | `uv run pytest tests/test_calibrate_matrix.py -x`(标记 integration) | ❌ Wave 0(真实 sb 角点跑全矩阵:产物齐全带血缘、upstream 哈希正确;M1/M2 精化不劣于纯 OpenCV(同模型比较);新进程重跑一次结果可复现;存疑复核产物生成) |
| 装载契约 | SOL-01 | unit | tests/test_solve_observation.py | `uv run pytest tests/test_solve_observation.py -x` | ❌ Wave 0(行序对齐、剔除清单、≥2 视图保护、NaN 断言) |

### Sampling Rate

- **Per task commit:** `uv run pytest -m "not integration"`(全绿才提交)
- **Per wave merge:** `uv run pytest`(含 integration,真实数据端到端)
- **Phase gate:** 全量绿 + 成功标准 5 条逐条对照(④⑥ 直接对应标准 3/4;①② 标准 1;③ 标准 2;⑤ 标准 5)后才 `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `uv add --group dev numdifftools`(0.11.1)
- [ ] tests/test_models_parity.py / test_models_jacobian.py —— **必须先于任何求解代码合入**(D-06,建议 Wave 1 就位)
- [ ] tests/test_models_registry.py / test_solve_observation.py / test_solve_solvers.py / test_solve_scipy_residuals.py
- [ ] tests/test_calibrate_matrix.py(integration)
- [ ] 既有资产可复用:conftest.py `make_chessboard_image`(已知真值合成棋盘——恢复精度断言的标准答案来源)、test_solve_baseline.py 的组织范式

---

## Security Domain

> security_enforcement 启用(ASVS Level 1)。离线单机数值工具,无网络面、无认证面。

### Applicable ASVS Categories

| ASVS 类别 | 适用 | 本阶段控制 |
|-----------|------|------------|
| V2 认证 / V3 会话 / V4 访问控制 | 否 | 无用户/会话概念 |
| V5 输入校验 | 是 | 既有模式延续:配置经 io/config.py 冻结 dataclass + 中文校验;npz 读取强制 `allow_pickle=False`(防 pickle 任意代码执行,阶段一已有测试锁定);yaml safe_load |
| V6 密码学 | 否 | 无 secrets;sha256 仅作内容指纹非安全用途 |

### Known Threat Patterns(本项目现实面)

| 模式 | 类别 | 缓解 |
|------|------|------|
| 恶意 npz(pickle 载荷) | 执行 | allow_pickle=False 全链强制(新读盘代码照抄) |
| 产物写出项目根之外 | 篡改 | 全部写 output/<阶段>/run_<时间戳>/(既有约定) |
| 配置注入(yaml 反序列化) | 执行 | yaml.safe_load(既有) |

---

## Sources

### Primary(HIGH confidence)

- **本地实证(本研究,2026-10-17,项目 uv 环境)**:四模型投影 parity(float64 ≤4.5e-13px)、fisheye skew 忽略(max|Δ|=0.0 @skew=0.5)、解析雅可比 vs 中心差分(6.7e-10)、flags↔D 布局(M1 切 4/M2 切 5/M3 切 8 于 14 长度)、fisheye D=4 与 FIX_SKEW 行为、solvePnP 4/5/8 系数与 fisheye.solvePnP、cv2 雅可比 dtype 行为、scipy trf+lsmr+jac_sparsity 真实规模收敛(linear 15 njev/13s;soft_l1/huber status=0 但 RMS 达噪声水平)、耗时(calibrateCamera 105~115s、fisheye 27~32s、scipy 13~115s @640 视图)、稀疏规模(nnz 169 万/稠密等价 3.47GB)——验证脚本存于会话临时目录,关键公式已内嵌本文件"Code Examples"
- **OpenCV 4.x 官方文档**(docs.opencv.org/4.x calib3d 与 fisheye 页):畸变模型公式、flags 语义、KB4 θd 公式 —— [CITED: docs.opencv.org/4.x](与本地实测互证)
- **scipy 官方文档**(docs.scipy.org least_squares):trf/lsmr/jac_sparsity/loss 契约 —— [CITED](与本地实测互证)
- 项目内既有研究:.planning/research/ARCHITECTURE.md(models/solve 模块设计原型)、PITFALLS.md(robust 两段式、fisheye 陷阱、初始化/scaling)

### Secondary(MEDIUM confidence)

- 阶段一产物数值(sb fisheye RMS 0.083px、skew 0.1199、sb pinhole RMS 1.24px / fx 1872)——来自 output/calibration/run_20261016_090004(旧角点轮次,600 视图;供量级参考)

### Tertiary(LOW confidence)

- 无(本阶段结论均有本地实证或官方文档支撑)

---

## Metadata

**Confidence breakdown:**
- 模型公式与参数布局: HIGH —— 全部数值实测,公式可直接进实现与测试
- 求解器设计(稀疏/scipy 配置): HIGH —— 真实规模合成数据端到端验证收敛与耗时
- 试车矩阵真实数据表现: MEDIUM —— 合成验证充分,真实数据行为(尤其 M1/M2 精化幅度、鲁棒核 status)待执行期观察,失败契约兜底
- D-04 对比结论走向: LOW(本就该由数据说话,框架/载体设计为 HIGH)

**Research date:** 2026-10-17
**Valid until:** 2026-11-16(依赖锁定的 4.x/1.18 栈稳定;公式类结论长期有效)
