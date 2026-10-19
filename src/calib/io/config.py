"""配置解析与校验:board.yaml(棋盘规格)与 calibration.yaml(算法参数)的唯一入口。

硬性规则:
- 只用 yaml.safe_load(绝不 yaml.load——后者能执行配置里的任意代码,是安全漏洞);
- 解析失败必须给出带字段名与实际值的通俗中文错误,让人一看就知道改哪里;
- 下游永远拿到不可变数据类(dataclass),不接触原始字典——
  字段拼错在开发阶段就暴露,而不是算到一半才炸。

数据类字段名与 yaml 键一一对应,嵌套关系也一致。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# 经典检测器允许的选项名(写在 calibration.yaml 里的名字,代码内翻译成 cv2 常量)
_ALLOWED_CLASSIC_FLAGS = {"adaptive_thresh", "normalize_image", "fast_check"}

# 亚像素窗口粒度的合法取值:frame = 每帧一个窗口;corner = 逐角点局部自适应
_ALLOWED_GRANULARITY = {"frame", "corner"}

# calibrate 段的白名单(全部在本文件校验,CLI 零参数,D-12):
# 角点路线:标定默认吃 sb 路线(SB 检测为主 + classic 兜底那份,D-02)
_ALLOWED_CORNER_ROUTES = {"sb", "classic"}
# 模型短名与模型注册表的内置清单一致(02-01 已注册 m1–m4)
_ALLOWED_MODEL_IDS = {"m1", "m2", "m3", "m4"}
# 求解路径:opencv = OpenCV 原生;scipy = OpenCV 初值 → scipy 精化
_ALLOWED_SOLVERS = {"opencv", "scipy"}
# 损失核:linear 主线;soft_l1/huber 是鲁棒核(对照路径,压低大残差权重)
_ALLOWED_LOSSES = {"linear", "soft_l1", "huber"}
# scipy 优化方法:只允许 trf——'lm' 只支持稠密雅可比,640 视图等价 3.47 GB
# 内存,实测不可用(研究 §Pattern 2),配错直接在配置层拦下
_ALLOWED_SCIPY_METHODS = {"trf"}
# scipy 内部线性求解器:lsmr 是稀疏雅可比的标配搭配(trf 专属)
_ALLOWED_TR_SOLVERS = {"lsmr"}
# noise 段的白名单(第③步测噪声的全部参数,CLI 零参数,D-12):
# Immerkaer 估计(用固定 3×3 卷积核从单张照片估像素噪声)的主线口径:
# std = 整图标准式;robust = 稳健中位数式(JPEG 痕迹/纹理下更稳,默认主线)
_ALLOWED_IMMERKAER_VARIANTS = {"std", "robust"}
# 逐姿态 σ 汇成全局标量时的统计量:rms = 方均根(标量噪声假设下的自然聚合);
# median = 中位数(离群姿态更不敏感)
_ALLOWED_TEMPORAL_POOLING = {"rms", "median"}
# 噪声分析处理哪些数据划分:train 连拍 + test 独立照默认都算,逐张来源标注
_ALLOWED_NOISE_SPLITS = {"train", "test"}
# evaluate 段的鲁棒核白名单(阶段四优化器对比 D-14 的四核矩阵):
# 比 calibrate.scipy 的名单多 cauchy(Cauchy 核,对离群残差压制更狠)——
# 阶段四做全核对比,calibrate 段的主线配置不必跟进
_ALLOWED_EVAL_LOSSES = {"linear", "soft_l1", "huber", "cauchy"}


@dataclass(frozen=True)
class BoardSpec:
    """棋盘规格(来自 config/board.yaml)。

    square_size_mm 只影响外参平移的物理尺度,不影响内参——
    它绝不进入任何像素级的逻辑(角点窗口等用实测方格像素边长,不用它)。
    """

    cols: int = 11          # 内部角点列数
    rows: int = 8           # 内部角点行数
    square_size_mm: float = 25.0
    unit: str = "mm"


@dataclass(frozen=True)
class DataDirs:
    """数据目录(相对当前工作目录)。"""

    train: str = "input_data/train"
    test: str = "input_data/test"


@dataclass(frozen=True)
class PrecheckConfig:
    """入库体检的判定阈值。

    blur_z_threshold:同姿态连拍内模糊度稳健 z 分数的报警线;
    global_quantile_low/high:全局兜底分位线(单张姿态没有连拍可比时用)。
    """

    blur_z_threshold: float = 3.5
    global_quantile_low: float = 0.01
    global_quantile_high: float = 0.99


@dataclass(frozen=True)
class IngestConfig:
    """第①步入库体检的参数。"""

    precheck: PrecheckConfig = field(default_factory=PrecheckConfig)


@dataclass(frozen=True)
class SubpixConfig:
    """亚像素精化参数(把角点位置从整数像素细化到小数像素)。

    窗口从每帧实测的方格像素边长按比例推导,这里只控制粒度与下限;
    半窗口绝不允许大到跨过方格对边,否则会产生重投影误差里完全看不见的
    系统性定位偏差(本项目头号陷阱)。
    """

    window_granularity: str = "frame"
    min_win: int = 3
    max_iter: int = 30
    epsilon: float = 0.01


@dataclass(frozen=True)
class BurstCheckConfig:
    """同姿态连拍一致性断言的容差(以方格像素边长的倍数表达,不写死像素)。"""

    median_tol: float = 0.05
    max_tol: float = 0.3


@dataclass(frozen=True)
class CornersConfig:
    """第②步找角点参数:经典与新款(SB)双检测器并跑,各存一份。"""

    classic_flags: tuple[str, ...] = ("adaptive_thresh", "normalize_image")
    sb_flags: int = 0
    # SB 未检出的帧是否用经典检测器的结果兜底(填进 SB 路线的槽位)。
    # 三轮实验结论(见 corners 产物 review_sb_failed/DIAGNOSIS.md):SB 在大倾斜/
    # 远距视角下会系统性拒绝共 41 帧,而"缩小档救援 + 全分辨率精化"的产物与
    # classic 完全重合(偏差 0.01px 级)——兜底比多尺度重试更简单且多覆盖
    # 17 张穷举也救不回的帧,故默认开。false = 关闭,行为与只有 SB 时完全一致。
    sb_fallback_classic: bool = True
    subpix: SubpixConfig = field(default_factory=SubpixConfig)
    burst_check: BurstCheckConfig = field(default_factory=BurstCheckConfig)


@dataclass(frozen=True)
class FisheyeConfig:
    """鱼眼 KB4 模型的求解参数(仅供阶段一基线复现)。

    本段只被 solve/baseline.py 的 2×2 基线复现读取;阶段二的 M4 走模型
    注册机制,flags 由模型适配器自带(= 10 = 每轮重算位姿 + skew 固定 0),
    不再读这里——保留是为了"阶段一产物可独立重跑"的架构原则。

    flags 默认 2 = CALIB_RECOMPUTE_EXTRINSIC(每轮迭代都重算棋盘位姿):
    实测不加它时 OpenCV 只按初始内参算一次位姿,会收敛到 RMS 上百像素的
    假结果(哪怕数据完美符合 KB4);skew 倾斜因子仍被估计(未加 FIX_SKEW)。
    """

    flags: int = 2
    max_iter: int = 100
    epsilon: float = 1e-6


@dataclass(frozen=True)
class ScipyConfig:
    """scipy 精化路径的优化器参数(OpenCV 初值 → scipy least_squares 精化)。

    白话背景:非线性最小二乘 = 反复"按当前参数算残差(重投影位置与实测
    角点的差)→ 朝让残差变小的方向调参数",直到调不动为止。下面的字段
    控制"怎么调、什么时候停"。

    method 只允许 trf(信赖域反射法:每步只允许参数在当前解附近一个
    "可信范围"里动)。scipy 的 'lm' 只支持稠密雅可比(雅可比 = 残差对
    参数的偏导数矩阵;稠密 = 每个元素都存下来),640 视图的稠密雅可比
    约需 3.47 GB 内存,实测不可用——配错在配置层直接报错防呆。

    losses:要跑哪些损失核。linear 是主线(普通最小二乘,每个残差原样
    进代价);soft_l1 / huber 是鲁棒核(把大残差的权重压低,让个别离群
    角点不那么拖动整体),作对照路径——是否提升精度由数据说话,不预设。

    f_scale:鲁棒核的分界尺度,默认 "auto" = 3 × 1.4826 × MAD。
    MAD = 中位数绝对差,即用中位数衡量的初值残差波动;1.4826 是把它
    换算成标准差的惯例系数,再放大 3 倍 ≈ "残差大过波动 3 倍才开始被
    压低"。也可以写死一个 > 0 的数值覆盖。
    """

    method: str = "trf"
    tr_solver: str = "lsmr"
    losses: tuple[str, ...] = ("linear", "soft_l1")
    f_scale: str | float = "auto"
    max_nfev: int = 300
    ftol: float = 1e-10
    xtol: float = 1e-12
    gtol: float = 1e-10


@dataclass(frozen=True)
class CalibrateConfig:
    """第④步求解内参参数:模型矩阵 × 求解路径,全部在配置里(CLI 零参数,D-12)。"""

    # 角点输入路线(D-02):sb = SB 为主 + classic 兜底那份(默认);
    # classic = 仅经典检测器那份,留作对照
    corner_route: str = "sb"
    # 试车矩阵的模型短名(D-01 全矩阵默认跑齐四个)
    models: tuple[str, ...] = ("m1", "m2", "m3", "m4")
    # 求解路径:opencv 原生求解;scipy = OpenCV 结果做初值再精化
    solvers: tuple[str, ...] = ("opencv", "scipy")
    # 存疑帧复核(D-04)用哪个模型的精化结果做对照:固定一个模型全量
    # 对比,不逐模型重复(研究遗留问题 1 的定夺);默认 m2 = 阶段一针孔
    # 基线同款,与"体检看着糊的帧是不是真有害"这个问题直接衔接
    suspect_review_model: str = "m2"
    scipy: ScipyConfig = field(default_factory=ScipyConfig)
    fisheye: FisheyeConfig = field(default_factory=FisheyeConfig)


@dataclass(frozen=True)
class NoiseConfig:
    """第③步测噪声底的全部参数(时域连拍 + Immerkaer 单帧 + 注入实验)。

    白话背景:这一步回答"这批照片天生的噪声有多大",分三条线——
    同姿态连拍互减实测时域波动;单张照片用 Immerkaer 方法估计像素噪声;
    往真实照片上加已知强度的噪声重跑检测器,量出"像素噪声传到角点上
    被放大成多大的定位误差"(传递曲线)。下面的字段控制这三条线怎么跑。

    注入实验的合成加噪图不落盘(640 张 × 三十多个副本会有几十 GB),
    靠 (图像, σ, 随机种子) 三元组精确复现——所以种子必须落在配置里。
    """

    # Immerkaer 估计的主线口径:std = 整图标准式;robust = 稳健中位数式。
    # 默认 robust:JPEG 压缩痕迹与图像纹理会把 std 式抬高,中位数天然压低
    # 结构区域的权重(研究 Pitfall 5);两种估计值都会逐张落盘供对照
    immerkaer_variant: str = "robust"
    # 注入实验 σ 网格的下限/上限(单位:灰阶,即像素噪声的标准差)。
    # None = 自动锚定:下限取 max(0.25, 实测中位数σ/4)、上限取
    # min(128, 实测中位数σ×32),覆盖实测噪声邻域并向上延伸到检测器失效区;
    # 写死正数可覆盖自动锚定
    injection_sigma_min: float | None = None
    injection_sigma_max: float | None = None
    injection_n_levels: int = 8      # σ 对数网格档数(另加一个 σ=0 参照档,不计入本数)
    injection_n_repeats: int = 4     # 每档重复次数,压掉注入随机性(研究定 3–5)
    injection_seed: int = 20261018   # 随机种子:逐张逐档逐重复可精确复现的根
    # 并行进程上限:注入实验按图像切任务并行跑"加噪-检测";封顶 8 给系统留核
    # (总核时估算见 03-RESEARCH Pattern 2);1 = 不开子进程,当前进程顺序跑
    injection_max_workers: int = min(8, os.cpu_count() or 1)
    # 逐姿态 σ 汇成全局标量时用哪种统计量:rms = 方均根(标量噪声假设下
    # 的自然聚合,推荐);median = 中位数(个别离群姿态不拖总数)
    temporal_pooling: str = "rms"
    # 处理哪些数据划分:train 连拍与 test 独立照都算(研究开放问题 4 裁定),
    # 逐张来源标注;只想算训练集可改成 [train]
    noise_splits: tuple[str, ...] = ("train", "test")


@dataclass(frozen=True)
class FisherConfig:
    """第⑤步算理论精度极限(CRLB)的全部参数(CLI 零参数,D-12)。

    白话背景:这一步回答"这批数据天生最准能到多少"——在标定解的邻域
    把问题拉直(线性化),算 Fisher 信息矩阵(数据里"关于参数的信息量"),
    取逆开方得到每个内参标准差的理论下界(CRLB)。下面的字段控制
    算哪些模型、什么时候算"这个摆位有退化"、以及敏感性带开多宽。
    """

    # 算哪些模型的 CRLB:默认四个全跑(阶段二"全矩阵多存"风格,
    # 阶段四选型直接引用不回头补算,D-05)
    models: tuple[str, ...] = ("m1", "m2", "m3", "m4")
    # 条件数警戒线:条件数 = 信息矩阵"最软方向与最硬方向之比",越大说明
    # 有些参数组合越难分辨。达到警戒线记 warning(该组合方向上精度差)
    cond_warning: float = 1e4
    # 条件数退化线:达到它判 degenerate(该摆位基本定不住这些参数组合)
    # 校验要求 warning < degenerate(分档必须有序)
    cond_degenerate: float = 1e6
    # Σ 敏感性带的相对半宽:σ(角点噪声)由 8 张连拍估出时自带约 ±26.7%
    # 的相对不确定度(1/√14),CRLB 随之乘 (1±半宽) 成"带",不输出裸下界
    sigma_band_rel: float = 0.267
    # 是否做线性化点敏感性检查:在 OpenCV 解的点上重算一遍 CRLB,与
    # scipy 精化点的结果比相对差(预期 <1%;差异大 = 解未收敛透的信号)
    linearization_check: bool = True
    # 是否做 OpenCV stdDev 交叉验证(把我们的精度界与 OpenCV 自己报的
    # 标准差对账,两套数学互相印证);失败不阻塞 CRLB 主产物
    crosscheck_enabled: bool = True


@dataclass(frozen=True)
class EvaluateConfig:
    """第⑥步防作弊考试(交叉验证 / bootstrap / 重复张分组 / 留一姿态)的全部参数。

    白话背景:这一步回答"这批参数换了没见过的照片还准不准"——把 80 个姿态
    反复换着分组当"考题",每个相机模型都考一遍,再配合参数稳定性检查与
    理论精度下界对照,挑出当前数据下最合适的模型。下面的字段控制考试怎么
    出题(切分次数)、怎么判卷(判据阈值)、以及并行开多大。

    契约承诺:本阶段(04)全部实验计划只读本配置,键集合一次定死,
    后续计划不改不删——改契约必须回到本文件与 yaml evaluate 段同步改。
    """

    # 参选模型(D-11:四个全部参选,不因 Fisher 退化预警预否决——
    # 过拟合会在交叉验证的考题上自己显形)
    models: tuple[str, ...] = ("m1", "m2", "m3", "m4")
    # 优化器对比的鲁棒核名单(D-14:4 核 × 4 模型全矩阵;
    # linear 是主线,必须包含——它是"其他核有没有更好"的归一参照)
    optimizer_losses: tuple[str, ...] = ("linear", "soft_l1", "huber", "cauchy")
    # 交叉验证折数与重复轮数(D-01:5 折 × 5 轮,每姿态恰好 5 次进验证;
    # 折数 ≥ 2 是交叉验证的最低要求,1 折等于不考)
    cv_n_folds: int = 5
    cv_n_rounds: int = 5
    # bootstrap 重采样次数(D-04:按姿态有放回重采 200 次,4 模型共用同批)
    bootstrap_n: int = 200
    # 重复张分组实验开关(D-05:每姿态取第 i 张组成数据组,量纯噪声影响)
    slice_enabled: bool = True
    # 留一姿态实验开关(D-08:每次留 1 个姿态当考题,排查问题姿态)
    lopo_enabled: bool = True
    # 并行进程上限(实验按"一次完整求解"切任务;1 = 不开子进程,串行调试;
    # 参照 injection_max_workers 体例,默认 8 给系统留核)
    evaluate_max_workers: int = 8
    # 统计平局的等价带(D-09):两模型同折配对误差差落入
    # ±max(tie_band_px 像素, tie_band_rel × 较简模型均值)内判"统计平局",
    # 平局取更简模型。像素绝对带 > 0;相对带在 0 到 1 之间
    tie_band_px: float = 0.005
    tie_band_rel: float = 0.02
    # BIC 平局裁决器开关与有效姿态数(D-10:BIC = 惩罚参数数量的模型比较量,
    # 只在统计平局时启用;n 锁有效姿态数 80,不给 640——640 会把复杂模型
    # 的惩罚弱化约 8 倍,Phase 3 研究已锁死)
    bic_enabled: bool = True
    bic_n_poses: int = 80
    # 问题姿态的离群判据(D-08):留一姿态误差的稳健 z 分数超过它判"问题姿态",
    # 与入库体检的模糊度判据同源(3.5 ≈ 千分之二误报率的统计惯例)
    mad_z_threshold: float = 3.5
    # "实测好于理论下界"硬告警的两档显著性(D-06:实测不确定度低于 CRLB
    # 下界在物理上不可能,说明对照口径错了)。hard 达到判 FAIL + 告警,
    # soft 只提示边缘;两者都在 0 到 0.5 之间,且 hard < soft(告警线有序)
    crlb_alert_alpha_hard: float = 0.01
    crlb_alert_alpha_soft: float = 0.05
    # 泛化判据的执行位:折级"验证误差 / 训练误差"比值上限——比值超它判
    # 过拟合显形(训练好考题差)。宽松默认 1.5,判行表同时呈现绝对数值
    gen_overfit_ratio_max: float = 1.5
    # 稳定判据的执行位:参数扰动换算成典型视场投影位移的上限(像素当量)。
    # 宽松默认待真实数据校准(04-RESEARCH Open Q1),合法范围 > 0
    stability_disp_px_max: float = 0.05
    # 多起点一致性检验次数(D-15:从 OpenCV 初值加扰动重复求解,验"解唯一";
    # 鱼眼 m4 初值敏感,次数加倍)
    multistart_n: int = 10
    multistart_n_fisheye: int = 20
    # 多起点扰动幅度表(04-RESEARCH Pattern 6:内参相对扰动 ±2%、
    # 畸变相对 ±10%、旋转约 1°、平移相对 ±2%,全部 > 0)
    multistart_intr_rel: float = 0.02
    multistart_dist_rel: float = 0.10
    multistart_rot_deg: float = 1.0
    multistart_trans_rel: float = 0.02
    # 多起点收敛判据的双容差:重投影 RMS 差 < 0.001 px 且内参位移当量
    # < 0.1 px 才算"收敛到同一解"(双容差同时满足,单一容差容易被
    # 参数漂移糊弄过去)
    multistart_rms_tol_px: float = 0.001
    multistart_param_tol_px: float = 0.1


@dataclass(frozen=True)
class MosaicConfig:
    """汇总拼图的布局参数(拼图只是索引,不是逐帧副本的替代)。"""

    per_pose_sample: int = 1
    worst_n: int = 20
    thumb_width: int = 320
    cols: int = 8


@dataclass(frozen=True)
class VisualizationConfig:
    """可视化与副本留痕参数。"""

    jpeg_quality: int = 90
    mosaic: MosaicConfig = field(default_factory=MosaicConfig)


@dataclass(frozen=True)
class CalibrationConfig:
    """calibration.yaml 解析结果:全流程算法参数的唯一载体(D-12)。"""

    data: DataDirs = field(default_factory=DataDirs)
    seed: int = 42
    cv2_threads: int = 0
    exclude_images: tuple[str, ...] = ()
    ingest: IngestConfig = field(default_factory=IngestConfig)
    corners: CornersConfig = field(default_factory=CornersConfig)
    calibrate: CalibrateConfig = field(default_factory=CalibrateConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    fisher: FisherConfig = field(default_factory=FisherConfig)
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


def _fail(field_name: str, actual, requirement: str) -> None:
    """抛出带字段名与实际值的中文配置错误(统一格式,便于定位)。"""
    raise ValueError(
        f"配置字段 {field_name} 不合法:{requirement};实际读到的值是 {actual!r}。"
        f"请检查对应的 yaml 文件。"
    )


def _check_is_dict(raw, field_name: str) -> dict:
    """确认 yaml 里某一段是"键: 值"结构,否则给中文报错。"""
    if not isinstance(raw, dict):
        _fail(field_name, raw, "应当是一组缩进的配置项(字典)")
    return raw


def _read_yaml(path: Path | str) -> object:
    """读 yaml 文件,safe_load(禁 yaml.load,防配置执行任意代码)。

    yaml **语法**错误(如缩进用了 tab)也归入中文报错防线:转成带文件
    路径的 ValueError,消息附 PyYAML 原始信息(行号/列号对定位有用),
    并保留因果链(raise ... from)——PyYAML 自己抛的 ScannerError 不是
    ValueError 子类,不转的话两道防线(中文报错、_setup_runtime 的
    except (ValueError, OSError))都会被击穿。
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"配置文件不存在:{path}。请确认路径(相对当前工作目录)是否正确。")
    with open(path, encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(
                f"配置文件 {path} 的 yaml 语法有误,请检查缩进与冒号:{e}"
            ) from e


def load_board(path: Path | str) -> BoardSpec:
    """解析并校验棋盘规格 board.yaml,返回不可变的 BoardSpec。

    校验规则:board_type 必须是 chessboard(当前只支持棋盘格);
    cols / rows >= 3(再少无法构成棋盘);square_size_mm > 0。
    任何不合:抛 ValueError,消息含字段名与实际值。
    """
    raw = _read_yaml(path)
    if not isinstance(raw, dict) or "board" not in raw:
        _fail("board", raw, "文件顶层必须有 board: 这一段")
    board = _check_is_dict(raw["board"], "board")

    board_type = board.get("board_type")
    if board_type != "chessboard":
        _fail("board.board_type", board_type, "当前只支持 chessboard(棋盘格)")

    cols = board.get("cols")
    if not isinstance(cols, int) or isinstance(cols, bool) or cols < 3:
        _fail("board.cols", cols, "必须是 >= 3 的整数(内部角点列数)")

    rows = board.get("rows")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 3:
        _fail("board.rows", rows, "必须是 >= 3 的整数(内部角点行数)")

    square = board.get("square_size_mm")
    if not isinstance(square, int | float) or isinstance(square, bool) or square <= 0:
        _fail("board.square_size_mm", square, "必须是 > 0 的数(方格物理边长,毫米)")

    unit = board.get("unit", "mm")
    if not isinstance(unit, str) or not unit:
        _fail("board.unit", unit, "必须是长度单位的文字(如 mm)")

    return BoardSpec(cols=cols, rows=rows, square_size_mm=float(square), unit=unit)


def _parse_precheck(raw: dict) -> PrecheckConfig:
    section = _check_is_dict(raw.get("precheck", {}), "ingest.precheck")
    z = section.get("blur_z_threshold", PrecheckConfig.blur_z_threshold)
    if not isinstance(z, int | float) or isinstance(z, bool) or z <= 0:
        _fail("ingest.precheck.blur_z_threshold", z, "必须是 > 0 的数(稳健 z 分数报警线)")
    low = section.get("global_quantile_low", PrecheckConfig.global_quantile_low)
    high = section.get("global_quantile_high", PrecheckConfig.global_quantile_high)
    if not isinstance(low, int | float) or isinstance(low, bool) or not 0 < low < 1:
        _fail("ingest.precheck.global_quantile_low", low, "必须是 0 到 1 之间的小数(分位线)")
    if not isinstance(high, int | float) or isinstance(high, bool) or not 0 < high < 1:
        _fail("ingest.precheck.global_quantile_high", high, "必须是 0 到 1 之间的小数(分位线)")
    if low >= high:
        _fail("ingest.precheck.global_quantile_low", (low, high),
              f"下分位线必须小于上分位线(global_quantile_high = {high!r})")
    return PrecheckConfig(blur_z_threshold=float(z),
                          global_quantile_low=float(low),
                          global_quantile_high=float(high))


def _parse_subpix(raw: dict) -> SubpixConfig:
    cfg = SubpixConfig()
    section = _check_is_dict(raw.get("subpix", {}), "corners.subpix")
    granularity = section.get("window_granularity", cfg.window_granularity)
    if granularity not in _ALLOWED_GRANULARITY:
        _fail("corners.subpix.window_granularity", granularity,
              f"只能是 {' 或 '.join(sorted(_ALLOWED_GRANULARITY))}")
    min_win = section.get("min_win", cfg.min_win)
    if not isinstance(min_win, int) or isinstance(min_win, bool) or min_win < 1:
        _fail("corners.subpix.min_win", min_win, "必须是 >= 1 的整数(半窗口下限,像素)")
    max_iter = section.get("max_iter", cfg.max_iter)
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        _fail("corners.subpix.max_iter", max_iter, "必须是 >= 1 的整数(迭代次数上限)")
    epsilon = section.get("epsilon", cfg.epsilon)
    if not isinstance(epsilon, int | float) or isinstance(epsilon, bool) or epsilon <= 0:
        _fail("corners.subpix.epsilon", epsilon, "必须是 > 0 的数(收敛容差,像素)")
    return SubpixConfig(window_granularity=granularity, min_win=min_win,
                        max_iter=max_iter, epsilon=float(epsilon))


def _parse_burst_check(raw: dict) -> BurstCheckConfig:
    cfg = BurstCheckConfig()
    section = _check_is_dict(raw.get("burst_check", {}), "corners.burst_check")
    median_tol = section.get("median_tol", cfg.median_tol)
    max_tol = section.get("max_tol", cfg.max_tol)
    for name, value in (("median_tol", median_tol), ("max_tol", max_tol)):
        if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
            _fail(f"corners.burst_check.{name}", value, "必须是 > 0 的数(方格边长的倍数)")
    if median_tol > max_tol:
        _fail("corners.burst_check.median_tol", (median_tol, max_tol),
              f"中位数容差不应大于最大值容差(max_tol = {max_tol!r})")
    return BurstCheckConfig(median_tol=float(median_tol), max_tol=float(max_tol))


def _parse_corners(raw: dict) -> CornersConfig:
    section = _check_is_dict(raw.get("corners", {}), "corners")
    flags = section.get("classic_flags", CornersConfig.classic_flags)
    if not isinstance(flags, (list, tuple)) or not all(isinstance(f, str) for f in flags):
        _fail("corners.classic_flags", flags, "必须是一列选项名(如 [adaptive_thresh])")
    bad = [f for f in flags if f not in _ALLOWED_CLASSIC_FLAGS]
    if bad:
        _fail("corners.classic_flags", bad,
              f"含有不认识的选项;可选:{sorted(_ALLOWED_CLASSIC_FLAGS)}")
    sb_flags = section.get("sb_flags", CornersConfig.sb_flags)
    if not isinstance(sb_flags, int) or isinstance(sb_flags, bool) or sb_flags < 0:
        _fail("corners.sb_flags", sb_flags, "必须是 >= 0 的整数(选项位掩码)")
    fallback = section.get("sb_fallback_classic", CornersConfig.sb_fallback_classic)
    # 注意顺序:Python 里 True 同时也是 int(issubclass(bool, int)),但 1 不是 bool,
    # 所以先判 bool 再拒其余一切——整数 1/0 是 yaml 手写时最常见的笔误,必须拦下
    if not isinstance(fallback, bool):
        _fail("corners.sb_fallback_classic", fallback,
              "必须是 true 或 false(SB 未检出的帧是否用经典检测器结果兜底)")
    return CornersConfig(classic_flags=tuple(flags), sb_flags=sb_flags,
                         sb_fallback_classic=fallback,
                         subpix=_parse_subpix(section),
                         burst_check=_parse_burst_check(section))


def _parse_fisheye(raw: dict) -> FisheyeConfig:
    cfg = FisheyeConfig()
    section = _check_is_dict(raw.get("fisheye", {}), "calibrate.fisheye")
    flags = section.get("flags", cfg.flags)
    if not isinstance(flags, int) or isinstance(flags, bool) or flags < 0:
        _fail("calibrate.fisheye.flags", flags, "必须是 >= 0 的整数(选项位掩码)")
    max_iter = section.get("max_iter", cfg.max_iter)
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        _fail("calibrate.fisheye.max_iter", max_iter, "必须是 >= 1 的整数(迭代次数上限)")
    epsilon = section.get("epsilon", cfg.epsilon)
    if not isinstance(epsilon, int | float) or isinstance(epsilon, bool) or epsilon <= 0:
        _fail("calibrate.fisheye.epsilon", epsilon, "必须是 > 0 的数(收敛容差)")
    return FisheyeConfig(flags=flags, max_iter=max_iter, epsilon=float(epsilon))


def _parse_scipy(raw: dict) -> ScipyConfig:
    """逐字段校验 calibrate.scipy 段(体例照 _parse_fisheye:字段名 + 白话要求 + 实际值)。"""
    cfg = ScipyConfig()
    section = _check_is_dict(raw.get("scipy", {}), "calibrate.scipy")

    method = section.get("method", cfg.method)
    if method not in _ALLOWED_SCIPY_METHODS:
        _fail("calibrate.scipy.method", method,
              f"可选:{sorted(_ALLOWED_SCIPY_METHODS)};"
              f"scipy 的 'lm' 只支持稠密雅可比,640 视图约 3.47 GB 内存,实测不可用")

    tr_solver = section.get("tr_solver", cfg.tr_solver)
    if tr_solver not in _ALLOWED_TR_SOLVERS:
        _fail("calibrate.scipy.tr_solver", tr_solver,
              f"可选:{sorted(_ALLOWED_TR_SOLVERS)}(稀疏雅可比的标配内部求解器;"
              f"'exact' 要稠密矩阵,本项目规模不可用)")

    losses = section.get("losses", cfg.losses)
    if (not isinstance(losses, (list, tuple)) or not losses
            or not all(isinstance(x, str) for x in losses)):
        _fail("calibrate.scipy.losses", losses,
              "必须是非空的损失核名单(如 [linear, soft_l1])")
    bad_losses = [x for x in losses if x not in _ALLOWED_LOSSES]
    if bad_losses:
        _fail("calibrate.scipy.losses", bad_losses,
              f"含有不认识的损失核;可选:{sorted(_ALLOWED_LOSSES)}")

    f_scale = section.get("f_scale", cfg.f_scale)
    if isinstance(f_scale, str):
        if f_scale != "auto":
            _fail("calibrate.scipy.f_scale", f_scale,
                  "文字只接受 \"auto\"(= 3×1.4826×MAD,按初值残差自适应);"
                  "要写死请直接给一个 > 0 的数")
    elif not isinstance(f_scale, int | float) or isinstance(f_scale, bool) or f_scale <= 0:
        _fail("calibrate.scipy.f_scale", f_scale,
              "只接受 \"auto\" 或 > 0 的数(鲁棒核开始压低残差的分界尺度)")

    max_nfev = section.get("max_nfev", cfg.max_nfev)
    if not isinstance(max_nfev, int) or isinstance(max_nfev, bool) or max_nfev < 1:
        _fail("calibrate.scipy.max_nfev", max_nfev, "必须是 >= 1 的整数(函数评估次数上限)")

    tolerances = []
    for name in ("ftol", "xtol", "gtol"):
        value = section.get(name, getattr(cfg, name))
        if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
            requirement = {
                "ftol": "残差变化容差",
                "xtol": "参数变化容差",
                "gtol": "梯度(下一步最陡方向)容差",
            }[name]
            _fail(f"calibrate.scipy.{name}", value, f"必须是 > 0 的数({requirement})")
        tolerances.append(float(value))

    return ScipyConfig(method=method, tr_solver=tr_solver, losses=tuple(losses),
                       f_scale=f_scale, max_nfev=max_nfev,
                       ftol=tolerances[0], xtol=tolerances[1], gtol=tolerances[2])


def _parse_calibrate(raw: dict) -> CalibrateConfig:
    """解析 calibrate 段:角点路线 + 模型矩阵 + 求解路径 + scipy/fisheye 子段。"""
    section = _check_is_dict(raw, "calibrate")

    corner_route = section.get("corner_route", CalibrateConfig.corner_route)
    if corner_route not in _ALLOWED_CORNER_ROUTES:
        _fail("calibrate.corner_route", corner_route,
              f"可选:{sorted(_ALLOWED_CORNER_ROUTES)}"
              f"(D-02:标定默认吃 SB 为主、classic 兜底那份)")

    models = section.get("models", CalibrateConfig.models)
    if (not isinstance(models, (list, tuple)) or not models
            or not all(isinstance(x, str) for x in models)):
        _fail("calibrate.models", models,
              "必须是非空的模型短名名单(如 [m1, m2, m3, m4])")
    bad_models = [x for x in models if x not in _ALLOWED_MODEL_IDS]
    if bad_models:
        _fail("calibrate.models", bad_models,
              f"含有不认识的模型短名;可选:{sorted(_ALLOWED_MODEL_IDS)}")

    solvers = section.get("solvers", CalibrateConfig.solvers)
    if (not isinstance(solvers, (list, tuple)) or not solvers
            or not all(isinstance(x, str) for x in solvers)):
        _fail("calibrate.solvers", solvers,
              "必须是非空的求解路径名单(如 [opencv, scipy])")
    bad_solvers = [x for x in solvers if x not in _ALLOWED_SOLVERS]
    if bad_solvers:
        _fail("calibrate.solvers", bad_solvers,
              f"含有不认识的求解路径;可选:{sorted(_ALLOWED_SOLVERS)}")

    suspect_review_model = section.get(
        "suspect_review_model", CalibrateConfig.suspect_review_model)
    if suspect_review_model not in _ALLOWED_MODEL_IDS:
        _fail("calibrate.suspect_review_model", suspect_review_model,
              f"可选:{sorted(_ALLOWED_MODEL_IDS)}"
              f"(存疑帧复核用哪个模型的精化结果做对照,固定一个模型,不逐模型重复)")

    return CalibrateConfig(corner_route=corner_route,
                           models=tuple(models), solvers=tuple(solvers),
                           suspect_review_model=suspect_review_model,
                           scipy=_parse_scipy(section),
                           fisheye=_parse_fisheye(section))


def _parse_noise(raw: dict) -> NoiseConfig:
    """解析 noise 段:主线口径、注入网格、重复数、种子、并行上限全部白名单校验。

    注入实验是全项目最重的资源开关(网格 × 重复 × 并行一起放大),
    档数/重复数/进程上限在这里用硬范围拦下,防止一个手滑的配置把
    机器占满(研究 §Pitfall:DoS 自伤防线)。
    """
    cfg = NoiseConfig()
    section = _check_is_dict(raw.get("noise", {}), "noise")

    variant = section.get("immerkaer_variant", cfg.immerkaer_variant)
    if variant not in _ALLOWED_IMMERKAER_VARIANTS:
        _fail("noise.immerkaer_variant", variant,
              f"可选:{sorted(_ALLOWED_IMMERKAER_VARIANTS)}"
              f"(std = 整图标准式;robust = 稳健中位数式,默认主线)")

    sigma_min = section.get("injection_sigma_min", cfg.injection_sigma_min)
    sigma_max = section.get("injection_sigma_max", cfg.injection_sigma_max)
    for name, value in (("noise.injection_sigma_min", sigma_min),
                        ("noise.injection_sigma_max", sigma_max)):
        if value is not None and (not isinstance(value, int | float)
                                  or isinstance(value, bool) or value <= 0):
            _fail(name, value,
                  "留空(null,自动按实测噪声锚定)或 > 0 的数(注入 σ 网格上下限,灰阶)")
    if sigma_min is not None and sigma_max is not None and sigma_max <= sigma_min:
        _fail("noise.injection_sigma_min", (sigma_min, sigma_max),
              f"网格下限必须小于上限(injection_sigma_max = {sigma_max!r}),"
              f"否则对数网格没有跨度")

    n_levels = section.get("injection_n_levels", cfg.injection_n_levels)
    if (not isinstance(n_levels, int) or isinstance(n_levels, bool)
            or not 3 <= n_levels <= 16):
        _fail("noise.injection_n_levels", n_levels,
              "必须是 3 到 16 的整数(σ 对数网格档数,不含 σ=0 参照档;"
              "档太少曲线不准,档太多纯浪费算力)")

    n_repeats = section.get("injection_n_repeats", cfg.injection_n_repeats)
    if (not isinstance(n_repeats, int) or isinstance(n_repeats, bool)
            or not 3 <= n_repeats <= 5):
        _fail("noise.injection_n_repeats", n_repeats,
              "必须是 3 到 5 的整数(每档重复次数,少于 3 次压不掉注入随机性,"
              "多于 5 次算力收益趋平)")

    seed = section.get("injection_seed", cfg.injection_seed)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        _fail("noise.injection_seed", seed,
              "必须是 >= 0 的整数(注入实验随机种子,换种子 = 换一批加噪副本)")

    workers = section.get("injection_max_workers", cfg.injection_max_workers)
    cpu = os.cpu_count() or 1
    if (not isinstance(workers, int) or isinstance(workers, bool)
            or not 1 <= workers <= cpu):
        _fail("noise.injection_max_workers", workers,
              f"必须是 1 到 {cpu} 的整数(并行进程上限,本机 CPU 核数 = {cpu};"
              f"超过核数只会互相抢资源)")

    pooling = section.get("temporal_pooling", cfg.temporal_pooling)
    if pooling not in _ALLOWED_TEMPORAL_POOLING:
        _fail("noise.temporal_pooling", pooling,
              f"可选:{sorted(_ALLOWED_TEMPORAL_POOLING)}"
              f"(rms = 方均根,标量噪声假设下的自然聚合;median = 中位数)")

    splits = section.get("noise_splits", cfg.noise_splits)
    if (not isinstance(splits, (list, tuple)) or not splits
            or not all(isinstance(x, str) for x in splits)):
        _fail("noise.noise_splits", splits, "必须是非空的划分名单(如 [train, test])")
    bad_splits = [x for x in splits if x not in _ALLOWED_NOISE_SPLITS]
    if bad_splits:
        _fail("noise.noise_splits", bad_splits,
              f"含有不认识的划分名;可选:{sorted(_ALLOWED_NOISE_SPLITS)}")

    return NoiseConfig(
        immerkaer_variant=variant,
        injection_sigma_min=None if sigma_min is None else float(sigma_min),
        injection_sigma_max=None if sigma_max is None else float(sigma_max),
        injection_n_levels=n_levels,
        injection_n_repeats=n_repeats,
        injection_seed=seed,
        injection_max_workers=workers,
        temporal_pooling=pooling,
        noise_splits=tuple(splits),
    )


def _parse_fisher(raw: dict) -> FisherConfig:
    """解析 fisher 段:模型清单、条件数分档阈值、敏感性带半宽、两个开关。

    校验要点:模型短名走白名单(与 calibrate.models 同一份清单);
    条件数阈值必须 > 1(条件数 = 最软方向与最硬方向之比,恒 ≥ 1);
    警戒线必须小于退化线(分档有序);敏感性带半宽在 0 到 1 之间;
    布尔键先判 isinstance bool 再拒其余(yaml 手写 1/0 当布尔是常见笔误)。
    """
    cfg = FisherConfig()
    section = _check_is_dict(raw.get("fisher", {}), "fisher")

    models = section.get("models", cfg.models)
    if (not isinstance(models, (list, tuple)) or not models
            or not all(isinstance(x, str) for x in models)):
        _fail("fisher.models", models, "必须是非空的模型短名名单(如 [m1, m2, m3, m4])")
    bad_models = [x for x in models if x not in _ALLOWED_MODEL_IDS]
    if bad_models:
        _fail("fisher.models", bad_models,
              f"含有不认识的模型短名;可选:{sorted(_ALLOWED_MODEL_IDS)}")

    cond_warning = section.get("cond_warning", cfg.cond_warning)
    if (not isinstance(cond_warning, int | float) or isinstance(cond_warning, bool)
            or cond_warning <= 1):
        _fail("fisher.cond_warning", cond_warning,
              "必须是 > 1 的数(条件数警戒线;条件数 = 最软方向与最硬方向的比值,"
              "恒 ≥ 1,≤ 1 没有物理意义)")
    cond_degenerate = section.get("cond_degenerate", cfg.cond_degenerate)
    if (not isinstance(cond_degenerate, int | float) or isinstance(cond_degenerate, bool)
            or cond_degenerate <= 1):
        _fail("fisher.cond_degenerate", cond_degenerate,
              "必须是 > 1 的数(条件数退化线)")
    if cond_warning >= cond_degenerate:
        _fail("fisher.cond_warning", (cond_warning, cond_degenerate),
              f"警戒线必须小于退化线(cond_degenerate = {cond_degenerate!r}),"
              f"否则分档失去意义")

    band_rel = section.get("sigma_band_rel", cfg.sigma_band_rel)
    if (not isinstance(band_rel, int | float) or isinstance(band_rel, bool)
            or not 0 < band_rel < 1):
        _fail("fisher.sigma_band_rel", band_rel,
              "必须是 0 到 1 之间的小数(Σ 敏感性带的相对半宽,如 0.267 ≈ ±27%)")

    # 注意顺序:先判 bool 再拒其余(整数 1/0 是 yaml 手写时最常见的笔误)
    lin_check = section.get("linearization_check", cfg.linearization_check)
    if not isinstance(lin_check, bool):
        _fail("fisher.linearization_check", lin_check,
              "必须是 true 或 false(是否做线性化点敏感性检查)")
    crosscheck = section.get("crosscheck_enabled", cfg.crosscheck_enabled)
    if not isinstance(crosscheck, bool):
        _fail("fisher.crosscheck_enabled", crosscheck,
              "必须是 true 或 false(是否做 OpenCV stdDev 交叉验证)")

    return FisherConfig(models=tuple(models),
                        cond_warning=float(cond_warning),
                        cond_degenerate=float(cond_degenerate),
                        sigma_band_rel=float(band_rel),
                        linearization_check=lin_check,
                        crosscheck_enabled=crosscheck)


def _parse_evaluate(raw: dict) -> EvaluateConfig:
    """解析 evaluate 段:参选名单、考试规模、判据阈值、多起点与并行上限。

    校验要点(防手写配置放大成资源浪费或错误实验):
    - 模型短名与核名走白名单;核名单必须含 linear(linear 是"其他核有没有
      更好"的归一参照,缺了它对比就没有基准);
    - 计数键 ≥ 1、折数 ≥ 2(1 折等于不考)、并行进程 ≥ 1;
    - 两个告警显著性都在 0 到 0.5 之间且 hard < soft(告警线必须有序);
    - 等价带像素值 > 0、相对带在 0 到 1 之间、全部幅度/容差 > 0;
    - 布尔键先判 isinstance bool 再拒其余(整数 1/0 是 yaml 手写最常见笔误)。
    """
    cfg = EvaluateConfig()
    section = _check_is_dict(raw.get("evaluate", {}), "evaluate")

    models = section.get("models", cfg.models)
    if (not isinstance(models, (list, tuple)) or not models
            or not all(isinstance(x, str) for x in models)):
        _fail("evaluate.models", models, "必须是非空的模型短名名单(如 [m1, m2, m3, m4])")
    bad_models = [x for x in models if x not in _ALLOWED_MODEL_IDS]
    if bad_models:
        _fail("evaluate.models", bad_models,
              f"含有不认识的模型短名;可选:{sorted(_ALLOWED_MODEL_IDS)}")

    losses = section.get("optimizer_losses", cfg.optimizer_losses)
    if (not isinstance(losses, (list, tuple)) or not losses
            or not all(isinstance(x, str) for x in losses)):
        _fail("evaluate.optimizer_losses", losses,
              "必须是非空的鲁棒核名单(如 [linear, soft_l1, huber, cauchy])")
    bad_losses = [x for x in losses if x not in _ALLOWED_EVAL_LOSSES]
    if bad_losses:
        _fail("evaluate.optimizer_losses", bad_losses,
              f"含有不认识的鲁棒核;可选:{sorted(_ALLOWED_EVAL_LOSSES)}")
    if "linear" not in losses:
        _fail("evaluate.optimizer_losses", losses,
              f"必须包含 linear(它是其他核的归一参照);当前名单:{list(losses)}")

    # 计数键:全部 >= 1;折数另要求 >= 2(1 折没有"留出的考题"可言)
    counts = (("cv_n_rounds", cfg.cv_n_rounds), ("bootstrap_n", cfg.bootstrap_n),
              ("bic_n_poses", cfg.bic_n_poses), ("multistart_n", cfg.multistart_n),
              ("multistart_n_fisheye", cfg.multistart_n_fisheye))
    for name, default in counts:
        value = section.get(name, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            _fail(f"evaluate.{name}", value, "必须是 >= 1 的整数(次数/数量配置)")
    cv_n_folds = section.get("cv_n_folds", cfg.cv_n_folds)
    if not isinstance(cv_n_folds, int) or isinstance(cv_n_folds, bool) or cv_n_folds < 2:
        _fail("evaluate.cv_n_folds", cv_n_folds,
              "必须是 >= 2 的整数(1 折留不出考题,交叉验证至少两折)")

    workers = section.get("evaluate_max_workers", cfg.evaluate_max_workers)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        _fail("evaluate.evaluate_max_workers", workers,
              "必须是 >= 1 的整数(1 = 不开子进程串行调试;并行交给 joblib 调度,"
              "不必卡着本机核数写)")

    # 等价带(D-09):像素绝对带 > 0;相对带在 0 到 1 之间
    tie_px = section.get("tie_band_px", cfg.tie_band_px)
    if not isinstance(tie_px, int | float) or isinstance(tie_px, bool) or tie_px <= 0:
        _fail("evaluate.tie_band_px", tie_px,
              "必须是 > 0 的数(统计平局的等价带,像素)")
    tie_rel = section.get("tie_band_rel", cfg.tie_band_rel)
    if (not isinstance(tie_rel, int | float) or isinstance(tie_rel, bool)
            or not 0 < tie_rel < 1):
        _fail("evaluate.tie_band_rel", tie_rel,
              "必须是 0 到 1 之间的小数(等价带的相对半宽,如 0.02 = 较简模型均值的 2%)")

    # 离群判据与告警显著性:阈值 > 0;两个显著性在 (0, 0.5) 且 hard < soft
    mad_z = section.get("mad_z_threshold", cfg.mad_z_threshold)
    if not isinstance(mad_z, int | float) or isinstance(mad_z, bool) or mad_z <= 0:
        _fail("evaluate.mad_z_threshold", mad_z,
              "必须是 > 0 的数(留一姿态误差的稳健 z 分数报警线)")
    alpha_hard = section.get("crlb_alert_alpha_hard", cfg.crlb_alert_alpha_hard)
    alpha_soft = section.get("crlb_alert_alpha_soft", cfg.crlb_alert_alpha_soft)
    for name, value in (("crlb_alert_alpha_hard", alpha_hard),
                        ("crlb_alert_alpha_soft", alpha_soft)):
        if (not isinstance(value, int | float) or isinstance(value, bool)
                or not 0 < value < 0.5):
            _fail(f"evaluate.{name}", value,
                  "必须是 0 到 0.5 之间的小数(显著性水平,越小告警越严)")
    if alpha_hard >= alpha_soft:
        _fail("evaluate.crlb_alert_alpha_hard", (alpha_hard, alpha_soft),
              f"硬告警显著性必须小于边缘提示显著性(crlb_alert_alpha_soft = "
              f"{alpha_soft!r}),否则告警线失去分档意义")

    # 判据执行位:过拟合比值上限 > 1(等于 1 意味着验证误差必须不高于训练误差,
    # 噪声下不可能);稳定位移上限 > 0
    ratio_max = section.get("gen_overfit_ratio_max", cfg.gen_overfit_ratio_max)
    if not isinstance(ratio_max, int | float) or isinstance(ratio_max, bool) or ratio_max <= 1:
        _fail("evaluate.gen_overfit_ratio_max", ratio_max,
              "必须是 > 1 的数(折级验证/训练误差比上限;噪声的存在让恰好 1 不可能,"
              "过拟合显形要留出噪声余量)")
    disp_max = section.get("stability_disp_px_max", cfg.stability_disp_px_max)
    if not isinstance(disp_max, int | float) or isinstance(disp_max, bool) or disp_max <= 0:
        _fail("evaluate.stability_disp_px_max", disp_max,
              "必须是 > 0 的数(参数扰动的投影位移上限,像素当量)")

    # 多起点扰动幅度与双容差:全部 > 0
    amps = (("multistart_intr_rel", cfg.multistart_intr_rel),
            ("multistart_dist_rel", cfg.multistart_dist_rel),
            ("multistart_rot_deg", cfg.multistart_rot_deg),
            ("multistart_trans_rel", cfg.multistart_trans_rel),
            ("multistart_rms_tol_px", cfg.multistart_rms_tol_px),
            ("multistart_param_tol_px", cfg.multistart_param_tol_px))
    for name, default in amps:
        value = section.get(name, default)
        if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
            _fail(f"evaluate.{name}", value,
                  "必须是 > 0 的数(多起点扰动幅度/收敛容差,幅度为零等于不扰动)")

    # 三个布尔键:先判 bool 再拒其余(整数 1/0 是 yaml 手写时最常见的笔误)
    bools = (("slice_enabled", cfg.slice_enabled, "是否做重复张分组实验"),
             ("lopo_enabled", cfg.lopo_enabled, "是否做留一姿态实验"),
             ("bic_enabled", cfg.bic_enabled, "统计平局时是否用 BIC 裁决"))
    for name, default, purpose in bools:
        value = section.get(name, default)
        if not isinstance(value, bool):
            _fail(f"evaluate.{name}", value, f"必须是 true 或 false({purpose})")

    return EvaluateConfig(
        models=tuple(models),
        optimizer_losses=tuple(losses),
        cv_n_folds=cv_n_folds,
        cv_n_rounds=section.get("cv_n_rounds", cfg.cv_n_rounds),
        bootstrap_n=section.get("bootstrap_n", cfg.bootstrap_n),
        slice_enabled=section.get("slice_enabled", cfg.slice_enabled),
        lopo_enabled=section.get("lopo_enabled", cfg.lopo_enabled),
        evaluate_max_workers=workers,
        tie_band_px=float(tie_px),
        tie_band_rel=float(tie_rel),
        bic_enabled=section.get("bic_enabled", cfg.bic_enabled),
        bic_n_poses=section.get("bic_n_poses", cfg.bic_n_poses),
        mad_z_threshold=float(mad_z),
        crlb_alert_alpha_hard=float(alpha_hard),
        crlb_alert_alpha_soft=float(alpha_soft),
        gen_overfit_ratio_max=float(ratio_max),
        stability_disp_px_max=float(disp_max),
        multistart_n=section.get("multistart_n", cfg.multistart_n),
        multistart_n_fisheye=section.get("multistart_n_fisheye", cfg.multistart_n_fisheye),
        multistart_intr_rel=float(section.get("multistart_intr_rel", cfg.multistart_intr_rel)),
        multistart_dist_rel=float(section.get("multistart_dist_rel", cfg.multistart_dist_rel)),
        multistart_rot_deg=float(section.get("multistart_rot_deg", cfg.multistart_rot_deg)),
        multistart_trans_rel=float(section.get("multistart_trans_rel", cfg.multistart_trans_rel)),
        multistart_rms_tol_px=float(section.get("multistart_rms_tol_px",
                                                cfg.multistart_rms_tol_px)),
        multistart_param_tol_px=float(section.get("multistart_param_tol_px",
                                                  cfg.multistart_param_tol_px)),
    )


def _parse_visualization(raw: dict) -> VisualizationConfig:
    section = _check_is_dict(raw.get("visualization", {}), "visualization")
    quality = section.get("jpeg_quality", VisualizationConfig.jpeg_quality)
    if not isinstance(quality, int) or isinstance(quality, bool) or not 1 <= quality <= 100:
        _fail("visualization.jpeg_quality", quality, "必须是 1 到 100 的整数(JPEG 质量)")
    mosaic = _check_is_dict(section.get("mosaic", {}), "visualization.mosaic")
    cfg = MosaicConfig()
    per_pose = mosaic.get("per_pose_sample", cfg.per_pose_sample)
    if not isinstance(per_pose, int) or isinstance(per_pose, bool) or per_pose < 1:
        _fail("visualization.mosaic.per_pose_sample", per_pose, "必须是 >= 1 的整数(每姿态抽样张数)")
    worst_n = mosaic.get("worst_n", cfg.worst_n)
    if not isinstance(worst_n, int) or isinstance(worst_n, bool) or worst_n < 0:
        _fail("visualization.mosaic.worst_n", worst_n, "必须是 >= 0 的整数(最差张数)")
    thumb = mosaic.get("thumb_width", cfg.thumb_width)
    if not isinstance(thumb, int) or isinstance(thumb, bool) or thumb < 1:
        _fail("visualization.mosaic.thumb_width", thumb, "必须是 >= 1 的整数(缩略图宽,像素)")
    cols = mosaic.get("cols", cfg.cols)
    if not isinstance(cols, int) or isinstance(cols, bool) or cols < 1:
        _fail("visualization.mosaic.cols", cols, "必须是 >= 1 的整数(拼图每行格数)")
    return VisualizationConfig(jpeg_quality=quality,
                               mosaic=MosaicConfig(per_pose_sample=per_pose,
                                                   worst_n=worst_n,
                                                   thumb_width=thumb, cols=cols))


def load_calibration(path: Path | str) -> CalibrationConfig:
    """解析并校验 calibration.yaml,返回不可变的 CalibrationConfig。

    缺省键用内置默认值(与随仓库的 calibration.yaml 一致);
    类型或范围不合给带字段名与实际值的中文错误;
    exclude_images 缺省时为空元组(人工审阅后填 image_id,程序只做逻辑剔除)。
    """
    raw = _read_yaml(path)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        _fail("calibration.yaml 顶层", raw, "应当是一组配置项(字典)")
    top = raw

    data_section = _check_is_dict(top.get("data", {}), "data")
    train = data_section.get("train", DataDirs.train)
    test = data_section.get("test", DataDirs.test)
    for name, value in (("data.train", train), ("data.test", test)):
        if not isinstance(value, str) or not value.strip():
            _fail(name, value, "必须是数据目录路径文字(相对当前工作目录)")

    seed = top.get("seed", CalibrationConfig.seed)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        _fail("seed", seed, "必须是 >= 0 的整数(随机种子)")

    cv2_threads = top.get("cv2_threads", CalibrationConfig.cv2_threads)
    if not isinstance(cv2_threads, int) or isinstance(cv2_threads, bool) or cv2_threads < 0:
        _fail("cv2_threads", cv2_threads, "必须是 >= 0 的整数(OpenCV 线程数)")

    exclude = top.get("exclude_images", [])
    if exclude is None:
        exclude = []
    if not isinstance(exclude, list) or not all(isinstance(x, str) for x in exclude):
        _fail("exclude_images", exclude, "必须是一列 image_id 文字(人工剔除清单)")

    ingest_section = _check_is_dict(top.get("ingest", {}), "ingest")

    return CalibrationConfig(
        data=DataDirs(train=train, test=test),
        seed=seed,
        cv2_threads=cv2_threads,
        exclude_images=tuple(exclude),
        ingest=IngestConfig(precheck=_parse_precheck(ingest_section)),
        corners=_parse_corners(top),
        calibrate=_parse_calibrate(top.get("calibrate", {})),
        noise=_parse_noise(top),
        fisher=_parse_fisher(top),
        evaluate=_parse_evaluate(top),
        visualization=_parse_visualization(top),
    )
