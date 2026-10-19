"""配置解析与校验测试(IO-03)+ 血缘指纹测试。

覆盖四类要求:合法加载的值断言、缺字段的中文报错、非法值的中文报错、
源码安全断言(只许 safe_load,禁 yaml.load)。
"""

import hashlib
from pathlib import Path

import pytest

from calib.evaluation.parallel import run_parallel
from calib.io.config import load_board, load_calibration
from calib.util.lineage import make_lineage, sha256_file

# 仓库根目录(测试文件在 <根>/tests/ 下)
REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_BOARD = REPO_ROOT / "config" / "board.yaml"
REAL_CALIB = REPO_ROOT / "config" / "calibration.yaml"


# ---- 合法加载:真实配置的值必须与随仓库文件一致 ----

def test_load_real_board_values():
    """真实 board.yaml 解析结果:11 列 × 8 行内角点,方格 25.0mm。"""
    spec = load_board(REAL_BOARD)
    assert (spec.cols, spec.rows, spec.square_size_mm, spec.unit) == (11, 8, 25.0, "mm")


def test_load_real_calibration_values():
    """真实 calibration.yaml:窗口粒度 frame、亚像素参数、连拍容差等关键默认值。"""
    cfg = load_calibration(REAL_CALIB)
    assert cfg.data.train == "input_data/train"
    assert cfg.data.test == "input_data/test"
    assert cfg.seed == 42
    assert cfg.corners.subpix.window_granularity == "frame"
    assert cfg.corners.subpix.min_win == 3
    assert cfg.corners.burst_check.median_tol == 0.05
    assert cfg.corners.burst_check.max_tol == 0.3
    assert cfg.calibrate.fisheye.max_iter == 100
    assert cfg.visualization.mosaic.cols == 8
    assert cfg.corners.sb_fallback_classic is True  # 随仓库配置默认开 classic 兜底


def test_missing_exclude_images_defaults_to_empty(tmp_path):
    """calibration.yaml 缺 exclude_images 时:默认空清单(人工审阅后才填)。"""
    p = tmp_path / "calibration.yaml"
    p.write_text("data:\n  train: a\n  test: b\n", encoding="utf-8")
    cfg = load_calibration(p)
    assert cfg.exclude_images == ()


def test_sb_fallback_classic_defaults_true_when_missing(tmp_path):
    """corners 段缺 sb_fallback_classic 键:默认 True(SB 未检出的帧用 classic 兜底)。"""
    p = tmp_path / "calibration.yaml"
    p.write_text("data:\n  train: a\n  test: b\n", encoding="utf-8")
    cfg = load_calibration(p)
    assert cfg.corners.sb_fallback_classic is True


def test_sb_fallback_classic_explicit_false(tmp_path):
    """显式写 sb_fallback_classic: false:解析为 False(关闭兜底,SB 缺就缺)。"""
    p = tmp_path / "calibration.yaml"
    p.write_text("corners:\n  sb_fallback_classic: false\n", encoding="utf-8")
    cfg = load_calibration(p)
    assert cfg.corners.sb_fallback_classic is False


# ---- 第④步求解矩阵:模型/路径/scipy 参数进配置(D-01/D-12)----

def test_calibrate_matrix_values_from_repo_yaml():
    """真实 calibration.yaml 的求解矩阵:m1–m4 全矩阵 × 双路径 × 默认 scipy 参数。"""
    cfg = load_calibration(REAL_CALIB)
    assert cfg.calibrate.corner_route == "sb"
    assert cfg.calibrate.models == ("m1", "m2", "m3", "m4")
    assert cfg.calibrate.solvers == ("opencv", "scipy")
    assert cfg.calibrate.scipy.method == "trf"
    assert cfg.calibrate.scipy.tr_solver == "lsmr"
    assert cfg.calibrate.scipy.losses == ("linear", "soft_l1")
    assert cfg.calibrate.scipy.f_scale == "auto"
    assert cfg.calibrate.scipy.max_nfev == 300
    assert cfg.calibrate.scipy.ftol == 1e-10
    assert cfg.calibrate.scipy.xtol == 1e-12
    assert cfg.calibrate.scipy.gtol == 1e-10
    # 鱼眼段原样保留(阶段一基线复现仍读它)
    assert cfg.calibrate.fisheye.flags == 2


def test_calibrate_matrix_defaults_when_section_missing(tmp_path):
    """整个 calibrate 段缺省:用内置默认(与随仓库 yaml 一致)。"""
    p = tmp_path / "calibration.yaml"
    p.write_text("data:\n  train: a\n  test: b\n", encoding="utf-8")
    cfg = load_calibration(p)
    assert cfg.calibrate.corner_route == "sb"
    assert cfg.calibrate.models == ("m1", "m2", "m3", "m4")
    assert cfg.calibrate.solvers == ("opencv", "scipy")
    assert cfg.calibrate.scipy.losses == ("linear", "soft_l1")
    assert cfg.calibrate.scipy.f_scale == "auto"


def test_f_scale_numeric_override(tmp_path):
    """f_scale 写死数值:0.25 被接受(鱼眼数据量级的实测推荐值)。"""
    p = tmp_path / "calibration.yaml"
    p.write_text("calibrate:\n  scipy:\n    f_scale: 0.25\n", encoding="utf-8")
    cfg = load_calibration(p)
    assert cfg.calibrate.scipy.f_scale == 0.25


# ---- 非法配置:必须抛 ValueError,消息含字段名(中文)----

def _write_board(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "board.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_missing_cols_field_chinese_error(tmp_path):
    """缺 cols 字段:报错必须点名 board.cols,而不是裸 KeyError。"""
    p = _write_board(
        tmp_path,
        "board:\n  board_type: chessboard\n  rows: 8\n"
        "  square_size_mm: 25.0\n  unit: mm\n",
    )
    with pytest.raises(ValueError) as exc_info:
        load_board(p)
    assert "board.cols" in str(exc_info.value)
    assert "实际读到的值" in str(exc_info.value)  # 错误里带了实际值,方便定位


def test_square_size_zero_chinese_error(tmp_path):
    """square_size_mm = 0:物理边长必须是正数。"""
    p = _write_board(
        tmp_path,
        "board:\n  board_type: chessboard\n  cols: 11\n  rows: 8\n"
        "  square_size_mm: 0\n  unit: mm\n",
    )
    with pytest.raises(ValueError, match="square_size_mm"):
        load_board(p)


def test_wrong_board_type_chinese_error(tmp_path):
    """board_type 不是 chessboard:当前只支持棋盘格,报错点名该字段。"""
    p = _write_board(
        tmp_path,
        "board:\n  board_type: charuco\n  cols: 11\n  rows: 8\n"
        "  square_size_mm: 25.0\n  unit: mm\n",
    )
    with pytest.raises(ValueError, match="board_type"):
        load_board(p)


def test_quantile_order_violation(tmp_path):
    """分位线上下颠倒:global_quantile_low 必须小于 high。"""
    p = tmp_path / "calibration.yaml"
    p.write_text(
        "ingest:\n  precheck:\n    global_quantile_low: 0.9\n"
        "    global_quantile_high: 0.1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="global_quantile_low"):
        load_calibration(p)


def test_sb_fallback_classic_rejects_int_one(tmp_path):
    """sb_fallback_classic: 1(整数而非布尔):必须报错并点名 corners.sb_fallback_classic。

    Python 里 isinstance(1, int) 与 isinstance(True, int) 都成立,极容易把
    整数 1/0 误当布尔放行;yaml 手写配置时这又是常见笔误,必须显式拒绝。
    """
    p = tmp_path / "calibration.yaml"
    p.write_text("corners:\n  sb_fallback_classic: 1\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        load_calibration(p)
    assert "corners.sb_fallback_classic" in str(exc_info.value)
    assert "实际读到的值" in str(exc_info.value)


# ---- 求解矩阵非法取值:报错点名字段 + 列出可选(白名单防线)----

def _write_calibrate(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "calibration.yaml"
    p.write_text(f"calibrate:\n{text}", encoding="utf-8")
    return p


def test_bad_model_name_rejected(tmp_path):
    """models 里混入 "m5":报错点名 calibrate.models 并给出可选清单。"""
    p = _write_calibrate(tmp_path, "  models: [m1, m5]\n")
    with pytest.raises(ValueError) as exc_info:
        load_calibration(p)
    message = str(exc_info.value)
    assert "calibrate.models" in message
    assert "可选" in message and "m5" in message


def test_empty_models_rejected(tmp_path):
    """models 空名单:矩阵至少要一个模型,报错点名 calibrate.models。"""
    p = _write_calibrate(tmp_path, "  models: []\n")
    with pytest.raises(ValueError, match="calibrate.models"):
        load_calibration(p)


def test_bad_solver_name_rejected(tmp_path):
    """solvers 里混入 "ceres":报错点名 calibrate.solvers 并给出可选清单。"""
    p = _write_calibrate(tmp_path, "  solvers: [opencv, ceres]\n")
    with pytest.raises(ValueError) as exc_info:
        load_calibration(p)
    message = str(exc_info.value)
    assert "calibrate.solvers" in message
    assert "可选" in message and "ceres" in message


def test_bad_loss_name_rejected(tmp_path):
    """losses 里混入 "l1":报错点名 calibrate.scipy.losses 并给出可选清单。"""
    p = _write_calibrate(tmp_path, "  scipy:\n    losses: [linear, l1]\n")
    with pytest.raises(ValueError) as exc_info:
        load_calibration(p)
    message = str(exc_info.value)
    assert "calibrate.scipy.losses" in message
    assert "可选" in message and "l1" in message


def test_f_scale_zero_and_negative_rejected(tmp_path):
    """f_scale = 0 与 -1:不是 "auto" 也不是正数,两条都报错点名 calibrate.scipy.f_scale。"""
    for bad in ("0", "-1"):
        p = _write_calibrate(tmp_path, f"  scipy:\n    f_scale: {bad}\n")
        with pytest.raises(ValueError) as exc_info:
            load_calibration(p)
        assert "calibrate.scipy.f_scale" in str(exc_info.value)


def test_method_lm_rejected_with_dense_reason(tmp_path):
    """method: lm:白名单拒绝,报错消息说明稠密雅可比限定(640 视图不可用)。"""
    p = _write_calibrate(tmp_path, "  scipy:\n    method: lm\n")
    with pytest.raises(ValueError) as exc_info:
        load_calibration(p)
    message = str(exc_info.value)
    assert "calibrate.scipy.method" in message
    assert "稠密" in message and "lm" in message


def test_bad_corner_route_rejected(tmp_path):
    """corner_route 非法值(如 "sift"):报错点名 calibrate.corner_route。"""
    p = _write_calibrate(tmp_path, "  corner_route: sift\n")
    with pytest.raises(ValueError) as exc_info:
        load_calibration(p)
    assert "calibrate.corner_route" in str(exc_info.value)
    assert "可选" in str(exc_info.value)


# ---- 第③步噪声段:参数进配置(D-12),白名单 + 硬范围防线 ----

def _write_noise(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "calibration.yaml"
    p.write_text(f"noise:\n{text}", encoding="utf-8")
    return p


def test_noise_defaults_from_repo_yaml():
    """真实 calibration.yaml 的 noise 段:稳健式主线、8 档 × 4 重复、
    种子与并行上限与随仓库文件一致,网格上下限留空 = 自动锚定。"""
    import os

    cfg = load_calibration(REAL_CALIB)
    assert cfg.noise.immerkaer_variant == "robust"
    assert cfg.noise.injection_sigma_min is None
    assert cfg.noise.injection_sigma_max is None
    assert cfg.noise.injection_n_levels == 8
    assert cfg.noise.injection_n_repeats == 4
    assert cfg.noise.injection_seed == 20261018
    assert cfg.noise.injection_max_workers == min(8, os.cpu_count() or 1)
    assert cfg.noise.temporal_pooling == "rms"
    assert cfg.noise.noise_splits == ("train", "test")


def test_noise_section_defaults_when_missing(tmp_path):
    """整个 noise 段缺省:用内置默认(与随仓库 yaml 一致)。"""
    p = tmp_path / "calibration.yaml"
    p.write_text("data:\n  train: a\n  test: b\n", encoding="utf-8")
    cfg = load_calibration(p)
    assert cfg.noise.immerkaer_variant == "robust"
    assert cfg.noise.injection_n_levels == 8
    assert cfg.noise.injection_n_repeats == 4


def test_noise_empty_section_ok(tmp_path):
    """noise: 写成空段(键全缺):走内置默认,不报错。"""
    p = tmp_path / "calibration.yaml"
    p.write_text("noise: {}\n", encoding="utf-8")
    cfg = load_calibration(p)
    assert cfg.noise.temporal_pooling == "rms"
    assert cfg.noise.noise_splits == ("train", "test")


def test_noise_full_keys_parsed(tmp_path):
    """noise 段写全键:全部键逐一生效(网格上下限写死数值覆盖自动锚定)。"""
    p = _write_noise(
        tmp_path,
        "  immerkaer_variant: std\n"
        "  injection_sigma_min: 4.0\n"
        "  injection_sigma_max: 96.0\n"
        "  injection_n_levels: 6\n"
        "  injection_n_repeats: 5\n"
        "  injection_seed: 7\n"
        "  injection_max_workers: 1\n"
        "  temporal_pooling: median\n"
        "  noise_splits: [train]\n",
    )
    cfg = load_calibration(p)
    assert cfg.noise.immerkaer_variant == "std"
    assert cfg.noise.injection_sigma_min == 4.0
    assert cfg.noise.injection_sigma_max == 96.0
    assert cfg.noise.injection_n_levels == 6
    assert cfg.noise.injection_n_repeats == 5
    assert cfg.noise.injection_seed == 7
    assert cfg.noise.injection_max_workers == 1
    assert cfg.noise.temporal_pooling == "median"
    assert cfg.noise.noise_splits == ("train",)


def test_bad_immerkaer_variant_rejected(tmp_path):
    """immerkaer_variant 非法值:报错点名 noise.immerkaer_variant 并列可选。"""
    p = _write_noise(tmp_path, "  immerkaer_variant: median_of_means\n")
    with pytest.raises(ValueError) as exc_info:
        load_calibration(p)
    message = str(exc_info.value)
    assert "noise.immerkaer_variant" in message
    assert "可选" in message and "robust" in message


def test_bad_temporal_pooling_rejected(tmp_path):
    """temporal_pooling 非法值:报错点名 noise.temporal_pooling 并列可选。"""
    p = _write_noise(tmp_path, "  temporal_pooling: mean\n")
    with pytest.raises(ValueError) as exc_info:
        load_calibration(p)
    message = str(exc_info.value)
    assert "noise.temporal_pooling" in message
    assert "可选" in message and "rms" in message


def test_n_repeats_two_rejected(tmp_path):
    """injection_n_repeats = 2:低于硬下限 3(压不掉注入随机性),报错点名。"""
    p = _write_noise(tmp_path, "  injection_n_repeats: 2\n")
    with pytest.raises(ValueError) as exc_info:
        load_calibration(p)
    message = str(exc_info.value)
    assert "noise.injection_n_repeats" in message
    assert "3" in message  # 消息里带合法范围


def test_max_workers_over_cpu_rejected(tmp_path):
    """injection_max_workers 超过本机核数:配置层拦下(DoS 自伤防线)。"""
    import os

    cpu = os.cpu_count() or 1
    p = _write_noise(tmp_path, f"  injection_max_workers: {cpu + 1}\n")
    with pytest.raises(ValueError) as exc_info:
        load_calibration(p)
    message = str(exc_info.value)
    assert "noise.injection_max_workers" in message
    assert str(cpu) in message  # 消息里带本机核数上限


def test_sigma_max_le_min_rejected(tmp_path):
    """网格上限 <= 下限:对数网格没有跨度,报错点名 noise.injection_sigma_min。"""
    p = _write_noise(
        tmp_path, "  injection_sigma_min: 8.0\n  injection_sigma_max: 8.0\n")
    with pytest.raises(ValueError) as exc_info:
        load_calibration(p)
    message = str(exc_info.value)
    assert "noise.injection_sigma_min" in message
    assert "8.0" in message  # 消息里带实际值


def test_bad_noise_splits_rejected(tmp_path):
    """noise_splits 混入 "val":报错点名 noise.noise_splits 并列可选。"""
    p = _write_noise(tmp_path, "  noise_splits: [train, val]\n")
    with pytest.raises(ValueError) as exc_info:
        load_calibration(p)
    message = str(exc_info.value)
    assert "noise.noise_splits" in message
    assert "可选" in message and "test" in message


def test_yaml_syntax_error_chinese_error_with_path(tmp_path):
    """yaml 语法错误(缩进用了 tab):报错必须是带文件路径的中文 ValueError,
    附 PyYAML 原始信息(行号/列号对定位有用),而不是裸抛英文堆栈。
    两个加载入口(load_calibration 与 load_board)共用 _read_yaml,都要兜住。"""
    p = tmp_path / "calibration.yaml"
    p.write_text("data:\n\ttrain: a\n  test: b\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        load_calibration(p)
    message = str(exc_info.value)
    assert "语法" in message
    assert str(p) in message

    board_p = tmp_path / "board.yaml"
    board_p.write_text("board:\n\tcols: 11\n", encoding="utf-8")
    with pytest.raises(ValueError, match="语法"):
        load_board(board_p)


def test_setup_runtime_survives_broken_yaml(tmp_path, monkeypatch):
    """坏 yaml 不再击穿 --help 防线:配置目录里只有语法坏的 calibration.yaml
    时,_setup_runtime 安静返回——该回调承诺"--help / 配置还没就位时不拦着
    用户",一个坏文件不该让 calib ingest --help 直接崩溃。"""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "calibration.yaml").write_text(
        "data:\n\ttrain: a\n  test: b\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    from calib.cli.main import _setup_runtime
    _setup_runtime()  # 不抛任何异常即通过


# ---- 源码安全断言(T-01-01:配置解析只许 safe_load)----

def test_config_source_uses_safe_load_only():
    """config.py 源码必须含 safe_load、绝不含 yaml.load((防反序列化漏洞)。"""
    source = (REPO_ROOT / "src" / "calib" / "io" / "config.py").read_text(encoding="utf-8")
    assert "safe_load" in source
    assert "yaml.load(" not in source


# ---- 第⑤步理论精度极限:fisher 段进配置(D-12:参数只从 config 读)----

def test_fisher_defaults_when_section_missing(tmp_path):
    """fisher 段整体缺省:全部用内置默认(与随仓库 yaml 一致)。"""
    p = tmp_path / "calibration.yaml"
    p.write_text("data:\n  train: a\n  test: b\n", encoding="utf-8")
    cfg = load_calibration(p)
    assert cfg.fisher.models == ("m1", "m2", "m3", "m4")
    assert cfg.fisher.cond_warning == 1e4
    assert cfg.fisher.cond_degenerate == 1e6
    assert cfg.fisher.sigma_band_rel == pytest.approx(0.267)
    assert cfg.fisher.linearization_check is True
    assert cfg.fisher.crosscheck_enabled is True


def test_fisher_repo_yaml_values():
    """真实 calibration.yaml 的 fisher 段:逐键与随仓库默认一致。"""
    cfg = load_calibration(REAL_CALIB)
    assert cfg.fisher.models == ("m1", "m2", "m3", "m4")
    assert cfg.fisher.cond_warning == 1e4
    assert cfg.fisher.cond_degenerate == 1e6
    assert cfg.fisher.sigma_band_rel == pytest.approx(0.267)
    assert cfg.fisher.linearization_check is True
    assert cfg.fisher.crosscheck_enabled is True


def test_fisher_bad_model_name_chinese_error(tmp_path):
    """models 里混入不认识的短名:报错点名 fisher.models 并列出可选清单。"""
    p = tmp_path / "calibration.yaml"
    p.write_text("fisher:\n  models: [m1, m9]\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        load_calibration(p)
    message = str(exc_info.value)
    assert "fisher.models" in message
    assert "m9" in message and "可选" in message


def test_fisher_cond_threshold_le_one_rejected(tmp_path):
    """条件数阈值 ≤ 1:条件数是"最软方向与最硬方向的比值",恒 ≥ 1,
    阈值 ≤ 1 没有物理意义,配置层直接拦下。"""
    p = tmp_path / "calibration.yaml"
    p.write_text("fisher:\n  cond_warning: 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fisher\\.cond_warning"):
        load_calibration(p)


def test_fisher_cond_order_violation(tmp_path):
    """警戒线 ≥ 退化线:分档失去意义,报错点名两个字段。

    写法注意:yaml 的指数必须带符号(1.0e+8 才解析成数,1.0e8 是文字)。
    """
    p = tmp_path / "calibration.yaml"
    p.write_text(
        "fisher:\n  cond_warning: 1.0e+8\n  cond_degenerate: 1.0e+6\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc_info:
        load_calibration(p)
    message = str(exc_info.value)
    assert "fisher.cond_warning" in message and "cond_degenerate" in message


def test_fisher_sigma_band_rel_zero_rejected(tmp_path):
    """敏感性带半宽 ≤ 0(或 ≥ 1):带失去意义,报错点名字段。"""
    p = tmp_path / "calibration.yaml"
    p.write_text("fisher:\n  sigma_band_rel: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fisher\\.sigma_band_rel"):
        load_calibration(p)


def test_fisher_bool_field_rejects_int(tmp_path):
    """布尔键写 1/0(yaml 手写常见笔误):必须报错并点名字段。"""
    p = tmp_path / "calibration.yaml"
    p.write_text("fisher:\n  linearization_check: 1\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        load_calibration(p)
    assert "fisher.linearization_check" in str(exc_info.value)


# ---- 第⑥步防作弊考试:evaluate 段进配置(25 键契约,D-12:参数只从 config 读)----

def _write_evaluate(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "calibration.yaml"
    p.write_text(f"evaluate:\n{text}", encoding="utf-8")
    return p


def test_evaluate_defaults_when_section_missing(tmp_path):
    """evaluate 段整体缺省:25 个键全部走内置默认(契约一次定死)。"""
    p = tmp_path / "calibration.yaml"
    p.write_text("data:\n  train: a\n  test: b\n", encoding="utf-8")
    ev = load_calibration(p).evaluate
    assert ev.models == ("m1", "m2", "m3", "m4")
    assert ev.optimizer_losses == ("linear", "soft_l1", "huber", "cauchy")
    assert ev.cv_n_folds == 5
    assert ev.cv_n_rounds == 5
    assert ev.bootstrap_n == 200
    assert ev.slice_enabled is True
    assert ev.lopo_enabled is True
    assert ev.evaluate_max_workers == 8
    assert ev.tie_band_px == pytest.approx(0.005)
    assert ev.tie_band_rel == pytest.approx(0.02)
    assert ev.bic_enabled is True
    assert ev.bic_n_poses == 80
    assert ev.mad_z_threshold == pytest.approx(3.5)
    assert ev.crlb_alert_alpha_hard == pytest.approx(0.01)
    assert ev.crlb_alert_alpha_soft == pytest.approx(0.05)
    assert ev.gen_overfit_ratio_max == pytest.approx(1.5)
    assert ev.stability_disp_px_max == pytest.approx(0.05)
    assert ev.multistart_n == 10
    assert ev.multistart_n_fisheye == 20
    assert ev.multistart_intr_rel == pytest.approx(0.02)
    assert ev.multistart_dist_rel == pytest.approx(0.10)
    assert ev.multistart_rot_deg == pytest.approx(1.0)
    assert ev.multistart_trans_rel == pytest.approx(0.02)
    assert ev.multistart_rms_tol_px == pytest.approx(0.001)
    assert ev.multistart_param_tol_px == pytest.approx(0.1)


def test_evaluate_repo_yaml_values():
    """真实 calibration.yaml 的 evaluate 段:逐键与内置默认一致(契约锚定)。"""
    ev = load_calibration(REAL_CALIB).evaluate
    assert ev.models == ("m1", "m2", "m3", "m4")
    assert ev.optimizer_losses == ("linear", "soft_l1", "huber", "cauchy")
    assert (ev.cv_n_folds, ev.cv_n_rounds, ev.bootstrap_n) == (5, 5, 200)
    assert ev.slice_enabled is True and ev.lopo_enabled is True
    assert ev.evaluate_max_workers == 8
    assert ev.tie_band_px == pytest.approx(0.005)
    assert ev.tie_band_rel == pytest.approx(0.02)
    assert ev.bic_enabled is True and ev.bic_n_poses == 80
    assert ev.mad_z_threshold == pytest.approx(3.5)
    assert (ev.crlb_alert_alpha_hard, ev.crlb_alert_alpha_soft) == (0.01, 0.05)
    assert ev.gen_overfit_ratio_max == pytest.approx(1.5)
    assert ev.stability_disp_px_max == pytest.approx(0.05)
    assert (ev.multistart_n, ev.multistart_n_fisheye) == (10, 20)
    assert ev.multistart_intr_rel == pytest.approx(0.02)
    assert ev.multistart_dist_rel == pytest.approx(0.10)
    assert ev.multistart_rot_deg == pytest.approx(1.0)
    assert ev.multistart_trans_rel == pytest.approx(0.02)
    assert ev.multistart_rms_tol_px == pytest.approx(0.001)
    assert ev.multistart_param_tol_px == pytest.approx(0.1)


def test_evaluate_bad_model_name_chinese_error(tmp_path):
    """models 混入 "m9":报错点名 evaluate.models 并给出可选清单。"""
    p = _write_evaluate(tmp_path, "  models: [m1, m9]\n")
    with pytest.raises(ValueError) as exc_info:
        load_calibration(p)
    message = str(exc_info.value)
    assert "evaluate.models" in message
    assert "m9" in message and "可选" in message


def test_evaluate_bad_loss_name_chinese_error(tmp_path):
    """optimizer_losses 混入 "l1":报错点名 evaluate.optimizer_losses 并列可选。"""
    p = _write_evaluate(tmp_path, "  optimizer_losses: [linear, l1]\n")
    with pytest.raises(ValueError) as exc_info:
        load_calibration(p)
    message = str(exc_info.value)
    assert "evaluate.optimizer_losses" in message
    assert "l1" in message and "可选" in message


def test_evaluate_losses_missing_linear_rejected(tmp_path):
    """optimizer_losses 缺 linear:没有归一参照,对比没有基准,配置层拦下。"""
    p = _write_evaluate(tmp_path, "  optimizer_losses: [soft_l1, huber]\n")
    with pytest.raises(ValueError) as exc_info:
        load_calibration(p)
    message = str(exc_info.value)
    assert "evaluate.optimizer_losses" in message
    assert "linear" in message


def test_evaluate_cv_n_folds_one_rejected(tmp_path):
    """cv_n_folds = 1:1 折留不出考题,报错点名 evaluate.cv_n_folds。"""
    p = _write_evaluate(tmp_path, "  cv_n_folds: 1\n")
    with pytest.raises(ValueError) as exc_info:
        load_calibration(p)
    message = str(exc_info.value)
    assert "evaluate.cv_n_folds" in message
    assert "2" in message  # 消息里带合法下限


def test_evaluate_workers_zero_rejected(tmp_path):
    """evaluate_max_workers = 0:必须 >= 1,报错点名字段。"""
    p = _write_evaluate(tmp_path, "  evaluate_max_workers: 0\n")
    with pytest.raises(ValueError, match="evaluate\\.evaluate_max_workers"):
        load_calibration(p)


def test_evaluate_alpha_order_violation(tmp_path):
    """硬告警显著性 >= 边缘提示:告警线失去分档意义,报错点名两个字段。"""
    p = _write_evaluate(
        tmp_path, "  crlb_alert_alpha_hard: 0.05\n  crlb_alert_alpha_soft: 0.01\n")
    with pytest.raises(ValueError) as exc_info:
        load_calibration(p)
    message = str(exc_info.value)
    assert "crlb_alert_alpha_hard" in message and "crlb_alert_alpha_soft" in message


def test_evaluate_bool_field_rejects_int(tmp_path):
    """布尔键写 1/0(yaml 手写常见笔误):必须报错并点名字段。"""
    p = _write_evaluate(tmp_path, "  slice_enabled: 1\n")
    with pytest.raises(ValueError) as exc_info:
        load_calibration(p)
    assert "evaluate.slice_enabled" in str(exc_info.value)


# ---- 任务级并行助手:串行回退与并行同序同结果 ----

def _square_worker(task: tuple) -> int:
    """测试用小 worker:返回 (x+1)^2,验证任务顺序与结果一一对应。"""
    (x,) = task
    return (x + 1) ** 2


def test_run_parallel_serial_and_parallel_same_order_same_results():
    """run_parallel 在 max_workers=1 与 2 下对纯函数 worker 产出同序同结果。"""
    tasks = [(i,) for i in range(12)]
    serial = list(run_parallel(tasks, _square_worker, 1))
    parallel = list(run_parallel(tasks, _square_worker, 2))
    assert serial == [(i + 1) ** 2 for i in range(12)]
    assert parallel == serial


def test_run_parallel_progress_counts_to_total():
    """进度回调逐任务播报:done 从 1 数到 total;串行与并行播报节奏一致。"""
    tasks = [(i,) for i in range(5)]
    seen_serial: list[dict] = []
    list(run_parallel(tasks, _square_worker, 1,
                      progress=lambda d: seen_serial.append(d)))
    seen_parallel: list[dict] = []
    list(run_parallel(tasks, _square_worker, 2,
                      progress=lambda d: seen_parallel.append(d)))
    expected = [{"done": i, "total": 5} for i in range(1, 6)]
    assert seen_serial == expected
    assert seen_parallel == expected


# ---- 血缘指纹(ITER-01 地基)----

def test_sha256_file_matches_hashlib(tmp_path):
    """sha256_file 与 hashlib 直接计算一致(分块实现没有改变结果)。"""
    p = tmp_path / "data.bin"
    p.write_bytes(b"intrinsic-calibration lineage test")
    expected = hashlib.sha256(b"intrinsic-calibration lineage test").hexdigest()
    assert sha256_file(p) == expected


def test_make_lineage_fields(tmp_path):
    """make_lineage 返回的血缘字典字段齐全,指纹与文件内容对应。"""
    cfg_file = tmp_path / "board.yaml"
    cfg_file.write_text("board: {}", encoding="utf-8")
    upstream_file = tmp_path / "upstream.json"
    upstream_file.write_text("{}", encoding="utf-8")

    lineage = make_lineage(
        config_files={"board.yaml": cfg_file},
        upstream={"manifest.json": upstream_file},
        round_id="run_20260717_172523",
    )

    assert set(lineage.keys()) >= {"created", "calib_version", "config_sha256", "upstream", "round_id"}
    assert lineage["round_id"] == "run_20260717_172523"
    assert lineage["config_sha256"]["board.yaml"] == sha256_file(cfg_file)
    assert lineage["upstream"]["manifest.json"]["sha256"] == sha256_file(upstream_file)
    assert lineage["upstream"]["manifest.json"]["path"].endswith("upstream.json")
    # 内容一变,指纹必须变(血缘指纹的意义所在)
    cfg_file.write_text("board: {changed}", encoding="utf-8")
    assert make_lineage({"board.yaml": cfg_file}, {}, "r")["config_sha256"]["board.yaml"] != \
        lineage["config_sha256"]["board.yaml"]
