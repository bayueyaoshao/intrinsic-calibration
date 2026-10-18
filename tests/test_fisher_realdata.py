"""真实数据试车:`calib fisher` 全链路 + 成功标准 3/4/5 对账(NOI-03 收口)。

整文件标记 integration(需要真实 input_data 与 corners/noise/calibration
产物,日常开发用 `uv run pytest -m "not integration"` 自动跳过;前置产物
缺失时整文件跳过并写明原因,不误报失败)。

测的是什么(对应 ROADMAP Phase 3 成功标准):
1. 结构:crlb.json 契约键全,models 下 m1–m4 四键全;每模型 views640 /
   poses80 × detector / temporal = 16 份记录逐份可寻址,数值有限(或
   零信息参数记 inf,绝不出现 NaN);
2. 成功标准 3(摆位缺陷体检):每份记录 top_directions 非空(参数名与
   载荷成对),条件数为有限正值,超阈值 flag 正确落 warning/degenerate;
3. 成功标准 4(stdDev 交叉验证):M1/M2 的 max_rel_dev < 10%(D-07 硬
   门槛);M3 有结果记录(尽力档,不设门槛);M4 记 unavailable_fisheye;
   编排内的参数对账断言(status == "ok")保证喂进交叉验证的观测与阶段二
   是同一批;
4. 成功标准 5(Σ 敏感性带):band_low == se×(1−rel) 且 band_high ==
   se×(1+rel),rel 从记录读(rel 是 8 张连拍估 σ 的 ±26.7%,不写死);
5. D-01 对照:views640 与 poses80 同口径 CRLB 都存在且不相等(两版差值
   = 连拍相关性损耗,可计算);线性化点检查 rel_diff 落盘(软断言
   <0.05,超限在消息中标记为发现而非失败,D-06 预期 <1%);
6. 血缘:crlb.json 的 lineage.upstream 含 noise.json 与逐模型 scipy 产物
   路径且文件真实存在。

复用规则(体例照 test_noise_realdata.py):FIM 只吃几何,同输入逐位复现,
所以只要 output/fisher 下已有一份"完整(非失败轮)且血缘指纹与当前
corners npz + noise.json 都对得上"的轮子,就直接复用它做断言,不重跑
十几分钟的确定性计算;任一上游一变指纹就对不上,自动触发全新一轮。
没有可复用轮时才真正跑命令(独立进程,超时上限 2 小时)。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from calib.io.config import load_calibration

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[1]
FISHER_TIMEOUT_S = 7200  # 交叉验证要逐模型重解 OpenCV(640 视图分钟级),给足 2 小时
_EXPECTED_MODELS = ("m1", "m2", "m3", "m4")
_OBS_SETS = ("views640", "poses80")
_SIGMA_SOURCES = ("detector", "temporal")


def _latest_run(stage_dir: Path) -> Path:
    runs = sorted(p for p in stage_dir.iterdir()
                  if p.is_dir() and p.name.startswith("run_"))
    assert runs, f"{stage_dir} 下没有任何 run 目录"
    return runs[-1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _upstream_sha(doc: dict, key: str) -> str | None:
    return (doc.get("lineage", {}).get("upstream", {}).get(key, {})
            .get("sha256"))


def _reusable_run(corners_npz: Path, noise_json: Path) -> Path | None:
    """找最近一份"完整且血缘指纹与当前 corners + noise 都对得上"的
    CRLB 轮,没有则 None(复用规则见模块 docstring)。

    完整 = 非失败轮 + 血缘登记了逐模型 scipy 产物(血缘缺账的旧轮
    视为不完整,触发全新一轮,绝不拿半成品充数)。
    """
    fisher_root = REPO_ROOT / "output" / "fisher"
    if not fisher_root.is_dir():
        return None
    actual = {"corners_train_sb.npz": _sha256(corners_npz),
              "noise.json": _sha256(noise_json)}
    for run in sorted((p for p in fisher_root.iterdir()
                       if p.is_dir() and p.name.startswith("run_")),
                      reverse=True):
        doc_path = run / "crlb.json"
        if not doc_path.is_file():
            continue
        with open(doc_path, encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("failure_note") is not None:
            continue
        upstream_keys = set(doc.get("lineage", {}).get("upstream", {}))
        required = {"m1__scipy/result.json", "m4__scipy/extrinsics.npz"}
        if not required <= upstream_keys:
            continue
        if all(_upstream_sha(doc, key) == sha
               for key, sha in actual.items()):
            return run
    return None


@pytest.fixture(scope="module")
def fisher_tree() -> dict:
    """在仓库根目录独立进程跑 calib fisher,返回运行目录与读回的报告。

    前置产物(真实照片、corners / noise / calibration)缺失时整文件跳过;
    已有完整且血缘对得上的历史轮则直接复用;命令失败带着完整输出报错。
    """
    if not (REPO_ROOT / "input_data" / "train").is_dir():
        pytest.skip("真实 input_data 不在,跳过 CRLB 集成测试")
    corners_root = REPO_ROOT / "output" / "corners"
    noise_root = REPO_ROOT / "output" / "noise"
    calib_root = REPO_ROOT / "output" / "calibration"
    for stage_root, hint in ((corners_root, "calib corners"),
                             (noise_root, "calib noise"),
                             (calib_root, "calib calibrate")):
        if not stage_root.is_dir():
            pytest.skip(f"还没有 {stage_root.name} 产物,先跑 {hint} 再来")

    corners_run = _latest_run(corners_root)
    noise_run = _latest_run(noise_root)
    corners_npz = corners_run / "corners_train_sb.npz"
    noise_json = noise_run / "noise.json"

    fisher_run = _reusable_run(corners_npz, noise_json)
    result = None
    if fisher_run is not None:
        print(f"\n复用完整 CRLB 轮(血缘指纹与当前 corners + noise 对得上):"
              f"{fisher_run}")
    else:
        result = subprocess.run(
            ["uv", "run", "--project", str(REPO_ROOT), "calib", "fisher"],
            check=False,
            cwd=REPO_ROOT, capture_output=True, text=True,
            timeout=FISHER_TIMEOUT_S, env={**os.environ},
        )
        if result.returncode != 0:
            raise AssertionError(
                f"calib fisher 退出码 {result.returncode}\n"
                f"--- 标准输出 ---\n{result.stdout}\n--- 标准误差 ---\n{result.stderr}"
            )
        fisher_run = _latest_run(REPO_ROOT / "output" / "fisher")
        assert (fisher_run / "crlb.json").is_file(), \
            f"命令退出码 0 但最新轮没有 crlb.json:{fisher_run}"

    with open(fisher_run / "crlb.json", encoding="utf-8") as f:
        report = json.load(f)
    return {"result": result, "fisher_run": fisher_run, "report": report,
            "corners_run": corners_run, "noise_run": noise_run}


def _records(report: dict):
    """逐份遍历 16 条 CRLB 记录,产出 ((模型, 观测集, σ口径), 记录)。"""
    for model_id in _EXPECTED_MODELS:
        for obs_name in _OBS_SETS:
            for sigma_name in _SIGMA_SOURCES:
                key = (model_id, obs_name, sigma_name)
                record = (report["models"][model_id]["observation_sets"]
                          [obs_name][sigma_name])
                yield key, record


def _assert_no_nan(values, key) -> None:
    arr = np.asarray(values, dtype=float)
    assert not np.isnan(arr).any(), \
        f"{key} 出现 NaN(数据缺口必须显式记 None 或 inf,NaN 是脏数):{key}"


def test_crlb_contract_and_16_records_addressable(fisher_tree: dict):
    """结构对账:crlb.json 契约键全、m1–m4 四键全、16 份记录逐份可寻址,
    数值字段有限或 inf(零信息参数),绝不 NaN;本轮非失败轮。"""
    report = fisher_tree["report"]
    assert report["failure_note"] is None, \
        f"本轮是失败轮:{report['failure_note']}"
    for key in ("schema_version", "created", "lineage", "models", "meta",
                "failure_note"):
        assert key in report, f"crlb.json 缺契约键:{key}"
    assert set(report["models"]) == set(_EXPECTED_MODELS), \
        f"models 键应为 { _EXPECTED_MODELS },实际 {tuple(report['models'])}"

    for key, record in _records(report):
        assert record["status"] == "ok", \
            f"{key} 未算成:{record['failure_note']}"
        for field in ("se", "band_low", "band_high", "condition_number"):
            assert record[field] is not None, f"{key} 的 {field} 缺数值"
        _assert_no_nan(record["se"], key)
        _assert_no_nan(record["band_low"], key)
        _assert_no_nan(record["band_high"], key)
        _assert_no_nan(record["eigvals"], key)
        cond = record["condition_number"]
        assert np.isfinite(cond) and cond > 0, \
            f"{key} 的条件数应为有限正值,实际 {cond!r}"


def test_success_criterion_3_spectrum_and_flags(fisher_tree: dict):
    """成功标准 3(摆位缺陷体检):top_directions 非空且参数名/载荷成对;
    条件数分档与阈值一致(阈值从配置读,不写死)——warning / degenerate
    是要上报的发现,flag 错了等于体检失灵。"""
    cfg = load_calibration(REPO_ROOT / "config" / "calibration.yaml")
    for key, record in _records(fisher_tree["report"]):
        tops = record["top_directions"]
        assert tops, f"{key} 的软方向表为空(特征谱报告缺账)"
        for item in tops:
            assert item["params"] and item["loadings"], \
                f"{key} 的软方向参数名/载荷不成对"
            assert len(item["params"]) == len(item["loadings"]), \
                f"{key} 的软方向参数名与载荷个数不一致"
        cond = record["condition_number"]
        expect = ("degenerate" if cond >= cfg.fisher.cond_degenerate
                  else "warning" if cond >= cfg.fisher.cond_warning else "")
        assert record["flag"] == expect, \
            (f"{key} 的分档旗标不对:条件数 {cond:.3e} 应记 {expect!r},"
             f"实际 {record['flag']!r}(阈值 {cfg.fisher.cond_warning} / "
             f"{cfg.fisher.cond_degenerate})")


def test_success_criterion_4_crosscheck_verdicts(fisher_tree: dict):
    """成功标准 4(stdDev 交叉验证,D-07 分级):M1/M2 硬门槛
    max_rel_dev < 10% 且判 pass;M3 尽力档有结果记录;M4 记
    unavailable_fisheye;编排内参数对账成立(交叉验证 status == ok)。"""
    report = fisher_tree["report"]
    for model_id in ("m1", "m2"):
        cc = report["models"][model_id]["opencv_crosscheck"]
        assert cc is not None and cc["status"] == "ok", \
            f"{model_id} 的交叉验证没跑成:{cc and cc.get('failure_note')}"
        assert cc["verdict"] == "pass", \
            (f"{model_id} 交叉验证未过硬门槛:{cc.get('verdict_note')}")
        max_dev = cc["compare"]["max_rel_dev"]
        assert max_dev < 0.10, \
            (f"{model_id} 的 max_rel_dev = {max_dev:.3%} 未进个位数百分比"
             f"执行线(<10%):两套独立数学对不上账,按发现上报")
    m3_cc = report["models"]["m3"]["opencv_crosscheck"]
    assert m3_cc is not None and m3_cc["status"] == "ok", \
        f"m3 尽力档也应有结果记录:{m3_cc and m3_cc.get('failure_note')}"
    assert m3_cc["compare"] is not None
    m4_cc = report["models"]["m4"]["opencv_crosscheck"]
    assert m4_cc is not None and m4_cc["status"] == "unavailable_fisheye", \
        f"m4 鱼眼族应记 unavailable_fisheye,实际 {m4_cc and m4_cc.get('status')!r}"


def test_success_criterion_5_sigma_band_identity(fisher_tree: dict):
    """成功标准 5(Σ 敏感性带):band_low == se×(1−rel)、band_high ==
    se×(1+rel) 逐份逐位成立,rel 从记录读(不写死 0.267);
    rel 合法范围 (0, 1)。带是纯乘法,等式必须逐位精确。"""
    for key, record in _records(fisher_tree["report"]):
        rel = record["band_rel"]
        assert 0.0 < rel < 1.0, f"{key} 的带半宽 rel 越界:{rel!r}"
        se = np.asarray(record["se"], dtype=float)
        np.testing.assert_array_equal(
            np.asarray(record["band_low"], dtype=float), se * (1.0 - rel),
            err_msg=f"{key} 的带下沿不是 se×(1−rel)——带被谁动过")
        np.testing.assert_array_equal(
            np.asarray(record["band_high"], dtype=float), se * (1.0 + rel),
            err_msg=f"{key} 的带上沿不是 se×(1+rel)——带被谁动过")


def test_d01_two_observation_sets_differ_and_linearization_check(fisher_tree: dict):
    """D-01 对照:views640 与 poses80 同口径 CRLB 都存在且不相等(两版差值
    = 连拍相关性损耗,可计算);线性化点检查 rel_diff 落盘,软断言 <0.05
    (超限在消息中标记为发现而非失败,D-06 预期 <1%)。"""
    report = fisher_tree["report"]
    for model_id in _EXPECTED_MODELS:
        for sigma_name in _SIGMA_SOURCES:
            se640 = np.asarray(report["models"][model_id]
                               ["observation_sets"]["views640"][sigma_name]["se"])
            se80 = np.asarray(report["models"][model_id]
                              ["observation_sets"]["poses80"][sigma_name]["se"])
            assert se640.shape == se80.shape
            assert not np.array_equal(se640, se80), \
                (f"{model_id}×{sigma_name}:640 全帧版与 80 聚合版的 CRLB 完全"
                 f"相等——聚合观测集没有生效,D-01 对照失效")
        check = report["models"][model_id].get("linearization_check")
        assert check is not None and check.get("rel_diff") is not None, \
            f"{model_id} 的线性化点检查没有落盘:{check}"
        rel = check["rel_diff"]
        if rel >= 0.05:
            # 软断言:超限是"解未收敛透"的发现,按发现记录,不判失败
            print(f"\n[发现] {model_id} 线性化点敏感性 rel_diff = {rel:.3%}"
                  f" ≥ 5%(D-06 预期 <1%)——解未收敛透的信号,按发现记录")


def test_lineage_upstream_files_exist(fisher_tree: dict):
    """血缘:crlb.json 的 lineage.upstream 含 noise.json 与逐模型 scipy
    result / extrinsics 路径,且文件真实存在(产物链可追溯的最后一环)。"""
    upstream = fisher_tree["report"]["lineage"]["upstream"]
    assert "noise.json" in upstream, \
        f"血缘缺 noise.json 上游:{tuple(upstream)}"
    for model_id in _EXPECTED_MODELS:
        for key in (f"{model_id}__scipy/result.json",
                    f"{model_id}__scipy/extrinsics.npz"):
            assert key in upstream, f"血缘缺 {key} 上游:{tuple(upstream)}"
    for key, entry in upstream.items():
        path = REPO_ROOT / entry["path"]
        assert path.is_file(), \
            f"血缘登记的上游文件不在:{key} → {path}"


def test_fisher_detail_npz_written_per_model(fisher_tree: dict):
    """明细落盘(D-13 保真):每个模型的 Fisher 明细 npz(边缘化矩阵等
    大数组)与 meta 旁车都在,json 里只留路径引用。"""
    run = fisher_tree["fisher_run"]
    for model_id in _EXPECTED_MODELS:
        npz_path = run / f"fisher_{model_id}.npz"
        assert npz_path.is_file(), f"缺逐模型明细 npz:{npz_path}"
        assert (run / f"fisher_{model_id}.meta.json").is_file(), \
            f"缺明细 npz 的血缘旁车:{npz_path.stem}.meta.json"
        with np.load(npz_path, allow_pickle=False) as data:
            assert {"F_marg_views640", "F_marg_poses80",
                    "param_names"} <= set(data.files), \
                f"{npz_path.name} 缺明细键:{data.files}"
        block = fisher_tree["report"]["models"][model_id]
        assert "_fisher_detail" not in block, \
            "明细数组不许进 json(D-13:汇总只是索引不是替代)"
    figures = sorted((run / "figures").glob("spectrum_*.png"))
    assert figures, "特征谱图一张都没有(plot_spectrum 整体未跑成,图是留痕的一环)"
