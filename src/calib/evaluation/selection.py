"""选型统计纯函数库:防作弊考试的判卷规则(配对检验 / BIC / 卡方告警线 / MAD 离群 / 三判据判行表)。

白话:模型选型的全部结论都从本模块的纯函数里判出来——输入数值,输出判定;
不做求解、不读落盘产物、不碰命令行框架。判定分支先在合成数据上证明正确
(tests/test_selection_stats.py),真实数据上的结论才拿得出辩护。

术语首现白话:
- 配对 t 检验(ttest_rel):同一批对象在两种处理下成对比较——这里是同一折
  切分下两个模型各考一次,误差逐折相减,把折与折之间的波动抵消掉,只看
  两模型的真实差距。
- 置信区间(CI):给"平均差距"圈出的高概率范围;这里用 95% 置信区间,
  区间整体落在哪里,结论就落在哪里。
- 等价带 δ:「差异小到可视为相同」的区间,±δ 之内算没差别。
- Shapiro-Wilk 检验:判断一组数像不像正态分布(钟形对称);p ≥ 0.05 视作
  "没有证据说它不是正态"。
- Wilcoxon 符号秩检验:不要求数据正态的成对比较兜底,基于秩(排位)计算。
- BIC(贝叶斯信息准则):惩罚参数数量的模型比较量,越小越好;只在统计
  平局时出场当裁决器(D-10)。
- 卡方分布:样本方差的波动遵循的分布;给"实测显著低于理论下界"定告警线
  用(D-06)。
- MAD(中位数绝对偏差):对离群值不敏感的稳健"标准差"当量;
  median_abs_deviation(x, scale='normal') 自带 1.4826 修正因子,对正态数据
  它就是标准差的一致估计。

判定体例(照 noise/fisher.py 的 verdict/verdict_note 与 solve/scoring.py 的
三分支结论):判定式与阈值随结论落盘,不凭空下结论。输入含 NaN/inf 一律
中文报错,绝不带着非有限值静默往下判(T-04-03-01)。
"""

from __future__ import annotations

import numpy as np
from scipy.stats import (
    median_abs_deviation,
    norm,
    shapiro,
    ttest_rel,
    wilcoxon,
)

from calib.io.config import CalibrationConfig

# 各模型内参数量("参数更简" = 这个数更小,D-09 平局取简的依据)。
# 出处:src/calib/models/ 注册表参数布局——
# m1 = fx,fy,cx,cy,k1,k2,p1,p2(8);m2 = m1 + k3(9);
# m3 = fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6(12);m4 = fx,fy,cx,cy,k1..k4(8)。
_MODEL_PARAM_COUNTS = {"m1": 8, "m2": 9, "m3": 12, "m4": 8}

_SHAPIRO_ALPHA = 0.05  # 正态性判断线:p >= 0.05 视作没有证据拒绝正态


def _require_finite_1d(values, what: str) -> np.ndarray:
    """一维数值输入的公共防线:非有限值(NaN/inf)当场中文报错(T-04-03-01)。"""
    arr = np.asarray(values, dtype=np.float64).ravel()
    n_bad = int(arr.size - np.count_nonzero(np.isfinite(arr)))
    if n_bad:
        raise ValueError(
            f"{what}里有 {n_bad} 个非有限值(NaN/inf):统计判定拒绝带病输入,"
            f"非有限值混进来会让结论静默失真。这些多半是求解失败的记录,"
            f"请在上游按失败记录剔除后再做统计。"
        )
    return arr


def _wilcoxon_ci(x: np.ndarray, alpha: float) -> tuple[float, float, str]:
    """Wilcoxon 符号秩置信区间的标准近似实现(教科书口径)。

    原理:把全部 Walsh 平均值(样本两两取均值,含自己和自己配对)排序,
    在正态近似下从两端各砍掉 k 个;k 由符号秩统计量的均值与方差算出,
    剩下最中间一段就是置信区间。零差值先丢弃(与 scipy.stats.wilcoxon 的
    处理一致)。返回 (区间下限, 区间上限, 方法标签)——方法标签按
    scipy.stats.wilcoxon(method='auto') 的同款规则判读:样本数 ≤ 25 且
    无零差、无并列走 "exact"(精确法),否则 "approx"(正态近似),
    调用方把它记进 meta(04-RESEARCH Pattern 3)。
    """
    x = x[x != 0.0]
    n = int(x.size)
    if n == 0:
        # 全是零差值:两模型逐项完全相同,区间退化为一个点
        return 0.0, 0.0, "exact"
    pairs = (x[:, None] + x[None, :])[np.triu_indices(n)] / 2.0
    walsh = np.sort(pairs)
    total = int(walsh.size)  # = n(n+1)/2
    z = norm.ppf((1.0 + alpha) / 2.0)
    mu = total / 2.0  # 符号秩统计量在零假设下的均值
    sd = float(np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0))  # 同一零假设下的标准差
    k = round(mu - z * sd)
    k = min(max(k, 0), total - 1)
    has_ties = bool(np.unique(np.abs(x)).size < n)
    w_method = "exact" if (n <= 25 and not has_ties) else "approx"
    return float(walsh[k]), float(walsh[total - 1 - k]), w_method


def paired_comparison(
    diff,
    tie_band_px: float,
    tie_band_rel: float,
    simpler_mean: float | None = None,
    alpha: float = 0.95,
) -> dict:
    """两模型 held-out 误差的配对比较,三分支判定(D-09/D-10)。

    diff:同折配对相减后的差值样本(模型甲误差 − 模型乙误差,逐折对齐,
    消除折间波动);LOPO 的逐姿态差值同法可用。等价带
    δ = max(tie_band_px, tie_band_rel × |simpler_mean|);simpler_mean 缺省
    (None)时只用绝对档,不拿"较简模型均值"放大带宽。

    三分支(依据是差值均值的置信区间与 ±δ 的关系,不看 p 值——"不显著"
    ≠"等价",样本小时区间宽,什么都"不显著"):
      1. 区间整体在 [−δ, +δ] 内        → "equivalent"(实际等价 → 取简,D-09)
      2. 区间含 0 且超出 ±δ(太宽)    → "tie"(统计平局 → 交 BIC 裁决,D-10)
      3. 其余(区间不含 0 且已出带)   → "different"(差异真实 → held-out 小者胜)
    数学上三支互斥且穷尽:先判整体在带内,再判含 0,剩下的必然不含 0
    且超出带(此时 |均值差| 必超过 δ 的一半以上,方向由区间端点定)。

    正态性:Shapiro-Wilk p ≥ 0.05 走 ttest_rel 的置信区间;否则 Wilcoxon
    符号秩区间兜底。所用方法、正态性 p 值与结论说明全部记进返回值
    (判定式随结论落盘,体例照 solve/scoring.py 三分支)。
    """
    arr = _require_finite_1d(diff, "配对差值样本")
    n = int(arr.size)
    if n < 3:
        raise ValueError(
            f"配对差值样本只有 {n} 个:至少要 3 个才能做正态性判断与置信区间,"
            f"请检查折级(或姿态级)误差表是否齐全。"
        )
    if not (float(tie_band_px) > 0.0) or not (float(tie_band_rel) >= 0.0):
        raise ValueError(
            f"等价带不合法:tie_band_px={tie_band_px} 必须大于 0,"
            f"tie_band_rel={tie_band_rel} 必须 >= 0(区间校验在 config 侧已锁,"
            f"这里是最后防线)。"
        )
    if simpler_mean is not None:
        sm = float(simpler_mean)
        if not np.isfinite(sm) or sm < 0.0:
            raise ValueError(
                f"simpler_mean = {simpler_mean!r} 不合法:必须是 >= 0 的有限数"
                f"(较简模型的平均 held-out 误差),相对等价带才算得出来。"
            )
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"置信水平 alpha = {alpha} 不合法:必须在 0 到 1 之间。")

    delta = (
        float(tie_band_px)
        if simpler_mean is None
        else float(max(float(tie_band_px), float(tie_band_rel) * abs(float(simpler_mean))))
    )
    mean_diff = float(np.mean(arr))

    if float(np.ptp(arr)) == 0.0:
        # 差值全是同一个数(含全 0):没有波动可言,区间退化成一个点
        ci_low = ci_high = mean_diff
        sp, w_method, w_p = float("nan"), "-", float("nan")
        method = "ttest_rel"  # 无波动,任何检验都给出同一答案;记 ttest_rel 名义路径
        normal = False
    else:
        sp = float(shapiro(arr).pvalue)
        normal = bool(sp >= _SHAPIRO_ALPHA)
        if normal:
            res = ttest_rel(arr, np.zeros_like(arr))
            ci = res.confidence_interval(alpha)
            ci_low, ci_high = float(ci.low), float(ci.high)
            method, w_method, w_p = "ttest_rel", "-", float("nan")
        else:
            ci_low, ci_high, w_method = _wilcoxon_ci(arr, alpha)
            w_res = wilcoxon(arr, method="auto")
            w_p = float(w_res.pvalue)
            method = "wilcoxon"

    band_txt = (
        f"±{delta:.4f}(绝对档 {tie_band_px} px)"
        if simpler_mean is None
        else f"±{delta:.4f}(= max({tie_band_px} px, {tie_band_rel:.0%} × 较简模型均值 "
        f"{abs(float(simpler_mean)):.4f}))"
    )
    if ci_high <= delta and ci_low >= -delta:
        verdict = "equivalent"
        note = (
            f"实际等价:差值均值 {mean_diff:+.4f} px 的 {alpha:.0%} 置信区间 "
            f"[{ci_low:.4f}, {ci_high:.4f}] 整体在等价带 {band_txt} 内,判 equivalent"
            f"——两模型差距小到可视为相同,按 D-09 取参数更简者。"
        )
    elif ci_low <= 0.0 <= ci_high:
        verdict = "tie"
        note = (
            f"统计平局:置信区间 [{ci_low:.4f}, {ci_high:.4f}] 含 0 但超出等价带 "
            f"{band_txt}(区间太宽,谁好谁坏判不定),按 D-10 交 BIC 裁决"
            f"(BIC 更小者胜,不做硬门槛)。"
        )
    else:
        verdict = "different"
        note = (
            f"真实差异:置信区间 [{ci_low:.4f}, {ci_high:.4f}] 不含 0(均值差 "
            f"{mean_diff:+.4f} px 已出等价带 {band_txt}),判 different——差异真实存在,"
            f"取 held-out 误差更小者。"
        )

    return {
        "verdict": verdict,
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "delta": delta,
        "n": n,
        "alpha": alpha,
        "normal": normal,
        "shapiro_p": sp,
        "method": method,
        "wilcoxon_method": w_method,
        "wilcoxon_p": w_p,
        "note": note,
    }


def bic_full_fit(rss: float, n_params: int, n_poses: int = 80) -> float:
    """全量拟合解的 BIC:n·ln(RSS/n) + k·ln(n)(D-10,只在统计平局时出场)。

    白话:BIC 给"拟合得好"与"参数多"各记一笔账,参数越多罚得越重,
    越小越好。n 锁定 80 个有效姿态——残差个数虽有 11 万+,但被 80 个
    姿态强相关;若写 640,ln(n) 对复杂模型的惩罚弱化约 8 倍,选型会系统
    性偏向复杂模型(Phase 3 研究已锁死)。k 写全部被估参数(内参数 +
    6×视图数外参):各模型都加同一常数 6V·ln(n),比较结果不变,但"按全部
    被估参数计"是教科书口径,写全可辩护。

    rss = 0(完美拟合,对数无定义且物理上可疑)返回 +inf,让它在 BIC
    比较里绝不获胜;rss < 0 / NaN、n_params < 1、n_poses < 1 中文报错。
    n_poses 由调用方传 cfg.evaluate.bic_n_poses(默认 80)。
    """
    rss = float(rss)
    if np.isnan(rss) or rss < 0.0:
        raise ValueError(
            f"BIC 输入 rss = {rss!r} 不合法:残差平方和必须是非负有限数,请检查上游的残差统计。"
        )
    if int(n_params) < 1 or int(n_poses) < 1:
        raise ValueError(
            f"BIC 输入不合法:n_params={n_params}、n_poses={n_poses},"
            f"参数个数与有效姿态数都必须 >= 1。"
        )
    if rss == 0.0 or not np.isfinite(rss):
        return float("inf")
    n, k = int(n_poses), int(n_params)
    return float(n * np.log(rss / n) + k * np.log(n))


def mad_outlier_poses(values: dict[str, float], threshold: float) -> list[tuple[str, float]]:
    """留一姿态误差的问题姿态点名:稳健 z 分数超过阈值判离群(D-08)。

    z = |x − 中位数| / MAD(scale='normal')。用 MAD(中位数绝对偏差)而
    不用"均值 ± 标准差",是因为离群点自己会把标准差撑大,躲过点名;
    MAD 对离群值不敏感,点名才点得准。MAD = 0(全体值相同)时返回空表:
    全体一个样,无从谈离群——此兜底防"分母为零"误报;若真出现全体相同,
    说明实验本身有问题,该去看逐姿态误差表,而不是在这里删姿态。阈值由
    调用方传 cfg.evaluate.mad_z_threshold(config 化,默认 3.5)。按 z 从
    大到小返回 [(姿态编号, z 分数)]。
    """
    if not values:
        return []
    names = [str(k) for k in values]
    arr = np.asarray([float(v) for v in values.values()], dtype=np.float64)
    bad = [nm for nm, v in zip(names, arr) if not np.isfinite(v)]
    if bad:
        raise ValueError(
            f"留一姿态误差里有 {len(bad)} 个非有限值(NaN/inf):{bad[:5]}"
            f"{'...' if len(bad) > 5 else ''}。这些多半是求解失败的姿态,"
            f"请在上游按失败记录剔除后再点名,非有限值不能静默参与判定。"
        )
    med = float(np.median(arr))
    mad = float(median_abs_deviation(arr, scale="normal"))
    if mad == 0.0:
        return []
    z = np.abs(arr - med) / mad
    out = [(nm, float(zi)) for nm, zi in zip(names, z) if zi > float(threshold)]
    out.sort(key=lambda item: item[1], reverse=True)
    return out


def select_model(verdicts: list[dict], cv_summary: dict, cfg: CalibrationConfig) -> dict:
    """选型规则:三判据全 PASS 才当选;held-out 最小者胜;平局取简、仍平按 BIC。

    占位防误用:本函数在任务 2(卡方告警线 + 三判据判行表 + 选型规则)
    实现;先立签名把接口定死,调用方(04-04 之后的编排层)不会误以为
    它已经能判。
    """
    raise NotImplementedError(
        "select_model 尚未实现(任务 2 交付):三判据判行表组装与选型规则还在路上,"
        "请不要在生产路径调用。"
    )
