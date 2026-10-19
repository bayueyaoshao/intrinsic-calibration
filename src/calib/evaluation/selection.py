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
    chi2,
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


def crlb_alert_check(
    empirical_std: dict[str, float],
    crlb_se_mean: dict[str, float],
    n_groups: int,
    alpha_hard: float,
    alpha_soft: float,
) -> dict:
    """「实测好于理论下界」的两档告警检查(D-05/D-06)。

    白话:重复张分组实验给出每个参数的经验标准差(同样的 80 个姿态分
    成 n_groups 组各标一遍,看参数抖多厉害);CRLB(克拉美-罗下界)是
    "这批数据在理论上能把参数定到多准"的下界。实测抖动若显著低于理论
    下界,物理上不可能——多半是对照口径错了(比如拿被 σ/√8 折减过的
    聚合记录来比),必须硬告警不静默通过(ROADMAP 成功标准 3)。

    数学:每组 n_groups 份参数,经验标准差 s 满足 (n−1)·s²/σ² 服从卡方
    (自由度 = n−1)分布(样本方差的波动规律);因此
    s < sqrt(chi2.ppf(alpha, df) / df) × σ_CRLB 即"以显著性 alpha 显著
    低于下界"。alpha=0.01 硬告警(df=7 时线约 0.42,判 FAIL + D-06 告警),
    alpha=0.05 边缘提示(线约 0.56,只记录不拦);df = 组数 − 1,随数据
    自适应。σ_CRLB 为 inf 的参数(数据里没有信息定住它的方向)跳过检查、
    计入 n_no_bound——既不进 FAIL 也不进 PASS 的分母;实测值非有限的参数
    同样跳过(测不出来 ≠ 好于下界)。

    返回体例照 noise/fisher.py 的 verdict/verdict_note:判定式与阈值随
    结论落盘,不凭空下结论。
    """
    if int(n_groups) < 2:
        raise ValueError(
            f"分组实验组数 n_groups = {n_groups} 不合法:至少要 2 组才谈得上"
            f"经验标准差(自由度 = 组数 − 1,组数 1 时自由度为 0)。"
        )
    df = int(n_groups) - 1
    s_hard = float(np.sqrt(chi2.ppf(float(alpha_hard), df) / df))
    s_soft = float(np.sqrt(chi2.ppf(float(alpha_soft), df) / df))

    per_param: dict[str, dict] = {}
    n_below_hard = 0
    n_below_soft = 0
    n_no_bound = 0
    hard_hits: list[str] = []
    soft_only_hits: list[str] = []
    n_checked = 0
    for name, s_emp in empirical_std.items():
        name = str(name)
        sigma = crlb_se_mean.get(name)
        s_emp = float(s_emp)
        if sigma is None or not np.isfinite(s_emp) or not np.isfinite(float(sigma)):
            n_no_bound += 1
            per_param[name] = {
                "ratio": None,
                "below_hard": False,
                "below_soft": False,
                "skipped": (
                    "理论下界缺失或为 inf(数据定不住的方向),或实测值非有限"
                    "——跳过检查,计入 n_no_bound"
                ),
            }
            continue
        n_checked += 1
        ratio = s_emp / float(sigma)
        below_hard = bool(ratio < s_hard)
        below_soft = bool(ratio < s_soft)
        n_below_hard += int(below_hard)
        n_below_soft += int(below_soft)
        per_param[name] = {
            "ratio": float(ratio),
            "below_hard": below_hard,
            "below_soft": below_soft,
            "skipped": None,
        }
        if below_hard:
            hard_hits.append(f"{name}(实测/理论 = {ratio:.3f},硬告警线 {s_hard:.2f})")
        elif below_soft:
            soft_only_hits.append(f"{name}(实测/理论 = {ratio:.3f},软提示线 {s_soft:.2f})")

    lines_txt = (
        f"告警线:s < sqrt(chi2.ppf(alpha, df)/df) × σ_CRLB,df = {df}"
        f"(= 组数 {n_groups} − 1),硬线 {s_hard:.4f}(alpha={alpha_hard}),"
        f"软线 {s_soft:.4f}(alpha={alpha_soft})。"
    )
    if hard_hits:
        verdict = "fail"
        verdict_note = (
            f"硬告警(D-06):{n_below_hard} 个参数实测显著低于同口径 CRLB 下界,"
            f"好于理论极限物理上不可能,判 fail 不静默——"
            + ";".join(hard_hits)
            + "。最可疑的原因是对照口径不一致(禁止拿 poses80 聚合 CRLB 对照"
            f"单张口径,见 D-05);另有 {n_no_bound} 个参数无理论下界跳过检查。" + lines_txt
        )
    else:
        verdict = "pass"
        verdict_note = (
            f"全部 {n_checked} 个可查参数的实测/理论比值都不低于硬告警线"
            f" {s_hard:.2f},判 pass"
            + (
                f";{n_below_soft} 个参数触发软提示线(边缘,只记录不拦):" + ";".join(soft_only_hits)
                if soft_only_hits
                else ""
            )
            + f";另有 {n_no_bound} 个参数无理论下界跳过检查。"
            + lines_txt
        )
    return {
        "per_param": per_param,
        "n_below_hard": n_below_hard,
        "n_below_soft": n_below_soft,
        "n_no_bound": n_no_bound,
        "n_checked": n_checked,
        "df": df,
        "lines": {"hard": s_hard, "soft": s_soft},
        "verdict": verdict,
        "verdict_note": verdict_note,
    }


def build_verdict_table(
    cv_summary: dict,
    stability_summary: dict,
    crlb_summary: dict,
    fisher_warnings: dict,
    thresholds: dict,
) -> dict:
    """三判据判行表:每个候选模型一行,三判据各记数值 + PASS/FAIL + 判定式(D-12)。

    输入全是 {model_id: {...}} 纯字典(接口先行,形状在此定死,
    04-04/04-05/04-06 照此供给):
      * cv_summary[mX] = {"mean_heldout_px", "mean_train_px",
                          "fold_heldout"(折级 held-out 列表), "n_failed_folds"}
      * stability_summary[mX] = {"max_disp_px", "slice_max_disp_px"}
        (bootstrap 与分组离散的像素位移当量)
      * crlb_summary[mX] = crlb_alert_check 的返回值
      * fisher_warnings[mX] = str(crlb.json 谱退化结论,可缺;原样进 notes,D-11)
      * thresholds = {"gen_overfit_ratio_max", "stability_disp_px_max"}
        (从 EvaluateConfig 取)
    判据(不搞加权打分,D-12):
      泛化 PASS = mean_heldout / mean_train < gen_overfit_ratio_max
      (过拟合比:考题分数相对练习分数的倍数,超线即过拟合显形,D-11 机制)
      稳定 PASS = max(bootstrap, 分组) 像素位移当量 < stability_disp_px_max
      逼近 CRLB PASS = n_below_hard == 0(实测好于理论下界即 FAIL + 硬告警)
    泛化行记录失败折数(n_failed_folds)——失败率进判行表备注(D-11)。
    返回 {"schema": "evaluate-verdict-v1", "verdicts": [...], "thresholds":
    实际用值}(T-04-03-02:阈值越界会扭曲判定,把实际用值随表落盘可复核)。
    """
    for key in ("gen_overfit_ratio_max", "stability_disp_px_max"):
        if key not in thresholds:
            raise ValueError(
                f"判行表阈值缺键:{key}(应有 gen_overfit_ratio_max 与 "
                f"stability_disp_px_max 两键,从 cfg.evaluate 取)。"
            )
    gen_max = float(thresholds["gen_overfit_ratio_max"])
    stab_max = float(thresholds["stability_disp_px_max"])

    verdicts: list[dict] = []
    for model_id, cv in cv_summary.items():
        model_id = str(model_id)
        if model_id not in stability_summary:
            raise ValueError(
                f"模型 {model_id} 缺稳定性摘要(stability_summary):判行表三判据"
                f"缺一不可,请检查 04-04/04-05 的实验汇总是否齐全。"
            )
        if model_id not in crlb_summary:
            raise ValueError(
                f"模型 {model_id} 缺 CRLB 对照摘要(crlb_summary):判行表三判据"
                f"缺一不可,请检查重复张分组实验与同口径 CRLB 是否已算。"
            )
        held = float(cv["mean_heldout_px"])
        train = float(cv["mean_train_px"])
        if not (np.isfinite(held) and np.isfinite(train)):
            raise ValueError(
                f"模型 {model_id} 的 CV 误差含非有限值(heldout={held}, "
                f"train={train}):统计判定拒绝带病输入,请先在上游剔除失败记录。"
            )
        if train <= 0.0:
            raise ValueError(
                f"模型 {model_id} 的训练侧平均误差 = {train}:必须是正数,"
                f"过拟合比(held-out/训练)才算得出来。"
            )
        ratio = held / train
        n_failed = int(cv.get("n_failed_folds", 0))
        gen_pass = bool(ratio < gen_max)
        gen_note = (
            f"泛化判据:mean_heldout/mean_train = {held:.4f}/{train:.4f} = "
            f"{ratio:.3f},须 < {gen_max}(过拟合比执行位),判 "
            f"{'PASS' if gen_pass else 'FAIL'}。"
        )

        d_boot = float(stability_summary[model_id]["max_disp_px"])
        d_slice = float(stability_summary[model_id]["slice_max_disp_px"])
        if not (np.isfinite(d_boot) and np.isfinite(d_slice)):
            raise ValueError(
                f"模型 {model_id} 的参数位移当量含非有限值(boot={d_boot}, "
                f"slice={d_slice}):统计判定拒绝带病输入。"
            )
        d_overall = max(d_boot, d_slice)
        stab_pass = bool(d_overall < stab_max)
        stab_note = (
            f"稳定判据:max(bootstrap {d_boot:.4f}, 分组 {d_slice:.4f}) = "
            f"{d_overall:.4f} px,须 < {stab_max} px(位移当量执行位),判 "
            f"{'PASS' if stab_pass else 'FAIL'}。"
        )

        crlb_block = crlb_summary[model_id]
        crlb_pass = bool(crlb_block.get("n_below_hard", 0) == 0)
        crlb_note = str(crlb_block.get("verdict_note", ""))

        notes: list[str] = []
        warning = fisher_warnings.get(model_id)
        if warning:
            notes.append(str(warning))  # Fisher 预警原样进 notes(D-11)
        if n_failed:
            notes.append(
                f"折失败率:{n_failed}/{len(cv.get('fold_heldout', []))} 折求解失败"
                f"(失败解未计入 held-out 均值;复杂模型小训练集不收敛是数据不是事故,"
                f"详见逐折明细)。"
            )
        overall = "PASS" if (gen_pass and stab_pass and crlb_pass) else "FAIL"
        verdicts.append(
            {
                "model_id": model_id,
                "generalization": {
                    "mean_heldout_px": held,
                    "mean_train_px": train,
                    "overfit_ratio": float(ratio),
                    "n_failed_folds": n_failed,
                    "pass": gen_pass,
                    "note": gen_note,
                },
                "stability": {
                    "max_disp_px": d_boot,
                    "slice_max_disp_px": d_slice,
                    "max_disp_px_overall": float(d_overall),
                    "pass": stab_pass,
                    "note": stab_note,
                },
                "crlb": {
                    "n_below_hard": crlb_block.get("n_below_hard"),
                    "n_below_soft": crlb_block.get("n_below_soft"),
                    "n_no_bound": crlb_block.get("n_no_bound"),
                    "lines": crlb_block.get("lines"),
                    "pass": crlb_pass,
                    "note": crlb_note,
                },
                "overall": overall,
                "notes": notes,
            }
        )
    return {
        "schema": "evaluate-verdict-v1",
        "verdicts": verdicts,
        "thresholds": {"gen_overfit_ratio_max": gen_max, "stability_disp_px_max": stab_max},
    }


def select_model(verdicts: list[dict], cv_summary: dict, cfg: CalibrationConfig) -> dict:
    """选型规则:三判据全 PASS 才当选;held-out 最小者胜;平局取简、仍平按 BIC(D-09/D-10/D-12)。

    verdicts:build_verdict_table 返回的 "verdicts" 行列表;行内可带 "bic" 键
    (调用方在全量重训解上用 bic_full_fit 算好写入——BIC 只在统计平局时
    真正出场当裁决器,D-10,不做硬门槛)。cv_summary 同判行表输入(要用
    fold_heldout 与 mean_heldout_px)。cfg:CalibrationConfig(读
    evaluate.tie_band_px / tie_band_rel,照切分函数"取全量配置"的既定体例)。

    判定顺序:三判据全 PASS 者才进候选(D-12,不搞加权打分);候选里
    held-out 均值最小者排第一;第一名与第二名做配对检验——
      * 实际等价 → 取参数更简者(D-09,_MODEL_PARAM_COUNTS);
      * 统计平局 → BIC 更小者裁决(D-10);行内没有可用 BIC 时退回取简,
        并在理由里说明 BIC 缺席(不静默冒充);
      * 真实差异 → held-out 最小者胜。
    无全 PASS 候选 → selected 为 None,reasons 逐模型记录未当选原因。

    返回 {"selected", "reason", "reasons", "selection_rule", "comparison"}
    (selection_rule 为中文规则说明,选型结论可考,T-04-03-03;comparison
    是 paired_comparison 的完整 meta,等价带值/区间/所用方法全在)。
    """
    rule = (
        "三判据(泛化 / 稳定 / 逼近 CRLB)全 PASS 才能当选,不搞加权打分(D-12);"
        "多个全 PASS 时 held-out 最小者胜;第一名与第二名的差距落进等价带判实际等价,"
        "取参数更简者(D-09);统计平局(置信区间含 0 但超出等价带)按 BIC 更小者裁决"
        "(D-10)。"
    )
    passing = [v for v in verdicts if v.get("overall") == "PASS"]
    if not passing:
        reasons: dict[str, str] = {}
        for v in verdicts:
            mid = str(v.get("model_id"))
            failed = [
                cname
                for key, cname in (
                    ("generalization", "泛化"),
                    ("stability", "稳定"),
                    ("crlb", "逼近 CRLB"),
                )
                if not (v.get(key) or {}).get("pass")
            ]
            reasons[mid] = (
                f"未当选:三判据未全过({'、'.join(failed) if failed else '无判定'})"
                f"——各项数值与判定式见判行表该行。"
            )
        return {
            "selected": None,
            "reason": (
                "没有任何候选模型三判据全 PASS——当前数据条件下选不出可辩护的模型,"
                "未当选原因逐模型记录在 reasons(不硬选,等数据条件改善或模型调整)。"
            ),
            "reasons": reasons,
            "selection_rule": rule,
            "comparison": None,
        }

    ranked = sorted(passing, key=lambda v: float(cv_summary[v["model_id"]]["mean_heldout_px"]))
    best = ranked[0]
    best_cv = cv_summary[best["model_id"]]
    if len(ranked) == 1:
        mid = best["model_id"]
        return {
            "selected": mid,
            "reason": (
                f"{mid} 是唯一三判据全 PASS 的候选,直接当选(held-out 均值 "
                f"{float(best_cv['mean_heldout_px']):.4f} px)。"
            ),
            "reasons": {},
            "selection_rule": rule,
            "comparison": None,
        }

    second = ranked[1]
    second_cv = cv_summary[second["model_id"]]
    # 前两名配对比较:同折相减消折间波动;等价带用两者中较简模型的均值当相对档
    simpler_of_two = (
        best["model_id"]
        if _MODEL_PARAM_COUNTS[best["model_id"]] <= _MODEL_PARAM_COUNTS[second["model_id"]]
        else second["model_id"]
    )
    cmp_res = paired_comparison(
        np.asarray(best_cv["fold_heldout"], dtype=np.float64)
        - np.asarray(second_cv["fold_heldout"], dtype=np.float64),
        tie_band_px=cfg.evaluate.tie_band_px,
        tie_band_rel=cfg.evaluate.tie_band_rel,
        simpler_mean=float(cv_summary[simpler_of_two]["mean_heldout_px"]),
    )
    if cmp_res["verdict"] == "equivalent":
        selected = simpler_of_two
        other = second["model_id"] if selected == best["model_id"] else best["model_id"]
        reason = (
            f"实际等价取简({cmp_res['note']})——按 D-09 取参数更简的 {selected}"
            f"({_MODEL_PARAM_COUNTS[selected]} 参,{other} 为 "
            f"{_MODEL_PARAM_COUNTS[other]} 参)。"
        )
    elif cmp_res["verdict"] == "tie":
        bic_best = best.get("bic")
        bic_second = second.get("bic")
        if (
            bic_best is not None
            and bic_second is not None
            and np.isfinite(float(bic_best))
            and np.isfinite(float(bic_second))
        ):
            selected = (
                best["model_id"] if float(bic_best) <= float(bic_second) else second["model_id"]
            )
            winner_bic = min(float(bic_best), float(bic_second))
            reason = (
                f"统计平局({cmp_res['note']})——按 D-10 由 BIC 裁决:{selected} 的 "
                f"BIC {winner_bic:.2f} 更小,当选(BIC 只作裁决器,不做硬门槛)。"
            )
        else:
            selected = simpler_of_two
            reason = (
                f"统计平局({cmp_res['note']})——判行表行内没有可用的 BIC 数值,"
                f"退回 D-09 取参数更简的 {selected}。若需 BIC 裁决,请检查全量重训解"
                f"是否已用 bic_full_fit 算好并写进行内 bic 键。"
            )
    else:
        selected = best["model_id"]
        reason = (
            f"真实差异({cmp_res['note']})——held-out 最小的 {selected} 当选"
            f"(均值 {float(best_cv['mean_heldout_px']):.4f} px vs "
            f"{float(second_cv['mean_heldout_px']):.4f} px)。"
        )
    return {
        "selected": selected,
        "reason": reason,
        "reasons": {},
        "selection_rule": rule,
        "comparison": cmp_res,
    }
