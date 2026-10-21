"""姿态切分纯函数库:防作弊考试出题的全部规则(分折 / 重采样 / 取第 i 张 / 留一)。

白话:阶段四的所有实验都遵循同一条铁律——**一切切分只操作姿态 ID 列表;
同一姿态的全部连拍照片同进同出,绝不按图像行随机划分**。这是结构性防
数据泄漏:同一姿态的连拍几乎是同一张照片,若让它的一部分进训练侧、
另一部分留在验证侧,模型等于"考试时见过原题",held-out 泛化误差会
假性好看。观测集的姿态编号字段(pose_ids)就是为这条铁律预留的连接键。

零重叠硬门禁(ROADMAP 成功标准 1):split_by_pose_ids 在函数体内断言
训练/验证两侧姿态交集为空;assign_cv_folds 断言每轮折间零交集且并集
等于全体姿态。单元测试(test_evaluation_split.py)再对全部轮次/折做
显式断言——函数体断言 + 单元测试双保险。

可复现承诺:分折与 bootstrap 重采样的随机性只从全局种子(config 的
seed,默认 42)按标签确定性派生(_seed_for,体例同 noise/injection.py)
——同配置重跑逐位一致,这是折分配表可落盘复查(D-03)与实验可复现的
根基。本模块全是纯函数:输入姿态清单或观测集,输出纯 str/int 的分配表
(可 json/npz 落盘),不改数据、不做求解、不落盘。
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence

import numpy as np

from calib.io.config import CalibrationConfig
from calib.solve.observation import ObservationSet


def _pose_groups(obs: ObservationSet) -> dict[str, list[int]]:
    """按姿态归组行号(保持清单里首次出现的顺序,读产物的人好对照)。"""
    groups: dict[str, list[int]] = {}
    for k, pose in enumerate(obs.pose_ids):
        groups.setdefault(pose, []).append(k)
    return groups


def unique_poses(obs: ObservationSet) -> list[str]:
    """观测集的去重姿态清单(保持首现顺序,体例照 noise/fisher.py 的按姿态归组)。"""
    return list(_pose_groups(obs))


def pose_row_mask(obs: ObservationSet, poses: Sequence[str] | set[str]) -> np.ndarray:
    """给定姿态集合,返回长度 = 观测集行数的布尔掩码(这些姿态的行全为 True)。"""
    wanted = set(poses)
    return np.array([p in wanted for p in obs.pose_ids], dtype=bool)


def obs_rows(obs: ObservationSet, mask: np.ndarray) -> ObservationSet:
    """按布尔行掩码选行,重建观测集(行序契约原样保持)。

    六字段逐一切片:image_ids / pose_ids / image_points / corner_source
    跟着掩码走;object_points(物点网格)与 image_size(图像尺寸)与行
    无关,原样传递。掩码形状与行数不符当场中文报错——行序错位会静默
    选错行,是切分层最不能容忍的事故(T-04-01-02)。
    """
    m = np.asarray(mask, dtype=bool)
    n = len(obs.image_ids)
    if m.shape != (n,):
        raise ValueError(
            f"行掩码形状 {tuple(m.shape)} 与观测集行数 {n} 对不上"
            f"(掩码长度必须是 {n})。掩码应由 pose_row_mask 生成,"
            f"或检查调用处传入的数组是否是同一份观测集的。")
    keep = np.flatnonzero(m)
    return ObservationSet(
        image_ids=tuple(obs.image_ids[i] for i in keep),
        pose_ids=tuple(obs.pose_ids[i] for i in keep),
        object_points=obs.object_points,
        image_points=obs.image_points[m],
        image_size=obs.image_size,
        corner_source=tuple(obs.corner_source[i] for i in keep),
    )


def split_by_pose_ids(obs: ObservationSet,
                      train_poses: Sequence[str] | set[str],
                      ) -> tuple[ObservationSet, ObservationSet]:
    """按训练姿态集合把观测集切成 (训练侧, 验证侧)。

    防泄漏铁律:验证姿态 = 全体姿态减去训练姿态,同一姿态的全部连拍
    同进同出。零重叠断言写在本函数体内(第一道防线)——训练/验证姿态
    交集必须为空,违反即 AssertionError,绝不带着泄漏往下算。

    用法约定(04-RESEARCH Pitfall 6):验证侧一律用 solve_pose_only 只估
    位姿来评分;验证姿态的外参绝不参与训练,否则 held-out 误差失去意义。
    """
    all_poses = set(unique_poses(obs))
    train_set = {str(p) for p in train_poses}
    unknown = sorted(train_set - all_poses)
    if unknown:
        raise ValueError(
            f"训练姿态清单里有观测集不存在的姿态 ID:{unknown}。"
            f"观测集实际包含 {len(all_poses)} 个姿态;两份产物多半不是同一轮的,"
            f"请检查切分清单与观测集是否同源。")
    if not train_set:
        raise ValueError(
            "训练侧姿态集合为空:没有训练姿态就无法标定内参,"
            "请至少留一个姿态在训练侧。")
    val_set = all_poses - train_set
    if not val_set:
        raise ValueError(
            f"验证侧姿态集合为空:训练姿态({len(train_set)} 个)占满了全体姿态"
            f"({len(all_poses)} 个),留不出考题。请至少留一个姿态在验证侧。")

    # ——零重叠硬门禁(结构性防泄漏,违反即停)——
    overlap = sorted(train_set & val_set)
    if overlap:
        raise AssertionError(
            f"切分泄漏:姿态 {overlap} 同时出现在训练侧与验证侧。"
            f"同一姿态的连拍必须同进同出,否则模型考试时见过原题,"
            f"held-out 误差不可信。请检查训练姿态清单是否与验证侧混用。")
    train_mask = pose_row_mask(obs, train_set)
    val_mask = pose_row_mask(obs, val_set)
    row_overlap = int(np.logical_and(train_mask, val_mask).sum())
    if row_overlap:
        raise AssertionError(
            f"切分泄漏:{row_overlap} 行照片同时落在训练侧与验证侧。"
            f"请检查观测集 pose_ids 是否有同一姿态以不同写法出现"
            f"(如空格/大小写差异)。")
    return obs_rows(obs, train_mask), obs_rows(obs, val_mask)


def _seed_for(label: str, base_seed: int) -> int:
    """从标签推导随机种子:base_seed + sha1(标签) mod 2**32。

    sha1 = 确定性散列(同一标签永远算出同一个数,不同标签几乎必然不同)。
    公式与 noise/injection.py 的 _seed_for 同一体例——全项目的随机性都
    从 config 的 seed(42)这样派生:同配置重跑逐位一致,这是"随机切分
    不落盘也能精确复现"的正当性来源。
    """
    digest = hashlib.sha1(label.encode()).hexdigest()
    return (base_seed + int(digest, 16)) % 2**32


def _checked_unique(pose_ids: Sequence[str], what: str) -> list[str]:
    """姿态清单的公共校验:非空、无重复,返回 str 列表(违规中文报错)。"""
    poses = [str(p) for p in pose_ids]
    if not poses:
        raise ValueError(f"{what}的姿态清单为空:至少需要 1 个姿态才能出题。")
    dupes = sorted(p for p, c in Counter(poses).items() if c > 1)
    if dupes:
        raise ValueError(
            f"{what}的姿态清单有重复:{dupes}。切分要求每个姿态恰好出现一次,"
            f"请先过 unique_poses 去重。")
    return poses


def assign_cv_folds(pose_ids: Sequence[str], cfg: CalibrationConfig) -> list[dict]:
    """交叉验证出题:每轮把全体姿态随机分成折,每折轮流当考题(D-01/D-03)。

    每轮 r 用种子 _seed_for(f"cv|round{{r}}|{{姿态数}}") 派生随机排列,
    再切成 cfg.evaluate.cv_n_folds 折;全部轮次累计每个姿态恰好
    cv_n_rounds 次进验证。返回
    [{"round": r, "fold": k, "train_poses": [...], "val_poses": [...]}],
    纯 str/int,可 json/npz 落盘复查(D-03:折分配表可考)。

    这里只出题(姿态层面的分配表);把分配落到观测集的行上、折内标定
    与评分,由 experiments 层消费本函数完成。
    """
    poses = _checked_unique(pose_ids, "交叉验证")
    n = len(poses)
    n_folds = cfg.evaluate.cv_n_folds
    if n_folds < 2 or n_folds > n:
        raise ValueError(
            f"交叉验证折数 {n_folds} 不合法:必须 >= 2 且 <= 姿态数 {n}"
            f"(折数多于姿态数会切出空折)。请检查 evaluate.cv_n_folds。")

    assignment: list[dict] = []
    for r in range(cfg.evaluate.cv_n_rounds):
        rng = np.random.default_rng(_seed_for(f"cv|round{r}|{n}", cfg.seed))
        folds = np.array_split(rng.permutation(n), n_folds)
        # ——零重叠硬门禁:本轮折间零交集且并集 = 全体姿态——
        seen: set[str] = set()
        for fold in folds:
            fold_poses = [poses[i] for i in fold.tolist()]
            dup = seen & set(fold_poses)
            if dup:
                raise AssertionError(
                    f"第 {r} 轮分折泄漏:姿态 {sorted(dup)} 进了多个折。"
                    f"随机排列切分本不该重复,出现说明种子派生或切片逻辑被改坏,"
                    f"请勿绕过本断言,先查 assign_cv_folds 的实现。")
            seen |= set(fold_poses)
        if seen != set(poses):
            missing = sorted(set(poses) - seen)
            raise AssertionError(
                f"第 {r} 轮分折不完整:姿态 {missing} 没进任何折。"
                f"每轮每个姿态必须恰进一个折,请检查 assign_cv_folds 的切片逻辑。")
        for k, fold in enumerate(folds):
            val_set = {poses[i] for i in fold.tolist()}
            assignment.append({
                "round": r,
                "fold": k,
                "train_poses": [p for p in poses if p not in val_set],
                "val_poses": [poses[i] for i in fold.tolist()],
            })
    return assignment


def bootstrap_resample(pose_ids: Sequence[str], cfg: CalibrationConfig) -> list[dict]:
    """bootstrap 出题:按姿态有放回重采样,模拟"换一批姿态重新标定"(D-04)。

    第 b 次重采样用种子 _seed_for(f"boot|rep{{b}}|{{姿态数}}") 有放回抽
    n 次(同一姿态可能抽中多次;抽中即带其全部连拍,由 experiments 层
    落行)。返回 [{"rep": b, "pose_ids": [...], "counts": {姿态: 抽中次数}}]
    ——纯 str/int 可落盘;4 个模型共用同一批分配表(共用由调用方保证,
    本函数只出一份考题)。
    """
    poses = _checked_unique(pose_ids, "bootstrap 重采样")
    n = len(poses)
    resamples: list[dict] = []
    for b in range(cfg.evaluate.bootstrap_n):
        rng = np.random.default_rng(_seed_for(f"boot|rep{b}|{n}", cfg.seed))
        picked = [poses[i] for i in rng.integers(0, n, size=n).tolist()]
        counts: dict[str, int] = {}
        for p in picked:
            counts[p] = counts.get(p, 0) + 1
        resamples.append({"rep": b, "pose_ids": picked, "counts": counts})
    return resamples


def slice_groups_by_index(obs: ObservationSet, cfg: CalibrationConfig) -> list[dict]:
    """重复张分组出题:每姿态按文件名排序后取第 i 张,组成数据组(D-05/EVA-03)。

    每组 = 每姿态 1 张,各组独立完整标定——量"纯噪声"对参数的影响
    (位姿集合固定,只有噪声不同)。组数 = 全体姿态的最少重复张数
    (数据驱动,不写死 8):重复数不足的姿态自然缺席更深的组,组间
    姿态集合单调不增(函数体内断言)。

    返回 [{"group": i, "pose_rows": {姿态: 行号}, "image_ids": [...]}]。
    """
    if not cfg.evaluate.slice_enabled:
        raise ValueError(
            "evaluate.slice_enabled 为 false,重复张分组实验已被配置关闭;"
            "请不要调用 slice_groups_by_index,或把开关改回 true。")
    ordered: dict[str, list[int]] = {
        pose: sorted(rows, key=lambda r: obs.image_ids[r])  # 文件名含时间戳,排序即时间序
        for pose, rows in _pose_groups(obs).items()}
    if not ordered:
        raise ValueError("观测集里没有任何姿态,无法分组。")
    n_groups = min(len(rows) for rows in ordered.values())
    out: list[dict] = []
    prev_poses: set[str] | None = None
    for i in range(n_groups):
        pose_rows = {pose: rows[i] for pose, rows in ordered.items()
                     if i < len(rows)}  # 重复数不足的姿态缺席该组
        rows_in_group = list(pose_rows.values())
        # ——断言:组内行互不重复(每姿态至多 1 张);组间姿态集合单调不增——
        if len(set(rows_in_group)) != len(rows_in_group):
            raise AssertionError(
                f"第 {i} 组里同一行照片被取了两次:{pose_rows}。"
                f"排序或取位逻辑被改坏,请检查 slice_groups_by_index。")
        if prev_poses is not None and not set(pose_rows) <= prev_poses:
            raise AssertionError(
                f"第 {i} 组的姿态集合没有单调不增:"
                f"出现了上一组没有的姿态 {sorted(set(pose_rows) - prev_poses)}。"
                f"分组应按'第 i 张'逐组加深,请检查 slice_groups_by_index。")
        prev_poses = set(pose_rows)
        out.append({"group": i, "pose_rows": pose_rows,
                    "image_ids": [obs.image_ids[r] for r in rows_in_group]})
    return out


def leave_one_pose_out(pose_ids: Sequence[str]) -> list[dict]:
    """留一姿态出题:每次留 1 个姿态当考题,其余全部训练(D-08)。

    无随机性:n 个姿态出 n 道题,两次调用恒等。
    返回 [{"val_pose": 姿态, "train_poses": 其余姿态}]。
    """
    poses = _checked_unique(pose_ids, "留一姿态")
    return [{"val_pose": p, "train_poses": [q for q in poses if q != p]}
            for p in poses]


def sample_pose_ids(pose_ids: Sequence[str], n: int, cfg: CalibrationConfig,
                    purpose: str = "lopo") -> list[str]:
    """从姿态名单里按种子抽 n 个(留一姿态抽查用,2026-10-21 裁定)。

    白话:全查 n 个姿态一个不落,抽查只考其中 n 个——抽哪几个由种子
    决定(体例同交叉验证分折:cfg.seed + 用途标签确定性派生,同配置
    重跑抽到同一批),不是"只看前 n 个"(那样排在后面的姿态永远没人查)。
    n >= 姿态数时等于全查,原样返回;抽样只决定"谁被单独考试",
    不改变训练侧用全体其余姿态的事实。

    返回保持原名单顺序的抽样子集(纯 str,可 json 落盘复查)。
    """
    poses = _checked_unique(pose_ids, f"{purpose}抽查")
    if n < 1:
        raise ValueError(
            f"{purpose} 抽查数 {n} 不合法:必须 >= 1。"
            f"请检查 evaluate.quick_lopo_n(不想抽查就把该档位数改成"
            f"姿态总数,或用 full 档全查)。")
    if n >= len(poses):
        return list(poses)
    rng = np.random.default_rng(_seed_for(f"{purpose}|sample{n}|{len(poses)}",
                                          cfg.seed))
    idx = np.sort(rng.choice(len(poses), size=n, replace=False))
    return [poses[i] for i in idx.tolist()]

