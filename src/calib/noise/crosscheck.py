"""OpenCV stdDev 交叉验证:把我们推导的精度界与 OpenCV 自己报的标准差对账。

本模块是成功标准 4 判据的落地(03-RESEARCH §Pattern 4 的固定三步协议):
同一个模型、同一批观测,分别算两份"每个内参定得多准"的数——
- 我们的 σ=1 精度界:边际化 Fisher 信息(03-02 的 marginalized_fisher,
  外参经 Schur 消元合并掉——白话:把不关心的位姿参数合并进误差计算,
  只留内参的精度)取逆开方,单位是"假设每个角点噪声 = 1 像素"的标准差
  理论下界(CRLB,任何无偏估计方法的方差都不会低于它);
- OpenCV 的 stdDeviationsIntrinsics:Extended 标定顺手报的每个内参
  "标准差",可理解为 OpenCV 自己算的精度界。

头号发现(先说破,免得被吓退——03-RESEARCH §Pattern 4 实测):两个向量
不差在参数顺序、也不差在数学结构,只差**一个公共倍数**——OpenCV 在
"单位噪声精度界"上乘了它自己从残差估出的噪声尺度(σ=0.1 注入下拟合出
的公共倍数 ≈ 0.1,与注入量一致)。所以直接比会"差一个倍数",那是两边
对"噪声多大"的约定不同,不是数学错。协议因此固定三步:
  ①经 models/ 适配器跑 Extended 标定,拿 OpenCV 解与它的标准差向量;
  ②把 OpenCV 解换成我们的参数布局,在该点上算我们的 σ=1 精度界
    (另在调用方给的线性化点上算一版作"换点不敏感"对照);
  ③拟合单一公共尺度 k 后逐参数比结构(< 个位数百分比 = 两边数学同构),
    绝对比对只作副报告并排存档(口径差,不设门槛)。
判读规则(03-RESEARCH Pitfall 1,也是本模块存在的理由):比值"所有参数
几乎同一个数" = 尺度因子问题(数学是对的,千万别回头改 FIM);逐参数
散乱 = 数学真有问题,才需要回头查。

边界与分级(D-07):只支持针孔族(适配器类型 calibrateCamera)。鱼眼
(M4)在 OpenCV 4.14 没有带标准差输出的标定函数(实测,只有普通标定),
返回 unavailable_fisheye 中文结论即算完成,绝不硬调鱼眼函数。M1/M2 硬
门槛 < 5%,M3 尽力对齐(对不上按发现记录),分级判断由调用方(03-05)
落盘时给出。失败契约:OpenCV 抛错转 status=failed + 中文说明,不抛异常
——交叉验证是 CRLB 主产物的附加件,它失败不阻塞主产物。

术语白话:线性化点 = 推导精度界时把非线性问题"拉直"所取的那组参数值;
公共尺度 k = 两套数之间只差的那一个倍数;检测器口径 = 把角点提取环节
实测出的噪声当作角点误差的计量约定。
"""

from __future__ import annotations

import cv2
import numpy as np

from calib.models.base import CameraModel
from calib.noise.fisher import crlb_se, marginalized_fisher
from calib.solve.observation import ObservationSet
from calib.solve.opencv import _prepare_view

# OpenCV stdDeviationsIntrinsics 的 18 元固定布局(4.14 实测,顺序与官方
# 畸变系数文档一致):针孔族三个模型的参数名是它的前 P 位同名同序(恒等
# 映射,tests/test_noise_crosscheck.py 静态锁定)。被固定选项锁死的参数
# (如 M1 的 k3,布局第 8 位)其标准差恒 0——按前 P 位取数天然把它剔除。
# s1..s4 / tau_x / tau_y 是本框架未启用的薄棱镜与倾斜镜头系数,只占位。
_OPENCV_INTRINSIC_ORDER: tuple[str, ...] = (
    "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3",
    "k4", "k5", "k6", "s1", "s2", "s3", "s4", "tau_x", "tau_y",
)


def _empty_result(status: str, note: str) -> dict:
    """失败/不可用状态的统一返回:结构键一个不少,数值字段一律 None。

    失败契约体例照 solve/result.py 的 make_failure:读到失败记录的代码
    不用先猜键在不在,直接按 status 分流即可。
    """
    return {
        "status": status,
        "failure_note": note,
        "params": None,
        "opencv_se": None,
        "opencv_se_raw": None,
        "rms": None,
        "per_view_errors": None,
        "rvecs": None,
        "tvecs": None,
    }


def opencv_stddev_intrinsics(model: CameraModel, obs: ObservationSet) -> dict:
    """经 models/ 适配器跑 Extended 标定,拿 OpenCV 解与它的内参标准差向量。

    协议第①步。返回 dict:
        status="ok" 时:
            params           OpenCV 解换成我们的参数布局(经适配器换算);
            opencv_se        OpenCV 的内参标准差,按参数表前 P 位对齐 (P,);
            opencv_se_raw    未对齐的完整 18 元向量(固定参数位恒 0,
                             天然对账位,审计用);
            rms / per_view_errors / rvecs / tvecs   OpenCV 解的其余产出;
        status="failed"      数值键全 None + 中文 failure_note(失败契约);
        status="unavailable_fisheye"   鱼眼族无可比输出(D-07 记结论即完成)。

    标定选项与 K/D 换算只走 model.opencv_adapter()(唯一入口,绝不手拼:
    M1 锁死 k3、M3 开有理模型的差异已封装在适配器里)。float32 降精度
    装包与 (n,1,dim) 变形体例照 solve/opencv.py(OpenCV 绑定只收 32 位
    浮点,像素坐标量级下约 1e-4 像素的表达精度,不污染 5% 门槛)。
    4.14 的 calibrateCameraExtended 返回 **8 元组**(实测,不是旧资料的
    11 元组):(retval, K, D, rvecs, tvecs, stdDevInt, stdDevExt,
    perViewErrors)——元组个数若随版本漂移,解包当场报错,测试覆盖
    真实调用路径会先红。
    """
    adapter = model.opencv_adapter()
    if adapter is None:
        raise AssertionError(
            f"模型 {model.model_id} 没有 OpenCV 适配器,做不了 stdDev 交叉验证:"
            f"交叉验证要求经适配器调用 Extended 标定、拿 OpenCV 自己报的标准差,"
            f"没有适配器的模型(玩具模型)没有这层入口。请改用针孔族模型(m1/m2/m3)。"
        )
    if adapter.kind == "fisheye":
        # M4 分支(D-07):鱼眼只有普通标定,没有带标准差输出的版本(4.14
        # 实测)——直接记中文结论返回,绝不硬调任何鱼眼标定函数
        return _empty_result(
            "unavailable_fisheye",
            "OpenCV 的鱼眼模块在 4.14 没有带标准差输出的标定函数(实测,"
            "只有普通标定),M4 无可比输出;按 D-07 记录此结论即算完成,"
            "不影响 CRLB 主产物。",
        )

    n = int(obs.object_points.shape[0])
    n_views = len(obs.image_ids)
    objs = [_prepare_view(obs.object_points, n, 3, "物点(交叉验证)")
            for _ in range(n_views)]
    imgs = [_prepare_view(p, n, 2, "角点(交叉验证)") for p in obs.image_points]
    objs32 = [p.astype(np.float32) for p in objs]
    imgs32 = [p.astype(np.float32) for p in imgs]

    try:
        retval, K, D, rvecs, tvecs, sd_int, _sd_ext, per_view = (
            cv2.calibrateCameraExtended(
                objs32, imgs32, tuple(obs.image_size), None, None,
                flags=int(adapter.flags)))
    except cv2.error as e:
        return _empty_result(
            "failed",
            f"{model.model_id} 模型的 stdDev 交叉验证未跑成:OpenCV 报错——{e}。"
            f"交叉验证是附加产物,失败不影响 CRLB 主产物;排障可先核对观测集"
            f"与棋盘规格是否配套(角点数、视图数、画面尺寸)。",
        )

    params = adapter.params_from_cv(np.asarray(K, dtype=np.float64),
                                    np.asarray(D, dtype=np.float64).ravel())
    se_raw = np.ravel(np.asarray(sd_int, dtype=np.float64))
    n_int = len(model.param_names)
    rvecs_arr = np.asarray([np.asarray(r, dtype=np.float64).ravel() for r in rvecs])
    tvecs_arr = np.asarray([np.asarray(t, dtype=np.float64).ravel() for t in tvecs])
    return {
        "status": "ok",
        "failure_note": None,
        "params": params,
        "opencv_se": se_raw[:n_int].copy(),
        "opencv_se_raw": se_raw,
        "rms": float(retval),
        "per_view_errors": np.asarray(per_view, dtype=np.float64).ravel(),
        "rvecs": rvecs_arr,
        "tvecs": tvecs_arr,
    }


def compare_structural(ours_se: np.ndarray, opencv_se: np.ndarray,
                       active_mask: np.ndarray | None = None) -> dict:
    """主判据(协议第③步):拟合单一公共尺度后逐参数比结构。

    做法:先拟合"opencv ≈ k·ours"里最小二乘意义最优的那个公共倍数
        k = Σ(opencv·ours) / Σ(ours²),
    再把 opencv 除以 k 归一,与 ours 逐参数比相对偏差。k 本身是量级
    对账位:注入多大噪声它就约等于多大(= OpenCV 从残差自估的噪声尺度),
    跑偏说明两边输入没对齐。

    判读规则(03-RESEARCH Pitfall 1,写死在这里免得口头失传):
    per_param_rel_dev"所有参数几乎同一个数" = 尺度因子没除干净(数学是
    对的);逐参数散乱 = 数学真有问题。

    参数:
        active_mask: 布尔数组,只比对打勾的参数。被标定选项锁死的参数
                     (如 M1 的 k3)其 OpenCV 标准差恒 0,必须先剔除——
                     不剔除会在归一化时除零出 NaN。缺省全部参数参与。
    返回:
        scale_k            公共尺度 k(上面公式的最小二乘解);
        per_param_rel_dev  与输入同长的逐参数相对偏差数组,未参与的位记
                           NaN 占位(便于按参数名对齐读数);
        max_rel_dev        参与参数里最大的相对偏差(主判据就比它);
        n_active           实际参与比对的参数个数。
    """
    s = np.asarray(ours_se, dtype=np.float64).ravel()
    o = np.asarray(opencv_se, dtype=np.float64).ravel()
    if s.ndim != 1 or s.shape != o.shape:
        raise AssertionError(
            f"两个标准差向量长度不齐:ours 形状 {s.shape},opencv 形状 {o.shape}。"
            f"必须逐参数一一对应(同一模型的参数表同序),请检查是否取自同一模型。"
        )
    if active_mask is None:
        mask = np.ones(s.shape[0], dtype=bool)
    else:
        mask = np.asarray(active_mask, dtype=bool).ravel()
        if mask.shape != s.shape:
            raise AssertionError(
                f"active_mask 形状 {mask.shape} 与标准差向量形状 {s.shape} 不齐,"
                f"掩码必须逐参数给布尔值。"
            )
    if not (np.isfinite(s[mask]).all() and (s[mask] > 0).all()):
        raise AssertionError(
            f"我们的 σ=1 精度界在参与比对的参数上必须全为正的有限值,实际 {s}。"
            f"出现 0/负数/非有限值说明 Fisher 信息矩阵组装有错,先查上游再比对。"
        )
    if not (np.isfinite(o[mask]).all() and (o[mask] > 0).all()):
        raise AssertionError(
            f"OpenCV 标准差在参与比对的参数上必须全为正的有限值,实际 {o}。"
            f"出现 0 说明把被固定选项锁死的参数也放进来了(它的标准差恒 0),"
            f"请先用 active_mask 或按参数表前 P 位取数把它剔除。"
        )

    k = float(np.sum(o[mask] * s[mask]) / np.sum(s[mask] ** 2))
    rel = np.full(s.shape[0], np.nan)
    rel[mask] = np.abs(o[mask] / k - s[mask]) / s[mask]
    return {
        "scale_k": k,
        "per_param_rel_dev": rel,
        "max_rel_dev": float(rel[mask].max()),
        "n_active": int(mask.sum()),
    }


def crosscheck_model(model: CameraModel, obs: ObservationSet,
                     linearization=None,
                     sigma_detector: float | None = None) -> dict:
    """顶层入口:交叉验证三步协议一步到位,给 03-05 直接调用(crlb.json
    的交叉验证段从这里取数)。

    参数:
        model:          模型实例(create_model 按名取);
        obs:            观测集(与标定/主 CRLB 同一份,输入不对齐则 k 量级
                        对账位会先暴露);
        linearization:  (params, rvecs, tvecs) 三元组——调用方指定的线性化点
                        (白话:推导精度界时把非线性问题拉直所取的参数值;
                        03-05 传阶段二的 scipy 精化解)。另在这个点上算一版
                        我们的 σ=1 精度界作"换点不敏感"对照,给 D-06 选口径
                        时引用;传 None 则跳过该对照;
        sigma_detector: 检测器口径的角点噪声(像素标准差,03-05 从 noise.json
                        读)。只进副报告的绝对比对,不设门槛;None 占位。

    返回(status="ok" 时,全部数字与状态,交 03-05 落 crlb.json):
        se_ours_opencv_point         我们的 σ=1 精度界(OpenCV 解的点上);
        se_ours_linearization_point  同上,但在指定线性化点上(None=未算);
        point_sensitivity_max_rel    两点之间的最大相对差(换点不敏感的度量);
        compare                      主判据(compare_structural 的结果);
        secondary_report             副报告:我们的精度界乘 σ_detector 与
                                     OpenCV 原值并排,差异解读为"双方对噪声
                                     多大的口径差",不设门槛;
        fisher_diagnostics           Fisher 信息组装诊断(视图数/条件数等)。
    status="failed" / "unavailable_fisheye" 时:桥接层的失败记录原样透传
    (数值键全 None + 中文 failure_note)——交叉验证失败不阻塞 CRLB 主产物。
    """
    # 第①步:OpenCV 解与它的标准差
    base = opencv_stddev_intrinsics(model, obs)
    if base["status"] != "ok":
        # 失败/不可用照样返回完整结构键(数值全 None):读 crlb.json 的
        # 代码不用先猜键在不在(失败契约,体例照 solve/result.py)
        return {
            **base,
            "model_id": model.model_id,
            "param_names": tuple(model.param_names),
            "se_ours_opencv_point": None,
            "se_ours_linearization_point": None,
            "point_sensitivity_max_rel": None,
            "compare": None,
            "secondary_report": {
                "sigma_detector": None,
                "ours_se_scaled": None,
                "note": base["failure_note"],
            },
            "fisher_diagnostics": None,
        }

    # 第②步:我们的 σ=1 精度界——先在 OpenCV 解的点上,再在指定线性化点上
    params_cv = base["params"]
    f_marg, _f_full, diagnostics = marginalized_fisher(
        model, obs, params_cv, base["rvecs"], base["tvecs"])
    se_at_opencv = crlb_se(f_marg, 1.0)

    se_at_linearization = None
    point_sensitivity = None
    if linearization is not None:
        lin_params, lin_rvecs, lin_tvecs = linearization
        f_marg_lin, _f_full_lin, _diag_lin = marginalized_fisher(
            model, obs, lin_params, lin_rvecs, lin_tvecs)
        se_at_linearization = crlb_se(f_marg_lin, 1.0)
        point_sensitivity = float(np.max(
            np.abs(se_at_linearization - se_at_opencv) / se_at_opencv))

    # 第③步:主判据(尺度归一结构比对)+ 副报告(绝对比对,不设门槛)
    comparison = compare_structural(se_at_opencv, base["opencv_se"])
    scaled = (None if sigma_detector is None
              else se_at_opencv * float(sigma_detector))
    return {
        "status": "ok",
        "failure_note": None,
        "model_id": model.model_id,
        "param_names": tuple(model.param_names),
        "params_opencv": params_cv,
        "opencv_se": base["opencv_se"],
        "opencv_se_raw": base["opencv_se_raw"],
        "rms": base["rms"],
        "per_view_errors": base["per_view_errors"],
        "rvecs": base["rvecs"],
        "tvecs": base["tvecs"],
        "se_ours_opencv_point": se_at_opencv,
        "se_ours_linearization_point": se_at_linearization,
        "point_sensitivity_max_rel": point_sensitivity,
        "compare": comparison,
        "secondary_report": {
            "sigma_detector": (None if sigma_detector is None
                               else float(sigma_detector)),
            "ours_se_scaled": scaled,
            "note": ("副报告(绝对比对,不设门槛):我们的 σ=1 精度界乘检测器"
                     "口径噪声,与 OpenCV 标准差原值并排;差异解读为双方对"
                     "噪声多大的口径差,不是数学对错(主判据见 compare)。"),
        },
        "fisher_diagnostics": diagnostics,
    }
