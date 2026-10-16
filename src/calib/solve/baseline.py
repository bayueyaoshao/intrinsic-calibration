"""2×2 基线标定(D-10):两颗检测器的角点 × 两类相机模型 = 四个基线并排。

流水线第④步的最小子集:只调 OpenCV 现成的求解函数,不发明算法。
- 针孔模型(pinhole):普通透视相机,带 5 个畸变系数(k1,k2,p1,p2,k3,
  描述镜头把直线拍弯程度的数),对应项目模型清单里的 M2;
- 鱼眼模型(fisheye):KB4 模型——"等距投影",即假设入射角越大、成像点
  离中心按角度均匀拉开,用 4 个系数 k1..k4 描述偏离。普通广角镜头不满足
  这个假设,求解可能不收敛:这是 D-10 预期内的正常发现,失败会被完整
  记录(结构键一个不少、数值字段一律 None),绝不中断其余组合。

名词白话:
- 重投影误差:把标定算出的相机参数拿去"重画"棋盘角点,重画位置与实际
  检测位置的差距(像素);RMS = 均方根(先平方、取平均、再开方),
  给整体误差一个有代表性的单一数字。
- square_size(方格物理边长)只在本文件的 build_object_points 出现——
  它只决定物点坐标的尺度(进而只影响"相机离棋盘多远"这类距离量),
  与任何像素级逻辑彻底解耦(IO-03:填错它只污染距离类指标,可被测试抓住)。
"""

from __future__ import annotations

import cv2
import numpy as np

from calib.io.config import BoardSpec, CalibrationConfig

# 四个基线组合的固定顺序(组合名 = baseline_<检测器>_<模型>)
DETECTORS = ("classic", "sb")
MODELS = ("pinhole", "fisheye")


def build_object_points(board: BoardSpec) -> np.ndarray:
    """构造物点网格:棋盘内角点在"棋盘平面"上的坐标,(cols*rows, 3) float64。

    Z 一律为 0(棋盘是平的),X/Y 按方格边长排开,单位毫米。
    square_size 只进这里——改错它只会把"相机离棋盘多远"算错,
    内参(焦距/主点)与全部像素级逻辑都不受影响。
    """
    xs = np.arange(board.cols, dtype=np.float64) * board.square_size_mm
    ys = np.arange(board.rows, dtype=np.float64) * board.square_size_mm
    objp = np.zeros((board.rows * board.cols, 3), dtype=np.float64)
    objp[:, 0] = np.tile(xs, board.rows)
    objp[:, 1] = np.repeat(ys, board.cols)
    return objp


def _prepare_view(points: np.ndarray, n_points: int, dim: int, what: str) -> np.ndarray:
    """把单视图的角点/物点整成 OpenCV 要求的 (n,1,dim) 形状,进函数先断言(陷阱 5)。

    fisheye 系函数对输入形状极挑剔:传 (n,2) 而非 (n,1,2) 会报出难懂的错误
    甚至崩溃——所以在进求解器之前就把形状查清楚,报错要一眼看懂。
    """
    arr = np.asarray(points)
    if arr.ndim == 2 and arr.shape == (n_points, dim):
        arr = arr.reshape(n_points, 1, dim)
    if arr.shape != (n_points, 1, dim):
        raise AssertionError(
            f"{what} 形状不合:期望每视图 ({n_points},{dim}) 或 ({n_points},1,{dim}),"
            f"实际是 {arr.shape}。角点数组必须与棋盘的 {n_points} 个内角点一一对齐,"
            f"请检查角点文件的规范化顺序是否正确。"
        )
    return np.ascontiguousarray(arr, dtype=np.float64)


def calibrate_pinhole(image_points_per_view: list[np.ndarray],
                      image_size: tuple[int, int],
                      board: BoardSpec) -> dict:
    """针孔模型基线:cv2.calibrateCameraExtended 求解,返回内参 + 逐帧 RMS。

    参数:
        image_points_per_view: 每视图的角点数组 (88,2) 或 (88,1,2)
        image_size:            (宽, 高),即 gray.shape[::-1]
                               (陷阱 8:直觉写 numpy 的 shape 会把宽高写反)
        board:                 棋盘规格(物点网格由此构造)

    返回字典键:rms_overall / K(3×3)/ D(一维)/ rvecs / tvecs /
    per_view_rms(逐帧 RMS 数组)/ flags_used / converged / failure_note。
    """
    n = board.cols * board.rows
    objp64 = build_object_points(board).reshape(n, 1, 3)
    objs64 = [objp64.copy() for _ in image_points_per_view]
    imgs64 = [_prepare_view(p, n, 2, "角点(针孔)") for p in image_points_per_view]

    # OpenCV 4.14 的 calibrateCamera 绑定只收 32 位浮点(Point2f/Point3f),
    # 传 64 位会报"Unsupported format"——这里显式降精度再调用:
    # 像素坐标最大几千,float32 约有 1e-4 像素的表达精度,远小于 0.5px 的断言线。
    objs32 = [p.astype(np.float32) for p in objs64]
    imgs32 = [p.astype(np.float32) for p in imgs64]

    flags = 0  # 默认:估计 5 个畸变系数 k1,k2,p1,p2,k3(项目模型 M2)
    rms, K, D, rvecs, tvecs, _std_int, _std_ext, per_view = cv2.calibrateCameraExtended(
        objs32, imgs32, tuple(image_size), None, None, flags=flags)

    return {
        "rms_overall": float(rms),
        "K": np.asarray(K, dtype=np.float64),
        "D": np.asarray(D, dtype=np.float64).ravel(),
        "rvecs": [np.asarray(r, dtype=np.float64) for r in rvecs],
        "tvecs": [np.asarray(t, dtype=np.float64) for t in tvecs],
        "per_view_rms": np.asarray(per_view, dtype=np.float64).ravel(),
        "flags_used": flags,
        "converged": True,
        "failure_note": None,
    }


def calibrate_fisheye(image_points_per_view: list[np.ndarray],
                      image_size: tuple[int, int],
                      board: BoardSpec, cfg) -> dict:
    """鱼眼 KB4 模型基线:cv2.fisheye.calibrate 求解,失败记录为发现、不抛异常。

    参数 cfg 是 FisheyeConfig(flags / max_iter / epsilon,来自 calibration.yaml)。
    flags 默认 2 = RECOMPUTE_EXTRINSIC(每轮迭代重算棋盘位姿)——实测发现
    若不带这一位(flags=0),OpenCV 只按初始内参算一次位姿、之后不随内参更新,
    会收敛到 RMS 上百像素的假结果;skew(倾斜因子,成像平面与镜头不垂直的
    偏离)仍被估计。flags 的实际取值显式记进产物(flags_used),不凭记忆下结论。

    K/D 传单位阵/零向量,交由 OpenCV 内部初始化:cx,cy 置图像中心,
    焦距按 max(宽,高)/π 最小二乘起步。逐帧 RMS 用与标定同一套投影函数
    (cv2.fisheye.projectPoints)把角点回代算出,天然可比。

    失败值语义(就此定死):数值字段一律 None(落盘成 JSON null),
    结构键一个不少——KB4 对非鱼眼广角数据可能报病态矩阵(Ill-conditioned,
    即输入数据让求解方程组"接近无解",数值不稳),这是 D-10 预期内的发现。
    """
    n = board.cols * board.rows
    objp64 = build_object_points(board).reshape(n, 1, 3)
    objs = [objp64.copy() for _ in image_points_per_view]
    imgs = [_prepare_view(p, n, 2, "角点(鱼眼)") for p in image_points_per_view]

    flags = int(cfg.flags)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
                int(cfg.max_iter), float(cfg.epsilon))
    try:
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objs, imgs, tuple(image_size), np.eye(3, dtype=np.float64),
            np.zeros(4, dtype=np.float64), flags=flags, criteria=criteria)
    except cv2.error as e:
        return {
            "rms_overall": None, "K": None, "D": None,
            "rvecs": None, "tvecs": None, "per_view_rms": None,
            "flags_used": flags, "converged": False,
            "failure_note": f"鱼眼 KB4 求解未收敛:OpenCV 报错——{e}",
        }

    K = np.asarray(K, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64).ravel()
    rvecs = [np.asarray(r, dtype=np.float64) for r in rvecs]
    tvecs = [np.asarray(t, dtype=np.float64) for t in tvecs]

    # 逐帧 RMS:把物点按求得的位姿重新投影(KB4 模型的投影函数),与检测角点比差
    per_view = np.empty(len(imgs), dtype=np.float64)
    for i, (obj, img) in enumerate(zip(objs, imgs)):
        proj, _jacobian = cv2.fisheye.projectPoints(obj, rvecs[i], tvecs[i], K, D)
        diff = proj.reshape(-1, 2) - img.reshape(-1, 2)
        per_view[i] = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

    return {
        "rms_overall": float(rms),
        "K": K, "D": D, "rvecs": rvecs, "tvecs": tvecs,
        "per_view_rms": per_view,
        "flags_used": flags,
        "converged": True,
        "failure_note": None,
    }


def run_baselines(corners_train: dict[str, dict],
                  image_size: tuple[int, int],
                  cfg: CalibrationConfig, board: BoardSpec) -> dict:
    """跑全部 2×2 = 4 个基线组合,先按剔除清单做逻辑剔除(D-03)。

    参数:
        corners_train: {检测器名: 角点 npz 读出的字典}——字典须含
                       image_ids (N,) / uv (N,88,2) / detected (N,)
                       (即 load_corners 的返回结构,行与数据清单对齐)
        image_size:    (宽, 高)
        cfg:           全局配置(剔除清单 exclude_images、鱼眼参数)
        board:         棋盘规格

    剔除规则:exclude_images 里列出的照片逻辑剔除(不进标定,原数据不动);
    未检出的帧(NaN 角点)进不了标定,同样跳过。两类剔除的幸存视图不足 2 个
    时直接报中文错误——两三个视角定不了内参,硬算必错。

    返回:
        {
            "baselines": {组合名: calibrate_* 的结果字典(另附 detector /
                          model / image_ids / image_size / n_points_per_view /
                          objp,供产物落盘与残差回代使用)},
            "excluded_images": [本轮实际被剔除的照片编号(按字典序)],
            "n_views": {检测器名: 进入标定的视图数},
        }
    """
    exclude = set(cfg.exclude_images)
    kept: dict[str, tuple[list[np.ndarray], list[str]]] = {}
    excluded_seen: list[str] = []
    n_views: dict[str, int] = {}

    for detector in DETECTORS:
        if detector not in corners_train:
            raise ValueError(
                f"缺少检测器「{detector}」的角点数据:基线标定需要 classic 与 sb 两份"
                f"train 角点文件,请先运行 calib corners。"
            )
        data = corners_train[detector]
        ids = [str(x) for x in np.asarray(data["image_ids"]).ravel()]
        uv = np.asarray(data["uv"], dtype=np.float64)
        detected = np.asarray(data["detected"], dtype=bool)
        views: list[np.ndarray] = []
        kept_ids: list[str] = []
        for i, image_id in enumerate(ids):
            if image_id in exclude:
                excluded_seen.append(image_id)
                continue  # 人工剔除清单:逻辑剔除,原数据不动(D-03)
            if not detected[i]:
                continue  # 未检出的帧是 NaN 角点,进不了标定
            if not np.isfinite(uv[i]).all():
                raise AssertionError(
                    f"检测器「{detector}」的照片 {image_id} 标记为已检出,"
                    f"但角点里混入了非数值(NaN/无穷),角点文件可疑,请重新运行 calib corners。"
                )
            views.append(uv[i])
            kept_ids.append(image_id)
        if len(views) < 2:
            raise ValueError(
                f"检测器「{detector}」剔除后只剩 {len(views)} 个可用视图"
                f"(检出且未被人工剔除),至少需要 2 个才能标定内参。"
                f"请检查 exclude_images 清单与角点检出率。"
            )
        kept[detector] = (views, kept_ids)
        n_views[detector] = len(views)

    objp = build_object_points(board)
    baselines: dict[str, dict] = {}
    for detector in DETECTORS:
        views, kept_ids = kept[detector]
        for model in MODELS:
            name = f"baseline_{detector}_{model}"
            if model == "pinhole":
                result = calibrate_pinhole(views, image_size, board)
            else:
                result = calibrate_fisheye(views, image_size, board,
                                           cfg.calibrate.fisheye)
            # 附上落盘与残差回代需要的元信息(objp 是物点网格,不进 JSON)
            result["detector"] = detector
            result["model"] = model
            result["image_ids"] = kept_ids
            result["image_size"] = tuple(image_size)
            result["n_points_per_view"] = board.cols * board.rows
            result["objp"] = objp
            baselines[name] = result

    return {
        "baselines": baselines,
        # 同一张照片在两颗检测器的数据里各被剔一次,名单按照片编号去重
        "excluded_images": sorted(set(excluded_seen)),
        "n_views": n_views,
    }
