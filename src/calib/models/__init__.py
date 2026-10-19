"""相机模型层:候选内参模型的统一抽象与注册机制。

四类候选模型(M1 针孔+Brown 少系数 / M2 常用五系数 / M3 有理模型 /
M4 鱼眼等距 KB4)共享同一套投影与雅可比实现——
雅可比 = 残差对参数的偏导数矩阵,优化器靠它知道"往哪边调参数";
理论界与实际求解必须用同一套数学,否则不可比。

注册机制(白话):每个模型类写好自己的数学后,在类头上加一行
@register_model 装饰器,就把自己登记进一张全局表;之后任何代码用
create_model("m1") 按名字查表拿实例。**一处注册、两个求解器都能用**:
求解器代码只认识 CameraModel 接口与 create_model 查表,不出现任何
具体模型的名字——测试里注册一个玩具模型即可证明这一点(D-09)。
"""

from __future__ import annotations

from calib.models.base import CameraModel

# 内置模型短名清单(与全部规划文档一致的顺序;具体类在模块底部 import 触发注册)
BUILTIN_MODEL_IDS = ("m1", "m2", "m3", "m4")

# 全局注册表:模型短名 → 模型类。装饰器写入、create_model 查表,别处不改
_REGISTRY: dict[str, type[CameraModel]] = {}


def register_model(cls: type[CameraModel]) -> type[CameraModel]:
    """类装饰器:把 cls 按 cls.model_id 登记进注册表(一处注册,全局可用)。

    重复登记同名模型直接抛错——两个类抢同一个短名,后续所有按名查表
    的代码都会拿到说不清的那一个,必须在注册时就拦下。
    """
    if cls.model_id in _REGISTRY:
        raise ValueError(
            f"模型短名「{cls.model_id}」被重复注册:"
            f"先来的是 {_REGISTRY[cls.model_id].__name__},"
            f"后又来了 {cls.__name__}。模型短名必须唯一,请检查 model_id。"
        )
    _REGISTRY[cls.model_id] = cls
    return cls


def create_model(model_id: str) -> CameraModel:
    """按短名查注册表,返回该模型的一个新实例(如 create_model("m2"))。

    未注册的名字抛中文错误并列出全部可选名——调用方一眼知道该填什么,
    提示直接指向配置文件里写模型清单的那一段。
    """
    if model_id not in _REGISTRY:
        raise ValueError(
            f"不认识的模型名:{model_id!r}。可选:{sorted(_REGISTRY) or '暂无已注册模型'}。"
            f"请把模型名改成一个已注册的短名,"
            f"并核对 calibration.yaml 里 calibrate.models 段写的是否就是这些名字。"
        )
    return _REGISTRY[model_id]()


def registered_model_ids() -> tuple[str, ...]:
    """当前已注册的全部模型短名(按注册顺序;报告与 CLI 汇总用)。"""
    return tuple(_REGISTRY)


# 底部 import 触发 @register_model 注册(必须放在注册表定义之后:
# 模型文件 import 本模块的 register_model,循环在此闭合,Python 能处理
# 这种"包先定义、子模块后加载"的顺序)。四个内置模型此后可被 create_model
# 查到。导入只为注册的副作用,名字不对外再输出,故同时压 F401。
from calib.models import kb4, pinhole_brown  # noqa: F401

