"""任务级并行助手:把"一大批小任务"分给常驻工作进程跑,逐个交回结果。

白话:阶段四的考试要跑约 1300 次标定求解,每次求解只花几秒——并行加速
靠"多开几个进程同时算",而不是把单次求解改快。本模块把 joblib 的并行
调度收拢成一个生成器函数:调用方给任务清单和一个"干一件活"的函数,
逐个拿到结果,顺带驱动进度播报(进度回调由命令行层注入,库层不碰
命令行框架——既有硬性规则)。

两条硬约定:
- 任务的入参必须是可整读的小元组(全部内容能被序列化传递);大数组
  (如整份观测集)靠 joblib 对 numpy 数组的内存映射共享,不要逐任务
  复制进元组,否则并行开销反超收益;
- 单个任务的异常由调用方捕获并转成失败契约记录(参照 solve/result.py
  的 make_failure 体例),本函数不吞异常——坏一个组合不该静默丢结果,
  也不该由这里决定整个批次的中止与否。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

from joblib import Parallel, delayed


def run_parallel(tasks: Iterable[Any], worker: Callable[[Any], Any],
                 max_workers: int,
                 progress: Callable[[dict], None] | None = None) -> Iterator[Any]:
    """把 tasks 逐个交给 worker 执行,按完成顺序逐个产出结果(生成器)。

    参数:
        tasks:       任务清单,每项是传给 worker 的一个小元组(可整读);
        worker:      干一件活的函数(顶层函数,能被序列化传给子进程);
        max_workers: 并行进程上限;<= 1 时不开子进程、当前进程顺序跑
                     (小样与调试用,结果与并行完全一致——任务顺序与
                     结果顺序都相同,随机性只看任务里带的种子);
        progress:    可选进度回调,每完成一个任务调一次,
                     传入 {"done": 已完成数, "total": 总数}。

    产出:按任务清单顺序逐个 yield worker 的返回值(调用方可流式落盘,
    不必等整批跑完)。
    """
    task_list = list(tasks)  # 任务是小元组,物化一次好取总数(进度播报要 total)
    total = len(task_list)

    if max_workers <= 1:
        # 串行回退:不开子进程,与并行同序同果(种子只看任务元组)
        for done, task in enumerate(task_list, start=1):
            yield worker(task)
            if progress:
                progress({"done": done, "total": total})
        return

    # 并行:loky 常驻工作进程 + 生成器逐个消费——结果一到就交出去,
    # 进度跟着动,不必等整批算完(return_as="generator",joblib >= 1.3)
    results = Parallel(n_jobs=max_workers, return_as="generator")(
        delayed(worker)(task) for task in task_list)
    for done, result in enumerate(results, start=1):
        yield result
        if progress:
            progress({"done": done, "total": total})
