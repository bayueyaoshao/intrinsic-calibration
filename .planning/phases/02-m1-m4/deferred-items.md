# Phase 02 执行期间发现的范围外问题(不修,留给后续计划)

## 2026-10-18 · 02-03 执行者记录:上一波(02-01)模型文件的 ruff 违例

在 02-03 Task 3 提交前跑全量 `uv run ruff check src/ tests/` 时发现,
以下 5 处违例全部位于 02-01(模型层)交付的文件,与 02-03 的改动无关,
按"范围边界"规则记录不修:

| 文件 | 规则 | 说明 |
| ---- | ---- | ---- |
| `src/calib/models/__init__.py:65` | noqa 指令非法 + F401 ×2 | `# noqa: E402(故意置底触发注册)` 把中文说明直接粘在代码后面,整个 noqa 指令失效,E402 未被压制;同时触发两条 F401(kb4 / pinhole_brown 导入未再导出)。修法:改成 `# noqa: E402  # 故意置底触发注册`(空格 + 独立注释),或用冗余再导出别名 `kb4 as kb4` |
| `src/calib/models/base.py:103` | RET501 | `opencv_adapter()` 默认实现显式 `return None`,规则要求只返回 None 时省略返回值(改 `return` 或删行) |
| `src/calib/models/kb4.py:159` | UP037 | `"_KB4OpenCVAdapter"` 带引号注解——文件已有 `from __future__ import annotations`,引号多余 |
| `src/calib/models/pinhole_brown.py:145` | RUF059 | 解包变量 `drad_dr2` 未使用,应加下划线前缀(注:ruff 提示该修复属 unsafe,改前需确认确无副作用) |

影响:`uv run ruff check src/ tests/` 整体退出码非 0(5 errors)。02-03 自己的
文件(solve/、io/config.py、tests、config/)全部干净。建议在波次合并后的
下一个计划(或 02-01 执行者的返工)里一并清掉,保持仓库级 lint 可过。
