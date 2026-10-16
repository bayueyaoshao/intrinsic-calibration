---
phase: 1
slug: data-foundation-minimal-calibration
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-10-16
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x(uv 管理安装,01-01 Task 1 落地) |
| **Config file** | pyproject.toml 内 [tool.pytest.ini_options](01-01 Task 1 建;markers: integration) |
| **Quick run command** | `uv run pytest tests/ -x -q -m "not integration"` |
| **Full suite command** | `uv run pytest tests/ -q` |
| **Estimated runtime** | 快速 ~10–30s(合成夹具);全套 + integration 冒烟 ~2–5min |

---

## Sampling Rate

- After every task commit: Run `uv run pytest tests/ -x -q -m "not integration"`
- After every plan wave: Run `uv run pytest tests/ -q`(含真实数据小子集冒烟)
- Before `/gsd-verify-work`: Full suite must be green
- Max feedback latency: 60 秒(快速命令)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | IO-03(环境前提) | T-01-SC | uv.lock 锁定已审计包;无 [SUS]/[SLOP] | unit(冒烟) | `uv run pytest tests/test_env_smoke.py -q` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 1 | IO-03 | T-01-01 | yaml.safe_load only;字段校验中文报错 | unit | `uv run pytest tests/test_io_config.py -q` | ❌ W0 | ⬜ pending |
| 1-01-03 | 01 | 1 | IO-01 | T-01-02, T-01-03 | image_id 白名单;分辨率不一致硬报错;空目录警告不崩 | unit + CLI | `uv run pytest tests/test_io_dataset.py -q && uv run calib ingest && uv run calib --help \| grep -q 入库体检` | ❌ W0 | ⬜ pending |
| 1-02-01 | 02 | 2 | COR-01 | — | 窗口推导恒满足 2w+1<sq_px(质量断言) | unit | `uv run pytest tests/test_corners.py -q` | ❌ W0 | ⬜ pending |
| 1-02-02 | 02 | 2 | COR-01 | T-02-03 | canonicalize 四变体归一;cornerSubPix 仅经典分支(源码断言) | unit | `uv run pytest tests/test_corners.py -q` | ❌ W0 | ⬜ pending |
| 1-02-03 | 02 | 2 | COR-01 | T-02-01 | npz 读写 allow_pickle=False;未检出帧 NaN 占位不消失 | unit + CLI | `uv run pytest tests/test_corners_artifacts.py -q && uv run calib corners` | ❌ W0 | ⬜ pending |
| 1-03-01 | 03 | 3 | COR-01(最小标定) | T-03-03 | fisheye try/except 不崩;exclude_images 过滤生效 | unit | `uv run pytest tests/test_solve_baseline.py -q` | ❌ W0 | ⬜ pending |
| 1-03-02 | 03 | 3 | COR-01(最小标定) | T-03-01, T-03-02 | lineage 含 corners sha256;剔除名单落盘可审计 | unit + CLI | `uv run calib calibrate && uv run pytest tests/ -q -m "not integration"` | ❌ W0 | ⬜ pending |
| 1-03-03 | 03 | 3 | IO-01(血缘链) | T-03-01 | 冒烟断言血缘哈希闭合(manifest→corners→calibration) | integration | `uv run pytest tests/test_smoke_realdata.py -q -m integration` | ❌ W0 | ⬜ pending |
| 1-04-01 | 04 | 4 | IO-02 | T-04-01 | 坏图中文报错;FAST_CHECK 仅预筛 | unit | `uv run pytest tests/test_io_precheck.py -q` | ❌ W0 | ⬜ pending |
| 1-04-02 | 04 | 4 | IO-02, IO-01 | T-04-02 | precheck.py 无删除路径(grep 断言);判定仅标记 | unit | `uv run pytest tests/test_io_precheck.py -q && ! grep -qE "remove\|unlink\|delete" src/calib/io/precheck.py` | ❌ W0 | ⬜ pending |
| 1-04-03 | 04 | 4 | IO-02 | T-04-02 | 存疑副本数 == 存疑数;跨轮 diff 三桶 | CLI | `uv run calib ingest && uv run calib ingest`(断言脚本见 01-04 Task 3) | ❌ W0 | ⬜ pending |
| 1-05-01 | 05 | 5 | COR-01(可视化) | T-05-01 | 副本另存,原图永不被修改 | unit | `uv run pytest tests/test_visualization.py -q` | ❌ W0 | ⬜ pending |
| 1-05-02 | 05 | 5 | COR-01(可视化) | T-05-02 | 拼图零新依赖;quality 90 控制体积 | unit | `uv run pytest tests/test_visualization.py -q` | ❌ W0 | ⬜ pending |
| 1-05-03 | 05 | 5 | IO-02, COR-01 | — | 全量产物断言;文档同步断言 | CLI + human | `uv run pytest tests/ -q && ! grep -q "自动剔除" docs/pipeline.md` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `pyproject.toml` + `uv lock` + 依赖/开发依赖安装 + 三件套冒烟打印版本(01-01 Task 1;opencv 轮子经代理下载约 10 分钟级,预留超时)
- [ ] `tests/conftest.py` — 合成棋盘图生成夹具:最小版(01-01 Task 3,tiny jpg)→ 完整投影版(01-02 Task 1,已知真值 K/D + blur/noise/overexpose 变体)——COR-01、最小标定、体检三类测试的共同地基
- [ ] pytest markers 注册(integration,01-01 Task 1)
- [ ] 全部 7 个测试文件按上表任务逐步创建(❌ W0 列在各自任务内补齐,快速命令链保持秒级)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| 角点叠加图人工核查(角点是否贴准/系统性偏移) | COR-01 | 视觉判断无法自动化(Phase 5 COR-02 才有合成退化数字证据) | 打开 mosaic/overview_mosaic.png + 抽查 5–10 张 marked_copies(含最差视图与存疑照片),见 01-05 Task 3 human-check 清单 |
| 存疑清单误杀/漏杀审阅 | IO-02 | 需要人对"糊不糊"的语义判断 | 打开 suspect_mosaic.png 与 suspect_originals/ 抽查,见 01-05 Task 3 human-check 清单 |
| 2×2 基线结果合理性(D-05 收手) | COR-01(最小标定) | 量级判断 + 用户认可即收手的锁定决策 | 查看控制台对比表:RMS 亚像素~2px 量级;fisheye 失败的 failure_note 可理解 |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 60s(快速命令)
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending execution
