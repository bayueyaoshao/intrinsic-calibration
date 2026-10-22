# 阶段六 超范围遗留事项(deferred items)

本文件登记执行阶段六计划时发现的、**不属于当前计划范围**的既有问题。按纪律只登记不修。

## 2026-10-23(执行 06-01 全量白天档回归时发现)

- **tests/test_noise_injection.py::test_full_injection_real_data** 在 `-m "not realdata"`
  全量白天档中失败(613 passed / 1 failed)。
  - **非新问题**:与 `.planning/phases/05-error-attribution-corner-tuning/deferred-items.md`
    2026-10-22 第 3 条登记的是**同一测试、同一根因**(阶段三老测试的数据契约脱节:
    预期首维 n_train=640,实际 (705, 9, 4, 88, 2),test 划分 65 张无 train 基线 → NaN;
    σ=0 档另有 2 角点不归零待查检测来源对齐)。
  - **与 06-01 零关联**:06-01 三个提交(fa1d33d / ddb3bcf / 8a5b18a)只动
    report/build.py、report/templates/、cli/main.py 的 report 命令块与两个报告测试,
    不触及噪声/注入代码路径。
  - **修复方向**已在阶段五登记(改按 split 过滤后断言 + 另查 σ=0 双角点,需重跑约
    40 分钟全量注入),留待专门一轮,不夹带在阶段六。
  - 附注:执行 06-01 时曾单独重跑该测试以取证,约 50 分钟未跑完被中断;既有登记
    的取证材料已足够,不再重复消耗。

## 附:06-01 观察项(非失败,后续计划留意)

- `calib report` 同一秒内连续重跑会落到同一个 `run_<YYYYMMDD_HHMMSS>` 轮(时间戳
  秒级粒度,与各上游阶段命令的轮次命名约定一致),后一次完整覆盖前一次的四件产物。
  若 06-02 跨运行 diff 需要「每次运行必成一轮」的保证,可考虑给报告轮加亚秒后缀;
  当前保持与全仓库轮次命名一致,不做特殊化。
- `main.py` 的 `run_all()` 播报仍写「后续阶段(防作弊考试/误差归因/汇总报告)由后续
  版本实现,run-all 到第⑤步 crlb.json 为止」——evaluate/analyze/report 三个命令
  实际已落地,该文案过时。属 06-03/06-04(run-all 接全流程)的文件范围,本次不越界。
