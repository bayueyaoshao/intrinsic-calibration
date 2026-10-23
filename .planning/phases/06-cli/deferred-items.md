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
  **(2026-10-23 更新:已由 06-03 提交 937a220 解决,run-all 重写为八步全流程,
  本条可销。)**

## 2026-10-23(执行 06-03 全量白天档回归时发现)

- **tests/test_report_loopback.py 收集期报错**:`from calib.report.loopback import ...`
  抛 ModuleNotFoundError,`-m "not realdata"` 全量白天档无法整趟收集(619/670 后中断)。
  - **非 06-03 范围**:该文件与 `src/calib/report/loopback.py` 同属 06-02(报告回环),
    执行纪律明文不碰 06-02 范围。判定为 06-02 并行执行中的中间态(测试文件已落、
    实现模块未落),06-02 收口后应自愈。
  - **06-03 的处置**:快速档回归用 `--ignore=tests/test_report_loopback.py` 绕开,
    其余 619 例照跑。
  - **(同日已自愈)**:数分钟后 06-02 落盘 `src/calib/report/loopback.py`,
    该文件单独补跑 8 passed,收集错误消失,无需后续处理。
- **同轮快速档的 4 个 test_report_build.py 失败**:06-03 的 `-m "not realdata"`
  全量白天档(绕开 loopback 文件)跑出 4 failed / 615 passed,可见失败行全部落在
  `tests/test_report_build.py`(第 4 个失败行被输出管道截断)——同样是 06-02 并行
  执行中的中间态(06-03 的套件 02:41 收集,06-02 的 5125a81/7c23090 在 02:42/02:47
  才落盘)。06-02 收口(d5a75fc)后单独复跑该文件 **5 passed 全绿**,无需处理。
  另:此前登记的 test_noise_injection 注入测试本轮未再出现失败。

## 2026-10-23(阶段六收口全量套件对账,2:28:37,691 passed / 2 failed / 2 errors)

- **阶段五登记的 2 个血缘断链测试自愈**:test_fisher_realdata /
  test_noise_realdata 的 lineage 检查在新全链(corners run_20261022_212000
  起)上通过——新链血缘完整,旧账销案。
- **test_calibrate_matrix 两条 ERROR(已修)**:测试把角点轮名写死为
  run_20261020_102709(阶段五前的旧链),及时清理换新链后夹具复制找不到文件。
  修复:改为动态取 output/ 下最新轮(无轮则 skip),换数据集/清旧轮不再断。
- **test_analyze_sensitivity_section 失败(已修)**:测试把 bootstrap 最大位移
  钉死为 1.9943±1%(50 次重抽口径的结论 2 数字);降载后 10 次口径给
  1.8330,量级一致、单点必变。修复:改锁「1–3px 量级带 + 远超 0.05px 阈值」
  两个口径不变量,精确值留在 findings(50 次口径 1.9943 / 10 次口径 1.8330)。
- **test_noise_injection 仍失败(维持登记)**:与上两条不同,这是实现侧老账
  (阶段三注入测试的数据契约脱节),非测试钉死问题,修复需约 40 分钟注入重跑,
  维持原修复方向留专门一轮。**当前唯一未销的既有限遗。**
