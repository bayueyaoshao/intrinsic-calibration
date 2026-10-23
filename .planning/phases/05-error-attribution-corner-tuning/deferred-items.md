# 阶段五 超范围遗留事项(deferred items)

本文件登记执行计划时发现的、**不属于当前计划范围**的既有问题。按纪律只登记不修(修复需要重跑数小时的噪声/CRLB 流水线并产生新轮,与阶段五计划无关)。

## 2026-10-22(执行 05-04 全量集成测试时发现)

以下 3 个集成测试失败,根因全部是**运行产物轮次清理后的血缘链断裂**,与 05-04 的四个提交(05ad259 / 4592d97 / 21b8b13 / b43f9d2,只动 analysis / visualization / analyze 命令块与其测试)无关——噪声、角点、注入、CRLB 的代码与本计划零接触。

**暴露时机说明:** 阶段五 05-01~05-03 收官时的集成验证是子集运行(会话记录里最大一次为 `11 passed, 37 deselected`,即只跑 test_analysis_realdata 的集成段);全量 `-m integration` 套件(约 4h21m)在阶段五由 05-04 首次完整跑通,这 3 个既有失败因此首次暴露。断链本身发生在 10月20日角点轮清理(163039 被 102709 替代后按及时清理删除),早于阶段五所有计划。

1. **tests/test_noise_realdata.py::test_lineage_first_ring_and_detail_readback**
   - 报错:`血缘记录的上游 corners_train_sb.npz 不在:output/corners/run_20261018_163039/corners_train_sb.npz(血缘链第一环断裂)`
   - 根因:复用的噪声轮 run_20261018_165412 建轮时血缘登记指向当时的角点轮 163039;该角点轮后来被更新的轮 run_20261020_102709 替代并按「及时清理」删除,旧轮登记的路径随之失联。
   - **(2026-10-23 销案)**:阶段六全链重跑后新链血缘完整,该测试在新链上通过;旧轮断链随旧轮清理一并消失,无需处理。对账详见
     `.planning/phases/06-cli/deferred-items.md` 2026-10-23 收口对账段。
2. **tests/test_fisher_realdata.py::test_lineage_upstream_files_exist**
   - 报错:`血缘登记的上游文件不在:corners_train_sb.npz → output/corners/run_20261018_163039/corners_train_sb.npz`
   - 根因同上:CRLB 轮 run_20261019_025859 建轮时的上游角点轮已被清理。
   - **(2026-10-23 销案)**:同上,新链(CRLB 轮 run_20261022_230315)血缘指向现存角点轮,测试通过。
3. **tests/test_noise_injection.py::test_full_injection_real_data**
   - **失败现场取证**(阶段五首次全量跑通时失败;用 pytest 保留的失败运行临时目录 `pytest-91` 的产物对账):
     - 位移数组实际形状 `(705, 9, 4, 88, 2)`,测试预期首维 `n_train = 640` → 形状断言先挂;
     - 基线(σ=0 档)含 45760 个 NaN = **65(test 划分张数)× 4(重复)× 88(角点)× 2**——多出的 65 行全是 test 划分图像,不在 `corners_train_sb.npz`(train 专用)里,查无基线角点即 NaN;
     - train 640 行基线零 NaN,但其中有 **2 个角点位移非零:0.332 / −0.250 px**(量级恰为亚像素量化台阶,疑似 σ=0 时重检路线与基线检测来源不一致——基线 npz 的检测来源与注入重检在该 2 角点上没对齐);
   - **根因:** 这是阶段三写下的老测试,与后续演进的数据契约脱节——`noise_splits: [train, test]` 自 03-01(6dedd32)起就是双划分,但阶段四引入 hold-out 划分后 `corners_train_sb.npz` 收窄为 train 专用(640 张);测试仍按"全部清单图像都有基线角点 + σ=0 恰为零"的旧前提断言。注入库层本身按配置过滤、行为正确(65 张 test 图本来就不该有 train 基线)。
   - **修复方向(留待专门一轮,需重跑约 40 分钟全量注入):** 测试改为按 corners npz 的 image_ids 或 `split=='train'` 过滤后断言形状;σ=0 双角点不归零另行查检测来源对齐;不夹带在阶段五。
   - **(2026-10-23 仍维持登记)**:阶段六收口全量套件复现同一失败,根因是实现侧老账(非测试钉死),维持原修复方向,见
     `.planning/phases/06-cli/deferred-items.md` 末段——**当前唯一未销的既有限遗。**

**共性教训(留给后续阶段):**「及时清理」删除被替代旧轮是对的,但血缘类测试断言的是旧轮登记路径的物理存在——重跑 noise/fisher/注入使其血缘指向现存角点轮 run_20261020_102709,即需重跑三条小时级流水线并产生新轮。留待专门一轮处理,不夹带在阶段五。

## 附:05-04 集成测试当轮清理记录(非遗留,仅存档)

- `output/evaluate/run_20261021_183217`(test_evaluation_realdata 副产品,与权威轮 131613 逐位一致)已删。
- `output/evaluate/run_20261021_224410`(test_optimizer_realdata 副产品,仅 optimizer.json;多起点 m2 param_disp_px = 0.07399264201266795 与权威轮 evaluate.json 内嵌值逐位一致)已删。
- 权威轮保留:`output/evaluate/run_20261021_131613`(analyze 轮 run_20261021_182345 指纹它)、`output/noise/run_20261018_165412`、`output/corners/run_20261020_102709`。
