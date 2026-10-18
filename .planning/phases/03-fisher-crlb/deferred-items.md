# Deferred Items — Phase 03(fisher-crlb)

## 03-05 执行期间发现(超出本计划范围,未修)

- **预存的 ruff 告警 8 处**(本计划未触碰的文件,属先前轮次遗留):
  - `src/calib/models/__init__.py:65`(2 处)
  - `src/calib/noise/injection.py:24,46,92,253,265`(5 处)
  - `tests/test_noise_immerkaer.py:12`(1 处)
  - 本计划改动的文件(fisher.py / artifacts.py / cli/main.py / noise_figs.py /
    test_noise_fisher.py / test_fisher_realdata.py)ruff 全绿。建议后续轮次
    统一跑一次 `ruff check --fix` 收口。

- **真实数据发现(非缺陷,按设计上报)**:四个模型的 Fisher 信息矩阵条件数
  全部超过退化线 1e6(flag = degenerate),最软方向集中在 fx/fy(个别模型
  带上 cx/cy)。含义:这批摆拍在"部分参数组合方向"上精度先天受限,典型
  原因是姿态的旋转多样性不足。这是要带进阶段四选型与摆拍改进建议的输入,
  不是 CRLB 计算的错误(交叉验证 max_rel_dev ≈ 0.005% 证明两套独立数学
  对得上账)。
