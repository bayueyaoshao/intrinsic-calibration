"""相机内参标定与评估框架(calib)。

对一批预先采集好的标定照片,产出当前数据条件下可论证为最优的相机内参,
以及完整的证据链:噪声底与可达精度预测、角点提取证据、抗过拟合的模型选型、
误差预算与参数敏感度诊断。

模块布局与流水线阶段一一对应(见 docs/pipeline.md):
io(读入)→ corners(角点)→ noise(噪声)→ models/solve(标定)
→ evaluation(评估)→ analysis(归因)→ visualization/report(出图与报告)。
"""
