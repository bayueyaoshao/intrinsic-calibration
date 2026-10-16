"""数据读入层:目录扫描、配置解析、入库体检、数据清单(manifest)写盘。

流水线第①步「照片入库体检」的承载模块:
- dataset.py:姿态数据模型(目录名即姿态 ID)与 input_data 只读扫描
- config.py:board.yaml / calibration.yaml 的解析与校验(参数唯一来源)
- manifest.py:带血缘指纹的数据清单写盘(后续阶段的唯一输入)
"""
