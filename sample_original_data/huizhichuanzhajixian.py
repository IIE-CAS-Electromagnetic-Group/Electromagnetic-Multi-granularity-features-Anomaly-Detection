'''绘制船闸迹线'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
df = pd.read_csv("E:\预处理\\0-15MHz\\20240628233749_20240629053749_0.009_15.0045.csv")

# 解析时间列（假设时间列是第一列）
time = pd.to_datetime(df.iloc[:, 0])  # 将时间列转换为datetime格式

# 提取7.902355kHz频率列的能量数据
frequency = '7.902355'  # 你需要查看的频率
energy = pd.to_numeric(df[frequency], errors='coerce')  # 将能量列转换为数值，无效值设为NaN

# 定义判断信号行为的函数
def detect_signal(energy, threshold_low=-10, threshold_high=-1, window_size=5):
    signal_status = []
    for i in range(len(energy) - window_size + 1):
        window = energy[i:i + window_size]
        if np.nanmean(window) > threshold_high:  # 使用nanmean忽略NaN值
            signal_status.append('Signal Existing')
        elif np.nanmean(window) < threshold_low:
            signal_status.append('Signal No Existing')
        else:
            signal_status.append('Signal No Existing')  # 默认状态为Signal No Existing
    # 补充最后几个时间点的状态
    signal_status.extend(['Signal No Existing'] * (window_size - 1))
    return signal_status

# 判断信号状态
signal_status = detect_signal(energy)

# 将时间转换为数值（以秒为单位）
time_numeric = (time - time[0]).dt.total_seconds()  # 将时间转换为从0开始的秒数

# 绘制图形
plt.figure(figsize=(12, 6))
plt.plot(time, energy, label='Intensity (dBm)', color='blue', alpha=0.7)

# 填充Signal Existing信号和Signal No Existing信号区域
plt.fill_between(time_numeric, -10, -1, where=np.array(signal_status) == 'Signal Existing', color='red', alpha=0.3, label='Signal Existing信号')
plt.fill_between(time_numeric, -10, -1, where=np.array(signal_status) == 'Signal No Existing', color='blue', alpha=0.1, label='Signal No Existing信号')

# 添加阈值线和Background Noise线
plt.axhline(y=-10, color='gray', linestyle='--', label='Background Noise')
plt.axhline(y=-5, color='gray', linestyle='--', label='Strong signal threshold')

# 设置图形属性
plt.xlabel('Time (s)')
plt.ylabel('Intensity (dBm)')
plt.title('7.902355kHz频率能量波动及信号状态')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # 旋转时间标签，避免重叠
plt.tight_layout()  # 自动调整布局
plt.show()