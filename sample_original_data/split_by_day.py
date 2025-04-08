import os
import pandas as pd
from datetime import datetime

# 输入文件夹和输出文件夹路径
input_folder = 'E:/预处理2/葛洲坝/0_10MHz_7.7MHz附近数据/7.7MHz原始信号数据'
output_folder = 'E:/预处理2/葛洲坝/0_10MHz按天分割后的数据'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取输入文件夹中的所有CSV文件
csv_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.csv')])

# 初始化一个字典来存储按日期分组的数据
data_by_date = {}

# 遍历每个CSV文件
for csv_file in csv_files:
    print("开始遍历："+str(csv_file))
    file_path = os.path.join(input_folder, csv_file)
    df = pd.read_csv(file_path, sep=',')  # 假设CSV文件是用制表符分隔的

    # 将日期列转换为datetime对象
    df['date'] = pd.to_datetime(df['date'])

    # 获取当前文件的起始和结束时间
    start_time = df['date'].iloc[0]
    end_time = df['date'].iloc[-1]

    # 生成当前文件覆盖的日期范围
    date_range = pd.date_range(start=start_time.date(), end=end_time.date(), freq='D')

    # 遍历每个日期
    for date in date_range:
        # 获取当前日期的起始和结束时间
        day_start = pd.Timestamp(date)
        day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        # 使用二分查找快速定位当前日期的数据范围
        start_idx = df['date'].searchsorted(day_start)
        end_idx = df['date'].searchsorted(day_end, side='right')

        # 提取当前日期的数据
        day_data = df.iloc[start_idx:end_idx]

        # 如果当前日期有数据，则保存到字典中
        if not day_data.empty:
            if date not in data_by_date:
                data_by_date[date] = day_data
            else:
                data_by_date[date] = pd.concat([data_by_date[date], day_data], ignore_index=True)

# 将按日期分组的数据保存到新的CSV文件中
for date, df in data_by_date.items():
    # 生成输出文件名
    output_filename = date.strftime('%Y%m%d') + '.csv'
    output_path = os.path.join(output_folder, output_filename)

    # 保存DataFrame到CSV文件
    df.to_csv(output_path, index=False, sep=',')

print("处理完成，文件已保存到:", output_folder)