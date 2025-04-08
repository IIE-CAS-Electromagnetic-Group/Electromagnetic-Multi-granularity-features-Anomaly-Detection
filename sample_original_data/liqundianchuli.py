'''离群点处理'''
import os
import pandas as pd
import pandas as pd
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from pathlib import Path

def clean_outliers(input_folder, output_folder,target="7.902355"):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            print(f"Processing file: {file_path}")

            # 尝试读取CSV文件，自动检测分隔符
            try:
                df = pd.read_csv(file_path, sep=None, engine='python', parse_dates=['date'])
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

            # 检查列名是否正确
            if 'date' not in df.columns or target not in df.columns:
                print(f"Columns 'date' or "+target+" not found in file {file_path}. Skipping.")
                continue

            # 检测和清洗离群点
            df['energy_diff'] = df[target].diff().abs()  # 计算与前一个值的差异
            df['energy_diff_next'] = df[target].diff(-1).abs()  # 计算与后一个值的差异

            # 定义离群点阈值（可以根据实际情况调整）
            threshold = 2  # 假设差异大于2的被认为是离群点

            # 找到离群点
            outliers = df[(df['energy_diff'] > threshold) & (df['energy_diff_next'] > threshold)]

            # 替换离群点的值为前后值的平均值
            for idx in outliers.index:
                if idx > 0 and idx < len(df) - 1:
                    df.loc[idx, target] = (df.loc[idx - 1, target] + df.loc[idx + 1, target]) / 2

            # 删除辅助列
            df.drop(columns=['energy_diff', 'energy_diff_next'], inplace=True)

            # 保存清洗后的文件到输出文件夹
            output_file_path = os.path.join(output_folder, filename)
            # 确保保存为两列，使用逗号分隔
            df.to_csv(output_file_path, sep=',', index=False, encoding='utf-8')
            print(f"Cleaned file saved to: {output_file_path}")
    print("All files processed.")


def clean_csv_files_by_time(input_folder):
    """
    清洗文件夹下所有.csv文件中的数据。
    删除异常时间顺序的数据和重复行，并将清洗后的文件保存回原文件。

    参数：
    input_folder (str): 包含CSV文件的文件夹路径
    """
    # 获取文件夹下所有.csv文件
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the input folder.")
        return

    # 遍历所有CSV文件
    for filename in csv_files:
        file_path = os.path.join(input_folder, filename)

        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 将date列转换为datetime类型，错误的日期时间将被标记为NaT
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # 删除无效的日期时间行
        df = df.dropna(subset=['date'])

        # 删除重复行
        df = df.drop_duplicates(subset=['date'], keep='first')

        # 按时间排序
        df = df.sort_values(by='date').reset_index(drop=True)

        # 删除异常时间顺序的数据
        df['time_diff'] = df['date'].diff()
        df = df[df['time_diff'] >= pd.Timedelta(seconds=0)]
        df = df.drop(columns=['time_diff'])

        # 保存清洗后的文件
        df.to_csv(file_path, index=False)
        print(f"Cleaned and saved: {file_path}")






# 输入文件夹路径和输出文件夹路径
input_folder = "E:\预处理2\葛洲坝\\0_10MHz按天分割后的数据_只有平均值"  # 替换为你的输入文件夹路径

output_folder = "E:\预处理2\葛洲坝\\0_10MHz按天分割后的数据_离群点清洗后"  # 替换为你的输出文件夹路径

# 调用函数
clean_csv_files_by_time(input_folder)
clean_outliers(input_folder, output_folder,target="Average")
