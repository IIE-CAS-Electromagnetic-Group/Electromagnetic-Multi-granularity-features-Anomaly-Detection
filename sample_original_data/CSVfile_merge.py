'''文件合并'''
import pandas as pd
import plotly.express as px
import os
import csv
from pathlib import Path


def merge_csv_files(input_folder, output_file):
    """
    合并文件夹下所有.csv文件到一个文件中。
    只保留第一个.csv文件的第一行，其余文件从第二行开始合并。
    删除异常时间顺序的数据和重复行。

    参数：
    input_folder (str): 包含CSV文件的文件夹路径
    output_file (str): 合并后的输出文件路径
    """
    # 获取文件夹下所有.csv文件
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the input folder.")
        return

    # 按文件名排序，确保合并顺序一致
    csv_files.sort()

    # 初始化一个空的DataFrame用于存储合并后的数据
    merged_df = pd.DataFrame()

    # 遍历所有CSV文件
    for i, filename in enumerate(csv_files):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)

        # 如果是第一个文件，保留第一行；否则从第二行开始
        if i == 0:
            merged_df = df
        else:
            merged_df = pd.concat([merged_df, df.iloc[1:]], ignore_index=True)

    # 将date列转换为datetime类型，错误的日期时间将被标记为NaT
    merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')

    # 删除无效的日期时间行
    merged_df = merged_df.dropna(subset=['date'])

    # 删除重复行
    merged_df = merged_df.drop_duplicates(subset=['date'], keep='first')

    # 删除异常时间顺序的数据
    merged_df = merged_df.sort_values(by='date')
    merged_df = merged_df.reset_index(drop=True)
    merged_df['time_diff'] = merged_df['date'].diff()
    merged_df = merged_df[merged_df['time_diff'] >= pd.Timedelta(seconds=0)]
    merged_df = merged_df.drop(columns=['time_diff'])

    # 保存合并后的文件
    merged_df.to_csv(output_file, index=False)
    print(f"All CSV files have been merged into: {output_file}")


def clean_csv_file(input_file, output_file):
    '''
    使用pandas重新整理CSV文件，删除多余的列并处理非法值
    :param input_file: 输入文件路径
    :param output_file: 输出文件路径
    :return:
    '''
    if(output_file=="0"):
        output_file=input_file
    # 读取前5行，确定文件的列数
    sample = pd.read_csv(input_file, nrows=5)
    expected_columns = len(sample.columns)
    print(f"文件应有列数: {expected_columns}")

    # 读取整个文件，删除多余的列
    data = pd.read_csv(input_file, usecols=range(expected_columns))

    # 处理非法值
    # 第一列是时间列，保持不变
    time_column = data.columns[0]  # 假设第一列是时间列
    numeric_columns = data.columns[1:]  # 其他列是数值列

    # 将非数值型数据替换为NaN
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # 使用上一行的合法值填充NaN（向前填充）
    data[numeric_columns] = data[numeric_columns].ffill()

    # 保存整理后的文件
    data.to_csv(output_file, index=False)
    print(f"文件已整理并保存到: {output_file}")



if __name__=="__main__":
    merge_csv_files("E:\预处理\\0-15MHz——离群点清洗后","tf_离群点清洗后.csv")
