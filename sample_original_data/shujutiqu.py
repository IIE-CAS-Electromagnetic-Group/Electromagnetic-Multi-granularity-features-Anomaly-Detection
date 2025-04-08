'''简单来说，就是把某一列单独摘出来
20250203更新，如果不摘某一列的话，还得求所有列的平均值'''
import os
import pandas as pd
from datetime import datetime, timedelta



def take_single_line(input_folder,output_folder,target_frequency = '7.902355'):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 定义要保留的频率列
    #target_frequency = '7.902355'

    # 遍历输入文件夹中的所有CSV文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            # 读取CSV文件
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path)

            # 只保留date和目标频率列
            df = df[['date', target_frequency]]

            # 尝试将date列转换为datetime类型，错误的日期时间将被标记为NaT
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

            # 删除无效的日期时间行
            df = df.dropna(subset=['date'])

            # 如果数据为空，则跳过处理
            if df.empty:
                print(f"Skipping empty file: {filename}")
                continue

            # 保存处理后的文件到输出文件夹
            output_file_path = os.path.join(output_folder, filename)
            df.to_csv(output_file_path, index=False)

            print(f'Processed and saved: {output_file_path}')


def process_average_files(input_folder, output_folder):
    '''求平均值'''
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            # 构建输入文件的完整路径
            input_file_path = os.path.join(input_folder, filename)

            # 读取CSV文件
            df = pd.read_csv(input_file_path, sep=',')

            # 计算平均值（排除第一列时间）
            df['Average'] = df.iloc[:, 1:].mean(axis=1)

            # 构建输出文件的完整路径
            output_file_path = os.path.join(output_folder, filename)

            # 保存处理后的CSV文件
            df.to_csv(output_file_path, sep=',', index=False)

            print(f"Processed {filename} and saved to {output_file_path}")


#input_folder = 'E:\预处理2\葛洲坝\\0_10MHz按天分割后的数据'
#output_folder = 'E:\预处理2\葛洲坝\\0_10MHz按天分割后的数据_平均值'

#process_average_files(input_folder, output_folder)

input_folder = 'E:\预处理2\葛洲坝\\0_10MHz按天分割后的数据_平均值'
output_folder = 'E:\预处理2\葛洲坝\\0_10MHz按天分割后的数据_只有平均值'

take_single_line(input_folder,output_folder,target_frequency = 'Average')