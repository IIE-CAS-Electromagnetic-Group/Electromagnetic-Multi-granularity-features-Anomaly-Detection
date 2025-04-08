import os

import numpy as np
import random
import pandas as pd
from datetime import datetime, timedelta




def get_all_file_list(csv_file_dir="E:\Work\Workspace\Data\电磁频谱数据\pictures\所里频谱迹线"):
    # 获取该文件夹中所有的bin文件路径
    file_path = os.listdir(csv_file_dir)
    file_list = list(map(lambda x: os.path.join(csv_file_dir, x), file_path))
    # all_file_path = sorted(file_list, key=lambda s: int(s.split('\\')[-1].split('_')[0]))
    all_file_path = sorted(file_list, key=lambda s: int(s.split('/')[-1].split('_')[0]))
    return all_file_path
    pass


def complete_datetime_string(time_str):
    # 当前日期和时间
    now = datetime.now()
    # 检查并补全时间字符串
    try:
        if len(time_str) == 10:
            # 输入格式为 'YYYY-MM-DD'
            complete_str = f"{time_str} 00:00:00"
        elif len(time_str) == 8:
            # 输入格式为 'HH:MM:SS'
            complete_str = f"{now.strftime('%Y-%m-%d')} {time_str}"
        elif len(time_str) == 16:
            # 输入格式为 'YYYY-MM-DD HH:MM'
            complete_str = f"{time_str}:00"
        elif len(time_str) == 13:
            # 输入格式为 'YYYY-MM-DD HH'
            complete_str = f"{time_str}:00:00"
        elif len(time_str) == 19:
            # 输入格式为 'YYYY-MM-DD HH:MM:SS'
            complete_str = time_str
        elif len(time_str) == 23:
            complete_str, _ = time_str.split(".")
        else:
            raise ValueError("未知的时间格式")

        # 将补全后的字符串转换为 datetime 对象以验证其有效性
        complete_datetime = datetime.strptime(complete_str, '%Y-%m-%d %H:%M:%S')
        return complete_datetime
    except ValueError as e:
        print(f"输入时间字符串格式无效: {e}")
    return None


# 定义生成随机时间段的函数
def generate_random_time_period(start_date, end_date,  duration_seconds_low,duration_seconds_high):
    # 生成随机的开始时间（在某个范围内，例如2024年1月1日至2024年12月31日）
    # start_date = datetime(2024, 1, 1)
    # end_date = datetime(2024, 12, 31)
    start_date = complete_datetime_string(start_date)
    end_date = complete_datetime_string(end_date)
    delta_days = (end_date - start_date).days
    random_start_date = start_date + timedelta(days=random.randint(0, delta_days))

    # 添加随机的小时、分钟、秒
    random_start_time = random_start_date + timedelta(
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )

    # 生成随机的时间段长度（例如0到10天，每天的随机秒数）
    # random_duration_seconds = random.randint(0, 10 * 24 * 3600)  # 0到10天的秒数

    random_duration_seconds = random.randint(max(0, duration_seconds_low), duration_seconds_high)

    random_end_time = random_start_time + timedelta(seconds=random_duration_seconds)

    return random_start_time, random_end_time


def get_middle_sublist(lst, length):
    """
    从列表中获取中间指定长度的子列表。

    参数:
    lst (list): 原始列表。
    length (int): 子列表的长度。

    返回:
    list: 中间子列表。
    """
    # 确保子列表长度不超过原列表长度
    if length > len(lst):
        length = len(list)
        # raise ValueError("子列表长度不能超过原列表长度。")

    # 计算中间起始位置和结束位置
    start_index = (len(lst) - length) // 2
    end_index = start_index + length

    # 返回子列表
    return lst[start_index:end_index]


def write_abnormal_to_csv(abnormal_value_df, raw_data_dir,test_abnormal_data_dir_path):
    print("generate abnormal data : write abnormal to csv.")
    all_file_path = get_all_file_list(raw_data_dir)
    for file_path in all_file_path:
        # save_file_path=os.path.join(test_abnormal_data_dir_path,file_path.split("\\")[-1])
        save_file_path=os.path.join(test_abnormal_data_dir_path,file_path.split("/")[-1])
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        cols = list(df.columns)
        cols.remove('date')
        # mask=None
        for index, row in abnormal_value_df.iterrows():
            select_start_date, select_end_date = row['start_time'], row['end_time']
            bandwidth = row['bandwidth']
            power = row['power']
            signal_cols = get_middle_sublist(cols, bandwidth)
            if (select_end_date < df['date'].iloc[0]):
                continue
            if (select_start_date > df['date'].iloc[-1]):
                break
            # mask= (df['date'] >= pd.to_datetime(select_start_date)) & (df['date'] <= pd.to_datetime(select_end_date))
            # df.iloc[mask,signal_cols]=power
            df.loc[(df['date'] >= pd.to_datetime(select_start_date)) & (
                        df['date'] <= pd.to_datetime(select_end_date)), signal_cols] = power
            df.to_csv(save_file_path, index=False)
            print("generate abnormal data: write abnormal to csv:"+str(save_file_path))
    pass


def normal_value(signal_df, signal_happen_time_df, high_low_perc=[0.95, 0.05]):
    signal_df['date'] = pd.to_datetime(signal_df['date'])
    signal_happen_time_df['start_time'] = pd.to_datetime(signal_happen_time_df['start_time'])
    signal_happen_time_df['end_time'] = pd.to_datetime(signal_happen_time_df['end_time'])
    time_high, time_low = signal_happen_time_df['duration_seconds'].quantile(high_low_perc)
    # time_range=time_high-time_low
    mask = None
    for index, row in signal_happen_time_df.iterrows():
        select_start_date, select_end_date = row['start_time'], row['end_time']
        if (select_end_date < signal_df['date'].iloc[0]):
            continue
        if (select_start_date > signal_df['date'].iloc[-1]):
            break

        mask1 = (signal_df['date'] >= pd.to_datetime(select_start_date)) & (
                signal_df['date'] <= pd.to_datetime(select_end_date))
        if (mask is None):
            mask = mask1
        else:
            mask = mask | mask1
    filtered_df = signal_df.loc[mask]
    voltage_threshold = 0

    # 找出存在大于阈值的列
    columns_with_values_above_threshold = filtered_df.iloc[:, 1:].columns[
        (filtered_df.iloc[:, 1:] > voltage_threshold).any(axis=0)]
    signal_power_df = filtered_df[columns_with_values_above_threshold]
    # signal_power_df=signal_power_df[(signal_power_df>voltage_threshold).any(axis=1)]

    # print(columns_with_values_above_threshold)
    signal_power_array = signal_power_df.values
    # a=signal_power_df.quantile(high_low_perc)
    # print(a)
    # print(signal_power_array)
    # 计算千分位值
    power_high, power_low = np.percentile(signal_power_array, [value * 100 for value in high_low_perc])
    # power_range=power_high-power_low

    # print(power_high)
    # print(power_low)

    # 创建一个新列用于存储判断结果

    filtered_df = filtered_df.copy()

    filtered_df = filtered_df.iloc[:, 1:]
    filtered_df = filtered_df[(filtered_df > voltage_threshold).any(axis=1)]

    filtered_df['Threshold_Exceed_num'] = -1
    # filtered_df['Threshold_Exceed_num'] = filtered_df.iloc[:, 1:].gt(voltage_threshold).sum(axis=1)
    filtered_df['Threshold_Exceed_num'] = filtered_df.gt(voltage_threshold).sum(axis=1)

    bandwidth_high, bandwidth_low = filtered_df['Threshold_Exceed_num'].quantile(high_low_perc)
    # bandwidth_range=bandwidth_high-bandwidth_low
    # print(time_high)
    # print(time_low)
    # print(bandwidth_high)
    # print(bandwidth_low)
    return (power_high, power_low), (bandwidth_high, bandwidth_low), (time_high, time_low)

    pass


def generate_power_abnormity(start_date, end_date, power_high_and_low, bandwidth_high_and_low, time_high_and_low,
                             abnormal_num):
    # 选择合适的异常值范围，大功率的范围，小功率的范围，
    # 在正常信号发生过程进行修改，在未出现信号的过程进行修改
    # 统计正常信号的功率值，并将千分位0.1和99定位正常功率的最小和最大界限值，
    # 并将99和0.1之间的跨度值inter作为参考，小功率异常值为最小限度min-inter*n（>0),大功率值为最大限度值max+inter*n(>0)

    power_high, power_low = power_high_and_low[0], power_high_and_low[1]
    power_range = power_high - power_low

    bandwidth_high, bandwidth_low = bandwidth_high_and_low[0], bandwidth_high_and_low[1]
    # bandwidth_range=bandwidth_high-bandwidth_low

    time_high, time_low = int(time_high_and_low[0]), int(time_high_and_low[1])

    # time_range= time_high-time_low

    # 生成10个随机时间段并存储在DataFrame中
    time_periods = [generate_random_time_period(start_date, end_date, time_low, time_high) for _ in range(abnormal_num)]

    abnormal_power_list = []
    for time_period in time_periods:
        abnormal_power = []
        abnormal_power.append(time_period[0])
        abnormal_power.append(time_period[1])
        abnormal_power_low = power_low - random.uniform(0, 3) * power_range
        abnormal_power_high = power_high + random.uniform(0, 3) * power_range
        # # 修正
        if(abnormal_power_low <-40):
            abnormal_power_low=random.uniform(-40,power_low)
        if(abnormal_power_high>150):
            abnormal_power_low = random.uniform(power_high, 150)

        if (random.random() > 0.5):
            abnormal_power.append(abnormal_power_high)
        else:
            abnormal_power.append(abnormal_power_low)
        normal_bandwidth = random.randint(bandwidth_low, bandwidth_high)
        abnormal_power.append(normal_bandwidth)
        abnormal_power_list.append(abnormal_power)
        # normal_bandwidth

    abnormal_power_df = pd.DataFrame(abnormal_power_list, columns=['start_time', 'end_time', 'power', 'bandwidth'])

    # power_high_and_low

    return abnormal_power_df
    pass


def generate_bandwidth_abnormity(start_date, end_date, power_high_and_low, bandwidth_high_and_low, time_high_and_low,
                             abnormal_num):
    # 生成异常值的范围，过大带宽的范围，过小带宽的范围
    # 在正常信号发生过程进行修改，在未出现信号的过程进行修改


    power_high, power_low = power_high_and_low[0], power_high_and_low[1]
    # power_range = power_high - power_low

    bandwidth_high, bandwidth_low = bandwidth_high_and_low[0], bandwidth_high_and_low[1]
    bandwidth_range=bandwidth_high-bandwidth_low

    time_high, time_low = int(time_high_and_low[0]), int(time_high_and_low[1])

    # time_range= time_high-time_low

    # 生成10个随机时间段并存储在DataFrame中
    time_periods = [generate_random_time_period(start_date, end_date, time_low,time_high) for _ in range(abnormal_num)]
    abnormal_instance_list = []
    for time_period in time_periods:
        abnormal_instance = []

        abnormal_instance.append(time_period[0])
        abnormal_instance.append(time_period[1])
        power=random.uniform(power_low,power_high)
        abnormal_instance.append(power)

        abnormal_bandwidth_low=random.randint(1, bandwidth_low)
        # abnormal_bandwidth_high=random.randint(bandwidth_high, bandwidth_high+bandwidth_low)
        abnormal_bandwidth_high=random.randint(bandwidth_high, bandwidth_high*2)
        if (random.random() > 0.5):
            abnormal_instance.append(abnormal_bandwidth_high)
        else:
            abnormal_instance.append(abnormal_bandwidth_low)
        abnormal_instance_list.append(abnormal_instance)
        # normal_bandwidth

    abnormal_instance_df = pd.DataFrame(abnormal_instance_list, columns=['start_time', 'end_time', 'power', 'bandwidth'])
    return abnormal_instance_df
    pass


def generate_duration_abnormity(start_date, end_date, power_high_and_low, bandwidth_high_and_low, time_high_and_low,
                             abnormal_num):
    # 生成异常值的范围，过大带宽的范围，过小带宽的范围
    # 在正常信号发生过程进行修改，在未出现信号的过程进行修改

    power_high, power_low = power_high_and_low[0], power_high_and_low[1]
    # power_range = power_high - power_low

    bandwidth_high, bandwidth_low = bandwidth_high_and_low[0], bandwidth_high_and_low[1]
    bandwidth_range = bandwidth_high - bandwidth_low

    time_high, time_low = int(time_high_and_low[0]), int(time_high_and_low[1])

    # time_range= time_high-time_low

    # 生成n个随机时间段并存储在DataFrame中
    time_periods=[]
    for _ in range(abnormal_num):
        if (random.random() > 0.5):
            time_periods.append(generate_random_time_period(start_date, end_date,  time_high,int(time_high*random.uniform(1, 2))))

        else:
            time_periods.append(generate_random_time_period(start_date, end_date,  0,time_low))


    # time_periods = [generate_random_time_period(start_date, end_date, time_high, time_low) for _ in range(abnormal_num)]
    abnormal_instance_list = []
    for time_period in time_periods:
        abnormal_instance = []
        abnormal_instance.append(time_period[0])
        abnormal_instance.append(time_period[1])
        power = random.uniform(power_low, power_high)
        abnormal_instance.append(power)


        bandwidth=random.randint(bandwidth_low,bandwidth_high)
        abnormal_instance.append(bandwidth)
        # normal_bandwidth
        abnormal_instance_list.append(abnormal_instance )
    abnormal_instance_df = pd.DataFrame(abnormal_instance_list,
                                        columns=['start_time', 'end_time', 'power', 'bandwidth'])
    return abnormal_instance_df

pass


# generate_abnormal_data()
