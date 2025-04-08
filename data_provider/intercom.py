import os
import random

import numpy as np
import pandas as pd

from data_provider.generate_abnormal_data import *

train_raw_data_dir="raw_data/train_raw_data"
train_signal_happen_time_file="raw_data/signal_record/train_signal_happen_time.csv"
test_raw_data_dir="raw_data/test_raw_data"
test_abnormal_data_dir="raw_data/test_abnormal_data"
test_signal_happen_time_file="raw_data/signal_record/test_signal_happen_time.csv"
abnormal_process_file="raw_data/raw_label/abnormal_process.csv"
abnormal_label_file="raw_data/raw_label/abnormal_label.csv"
train_data_file="intercom_train.csv"
test_data_file="intercom_test.csv"
all_abnormal_num =100
random.seed(0)

def get_all_file_list(csv_file_dir):
    # 获取该文件夹中所有的bin文件路径
    file_path = os.listdir(csv_file_dir)
    file_list = list(map(lambda x: os.path.join(csv_file_dir, x).replace('\\', '/'), file_path))
    # all_file_path = sorted(file_list, key=lambda s: int(s.split('\\')[-1].split('_')[0]))
    all_file_path = sorted(file_list, key=lambda s: int(s.split('/')[-1].split('_')[0]))
    return all_file_path
    pass
def find_signal_from_one_file(file_path, voltage_threshold, continuous_time_threshold):
    df = pd.read_csv(file_path)

    # print(df.columns.str.strip())

    # filter_freq_list = list(
    #     filter(lambda x: x=='date'or ((float(x) >= signal_start_freq) and (float(x) <= signal_stop_freq)),
    #            df.columns))
    # df=df[filter_freq_list]

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df=df.resample('1S').mean()
    # 创建一个新列用于存储判断结果
    df['Threshold_Exceed'] = 0
    df['Threshold_Exceed'] = df.iloc[:, 1:].gt(voltage_threshold).any(axis=1)
    signal_df=df[df['Threshold_Exceed'] ==1]
    signal_df=signal_df.copy()
    signal_df['time_diff'] = signal_df.index.to_series().diff().dt.total_seconds().fillna(0)
    signal_df['Group'] = signal_df['time_diff'] > continuous_time_threshold
    signal_df['Group'] = signal_df['Group'].cumsum()
    signal_df=signal_df.reset_index()
    grouped=signal_df.groupby('Group')
    signal_time_df=grouped['date'].agg(['min', 'max']).reset_index()
    signal_time_df=signal_time_df.rename(columns={'min':'start_time','max':'end_time'})
    # print(type(signal_time_df['end_time'][9]))
    # print(signal_time_df['end_time'])
    # signal_time_df['duration_seconds'] = (signal_time_df['end_time'] - signal_time_df['start_time'])
    signal_time_df['duration_seconds'] = (signal_time_df['end_time'] - signal_time_df['start_time']).dt.total_seconds()+1
    signal_time_df=signal_time_df.drop(columns='Group')

    # print(signal_time_df)

    # 统计信号在十分钟间隔中，信号产生的次数，持续的总时间
    # signal_df.set_index('date', inplace=True)

    return signal_time_df
    pass

def get_earlist_and_lastest_time(test_raw_data_dir_path):
    all_file_path=get_all_file_list(test_raw_data_dir_path)
    earliest_signal_df=pd.read_csv(all_file_path[0])
    earliest_date=earliest_signal_df['date'].iloc[0]
    lastest_signal_df=pd.read_csv(all_file_path[-1])
    lastest_date=lastest_signal_df['date'].iloc[-1]
    return earliest_date,lastest_date
    pass

def find_signal_record(all_file_path ,voltage_threshold=0,continuous_time_threshold=15):
    signal_occur_time=None
    for file_path in all_file_path:
        signal_time=find_signal_from_one_file(file_path, voltage_threshold, continuous_time_threshold)
        if(signal_occur_time is None):
            signal_occur_time=signal_time
        else:
            signal_occur_time=pd.concat([signal_occur_time, signal_time], axis=0, ignore_index=True)
    signal_occur_time['start_time'] = pd.to_datetime(signal_occur_time['start_time'])
    signal_occur_time = signal_occur_time.sort_values(by='start_time')
    return signal_occur_time

    pass

def generate_abnormal_data(root_path):
    # 生成三类异常数据：功率过大或过小，带宽过大或过小，持续时间过大或过小
    train_data_file_path = os.path.join(root_path, train_data_file)
    train_signal_happen_time_file_path = os.path.join(root_path, train_signal_happen_time_file)
    test_raw_data_dir_path=os.path.join(root_path, test_raw_data_dir)
    abnormal_label_file_path=os.path.join(root_path, abnormal_label_file)
    normal_signal_df = pd.read_csv(train_data_file_path, error_bad_lines=False)
    train_signal_happen_time_df = pd.read_csv(train_signal_happen_time_file_path, error_bad_lines=False)
    power_high_and_low, bandwidth_high_and_low, time_high_and_low = normal_value(normal_signal_df, train_signal_happen_time_df)
    # print(time_high_and_low)
    start_date,end_date=get_earlist_and_lastest_time(test_raw_data_dir_path)

    abnormal_num = all_abnormal_num//3
    # 功率异常
    abnormal_power_df = generate_power_abnormity(start_date, end_date, power_high_and_low, bandwidth_high_and_low,
                                                 time_high_and_low,
                                                 abnormal_num)
    abnormal_df=abnormal_power_df
    # # 带宽异常
    # abnormal_bandwidth_df = generate_bandwidth_abnormity(start_date, end_date, power_high_and_low, bandwidth_high_and_low,time_high_and_low,abnormal_num)
    # abnormal_df=pd.concat([abnormal_df, abnormal_bandwidth_df], ignore_index=True)

    # # 持续时间异常
    # abnormal_duration_df = generate_duration_abnormity(start_date, end_date, power_high_and_low, bandwidth_high_and_low,
    #                                              time_high_and_low,
    #                                              abnormal_num)
    # abnormal_df=pd.concat([abnormal_df, abnormal_duration_df], ignore_index=True)
    abnormal_df = abnormal_df.sort_values(by='start_time')

    abnormal_df.to_csv(os.path.join(root_path,abnormal_process_file),index=False)
    # 将异常数据写回原始数据中
    test_abnormal_data_dir_path=os.path.join(root_path, test_abnormal_data_dir)
    if not os.path.exists(test_abnormal_data_dir_path):
        os.makedirs(test_abnormal_data_dir_path)
    write_abnormal_to_csv(abnormal_df, test_raw_data_dir_path,test_abnormal_data_dir_path)
    signal_raw_label_df =  abnormal_df[['start_time', 'end_time']]
    signal_raw_label_df = signal_raw_label_df.copy()
    signal_raw_label_df['duration_seconds'] = (signal_raw_label_df['end_time'] - signal_raw_label_df[
        'start_time']).dt.total_seconds() + 1
    signal_raw_label_df.to_csv(abnormal_label_file_path, index=False)
    pass


def process_train_data(root_path,left_win_size,right_wind_size ,voltage_threshold=0, continuous_time_threshold=15):

    # csv_file_dir = os.path.join(root_path, "raw_data/train_raw_data")
    csv_file_dir = os.path.join(root_path, train_raw_data_dir)
    all_file_path = get_all_file_list(csv_file_dir)

    # if (os.path.exists(os.path.join(root_path, "raw_data/signal_record/train_signal_happen_time.csv"))):
    if (os.path.exists(os.path.join(root_path, train_signal_happen_time_file))):
        # signal_happen_time_file = os.path.join(root_path, "raw_data/signal_record/train_signal_happen_time.csv")
        signal_happen_time_file = os.path.join(root_path, train_signal_happen_time_file)
        signal_happen_time_df = pd.read_csv(signal_happen_time_file)
    else:
        # find_all_signal_record()
        signal_happen_time_df=find_signal_record(all_file_path, voltage_threshold=voltage_threshold, continuous_time_threshold=continuous_time_threshold)
        # signal_happen_time_file = os.path.join(root_path, "raw_data/signal_record/train_signal_happen_time.csv")
        signal_happen_time_file = os.path.join(root_path, train_signal_happen_time_file)
        signal_happen_time_df.to_csv(signal_happen_time_file,index=False)
    signal_happen_time_df['start_time'] = pd.to_datetime(signal_happen_time_df['start_time']) - pd.to_timedelta(
        left_win_size, unit='s')
    signal_happen_time_df['end_time'] = pd.to_datetime(signal_happen_time_df['end_time']) + pd.to_timedelta(
        right_wind_size, unit='s')

    signal_happen_time_df = signal_happen_time_df.sort_values(by='start_time')
    # 考虑是否删掉持续时间过小的信号


    signal_df = None
    for file_path in all_file_path:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        mask = None
        for index, row in signal_happen_time_df.iterrows():
            select_start_date, select_end_date = row['start_time'], row['end_time']
            if (select_end_date < df['date'].iloc[0]):
                continue
            if (select_start_date > df['date'].iloc[-1]):
                break

            mask1 = (df['date'] >= pd.to_datetime(select_start_date)) & (
                    df['date'] <= pd.to_datetime(select_end_date))
            if (mask is None):
                mask = mask1
            else:
                mask = mask | mask1
        filtered_df = df.loc[mask]
        if (signal_df is None):
            signal_df = filtered_df
        else:
            signal_df = pd.concat([signal_df, filtered_df], axis=0, ignore_index=True)

    df_save_path=os.path.join(root_path, train_data_file)
    signal_df.to_csv(df_save_path,index=False)
    pass


def process_test_data(root_path,left_win_size,right_wind_size ,voltage_threshold=0, continuous_time_threshold=15):
    csv_file_dir = os.path.join(root_path, test_raw_data_dir)
    all_file_path = get_all_file_list(csv_file_dir)
    if (os.path.exists(os.path.join(root_path, test_signal_happen_time_file))):
        signal_happen_time_file = os.path.join(root_path, test_signal_happen_time_file)
        signal_happen_time_df = pd.read_csv(signal_happen_time_file)
    else:
        signal_happen_time_df=find_signal_record(all_file_path, voltage_threshold=voltage_threshold, continuous_time_threshold=continuous_time_threshold)
        # signal_happen_time_file = os.path.join(root_path, "raw_data/signal_record/test_signal_happen_time.csv")
        signal_happen_time_file = os.path.join(root_path, test_signal_happen_time_file)
        signal_happen_time_df.to_csv(signal_happen_time_file,index=False)
    # 除了有信号的时间段外，还要将异常发生的时间段加入进去

    # abnormal_label_file_path=os.path.join(root_path, abnormal_label_file)
    if not (os.path.exists(os.path.join(root_path, abnormal_label_file))):
        generate_abnormal_data(root_path)

    abnormal_label_df=pd.read_csv(os.path.join(root_path, abnormal_label_file))
    signal_happen_time_df= pd.concat([signal_happen_time_df, abnormal_label_df], axis=0, ignore_index=True)
    signal_happen_time_df['start_time'] = pd.to_datetime(signal_happen_time_df['start_time']) - pd.to_timedelta(
        left_win_size, unit='s')
    signal_happen_time_df['end_time'] = pd.to_datetime(signal_happen_time_df['end_time']) + pd.to_timedelta(
        right_wind_size, unit='s')

    signal_happen_time_df = signal_happen_time_df.sort_values(by='start_time')

    signal_df = None

    csv_file_dir = os.path.join(root_path, test_abnormal_data_dir)
    all_file_path = get_all_file_list(csv_file_dir)
    for file_path in all_file_path:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        mask = None
        for index, row in signal_happen_time_df.iterrows():
            select_start_date, select_end_date = row['start_time'], row['end_time']
            if (select_end_date < df['date'].iloc[0]):
                continue
            if (select_start_date > df['date'].iloc[-1]):
                break

            mask1 = (df['date'] >= pd.to_datetime(select_start_date)) & (
                    df['date'] <= pd.to_datetime(select_end_date))
            if (mask is None):
                mask = mask1
            else:
                mask = mask | mask1
        filtered_df = df.loc[mask]
        if (signal_df is None):
            signal_df = filtered_df
        else:
            signal_df = pd.concat([signal_df, filtered_df], axis=0, ignore_index=True)

    label_mask = None
    abnormal_label_df['start_time'] = pd.to_datetime(abnormal_label_df['start_time'])
    abnormal_label_df['end_time'] = pd.to_datetime(abnormal_label_df['end_time'])

    for index, row in abnormal_label_df.iterrows():
        select_start_date, select_end_date = row['start_time'], row['end_time']
        if (select_end_date < signal_df['date'].iloc[0]):
            continue
        if (select_start_date > signal_df['date'].iloc[-1]):
            break
        mask1 = (signal_df['date'] >= pd.to_datetime(select_start_date)) & (
                signal_df['date'] <= pd.to_datetime(select_end_date))
        if (label_mask  is None):
            label_mask = mask1
        else:
            label_mask = label_mask | mask1

    signal_df['label']=label_mask
    df_save_path = os.path.join(root_path, test_data_file)
    signal_df.to_csv(df_save_path,index=False)
    pass


def read_continuous_data(df,win_size,step=1):
    # 将日期列转换为datetime格式
    df['date'] = pd.to_datetime(df['date'])
    # 计算日期之间的差值，并找出不连续的点
    df['date_diff'] = df['date'].diff().dt.total_seconds().fillna(0)
    df['group'] = (df['date_diff'] > 1).cumsum()
    # 根据分组键进行分组
    grouped = df.groupby('group')
    # 打印每个分组的结果
    data = []
    for name, group in grouped:
        cols= list(group.columns)
        cols.remove('date')
        cols.remove('date_diff')
        cols.remove('group')
        group_data = group[cols]
        group_rows = group_data.values
        # df_stamp =group_data[['date']]
        for i in range((len(group) - win_size) // step + 1):
            data.append(group_rows[i:i + win_size])
        # print(f"Group {name}:")
        # print(group)
        # print("\n")
    return data
    pass


def read_data(root_path, left_win_size,right_wind_size ,flag="train"):
    if (flag == "train"):
        if not os.path.exists(os.path.join(root_path, train_data_file)):
            process_train_data(root_path, left_win_size,right_wind_size , voltage_threshold=0, continuous_time_threshold=15)
        train_raw_df=pd.read_csv(os.path.join(root_path, train_data_file))
        return train_raw_df
    elif (flag == "test"):
        if not os.path.exists(os.path.join(root_path, test_data_file)):
            process_test_data(root_path,left_win_size,right_wind_size , voltage_threshold=0, continuous_time_threshold=15)
        test_raw_df=pd.read_csv(os.path.join(root_path, test_data_file))
        return test_raw_df
    else:
        if not os.path.exists(os.path.join(root_path, train_data_file)):
            process_train_data(root_path, left_win_size,right_wind_size , voltage_threshold=0, continuous_time_threshold=15)
        train_raw_df=pd.read_csv(os.path.join(root_path, train_data_file))
        return train_raw_df
        pass
