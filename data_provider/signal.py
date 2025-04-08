#原始数据 → 信号事件分割 → 特征提取 → 异常注入 → 格式转换 → 模型输入
#主要负责从原始电磁信号数据中提取粗粒度和细粒度的特征，生成训练和测试数据集，并注入仿真异常。
import os
import random

import numpy as np
import pandas as pd
import itertools

from data_provider.generate_signal_abnormal_data import *

from scipy.spatial.distance import cosine
# from data_provider.intercom import generate_abnormal_data
import ast

def get_dataset_parameter(root_path):
    '''
    定义数据路径和参数（如时间窗口规则、阈值）。
    :param root_path:
    :return:
    '''
    dataset_parameter = {
        "train_raw_data_dir": "raw_data/train_raw_data",
        "train_signal_happen_time_file": "raw_data/signal_record/train_signal_happen_time.csv",
        "test_raw_data_dir": "raw_data/test_raw_data",
        "test_abnormal_data_dir": "raw_data/test_abnormal_data",
        "test_signal_happen_time_file": "raw_data/signal_record/test_signal_happen_time.csv",
        "abnormal_process_file": "raw_data/abnormal_label/abnormal_process.csv",
        "final_abnormal_label_file": "raw_data/abnormal_label/final_abnormal_label.csv",
        "fine_abnormal_label_file": "raw_data/abnormal_label/fine_abnormal_label.csv",
        "train_data_file": "intercom_train.csv",
        "test_data_file": "intercom_test.csv",
        "train_coarse_grained_data_file": "coarse_grained_data/train_coarse_grained_data.csv",
        "test_coarse_grained_data_file": "coarse_grained_data/test_coarse_grained_data.csv",
        "train_fine_grained_data_file": "fine_grained_data/train_fine_grained_data.csv",
        "test_fine_grained_data_file": "fine_grained_data/test_fine_grained_data.csv",
        "train_signal_record_and_feature_file": "raw_data/signal_record_and_feature/train_signal_record_and_feature.csv",
        "test_signal_record_and_feature_file": "raw_data/signal_record_and_feature/test_signal_record_and_feature.csv",
        "test_abnormal_signal_record_and_feature_file": "raw_data/signal_record_and_feature/test_abnormal_signal_record_and_feature.csv",
        "normal_value_list": [0.05, 0.5, 0.95],
        "adjust_abnormal_time":False
    }
    dataset_name = root_path.replace('\\', '/').split('/')[-1]
    if (dataset_name == 'EM_leakage'):
        # dataset_parameter["custom_intervals"]=[('00:00', '7:59'),('8:00', '17:59'),('18:00', '23:59')]
        # dataset_parameter["custom_rules"]=['10T','1T','5T']
        dataset_parameter["custom_intervals"]=[('00:00', '7:59'),('8:00', '18:59'),('19:00', '21:59'),('22:00', '23:59')]
        dataset_parameter["custom_rules"]=['8T','1T','2T','4T']
        # dataset_parameter["custom_rules"]=['16T','2T','4T','8T']



        dataset_parameter["SIR"]=None
        dataset_parameter["SNR"]=None
        dataset_parameter["normal_signal_change_ratio"] =None
        dataset_parameter["adjust_behavior_abnormal_time"]=True
        # dataset_parameter["custom_intervals"]=[('00:00', '23:59')]
        # dataset_parameter["custom_rules"]=['16T']
        # dataset_parameter["custom_intervals"] = [('00:00', '7:59'), ('8:00', '17:59'), ('18:00', '21:59'),
        #                                          ('22:00', '23:59')]
        # dataset_parameter["custom_rules"] = ['4T', '0.5T', '1T', '2T']

        # dataset_parameter["custom_intervals"]=[('00:00', '7:59'),('8:00', '18:59'),('19:00', '23:59')]
        # dataset_parameter["custom_rules"]=['8T','2T','4T']

        # dataset_parameter["custom_intervals"]=[('00:00', '7:59'),('8:00', '18:59'),('19:00', '21:59'),('22:00', '23:59')]
        # dataset_parameter["custom_rules"]=['4T','1T','2T','3T']

        dataset_parameter['fine_addit_information']=True
        dataset_parameter["voltage_threshold"] = -4
        dataset_parameter["communication_continuous_time_threshold"] = 6
        dataset_parameter["emission_continuous_time_threshold"] = 1
        dataset_parameter["abnormal_num"] = 200
        dataset_parameter['m_n']=(3, 7)
        dataset_parameter['abnormal_start_and_end_time'] = ('00:30:00', '23:59:00')
        dataset_parameter["coarse_columns"] = ['date', 'communication_num', 'duration_time_min', 'duration_time_max',
                                               'duration_time_mean', 'duration_time_median', 'duration_time_sum',
                                               'emission_time_min', 'emission_time_max', 'emission_time_mean',
                                               'emission_time_median', 'emission_time_sum',
                                               'emission_interval_time_max_min', 'emission_interval_time_max_max',
                                               'emission_interval_time_max_mean', 'emission_interval_time_max_median',
                                               'freq_bandwidth_min_min', 'freq_bandwidth_min_max',
                                               'freq_bandwidth_min_mean', 'freq_bandwidth_min_median',
                                               'freq_bandwidth_max_min', 'freq_bandwidth_max_max',
                                               'freq_bandwidth_max_mean', 'freq_bandwidth_max_median',
                                               'freq_bandwidth_mean_min', 'freq_bandwidth_mean_max',
                                               'freq_bandwidth_mean_mean', 'freq_bandwidth_mean_median',
                                               'freq_bandwidth_median_min', 'freq_bandwidth_median_max',
                                               'freq_bandwidth_median_mean', 'freq_bandwidth_median_median',
                                               'total_power_max_min', 'total_power_max_max', 'total_power_max_mean',
                                               'total_power_max_median', 'total_power_mean_min', 'total_power_mean_max',
                                               'total_power_mean_mean', 'total_power_mean_median',
                                               'total_power_median_min', 'total_power_median_max',
                                               'total_power_median_mean', 'total_power_median_median',
                                               'total_power_sum_sum', 'signal_power_max_min', 'signal_power_max_max',
                                               'signal_power_max_mean', 'signal_power_max_median',
                                               'signal_power_mean_min', 'signal_power_mean_max',
                                               'signal_power_mean_mean', 'signal_power_mean_median',
                                               'signal_power_median_min', 'signal_power_median_max',
                                               'signal_power_median_mean', 'signal_power_median_median',
                                               'signal_power_sum_sum', 'noise_power_max_min', 'noise_power_max_max',
                                               'noise_power_max_mean', 'noise_power_max_median', 'noise_power_mean_min',
                                               'noise_power_mean_max', 'noise_power_mean_mean',
                                               'noise_power_mean_median', 'noise_power_median_min',
                                               'noise_power_median_max', 'noise_power_median_mean',
                                               'noise_power_median_median', 'noise_power_sum_sum',
                                               'signal_first_singular_value_min', 'signal_first_singular_value_max',
                                               'signal_first_singular_value_mean', 'signal_first_singular_value_median',
                                               'start_time_diffs_max', 'start_time_diffs_min',
                                               'duration_time_diffs_max', 'duration_time_diffs_min',
                                               'emission_time_diffs_max', 'emission_time_diffs_min',
                                               'freq_bandwidth_max_diffs_max', 'freq_bandwidth_max_diffs_min',
                                               'signal_power_max_diffs_max', 'signal_power_max_diffs_min',
                                               'max_cosine_similarity', 'min_cosine_similarity', 'rule']
        # dataset_parameter["fine_addit_keys"]=['freq_bandwidth_min','freq_bandwidth_max','freq_bandwidth_mean','signal_power_max'
        #     ,'signal_power_mean','signal_power_sum','noise_power_mean','signal_first_singular_value','emission_time']
        dataset_parameter["fine_addit_keys"]=['emission_time_order','emission_time']
    elif dataset_name == '电梯信号':
        print("数据集:电梯信号")
        dataset_parameter["custom_intervals"] = [('00:00', '7:59'), ('8:00', '18:59'), ('19:00', '21:59'), ('22:00', '23:59')]
        # dataset_parameter["custom_rules"]=['15T','5T','10T']
        dataset_parameter["custom_rules"] = ['8T', '1T', '2T','4T']

        dataset_parameter["SIR"] = ''
        dataset_parameter["SNR"] = None
        dataset_parameter["normal_signal_change_ratio"] = ''
        dataset_parameter["adjust_behavior_abnormal_time"] = False
        dataset_parameter['fine_addit_information'] = False
        dataset_parameter["voltage_threshold"] = -4
        dataset_parameter["communication_continuous_time_threshold"] = 12

        dataset_parameter["emission_continuous_time_threshold"] = 1
        dataset_parameter["abnormal_num"] = 120
        dataset_parameter['m_n'] = (3, 7)
        dataset_parameter['abnormal_start_and_end_time'] = ('00:30:00', '23:59:00')
        dataset_parameter["coarse_columns"] = ['date', 'communication_num', 'duration_time_min', 'duration_time_max',
                                               'duration_time_mean', 'duration_time_median', 'duration_time_sum',
                                               'emission_time_min', 'emission_time_max', 'emission_time_mean',
                                               'emission_time_median', 'emission_time_sum',
                                               'emission_interval_time_max_min', 'emission_interval_time_max_max',
                                               'emission_interval_time_max_mean', 'emission_interval_time_max_median',
                                               'freq_bandwidth_min_min', 'freq_bandwidth_min_max',
                                               'freq_bandwidth_min_mean', 'freq_bandwidth_min_median',
                                               'freq_bandwidth_max_min', 'freq_bandwidth_max_max',
                                               'freq_bandwidth_max_mean', 'freq_bandwidth_max_median',
                                               'freq_bandwidth_mean_min', 'freq_bandwidth_mean_max',
                                               'freq_bandwidth_mean_mean', 'freq_bandwidth_mean_median',
                                               'freq_bandwidth_median_min', 'freq_bandwidth_median_max',
                                               'freq_bandwidth_median_mean', 'freq_bandwidth_median_median',
                                               'total_power_max_min', 'total_power_max_max', 'total_power_max_mean',
                                               'total_power_max_median', 'total_power_mean_min', 'total_power_mean_max',
                                               'total_power_mean_mean', 'total_power_mean_median',
                                               'total_power_median_min', 'total_power_median_max',
                                               'total_power_median_mean', 'total_power_median_median',
                                               'total_power_sum_sum', 'signal_power_max_min', 'signal_power_max_max',
                                               'signal_power_max_mean', 'signal_power_max_median',
                                               'signal_power_mean_min', 'signal_power_mean_max',
                                               'signal_power_mean_mean', 'signal_power_mean_median',
                                               'signal_power_median_min', 'signal_power_median_max',
                                               'signal_power_median_mean', 'signal_power_median_median',
                                               'signal_power_sum_sum', 'noise_power_max_min', 'noise_power_max_max',
                                               'noise_power_max_mean', 'noise_power_max_median', 'noise_power_mean_min',
                                               'noise_power_mean_max', 'noise_power_mean_mean',
                                               'noise_power_mean_median', 'noise_power_median_min',
                                               'noise_power_median_max', 'noise_power_median_mean',
                                               'noise_power_median_median', 'noise_power_sum_sum',
                                               'signal_first_singular_value_min', 'signal_first_singular_value_max',
                                               'signal_first_singular_value_mean', 'signal_first_singular_value_median',
                                               'start_time_diffs_max', 'start_time_diffs_min',
                                               'duration_time_diffs_max', 'duration_time_diffs_min',
                                               'emission_time_diffs_max', 'emission_time_diffs_min',
                                               'freq_bandwidth_max_diffs_max', 'freq_bandwidth_max_diffs_min',
                                               'signal_power_max_diffs_max', 'signal_power_max_diffs_min',
                                               'max_cosine_similarity', 'min_cosine_similarity', 'rule']
        dataset_parameter["fine_addit_keys"] = ['freq_bandwidth_min', 'freq_bandwidth_max', 'freq_bandwidth_mean',
                                                'signal_power_max'
            , 'signal_power_mean', 'signal_power_sum', 'noise_power_mean', 'signal_first_singular_value',
                                                'emission_time']
    else:
        # 时间的划分5分，10分，15分钟，20分钟，30分钟

        # dataset_parameter["custom_intervals"]=[('00:00', '23:59')]
        # dataset_parameter["custom_rules"]=['5T']

        dataset_parameter["custom_intervals"]=[('00:00', '6:59'),('7:00', '18:59'),('17:00', '23:59')]
        #dataset_parameter["custom_rules"]=['15T','5T','10T']
        dataset_parameter["custom_rules"] = ['30T', '30T', '30T']

        dataset_parameter["SIR"] =''
        dataset_parameter["SNR"] = None
        dataset_parameter["normal_signal_change_ratio"] =''
        dataset_parameter["adjust_behavior_abnormal_time"]=False
        dataset_parameter['fine_addit_information'] = False
        #dataset_parameter["voltage_threshold"] = -4
        dataset_parameter["voltage_threshold"] = -2
        #dataset_parameter["communication_continuous_time_threshold"] = 30
        dataset_parameter["communication_continuous_time_threshold"] = 120

        dataset_parameter["emission_continuous_time_threshold"] = 1
        #dataset_parameter["abnormal_num"] = 50
        dataset_parameter["abnormal_num"] = 3
        dataset_parameter['m_n'] = (3, 7)
        dataset_parameter['abnormal_start_and_end_time'] = ('00:30:00', '23:59:00')
        dataset_parameter["coarse_columns"] = ['date', 'communication_num', 'duration_time_min', 'duration_time_max',
                                               'duration_time_mean', 'duration_time_median', 'duration_time_sum',
                                               'emission_time_min', 'emission_time_max', 'emission_time_mean',
                                               'emission_time_median', 'emission_time_sum',
                                               'emission_interval_time_max_min', 'emission_interval_time_max_max',
                                               'emission_interval_time_max_mean', 'emission_interval_time_max_median',
                                               'freq_bandwidth_min_min', 'freq_bandwidth_min_max',
                                               'freq_bandwidth_min_mean', 'freq_bandwidth_min_median',
                                               'freq_bandwidth_max_min', 'freq_bandwidth_max_max',
                                               'freq_bandwidth_max_mean', 'freq_bandwidth_max_median',
                                               'freq_bandwidth_mean_min', 'freq_bandwidth_mean_max',
                                               'freq_bandwidth_mean_mean', 'freq_bandwidth_mean_median',
                                               'freq_bandwidth_median_min', 'freq_bandwidth_median_max',
                                               'freq_bandwidth_median_mean', 'freq_bandwidth_median_median',
                                               'total_power_max_min', 'total_power_max_max', 'total_power_max_mean',
                                               'total_power_max_median', 'total_power_mean_min', 'total_power_mean_max',
                                               'total_power_mean_mean', 'total_power_mean_median',
                                               'total_power_median_min', 'total_power_median_max',
                                               'total_power_median_mean', 'total_power_median_median',
                                               'total_power_sum_sum', 'signal_power_max_min', 'signal_power_max_max',
                                               'signal_power_max_mean', 'signal_power_max_median',
                                               'signal_power_mean_min', 'signal_power_mean_max',
                                               'signal_power_mean_mean', 'signal_power_mean_median',
                                               'signal_power_median_min', 'signal_power_median_max',
                                               'signal_power_median_mean', 'signal_power_median_median',
                                               'signal_power_sum_sum', 'noise_power_max_min', 'noise_power_max_max',
                                               'noise_power_max_mean', 'noise_power_max_median', 'noise_power_mean_min',
                                               'noise_power_mean_max', 'noise_power_mean_mean',
                                               'noise_power_mean_median', 'noise_power_median_min',
                                               'noise_power_median_max', 'noise_power_median_mean',
                                               'noise_power_median_median', 'noise_power_sum_sum',
                                               'signal_first_singular_value_min', 'signal_first_singular_value_max',
                                               'signal_first_singular_value_mean', 'signal_first_singular_value_median',
                                               'start_time_diffs_max', 'start_time_diffs_min',
                                               'duration_time_diffs_max', 'duration_time_diffs_min',
                                               'emission_time_diffs_max', 'emission_time_diffs_min',
                                               'freq_bandwidth_max_diffs_max', 'freq_bandwidth_max_diffs_min',
                                               'signal_power_max_diffs_max', 'signal_power_max_diffs_min',
                                               'max_cosine_similarity', 'min_cosine_similarity', 'rule']
        dataset_parameter["fine_addit_keys"]=['freq_bandwidth_min','freq_bandwidth_max','freq_bandwidth_mean','signal_power_max'
            ,'signal_power_mean','signal_power_sum','noise_power_mean','signal_first_singular_value','emission_time']
    return dataset_parameter
    pass


# train_raw_data_dir="raw_data/train_raw_data"
# train_signal_happen_time_file="raw_data/signal_record/train_signal_happen_time.csv"
# test_raw_data_dir="raw_data/test_raw_data"
# test_abnormal_data_dir="raw_data/test_abnormal_data"
# test_signal_happen_time_file="raw_data/signal_record/test_signal_happen_time.csv"
# abnormal_process_file="raw_data/raw_label/abnormal_process.csv"
# abnormal_label_file="raw_data/raw_label/abnormal_label.csv"
# train_data_file="intercom_train.csv"
# test_data_file="intercom_test.csv"
# train_coarse_grained_data_file="coarse_grained_data/train_coarse_grained_data.csv"
# test_coarse_grained_data_file="coarse_grained_data/test_coarse_grained_data.csv"
# train_signal_record_and_feature_file="signal_record_and_feature/train_signal_record_and_feature.csv"
# test_signal_record_and_feature_file="signal_record_and_feature/test_signal_record_and_feature.csv"
#
#
# all_abnormal_num =100
# random.seed(0)


def get_all_file_list(csv_file_dir):
    # 获取该文件夹中所有的bin文件路径
    file_path = os.listdir(csv_file_dir)
    file_list = list(map(lambda x: os.path.join(csv_file_dir, x).replace('\\', '/'), file_path))
    # all_file_path = sorted(file_list, key=lambda s: int(s.split('\\')[-1].split('_')[0]))
    all_file_path = sorted(file_list, key=lambda s: int(s.split('/')[-1].split('_')[0]))
    return all_file_path
    pass


def find_signal_from_one_file(file_path, voltage_threshold, emission_continuous_time_threshold,
                              communication_continuous_time_threshold):
    '''从单文件提取信号事件特征（时间、带宽、功率、SVD等）。'''
    print("从单文件提取信号事件特征（时间、带宽、功率、SVD等）:"+str(file_path))
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.resample('1S').mean()

    # 统计带宽
    df_cols = list(df.columns)
    # df_cols.remove("date")
    df_cols = sorted(df_cols, key=lambda s: float(s))
    df['freq_bandwidth'] = 0
    if (len(df_cols) > 1):
        freq_resolution = round(float(df_cols[1]) - float(df_cols[0]), 6)
        df['freq_bandwidth'] = df.iloc[:, 1:].gt(voltage_threshold).sum(axis=1) * freq_resolution
    # 创建一个新列用于存储判断结果
    df['threshold_exceed'] = 0
    df['threshold_exceed'] = df.iloc[:, 1:-2].gt(voltage_threshold).any(axis=1)
    signal_df = df[df['threshold_exceed'] == 1]
    signal_df = signal_df.copy()
    if (len(signal_df) == 0):
        signal_feature_df = pd.DataFrame()
        return signal_feature_df
    signal_df['time_diff'] = signal_df.index.to_series().diff().dt.total_seconds().fillna(0)


    #根据时间间隔进行辐射过程的分组
    signal_df['communication_group'] = signal_df['time_diff'] > communication_continuous_time_threshold
    signal_df['communication_group'] = signal_df['communication_group'].cumsum()
    signal_df = signal_df.reset_index()
    communication_grouped = signal_df.groupby('communication_group')

    # 统计生成每个通讯过程的特征

    # 统计通讯过程的相关的值（泄漏信号，通讯过程等于辐射过程）：开始，结束，中心时间，持续总时间，通讯过程和上一个过程的间隔。
    communication_date_df = communication_grouped['date'].agg(['min', 'max']).reset_index()
    communication_date_df = communication_date_df.rename(
        columns={'min': 'communication_start_time', 'max': 'communication_end_time'})
    communication_date_df['communication_mean_time'] = communication_date_df['communication_start_time'] + (
                communication_date_df['communication_end_time'] - communication_date_df['communication_start_time']) / 2
    communication_date_df['communication_mean_time'] = communication_date_df['communication_mean_time'].dt.floor('S')

    communication_date_df['communication_duration_time'] = (communication_date_df['communication_end_time'] -
                                                            communication_date_df[
                                                                'communication_start_time']).dt.total_seconds() + 1
    communication_interval_time_from_previous = communication_grouped.apply(
        lambda x: x['time_diff'][x['time_diff'] > communication_continuous_time_threshold].max()
    ).reset_index(name="communication_interval_time_from_previous").fillna(0)

    communication_stats=pd.concat([communication_date_df.set_index('communication_group'),
                                   communication_interval_time_from_previous.set_index('communication_group')], axis=1).reset_index()





    # 辐射过程的特征:辐射过程次数，辐射间隔最大时间，辐射总时间。
    emission_num = communication_grouped.apply(
        lambda x: ((x['time_diff'] > emission_continuous_time_threshold) & (
                    x['time_diff'] < communication_continuous_time_threshold)).sum()
                  + 1).reset_index(name="emission_num")

    emission_interval_time_max = communication_grouped.apply(
        lambda x: x['time_diff'][(x['time_diff'] > emission_continuous_time_threshold) & (
                    x['time_diff'] < communication_continuous_time_threshold)].max()
    ).reset_index(name="emission_interval_time_max").fillna(0)
    emission_time = communication_grouped['date'].count().reset_index(name='emission_time')

    emission_stats=pd.concat([emission_num.set_index('communication_group'),emission_interval_time_max.set_index('communication_group'),
                              emission_time.set_index('communication_group')], axis=1).reset_index()


    # 带宽的统计值：最大最小，均值，中位值
    freq_bandwidth_stats = communication_grouped['freq_bandwidth'].agg(['min', 'max', 'mean', 'median']).reset_index()
    freq_bandwidth_stats = freq_bandwidth_stats.rename(columns={
        'min': 'freq_bandwidth_min',
        'max': 'freq_bandwidth_max',
        'mean':'freq_bandwidth_mean',
        'median':'freq_bandwidth_median',
    })






    # 功率的统计置：整体功率的最大小，均值，总和，中位值；信号的最大最小，均值，总和，中位值，噪声的功率值的最大最小，均值，总和，中位值
    signal_df_cols = list(signal_df.columns)
    remove_keys = ['date', 'threshold_exceed', 'time_diff', 'communication_group', 'freq_bandwidth']
    power_keys_list = [x for x in signal_df_cols if x not in remove_keys]
    total_power_stats = communication_grouped[power_keys_list].apply(lambda x: pd.Series({
        'total_power_max': x[power_keys_list].max().max(),
        'total_power_min': x[power_keys_list].min().min(),
        'total_power_sum': x[power_keys_list].sum().sum(),
        'total_power_mean': x[power_keys_list].stack().mean(),
        'total_power_median': x[power_keys_list].stack().median()
    })).reset_index()

    # 定义按组统计的函数
    def signal_and_noise_power(group, threshold):
        # 计算每列大于阈值的比例
        proportion = (group > threshold).mean()
        # 找出比例大于0.1的列名
        signal_cols = proportion[proportion >= 0.1].index.tolist()
        noise_cols=proportion[proportion <0.1].index.tolist()

        # 对满足条件的列计算共同的最大值、最小值、均值、总和和中位数
        if signal_cols:
            signal_power_max = group[signal_cols].values.max()
            signal_power_min= group[signal_cols].values.min()
            signal_power_mean= group[signal_cols].values.mean()
            signal_power_sum= group[signal_cols].values.sum()
            signal_power_median= pd.Series(group[signal_cols].values.flatten()).median()
        else:
            signal_power_max =signal_power_min= signal_power_mean= signal_power_sum= signal_power_median= None

        # 对不满足条件的列计算共同的最大值、最小值、均值、总和和中位数
        if noise_cols:
            noise_power_max= group[noise_cols].values.max()
            noise_power_min= group[noise_cols].values.min()
            noise_power_mean= group[noise_cols].values.mean()
            noise_power_sum= group[noise_cols].values.sum()
            noise_power_median= pd.Series(group[noise_cols].values.flatten()).median()
        else:
            noise_power_max = noise_power_min=noise_power_mean =noise_power_sum = noise_power_median= None



        # 返回统计结果
        result = pd.Series({
            'signal_power_max':signal_power_max,
            'signal_power_min':signal_power_min,
            'signal_power_mean': signal_power_mean,
            'signal_power_sum': signal_power_sum,
            'signal_power_median': signal_power_median,
            'noise_power_max':noise_power_max,
            'noise_power_min': noise_power_min,
            'noise_power_mean': noise_power_mean,
            'noise_power_sum': noise_power_sum,
            'noise_power_median': noise_power_median
        })

        return result
    signal_and_noise_power_stats= communication_grouped[power_keys_list].apply(signal_and_noise_power, threshold=voltage_threshold).reset_index()



    # 对信号部分进行奇异值分解，选取第一个奇异值和奇异向量。
    def compute_first_svd_component(group,threshold):
        # 提取数值列，忽略分组列
        group[group <= threshold]=0
        matrix = group.values
        # 奇异值分解
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        # 获取第一个奇异值和奇异向量
        first_singular_value = S[0]
        first_left_singular_vector = U[:, 0]
        first_right_singular_vector = Vt[0, :]
        # 返回结果
        return pd.Series({
            'signal_first_singular_value': first_singular_value,
            'signal_first_left_singular_vector': np.around(first_left_singular_vector, decimals=3).tolist(),
            'signal_first_right_singular_vector': np.around(first_right_singular_vector, decimals=3).tolist()
        })
    svd_component=communication_grouped[power_keys_list].apply(compute_first_svd_component,threshold=voltage_threshold).reset_index()
    signal_feature_df = pd.concat([communication_stats.set_index('communication_group'),
                                   emission_stats.set_index('communication_group'),
                                   freq_bandwidth_stats.set_index('communication_group'),
                                   total_power_stats.set_index('communication_group'),
                                   signal_and_noise_power_stats.set_index('communication_group'),
                                   svd_component.set_index('communication_group')], axis=1)

    # new_cols = ['communication_start_time', 'communication_end_time', 'communication_mean_time',
    #             'communication_interval_time_max', 'communication_duration_time', 'power_max',
    #             'emission_num', 'emission_interval_time_max', 'emission_time', 'freq_bandwidth_min',
    #             'freq_bandwidth_max']

    signal_feature_df = signal_feature_df.reset_index(drop=True)
    # signal_feature_df = signal_feature_df[new_cols]

    signal_feature_df = signal_feature_df.rename(columns={
        'communication_start_time': 'start_time',
        'communication_end_time': 'end_time',
        'communication_mean_time': 'mean_time',
        'communication_duration_time': 'duration_time'
    })
    #为了便于分析，把signal_feature_df临时保留下来吧
    '''f1=str(file_path).split("train_raw_data")[0]
    f2=str(file_path).split("/")[-1].split("_")[0]
    signal_feature_df.to_csv(f1+f2+".csv",index=False)
    print("为了便于分析，将signal_feature_df临时保留下来")'''

    return signal_feature_df
    pass


def get_earlist_and_lastest_time(test_raw_data_dir_path):
    all_file_path = get_all_file_list(test_raw_data_dir_path)
    earliest_signal_df = pd.read_csv(all_file_path[0])
    earliest_date = earliest_signal_df['date'].iloc[0]
    lastest_signal_df = pd.read_csv(all_file_path[-1])
    lastest_date = lastest_signal_df['date'].iloc[-1]
    return earliest_date, lastest_date
    pass


# 定义计算时间跨度的函数
def is_time_span_exceeding_one_day(start_time, end_time, custom_rules, extend_length):
    # 将时间字符串转换为 Timestamp 对象
    start = pd.Timestamp(start_time)
    end = pd.Timestamp(end_time)

    # 计算原始时间跨度
    if end < start:
        # 处理跨天情况，计算从起始时间到当天结束 + 从午夜到结束时间的跨度
        midnight = pd.Timestamp('23:59:59')  # 当天最后一秒
        next_day_start = pd.Timestamp('00:00:00')  # 第二天开始
        original_span = (midnight - start + pd.Timedelta(seconds=1)) + (end - next_day_start)
    else:
        # 不跨天，直接计算时间跨度
        original_span = end - start

    # 加上扩展时间
    extension_delta = pd.Timedelta(custom_rules)
    total_span = original_span + extension_delta * extend_length * 2  # 扩展前后各加一次

    # 判断总时间跨度是否超过 24 小时
    exceeds_one_day = total_span > pd.Timedelta('1 day')

    return exceeds_one_day, total_span


# 定义扩展时间段函数，支持灵活的时间扩展（如 '10T' 表示 10 分钟）
def extend_time_range(start_and_end_time, custom_rules, extend_length):
    start_time, end_time = start_and_end_time[0], start_and_end_time[1]

    exceeds_one_day, total_span = is_time_span_exceeding_one_day(start_time, end_time, custom_rules, extend_length)
    if (exceeds_one_day is True):
        start_time = pd.Timestamp('00:00').time()
        end_time = pd.Timestamp('23:59').time()
    else:
        # 将时间扩展部分转换为 pd.Timedelta 类型
        extension_delta = pd.Timedelta(custom_rules)
        # 扩展起始时间，减去扩展分钟
        start_time = (pd.Timestamp(start_time) - extension_delta * extend_length).time()
        # 扩展结束时间，加上扩展分钟
        end_time = (pd.Timestamp(end_time) + extension_delta* extend_length).time()

    # 使用 between_time 过滤并返回结果
    # return df.between_time(start_time, end_time)
    return (start_time, end_time)


def expand_time_range_bidirectional(start_time, end_time, win_time,extend_before_num, extend_after_num, date='2023-11-01'):
    """
    双向扩展给定的时间范围，并处理跨天情况。
    如果扩展后的时间范围为 24 小时，直接返回 ('00:00:00', '23:59:00')。

    参数：
    - start_time: 开始时间 (字符串，格式 'HH:MM:SS')
    - end_time: 结束时间 (字符串，格式 'HH:MM:SS')
    - extend_before: 向前扩展的时间长度 (例如 '15T' 表示 15 分钟)
    - extend_after: 向后扩展的时间长度 (例如 '30T' 表示 30 分钟)
    - date: 基准日期，用于生成时间范围

    返回：
    - 包含扩展时间段的列表，格式为 (HH:MM:SS, HH:MM:SS)，如果扩展为 24 小时，则返回单个 ('00:00:00', '23:59:00') 时间段。
    """
    # 生成起始和结束时间的 datetime 对象
    date = pd.to_datetime(date)
    start = pd.to_datetime(f"{date.date()} {start_time}")
    end = pd.to_datetime(f"{date.date()} {end_time}")

    # 向前和向后扩展
    # extended_start = start - pd.to_timedelta(extend_before)
    # extended_end = end + pd.to_timedelta(extend_after)

    extended_start = start - extend_before_num*pd.to_timedelta(win_time)
    extended_end = end + extend_after_num*pd.to_timedelta(win_time)

    # 检查是否扩展到 24 小时
    full_day_start = pd.to_datetime(f"{date.date()} 00:00:00")
    full_day_end = pd.to_datetime(f"{date.date()} 23:59:59")
    if extended_start <= full_day_start and extended_end >= full_day_end:
        return [("00:00:00", "23:59:59")]

    ranges = []

    # 检查向前扩展是否跨天
    if extended_start.date() < date.date():
        # 跨到前一天，将前一天的部分分离
        # previous_day_end = pd.to_datetime(f"{date.date()} 23:59:59")

        ranges.append((extended_start.time().strftime("%H:%M:%S"), pd.to_datetime(f"{extended_start.date()} 23:59:59").time().strftime("%H:%M:%S")))

        # 更新扩展开始时间为当天的起始时间
        # previous_day_end = pd.to_datetime(f"{date.date()} 00:00:00")
        # extended_start = previous_day_end
        extended_start = pd.to_datetime(f"{date.date()} 00:00:00")
    # 检查向后扩展是否跨天
    if extended_end.date() > date.date():
        # 跨到下一天，将下一天的部分分离
        # next_day_start = pd.to_datetime(f"{extended_end.date()} 00:00:00")
        ranges.append((extended_start.time().strftime("%H:%M:%S"),  pd.to_datetime(f"{extended_start.date()} 23:59:59").time().strftime("%H:%M:%S")))  # 当天的部分
        ranges.append(( pd.to_datetime(f"{extended_start.date()} 00:00:00").time().strftime("%H:%M:%S"), extended_end.time().strftime("%H:%M:%S")))  # 下一天的部分
    else:
        # 不跨天，直接添加整个扩展后的时间段
        ranges.append((extended_start.time().strftime("%H:%M:%S"), extended_end.time().strftime("%H:%M:%S")))

    return ranges



def adjust_time_range_and_rule(custom_intervals,custom_rules,half_of_win):
    new_custom_intervals=[]
    new_custom_rules=[]
    for(custom_interval, custom_rule) in zip(custom_intervals, custom_rules):
        start_time, end_time=custom_interval[0],custom_interval[1]
        new_custom_interval=expand_time_range_bidirectional(start_time, end_time, custom_rule, half_of_win, half_of_win)
        new_custom_rule=[custom_rule for i in range(len(new_custom_interval))]
        new_custom_intervals.extend(new_custom_interval)
        new_custom_rules.extend(new_custom_rule)
    return new_custom_intervals,new_custom_rules
    pass






def find_signal_record(all_file_path, voltage_threshold=0, emission_continuous_time_threshold=1,
                       communication_continuous_time_threshold=30):
    '''

    :param all_file_path:包含多个文件路径的列表，这些文件中存储了信号数据。
    :param voltage_threshold:电压阈值，默认为 0。
    :param emission_continuous_time_threshold:发射信号的连续时间阈值，默认为 1。
    :param communication_continuous_time_threshold:通信信号的连续时间阈值，默认为 30。
    :return:
    '''
    # signal_occur_time=None
    signal_occur_process_feature = None
    for file_path in all_file_path:
        signal_feature = find_signal_from_one_file(file_path, voltage_threshold, emission_continuous_time_threshold,
                                                   communication_continuous_time_threshold)
        '''调用函数find_signal_from_one_file，从当前文件路径file_path中提取信号特征。
        这个函数的实现未在代码中给出，但可以推测它会根据传入的阈值参数（voltage_threshold、emission_continuous_time_threshold
        和communication_continuous_time_threshold）从文件中提取符合要求的信号特征，并返回一个包含这些特征的DataFrame。'''
        if (signal_occur_process_feature is None and len(signal_feature) != 0):
            signal_occur_process_feature = signal_feature
        else:
            signal_occur_process_feature = pd.concat([signal_occur_process_feature, signal_feature], axis=0,
                                                     ignore_index=True)
    signal_occur_process_feature['start_time'] = pd.to_datetime(signal_occur_process_feature['start_time'])
    signal_occur_process_feature = signal_occur_process_feature.sort_values(by='start_time')
    return signal_occur_process_feature
    pass



def resample_by_extend_custom_intervals(df, intervals, rules):
    """
    根据自定义的时间段和下采样规则对 DataFrame 进行不同粒度的下采样。

    :param df: 需要下采样的 DataFrame，必须具有时间索引。
    :param intervals: 自定义时间段列表，格式为 [('09:00', '09:59'), ('10:00', '11:59'), ...]。
    :param rules: 对应的下采样规则列表，格式为 ['5T', '10T', ...]。
    :return: 按不同时间段进行下采样后的 DataFrame。
    """

    # 定义自定义的下采样函数
    def custom_resample(df):
        df['start_time']=pd.to_datetime(df['start_time'])
        resampled_dfs = []
        for (start, end), rule in zip(intervals, rules):
            resampled_segment = df.between_time(start, end)

            # resampled_segments = resampled_segment.resample(rule)
            resampled_segments = resampled_segment.resample(rule,origin=start)

            resampled_segments_stats = resampled_segments.agg({
                'start_time': 'count',
                'duration_time': ['min', 'max', 'mean', 'median','sum'],
                'emission_time':['min', 'max', 'mean', 'median','sum'],
                'emission_interval_time_max':['min', 'max', 'mean', 'median'],
                'freq_bandwidth_min': ['min', 'max', 'mean', 'median'],
                'freq_bandwidth_max': ['min', 'max', 'mean', 'median'],
                'freq_bandwidth_mean': ['min', 'max', 'mean', 'median'],
                'freq_bandwidth_median': ['min', 'max', 'mean', 'median'],
                'total_power_max':['min', 'max', 'mean', 'median'],
                'total_power_mean':['min', 'max', 'mean', 'median'],
                'total_power_median':['min', 'max', 'mean', 'median'],
                'total_power_sum':'sum',
                'signal_power_max': ['min', 'max', 'mean', 'median'],
                'signal_power_mean': ['min', 'max', 'mean', 'median'],
                'signal_power_median': ['min', 'max', 'mean', 'median'],
                'signal_power_sum': 'sum',
                'noise_power_max': ['min', 'max', 'mean', 'median'],
                'noise_power_mean': ['min', 'max', 'mean', 'median'],
                'noise_power_median': ['min', 'max', 'mean', 'median'],
                'noise_power_sum': 'sum',
                'signal_first_singular_value': ['min', 'max', 'mean', 'median'],
            })

            def diff_stats(x):
                # 计算日期列的差值
                start_time_diffs = x['start_time'].diff().dt.total_seconds().fillna(0).abs()  # 日期差值的绝对值.dt.total_seconds().fillna(0).dropna()
                # 计算其他数值列（如 'value' 和 'other'）的差值
                duration_time_diffs = x['duration_time'].diff().fillna(0).abs()
                emission_time_diffs = x['emission_time'].diff().fillna(0).abs()
                freq_bandwidth_max_diffs = x['freq_bandwidth_max'].diff().fillna(0).abs()
                signal_power_max_diffs = x['signal_power_max'].diff().fillna(0).abs()
                return pd.Series({
                    'start_time_diffs_max': start_time_diffs.max(),
                    'start_time_diffs_min': start_time_diffs.min(),
                    'duration_time_diffs_max': duration_time_diffs.max(),
                    'duration_time_diffs_min': duration_time_diffs.min(),
                    'emission_time_diffs_max': emission_time_diffs.max(),
                    'emission_time_diffs_min': emission_time_diffs.min(),
                    'freq_bandwidth_max_diffs_max':freq_bandwidth_max_diffs.max(),
                    'freq_bandwidth_max_diffs_min': freq_bandwidth_max_diffs.min(),
                    'signal_power_max_diffs_max': signal_power_max_diffs.max(),
                    'signal_power_max_diffs_min': signal_power_max_diffs.min(),

                })

            resampled_segments_dif=resampled_segments.apply(diff_stats)
            resampled_segments_stats.columns = ['_'.join(col) for col in resampled_segments_stats.columns]
            resampled_segments_stats= resampled_segments_stats.rename(columns={
                'start_time_count': 'communication_num',
                'emission_num_sum': 'emission_num_sum',
                'duration_time_sum': 'duration_time_sum',
                'duration_time_max': 'duration_time_max',
            }).fillna(0)


            # 定义计算余弦相似性的函数
            def compute_cosine_similarities(group):
                singular_vectors=group['signal_first_right_singular_vector'].values
                similarities = []
                # 计算每组内所有可能的两两元素之间的余弦相似性
                for i in range(len(singular_vectors)):
                    for j in range(i + 1, len(singular_vectors)):
                        if isinstance(singular_vectors[i], str):
                            a=np.array(ast.literal_eval(singular_vectors[i].strip()))
                        else :
                            a=singular_vectors[i]
                        if isinstance(singular_vectors[i], str):
                            b=np.array(ast.literal_eval(singular_vectors[j].strip()))
                        else :
                            b=singular_vectors[j]
                        # b=np.array(ast.literal_eval(singular_vectors[j].strip()))
                        # a=np.array(ast.literal_eval(group[i]))
                        sim = abs(1 - cosine(a, b)) # 余弦相似性 = 1 - cosine距离,np.array(ast.literal_eval(group[i]))
                        similarities.append(sim)
                # 返回最大值和最小值
                if similarities:  # 如果列表非空
                    return pd.Series({
                        'max_cosine_similarity': max(similarities),
                        'min_cosine_similarity': min(similarities)
                    })
                else:
                    return pd.Series({
                        'max_cosine_similarity': np.nan,
                        'min_cosine_similarity': np.nan
                    })


            # 按照 'group' 列分组，并计算每组内的余弦相似性最大最小值
            # resampled_segments_cosine = resampled_segments['signal_first_right_singular_vector'].apply(compute_cosine_similarities)
            resampled_segments_cosine = resampled_segments.apply(compute_cosine_similarities)

            resampled_segments_df=pd.concat([resampled_segments_stats,resampled_segments_dif,resampled_segments_cosine],axis=1).fillna(0)

            if resampled_segments_df.empty:
                columns=['communication_num', 'duration_time_min', 'duration_time_max',
                                               'duration_time_mean', 'duration_time_median', 'duration_time_sum',
                                               'emission_time_min', 'emission_time_max', 'emission_time_mean',
                                               'emission_time_median', 'emission_time_sum',
                                               'emission_interval_time_max_min', 'emission_interval_time_max_max',
                                               'emission_interval_time_max_mean', 'emission_interval_time_max_median',
                                               'freq_bandwidth_min_min', 'freq_bandwidth_min_max',
                                               'freq_bandwidth_min_mean', 'freq_bandwidth_min_median',
                                               'freq_bandwidth_max_min', 'freq_bandwidth_max_max',
                                               'freq_bandwidth_max_mean', 'freq_bandwidth_max_median',
                                               'freq_bandwidth_mean_min', 'freq_bandwidth_mean_max',
                                               'freq_bandwidth_mean_mean', 'freq_bandwidth_mean_median',
                                               'freq_bandwidth_median_min', 'freq_bandwidth_median_max',
                                               'freq_bandwidth_median_mean', 'freq_bandwidth_median_median',
                                               'total_power_max_min', 'total_power_max_max', 'total_power_max_mean',
                                               'total_power_max_median', 'total_power_mean_min', 'total_power_mean_max',
                                               'total_power_mean_mean', 'total_power_mean_median',
                                               'total_power_median_min', 'total_power_median_max',
                                               'total_power_median_mean', 'total_power_median_median',
                                               'total_power_sum_sum', 'signal_power_max_min', 'signal_power_max_max',
                                               'signal_power_max_mean', 'signal_power_max_median',
                                               'signal_power_mean_min', 'signal_power_mean_max',
                                               'signal_power_mean_mean', 'signal_power_mean_median',
                                               'signal_power_median_min', 'signal_power_median_max',
                                               'signal_power_median_mean', 'signal_power_median_median',
                                               'signal_power_sum_sum', 'noise_power_max_min', 'noise_power_max_max',
                                               'noise_power_max_mean', 'noise_power_max_median', 'noise_power_mean_min',
                                               'noise_power_mean_max', 'noise_power_mean_mean',
                                               'noise_power_mean_median', 'noise_power_median_min',
                                               'noise_power_median_max', 'noise_power_median_mean',
                                               'noise_power_median_median', 'noise_power_sum_sum',
                                               'signal_first_singular_value_min', 'signal_first_singular_value_max',
                                               'signal_first_singular_value_mean', 'signal_first_singular_value_median',
                                               'start_time_diffs_max', 'start_time_diffs_min',
                                               'duration_time_diffs_max', 'duration_time_diffs_min',
                                               'emission_time_diffs_max', 'emission_time_diffs_min',
                                               'freq_bandwidth_max_diffs_max', 'freq_bandwidth_max_diffs_min',
                                               'signal_power_max_diffs_max', 'signal_power_max_diffs_min',
                                               'max_cosine_similarity', 'min_cosine_similarity']
                resampled_segments_df=pd.DataFrame(columns=columns)


            # 补充缺失时间点
            group_date = df.name
            start_time=f"{group_date} {start}"
            end_time=f"{group_date} {end}"
            full_index = pd.date_range(start=start_time, end=end_time, freq=rule)  # 生成完整时间索引
            resampled_segments_df= resampled_segments_df.reindex(full_index).fillna(0)  # 重建索引，填充缺失点
            resampled_segments_df['rule'] = rule
            resampled_segments_df.index.name = 'mean_time'
            # if not resampled_segments_df.empty:
            #     resampled_segments_df= resampled_segments_df.reindex(full_index).fillna(0)  # 重建索引，填充缺失点
            #     resampled_segments_df['rule'] = rule
            #     resampled_segments_df.index.name = 'mean_time'
            resampled_dfs.append(resampled_segments_df)
        resampled_dfs = pd.concat(resampled_dfs)
        return resampled_dfs


    all_resampled_dfs = df.groupby(df.index.date, group_keys=False).apply(custom_resample)



    # resampled_dfs = []
    #
    # for (start, end), rule in zip(intervals, rules):
    #     resampled_segment=df.between_time(start, end)
    #     resampled_segment =  resampled_segment.resample(rule).agg({
    #         'start_time': 'count',
    #         'emission_num': 'sum',
    #         'duration_time': ['sum', 'max'],
    #         'power_max': 'max',
    #         'emission_interval_time_max': 'max',
    #         'emission_time': 'sum',
    #         'freq_bandwidth_min': 'min',
    #         'freq_bandwidth_max': 'max'
    #
    #     })
    #     resampled_segment.columns = ['_'.join(col) for col in resampled_segment.columns]
    #     resampled_segment = resampled_segment.rename(columns={
    #         'start_time_count': 'communication_num',
    #         'emission_num_sum': 'emission_num_sum',
    #         'duration_time_sum': 'duration_time_sum',
    #         'duration_time_max': 'duration_time_max',
    #         'power_max_max': 'power_max',
    #         'emission_interval_time_max_max': 'emission_interval_time_max',
    #         'emission_time_sum': 'emission_time_sum',
    #         'freq_bandwidth_min_min': 'freq_bandwidth_min',
    #         'freq_bandwidth_max_max': 'freq_bandwidth_max'
    #     }).fillna(0)
    #
    #     # 根据时间段进行切片
    #     resampled_segment = resampled_segment.between_time(start, end)
    #     resampled_segment['rule'] = rule
    #
    #     resampled_dfs.append(resampled_segment)

    # # 合并所有时间段的下采样结果
    # resampled_dfs = pd.concat(resampled_dfs).sort_index()
    #
    all_resampled_dfs = all_resampled_dfs.reset_index()
    all_resampled_dfs = all_resampled_dfs.rename(columns={
        'mean_time': 'date', })
    return all_resampled_dfs


def process_train_coarse_data(root_path, left_win_size, right_win_size, dataset_parameter):
    '''生成训练用的粗粒度特征数据。'''
    print("生成训练用的粗粒度特征数据")
    train_raw_data_dir = dataset_parameter["train_raw_data_dir"]
    train_signal_record_and_feature_file = dataset_parameter["train_signal_record_and_feature_file"]
    train_coarse_grained_data_file = dataset_parameter["train_coarse_grained_data_file"]
    voltage_threshold = dataset_parameter["voltage_threshold"]#电压阈值
    communication_continuous_time_threshold = dataset_parameter["communication_continuous_time_threshold"]
    emission_continuous_time_threshold = dataset_parameter["emission_continuous_time_threshold"]

    csv_file_dir = os.path.join(root_path, train_raw_data_dir)
    all_file_path = get_all_file_list(csv_file_dir)

    if (os.path.exists(os.path.join(root_path, train_signal_record_and_feature_file))):

        train_signal_record_and_feature_file_path = os.path.join(root_path, train_signal_record_and_feature_file)
        print(str(train_signal_record_and_feature_file_path)+"存在，准备读取...")
        signal_record_and_feature_df = pd.read_csv(train_signal_record_and_feature_file_path)
    else:
        print(str(os.path.join(root_path, train_signal_record_and_feature_file))+"不存在，准备创建!")
        signal_record_and_feature_df = find_signal_record(all_file_path, voltage_threshold=voltage_threshold,
                                                          emission_continuous_time_threshold=emission_continuous_time_threshold,
                                                          communication_continuous_time_threshold=communication_continuous_time_threshold)
        os.makedirs(os.path.dirname(os.path.join(root_path, train_signal_record_and_feature_file)), exist_ok=True)
        signal_record_and_feature_df.to_csv(os.path.join(root_path, train_signal_record_and_feature_file), index=False)
        print("signal.py--process train coarse data--signal_record_and_feature_df:"+str(os.path.join(root_path, train_signal_record_and_feature_file)))
    custom_intervals = dataset_parameter["custom_intervals"]
    custom_rules = dataset_parameter["custom_rules"]
    # custom_intervals = [extend_time_range(custom_interval, custom_rule, max(left_win_size, right_win_size) // 2) for
    #                     (custom_interval, custom_rule) in zip(custom_intervals, custom_rules)]

    custom_intervals, custom_rules = adjust_time_range_and_rule(custom_intervals, custom_rules,
                                                                max(left_win_size, right_win_size) // 2)
    signal_record_and_feature_df['mean_time'] = pd.to_datetime(signal_record_and_feature_df['mean_time'])
    signal_record_and_feature_df.set_index('mean_time', inplace=True)
    coarse_grained_data_df = resample_by_extend_custom_intervals(signal_record_and_feature_df, custom_intervals,
                                                                 custom_rules)
    os.makedirs(os.path.dirname(os.path.join(root_path, train_coarse_grained_data_file)), exist_ok=True)
    coarse_grained_data_df.to_csv(os.path.join(root_path, train_coarse_grained_data_file), index=False)
    print("signal.py--process train coarse data--coarse_grained_data_df:"+str(os.path.join(root_path, train_coarse_grained_data_file)))
    pass


def get_fine_grained_data(raw_data_dir_path, signal_happen_time_df):
    all_file_path = get_all_file_list(raw_data_dir_path)
    # signal_record_and_feature_df=pd.read_csv(train_signal_record_and_feature_file)
    signal_happen_time_df['start_time'] = pd.to_datetime(signal_happen_time_df['start_time'])
    signal_happen_time_df['end_time'] = pd.to_datetime(signal_happen_time_df['end_time'])
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
    signal_df = signal_df.sort_values(by='date')
    return signal_df
    pass


def generate_signal_abnormal_data(root_path, dataset_parameter):
    # 在原始数据上添加异常成分，并记录异常标签
    print("在原始数据上添加异常成分，并记录异常标签")
    print("dataset_parameter的参数:"+str(dataset_parameter))
    # 先生成最简单的定频干扰的异常类型

    # 定义训练数据、测试数据、异常标签等文件的路径
    train_raw_data_dir_path = os.path.join(root_path, dataset_parameter["train_raw_data_dir"])
    test_raw_data_dir_path = os.path.join(root_path, dataset_parameter["test_raw_data_dir"])
    train_signal_record_and_feature_file_path = os.path.join(root_path,
                                                             dataset_parameter["train_signal_record_and_feature_file"])

    test_abnormal_data_dir_path = os.path.join(root_path, dataset_parameter["test_abnormal_data_dir"])

    abnormal_process_file_path = os.path.join(root_path, dataset_parameter["abnormal_process_file"])
    # abnormal_label_file_path = os.path.join(root_path, dataset_parameter["abnormal_label_file"])

    # 2. 加载正常信号的特征数据
    # 从训练数据中读取信号的起止时间和特征（如持续时间、功率等）
    print("从训练数据中读取信号的起止时间和特征:"+str(train_signal_record_and_feature_file_path))
    signal_record_and_feature_df = pd.read_csv(train_signal_record_and_feature_file_path)
    signal_happen_time_df = signal_record_and_feature_df[['start_time', 'end_time']]
    print("signal_happen_time_df:")
    print(signal_happen_time_df)

    # 3. 提取正常信号的细粒度数据（如频谱细节）
    # 根据信号起止时间，从原始数据中提取对应的频谱数据
    train_fine_grained_data_df = get_fine_grained_data(train_raw_data_dir_path, signal_happen_time_df)

    # 4. 计算正常信号的特征阈值
    # 通过分位数定义正常信号的持续时间、功率、带宽、噪声等范围
    normal_time_values = signal_record_and_feature_df['duration_time'].quantile(dataset_parameter["normal_value_list"])
    normal_time_values = tuple(normal_time_values)
    normal_power_values, normal_bandwidth_values, normal_noise_values= signal_normal_feature_value(train_fine_grained_data_df,
                                                                                  dataset_parameter)
    print("正常信号的normal_power_values:"+str(normal_power_values))
    print("正常信号的normal_bandwidth_values:" + str(normal_bandwidth_values))
    print("正常信号的normal_noise_values:" + str(normal_noise_values))

    # 5. 处理测试数据
    # 获取测试数据的所有文件路径，并检测信号记录
    all_file_path = get_all_file_list(test_raw_data_dir_path)
    test_signal_record_and_feature_df = find_signal_record(all_file_path,
                                                           voltage_threshold=dataset_parameter["voltage_threshold"],
                                                           emission_continuous_time_threshold=dataset_parameter[
                                                               "emission_continuous_time_threshold"],
                                                           communication_continuous_time_threshold=dataset_parameter[
                                                               "communication_continuous_time_threshold"])

    # 6. 确定时间范围
    # 获取测试数据的起止时间，并调整起始时间（例如从第二天开始）
    start_date, end_date = get_earlist_and_lastest_time(test_raw_data_dir_path)
    # 转换为 datetime 对象
    datetime_obj = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    #datetime_obj = datetime.strptime(start_date, "%Y/%m/%d %H:%M:%S")
    # 增加一天
    new_datetime_obj = datetime_obj + timedelta(days=1)
    # 转回字符串
    new_datetime_obj= new_datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
    if(new_datetime_obj<end_date):
        start_date=new_datetime_obj

    # 7. 获取频谱列名（频率点）
    # 例如：["7.01", "7.02", ..., "7.50"]
    pd.read_csv(all_file_path[0])
    df_cols = list(pd.read_csv(all_file_path[0]).columns)
    df_cols.remove("date")
    df_cols = sorted(df_cols, key=lambda s: float(s))

    # 8. 生成传输异常
    # 模拟信号传输过程中的异常（如功率超标、带宽突变）
    print("准备生成传输异常..")
    abnormal_num = dataset_parameter['abnormal_num']

    abnormal_transmission_df = generate_transmission_abnormity(abnormal_num, test_signal_record_and_feature_df,
                                                               (start_date, end_date), df_cols, normal_time_values,
                                                               normal_power_values, normal_bandwidth_values,normal_noise_values,dataset_parameter)
    print("abnormal_transmission_df前几行:")
    pd.set_option('display.max_columns', None)
    print(abnormal_transmission_df.head(3))
    # abnormal_df = abnormal_transmission_df.head(1)
    abnormal_df = abnormal_transmission_df

    # 9. 生成电磁干扰（EMI）异常
    # 模拟外部干扰（如插入随机频率的高能量信号）
    print("准备生成电磁干扰异常..")
    abnormal_emi_df = generate_emi_abnormity(abnormal_num, test_signal_record_and_feature_df, (start_date, end_date),
                                             df_cols, normal_time_values, normal_power_values, normal_bandwidth_values,normal_noise_values,dataset_parameter)
    print("abnormal_emi_df前几行:")
    pd.set_option('display.max_columns', None)
    print(abnormal_emi_df.head(3))
    # 10. 合并异常数据
    abnormal_df = pd.concat([abnormal_df, abnormal_emi_df], ignore_index=True)


    # 细粒度数据的异常标签
    # 11. 生成细粒度异常标签
    # 记录异常发生的时间段（用于后续评估）
    abnormal_df = abnormal_df.sort_values(by='start_time')
    fine_abnormal_label_df = abnormal_df[['abnormal_label_start_time','abnormal_label_end_time']]
    fine_abnormal_label_df = fine_abnormal_label_df .rename(columns={
        'abnormal_label_start_time': 'start_time',
        'abnormal_label_end_time':'end_time'
    })


    os.makedirs(os.path.dirname(os.path.join(root_path, dataset_parameter["fine_abnormal_label_file"])), exist_ok=True)

    fine_abnormal_label_df.to_csv(os.path.join(root_path, dataset_parameter["fine_abnormal_label_file"]), index=False)
    print("fine abnormal label df:"+os.path.join(root_path, dataset_parameter["fine_abnormal_label_file"]))

    # 12. 生成行为异常
    # 模拟信号行为的异常（如非预期的时间聚集）
    abnormal_behavior_df = generate_behavior_abnormity_new(abnormal_num,test_signal_record_and_feature_df,(start_date,end_date),df_cols,dataset_parameter)

    # 13. 合并所有异常数据
    print("合并所有异常数据.")
    abnormal_df = pd.concat([abnormal_df, abnormal_behavior_df], ignore_index=True)
    abnormal_df = abnormal_df.sort_values(by='start_time')

    # 14. 保存异常处理记录
    os.makedirs(os.path.dirname(abnormal_process_file_path), exist_ok=True)

    abnormal_df.to_csv(abnormal_process_file_path, index=False)
    print("在原始数据上添加异常成分,并记录异常标签,保存abnormal_df:"+str(abnormal_process_file_path))
    #abnormal_df是包含异常数据的dataframe(到底是怎么来的)

    # 15. 生成最终异常标签
    final_abnormal_label_df = abnormal_df[['abnormal_label_start_time','abnormal_label_end_time']]
    final_abnormal_label_df = final_abnormal_label_df .rename(columns={
        'abnormal_label_start_time': 'start_time',
        'abnormal_label_end_time':'end_time'
    })
    final_abnormal_label_df.to_csv(os.path.join(root_path, dataset_parameter["final_abnormal_label_file"]), index=False)

    # 16. 将异常写入测试数据
    # 将生成的异常数据插入原始测试数据中，生成含异常的测试集
    write_abnormal_to_csv(abnormal_df, test_raw_data_dir_path, test_abnormal_data_dir_path,dataset_parameter)


    pass


def produce_coarse_grained_data_label(coarse_grained_data_df, abnormal_label_df):
    abnormal_label_df=abnormal_label_df.dropna()
    coarse_grained_data_df['start_window']=coarse_grained_data_df['date']
    coarse_grained_data_df['end_window'] = coarse_grained_data_df['date'] + pd.to_timedelta(coarse_grained_data_df['rule'])
    abnormal_label_df['start_time']=pd.to_datetime(abnormal_label_df['start_time'])
    abnormal_label_df['end_time']=pd.to_datetime(abnormal_label_df['end_time'])

    # 定义一个函数来检查区间是否重叠
    def check_overlap(row):
        # 检查 dataframe1 中是否有时间区间与 dataframe2 的时间窗口重叠
        overlap = abnormal_label_df.apply(
            lambda x: not (row['end_window'] < x['start_time'] or row['start_window'] > x['end_time']), axis=1)
        # 如果存在重叠则返回 True，否则返回 False
        return overlap.any()

    # 应用函数给 dataframe2 增加一个新列 'overlap'
    coarse_grained_data_df['overlap'] = coarse_grained_data_df.apply(check_overlap, axis=1)

    coarse_grained_data_df = coarse_grained_data_df.drop(['start_window', 'end_window'], axis=1)
    coarse_grained_data_df = coarse_grained_data_df.rename(columns={
        'overlap': 'label'})

    return coarse_grained_data_df
    pass


def process_test_coarse_data(root_path, left_win_size, right_win_size, dataset_parameter):
    '''生成测试用的粗粒度特征数据。'''
    print("生成测试用的粗粒度特征数据。")
    # 处理测试输入数据和输入标签
    test_raw_data_dir = dataset_parameter["test_raw_data_dir"]
    # test_signal_record_and_feature_file = dataset_parameter["test_signal_record_and_feature_file"]
    test_abnormal_signal_record_and_feature_file = dataset_parameter["test_abnormal_signal_record_and_feature_file"]
    test_coarse_grained_data_file = dataset_parameter["test_coarse_grained_data_file"]
    voltage_threshold = dataset_parameter["voltage_threshold"]
    communication_continuous_time_threshold = dataset_parameter["communication_continuous_time_threshold"]
    emission_continuous_time_threshold = dataset_parameter["emission_continuous_time_threshold"]
    final_abnormal_label_file = dataset_parameter["final_abnormal_label_file"]
    test_abnormal_data_dir = dataset_parameter["test_abnormal_data_dir"]
    dataset_parameter["left_win_size"]=left_win_size
    dataset_parameter["right_win_size"]=right_win_size




    if (os.path.exists(os.path.join(root_path, test_abnormal_signal_record_and_feature_file)) and os.path.exists(os.path.join(root_path, final_abnormal_label_file))):
        test_signal_record_and_feature_file_path = os.path.join(root_path, test_abnormal_signal_record_and_feature_file)
        signal_record_and_feature_df = pd.read_csv(test_signal_record_and_feature_file_path)
    else:

        # abnormal_process_file = dataset_parameter["abnormal_process_file"]


        if (not os.path.exists(os.path.join(root_path, final_abnormal_label_file))) or (not os.path.exists(os.path.join(root_path, test_abnormal_data_dir))):
            generate_signal_abnormal_data(root_path, dataset_parameter)



        csv_file_dir = os.path.join(root_path, test_abnormal_data_dir)
        all_file_path = get_all_file_list(csv_file_dir)

        signal_record_and_feature_df = find_signal_record(all_file_path, voltage_threshold=voltage_threshold,
                                                          emission_continuous_time_threshold=emission_continuous_time_threshold,
                                                          communication_continuous_time_threshold=communication_continuous_time_threshold)

        os.makedirs(os.path.dirname(os.path.join(root_path, test_abnormal_signal_record_and_feature_file)), exist_ok=True)
        signal_record_and_feature_df.to_csv(os.path.join(root_path, test_abnormal_signal_record_and_feature_file), index=False)

    final_abnormal_label_df = pd.read_csv(os.path.join(root_path, final_abnormal_label_file))

    custom_intervals = dataset_parameter["custom_intervals"]
    custom_rules = dataset_parameter["custom_rules"]
    # custom_intervals = [extend_time_range(custom_interval, custom_rule, max(left_win_size, right_win_size) // 2) for
    #                     (custom_interval, custom_rule) in zip(custom_intervals, custom_rules)]

    custom_intervals,custom_rules=adjust_time_range_and_rule(custom_intervals,custom_rules,max(left_win_size, right_win_size) // 2)

    signal_record_and_feature_df['mean_time'] = pd.to_datetime(signal_record_and_feature_df['mean_time'])
    signal_record_and_feature_df.set_index('mean_time', inplace=True)

    coarse_grained_data_df = resample_by_extend_custom_intervals(signal_record_and_feature_df, custom_intervals,
                                                                 custom_rules)

    coarse_grained_data_df=produce_coarse_grained_data_label(coarse_grained_data_df,final_abnormal_label_df)


    os.makedirs(os.path.dirname(os.path.join(root_path, test_coarse_grained_data_file)), exist_ok=True)
    coarse_grained_data_df.to_csv(os.path.join(root_path, test_coarse_grained_data_file), index=False)

    pass


def read_coarse_grained_data(root_path, left_win_size, right_win_size, flag="train"):
    '''
    读取粗粒度数据
    :param root_path:
    :param left_win_size:
    :param right_win_size:
    :param flag:
    :return:
    '''
    print("读取粗粒度数据")
    dataset_parameter = get_dataset_parameter(root_path)
    train_coarse_grained_data_file = dataset_parameter["train_coarse_grained_data_file"]
    test_coarse_grained_data_file = dataset_parameter["test_coarse_grained_data_file"]
    coarse_columns=dataset_parameter["coarse_columns"]
    if (flag == "train"):
        if not os.path.exists(os.path.join(root_path, train_coarse_grained_data_file)):
            process_train_coarse_data(root_path, left_win_size, right_win_size, dataset_parameter)
        train_raw_df = pd.read_csv(os.path.join(root_path, train_coarse_grained_data_file))
        train_raw_df=train_raw_df[coarse_columns]
        return train_raw_df,dataset_parameter
    elif (flag == "test"):
        if not os.path.exists(os.path.join(root_path, train_coarse_grained_data_file)):
            process_train_coarse_data(root_path, left_win_size, right_win_size, dataset_parameter)
        train_raw_df = pd.read_csv(os.path.join(root_path, train_coarse_grained_data_file))
        train_raw_df = train_raw_df[coarse_columns]
        return train_raw_df,dataset_parameter

    elif (flag == "test_forecast_and_anomaly_detection"):
        if not os.path.exists(os.path.join(root_path, test_coarse_grained_data_file)):
            process_test_coarse_data(root_path, left_win_size, right_win_size, dataset_parameter)
        test_raw_df = pd.read_csv(os.path.join(root_path, test_coarse_grained_data_file))
        test_raw_df = test_raw_df [coarse_columns+['label']]
        return test_raw_df,dataset_parameter

        # else:
        #     if not os.path.exists(os.path.join(root_path, train_data_file)):
        #         process_coarse_train_data(root_path, left_win_size,right_win_size , voltage_threshold=0, continuous_time_threshold=15)
        #     train_raw_df=pd.read_csv(os.path.join(root_path, train_data_file))
        #     return train_raw_df
        pass



def read_signal_coarse_continuous_data(df,win_size,dataset_parameter,step=1):
    data=[]
    date_data=[]
    rules_grouped = df.groupby('rule')
    group_count_of_rules = rules_grouped.ngroups
    for rule, rule_group in rules_grouped:
        rule_group = rule_group.drop(columns='rule')
        # rule_group.set_index('date',inplace=True)
        # rule_group=rule_group.sort_index()
        rule_group = rule_group.sort_values(by='date')

        time_delta = pd.to_timedelta(rule)
        # 获取总秒数
        seconds = time_delta.total_seconds()
        # rule_group = rule_group.reset_index(drop=False)
        rule_group['date'] = pd.to_datetime(rule_group['date'])

        if(group_count_of_rules<=1):
            segment_grouped= rule_group.groupby(rule_group['date'].dt.date)
            segment_groupes = [group.reset_index(drop=True) for group_name, group in
                           segment_grouped]
        else:
            # 计算日期之间的差值，并找出不连续的点
            rule_group['date_diff'] = rule_group['date'].diff().dt.total_seconds().fillna(0)
            rule_group['segment_group'] = (rule_group['date_diff'] > seconds).cumsum()
            rule_group = rule_group.drop(columns='date_diff')
            segment_grouped = rule_group.groupby('segment_group')
            segment_groupes = [group.drop(columns='segment_group').reset_index(drop=True) for group_name, group in
                           segment_grouped]

        def groups_methods(groups, method="next"):
            if (method == "next"):
                # groups = list(grouped)
                return zip(groups, groups[1:])
            else:
                # 将每个组存储为一个列表（group_name, group_df）
                # groups = [(group_name, group_df) for group_name, group_df in grouped]
                # 使用 itertools.combinations 对分组进行两两组合
                combinations = itertools.combinations(groups, 2)
                return combinations

        input_and_output_combinations = groups_methods(segment_groupes, method="next")
        for (group_df1, group_df2) in input_and_output_combinations:

            group_df1=group_df1.drop_duplicates().sort_values(by='date')
            group_df2=group_df2.drop_duplicates().sort_values(by='date')
            # 提取时分秒
            group_df1['time'] = group_df1['date'].dt.strftime('%H:%M:%S')
            group_df2['time'] = group_df2['date'].dt.strftime('%H:%M:%S')


            # 找出时间上的交集
            intersection_times = pd.merge(group_df1[['time']], group_df2[['time']], on='time', how='inner')
            # 根据交集时间提取两个DataFrame的对应部分
            df1_intersection = group_df1[group_df1['time'].isin(intersection_times['time'])]
            df2_intersection = group_df2[group_df2['time'].isin(intersection_times['time'])]

            cols = list(df1_intersection.columns)
            cols.remove('date')
            cols.remove('time')

            # group_data = group[cols]
            group1_value_rows = df1_intersection[cols].values
            group2_value_rows = df2_intersection[cols].values

            # group1_date_rows = df1_intersection[['date']].astype(str).values
            # group2_date_rows = df2_intersection[['date']].astype(str).values

            # group1_date_rows = df1_intersection[['date']].values
            # group2_date_rows = df2_intersection[['date']].values

            group1_date_rows = df1_intersection['date'].apply(lambda x: x.timestamp()).values.reshape(-1,1)
            group2_date_rows = df2_intersection['date'].apply(lambda x: x.timestamp()).values.reshape(-1,1)

            for i in range((len(group1_value_rows) - win_size) // step + 1):
                data.append((group1_value_rows[i:i + win_size],group2_value_rows[i:i + win_size]))
                date_data.append((group1_date_rows[i:i + win_size],group2_date_rows[i:i + win_size]))


    return data,date_data
    pass


def get_fine_addit_data(df, signal_record_and_feature_df,dataset_parameter):
    signal_record_and_feature_df['start_time'] = pd.to_datetime(signal_record_and_feature_df['start_time'])
    signal_record_and_feature_df['end_time'] = pd.to_datetime(signal_record_and_feature_df['end_time'])
    fine_addit_keys=dataset_parameter['fine_addit_keys']
    if ('emission_time_order' in fine_addit_keys):
        fine_addit_keys.remove('emission_time_order')
        emission_time_order_flag=True
    else:
        emission_time_order_flag = False
    for index, row in signal_record_and_feature_df.iterrows():
        select_start_date, select_end_date = row['start_time'], row['end_time']
        if (select_end_date < df['date'].iloc[0]):
            continue
        if (select_start_date > df['date'].iloc[-1]):
            break
        # mask1 = (df['date'] >= pd.to_datetime(select_start_date)) & (
        #         df['date'] <= pd.to_datetime(select_end_date))

        # 为信号辐射过错排序
        if(emission_time_order_flag):
            count=((df['date'] >= pd.to_datetime(select_start_date)) & (
                df['date'] <= pd.to_datetime(select_end_date))).sum()
            count_order=np.arange(1,count+1)
            df.loc[(df['date'] >= pd.to_datetime(select_start_date)) & (
                    df['date'] <= pd.to_datetime(select_end_date)), ['emission_time_order']] = count_order


        for key in fine_addit_keys:
            df.loc[(df['date'] >= pd.to_datetime(select_start_date)) & (
                df['date'] <= pd.to_datetime(select_end_date)),key]=row[key]


    df=df.fillna(0)
    return df
    pass


def add_noise_to_spectrum(df, root_path,dataset_parameter):
    train_raw_data_dir_path = os.path.join(root_path, dataset_parameter["train_raw_data_dir"])
    train_signal_record_and_feature_file_path = os.path.join(root_path,
                                                             dataset_parameter["train_signal_record_and_feature_file"])
    signal_record_and_feature_df = pd.read_csv(train_signal_record_and_feature_file_path)
    signal_happen_time_df = signal_record_and_feature_df[['start_time', 'end_time']]
    train_fine_grained_data_df = get_fine_grained_data(train_raw_data_dir_path, signal_happen_time_df)

    normal_time_values = signal_record_and_feature_df['duration_time'].quantile(dataset_parameter["normal_value_list"])
    normal_time_values = tuple(normal_time_values)
    normal_power_values, normal_bandwidth_values, normal_noise_values= signal_normal_feature_value(train_fine_grained_data_df,
                                                                                  dataset_parameter)
    old_snr=normal_power_values[-1]-normal_noise_values[-1]
    new_snr=dataset_parameter["SNR"]
    change_noise_dbuv = normal_power_values[-1] -new_snr-normal_noise_values[-1]
    df.iloc[:, 1:] = df.iloc[:, 1:].where(df.iloc[:, 1:] > dataset_parameter["voltage_threshold"], df.iloc[:, 1:] + change_noise_dbuv)
    print(f"原始信噪比为：{old_snr:.2f}")
    print(f"新的信噪比为：{new_snr:.2f}")
    print(f"噪声电压变化均值为：{change_noise_dbuv:.2f}")
    pass


def process_train_fine_data(root_path, left_win_size, right_win_size, dataset_parameter):
    train_raw_data_dir = dataset_parameter["train_raw_data_dir"]
    train_signal_record_and_feature_file = dataset_parameter["train_signal_record_and_feature_file"]
    train_fine_grained_data_file = dataset_parameter["train_fine_grained_data_file"]
    voltage_threshold = dataset_parameter["voltage_threshold"]
    communication_continuous_time_threshold = dataset_parameter["communication_continuous_time_threshold"]
    emission_continuous_time_threshold = dataset_parameter["emission_continuous_time_threshold"]
    csv_file_dir = os.path.join(root_path, train_raw_data_dir)
    all_file_path = get_all_file_list(csv_file_dir)
    if (os.path.exists(os.path.join(root_path, train_signal_record_and_feature_file))):
        train_signal_record_and_feature_file_path = os.path.join(root_path, train_signal_record_and_feature_file)
        signal_record_and_feature_df = pd.read_csv(train_signal_record_and_feature_file_path)
    else:
        signal_record_and_feature_df = find_signal_record(all_file_path, voltage_threshold=voltage_threshold,
                                                          emission_continuous_time_threshold=emission_continuous_time_threshold,
                                                          communication_continuous_time_threshold=communication_continuous_time_threshold)
        os.makedirs(os.path.dirname(os.path.join(root_path, train_signal_record_and_feature_file)), exist_ok=True)
        signal_record_and_feature_df.to_csv(os.path.join(root_path, train_signal_record_and_feature_file), index=False)
    signal_happen_time_df = signal_record_and_feature_df[['start_time', 'end_time']]
    signal_happen_time_df['start_time'] = pd.to_datetime(signal_happen_time_df['start_time']) - pd.to_timedelta(
        left_win_size, unit='s')
    signal_happen_time_df['end_time'] = pd.to_datetime(signal_happen_time_df['end_time']) + pd.to_timedelta(
        right_win_size, unit='s')
    train_raw_data_dir_path = os.path.join(root_path, dataset_parameter["train_raw_data_dir"])
    train_fine_grained_data_df = get_fine_grained_data(train_raw_data_dir_path, signal_happen_time_df)

    if (dataset_parameter["SNR"] is not None and isinstance(dataset_parameter["SNR"], (int, float))):
        add_noise_to_spectrum(train_fine_grained_data_df,root_path, dataset_parameter)

    if(dataset_parameter['fine_addit_information']):
        train_fine_grained_data_df=get_fine_addit_data(train_fine_grained_data_df, signal_record_and_feature_df,dataset_parameter)

    os.makedirs(os.path.dirname(os.path.join(root_path, train_fine_grained_data_file)), exist_ok=True)
    train_fine_grained_data_df.to_csv((os.path.join(root_path, train_fine_grained_data_file)),index=False)
    pass


def process_test_fine_data(root_path, left_win_size, right_win_size, dataset_parameter):
    # 处理测试输入数据和输入标签
    print("处理测试输入数据和输入标签")

    test_raw_data_dir = dataset_parameter["test_raw_data_dir"]
    test_abnormal_signal_record_and_feature_file = dataset_parameter["test_abnormal_signal_record_and_feature_file"]
    voltage_threshold = dataset_parameter["voltage_threshold"]
    communication_continuous_time_threshold = dataset_parameter["communication_continuous_time_threshold"]
    emission_continuous_time_threshold = dataset_parameter["emission_continuous_time_threshold"]
    fine_abnormal_label_file = dataset_parameter["fine_abnormal_label_file"]
    test_abnormal_data_dir = dataset_parameter["test_abnormal_data_dir"]
    test_fine_grained_data_file= dataset_parameter["test_fine_grained_data_file"]
    dataset_parameter["left_win_size"]=left_win_size
    dataset_parameter["right_win_size"]=right_win_size

    if (os.path.exists(os.path.join(root_path, test_abnormal_signal_record_and_feature_file))):
        test_signal_record_and_feature_file_path = os.path.join(root_path, test_abnormal_signal_record_and_feature_file)
        signal_record_and_feature_df = pd.read_csv(test_signal_record_and_feature_file_path)
    else:
        if (not os.path.exists(os.path.join(root_path, fine_abnormal_label_file))) or (not os.path.exists(os.path.join(root_path, test_abnormal_data_dir))):
            generate_signal_abnormal_data(root_path, dataset_parameter)
        csv_file_dir = os.path.join(root_path, test_abnormal_data_dir)
        all_file_path = get_all_file_list(csv_file_dir)

        signal_record_and_feature_df = find_signal_record(all_file_path, voltage_threshold=voltage_threshold,
                                                          emission_continuous_time_threshold=emission_continuous_time_threshold,
                                                          communication_continuous_time_threshold=communication_continuous_time_threshold)

        os.makedirs(os.path.dirname(os.path.join(root_path, test_abnormal_signal_record_and_feature_file)), exist_ok=True)
        signal_record_and_feature_df.to_csv(os.path.join(root_path, test_abnormal_signal_record_and_feature_file), index=False)

    fine_abnormal_label_df=pd.read_csv(os.path.join(root_path, fine_abnormal_label_file))

    signal_happen_time_df = signal_record_and_feature_df[['start_time', 'end_time']]
    signal_happen_time_df= pd.concat([signal_happen_time_df, fine_abnormal_label_df], axis=0, ignore_index=True)
    signal_happen_time_df['start_time'] = pd.to_datetime(signal_happen_time_df['start_time']) - pd.to_timedelta(
        left_win_size, unit='s')
    signal_happen_time_df['end_time'] = pd.to_datetime(signal_happen_time_df['end_time']) + pd.to_timedelta(
        right_win_size, unit='s')
    signal_happen_time_df = signal_happen_time_df.sort_values(by='start_time')

    test_abnormal_data_dir_path = os.path.join(root_path, dataset_parameter["test_abnormal_data_dir"])
    test_abnormal_data_df = get_fine_grained_data(test_abnormal_data_dir_path, signal_happen_time_df)


    if (dataset_parameter["SNR"] is not None and isinstance(dataset_parameter["SNR"], (int, float))):
        add_noise_to_spectrum(test_abnormal_data_df,root_path, dataset_parameter)
    if(dataset_parameter['fine_addit_information']):
        test_abnormal_data_df=get_fine_addit_data(test_abnormal_data_df, signal_record_and_feature_df,dataset_parameter)


    label_mask = None
    fine_abnormal_label_df['start_time'] = pd.to_datetime(fine_abnormal_label_df['start_time'])
    fine_abnormal_label_df['end_time'] = pd.to_datetime(fine_abnormal_label_df['end_time'])

    for index, row in fine_abnormal_label_df.iterrows():
        select_start_date, select_end_date = row['start_time'], row['end_time']
        if (select_end_date < test_abnormal_data_df['date'].iloc[0]):
            continue
        if (select_start_date > test_abnormal_data_df['date'].iloc[-1]):
            break
        mask1 = (test_abnormal_data_df['date'] >= pd.to_datetime(select_start_date)) & (
                test_abnormal_data_df['date'] <= pd.to_datetime(select_end_date))
        if (label_mask is None):
            label_mask = mask1
        else:
            label_mask = label_mask | mask1

    test_abnormal_data_df['label'] = label_mask

    os.makedirs(os.path.dirname(os.path.join(root_path, test_fine_grained_data_file)), exist_ok=True)
    # df_save_path = os.path.join(root_path, test_data_file)
    test_abnormal_data_df.to_csv(os.path.join(root_path, test_fine_grained_data_file), index=False)



    pass


def read_fine_grained_data(root_path, left_win_size, right_win_size, flag="train"):
    '''
    读取细粒度数据
    :param root_path:
    :param left_win_size: 左侧窗口
    :param right_win_size: 右侧窗口
    :param flag: 默认为train
    :return:
    '''
    print("......读取细粒度数据......")
    dataset_parameter = get_dataset_parameter(root_path)
    train_fine_grained_data_file = dataset_parameter["train_fine_grained_data_file"]
    test_fine_grained_data_file = dataset_parameter["test_fine_grained_data_file"]
    if (flag == "train"):
        if not os.path.exists(os.path.join(root_path, train_fine_grained_data_file)):
            print("不存在" + str(os.path.join(root_path, train_fine_grained_data_file)) + ",准备创建")
            process_train_fine_data(root_path, left_win_size, right_win_size, dataset_parameter)
        train_raw_df = pd.read_csv(os.path.join(root_path, train_fine_grained_data_file))

        return train_raw_df,dataset_parameter
    elif (flag == "test"):
        if not os.path.exists(os.path.join(root_path, test_fine_grained_data_file)):
            print("不存在" + str(os.path.join(root_path, test_fine_grained_data_file)) + ",准备创建")
            process_test_fine_data(root_path, left_win_size, right_win_size, dataset_parameter)
        test_raw_df = pd.read_csv(os.path.join(root_path, test_fine_grained_data_file ))
        return test_raw_df ,dataset_parameter
    else:
        if not os.path.exists(os.path.join(root_path, train_fine_grained_data_file)):
            print("不存在"+str(os.path.join(root_path, train_fine_grained_data_file))+",准备创建")
            process_train_fine_data(root_path, left_win_size, right_win_size, dataset_parameter)
        train_raw_df = pd.read_csv(os.path.join(root_path, train_fine_grained_data_file))
        return train_raw_df,dataset_parameter
        pass


def read_signal_fine_continuous_data(df,win_size,fine_time_interval=1,step=1):
    # 将日期列转换为datetime格式
    df['date'] = pd.to_datetime(df['date'])
    # 计算日期之间的差值，并找出不连续的点
    df['date_diff'] = df['date'].diff().dt.total_seconds().fillna(0)
    df['group'] = (df['date_diff'] > fine_time_interval).cumsum()
    # 根据分组键进行分组
    grouped = df.groupby('group')
    # 打印每个分组的结果
    data = []
    date_data=[]
    for name, group in grouped:
        cols= list(group.columns)
        cols.remove('date')
        cols.remove('date_diff')
        cols.remove('group')
        group_data = group[cols]
        group_rows = group_data.values
        # df_stamp =group_data[['date']]

        group_date_rows = group['date'].apply(lambda x: x.timestamp()).values.reshape(-1, 1)

        for i in range((len(group) - win_size) // step + 1):
            data.append(group_rows[i:i + win_size])
            date_data.append(group_date_rows[i:i + win_size])
        # print(f"Group {name}:")
        # print(group)
        # print("\n")
    return data,date_data
    pass

if __name__=="__main__":
    pass