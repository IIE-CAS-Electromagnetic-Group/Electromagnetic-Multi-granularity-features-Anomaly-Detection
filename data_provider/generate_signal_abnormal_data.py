import os

import numpy as np
import random
import pandas as pd
from datetime import datetime, timedelta




def get_all_file_list(csv_file_dir="E:\Work\Workspace\Data\电磁频谱数据\pictures\所里频谱迹线"):
    # 获取该文件夹中所有的bin文件路径
    file_path = os.listdir(csv_file_dir)
    file_list = list(map(lambda x: os.path.join(csv_file_dir, x).replace('\\', '/'), file_path))
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
    '''
    在指定时间范围内生成随机时间段（起止时间）。生成随机的开始时间（在某个范围内，例如2024年1月1日至2024年12月31日）
    :param start_date:起始日期
    :param end_date:结束日期
    :param duration_seconds_low:最小持续时间描述
    :param duration_seconds_high:最大持续时间秒数
    :return:随机生成的起止时间（random_start_time, random_end_time）
    '''
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


def write_abnormal_to_csv(abnormal_value_df, raw_data_dir,test_abnormal_data_dir_path,dataset_parameter):
    '''将异常数据写入原始数据文件，生成包含异常信号的测试数据集。
        参数：
            - abnormal_value_df: 包含异常配置的DataFrame（时间范围、带宽键、功率值等）
            实际上就是raw_data/abnormal_label/abnormal_process.csv这个文件里的内容
            - raw_data_dir: 原始数据文件目录
            - test_abnormal_data_dir_path: 生成的异常测试数据保存目录
            - dataset_parameter: 数据集参数（如电压阈值）
    '''
    print("----执行:将异常数据写入原始数据文件，生成包含异常信号的测试数据集。----")
    all_file_path = get_all_file_list(raw_data_dir)
    for file_path in all_file_path:# 遍历每个原始数据文件
        # 生成保存路径（保留文件名，替换目录）
        # save_file_path=os.path.join(test_abnormal_data_dir_path,file_path.split("\\")[-1])
        save_file_path=os.path.join(test_abnormal_data_dir_path,file_path.split("/")[-1])
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        df = pd.read_csv(file_path)# 读取原始数据文件
        print("    读取原始数据文件："+str(file_path))
        df['date'] = pd.to_datetime(df['date'])# 转换日期列为datetime类型
        cols = list(df.columns)
        cols.remove('date')# 获取非日期列（数据特征列）
        # mask=None
        # 遍历每一条异常配置
        for index, row in abnormal_value_df.iterrows():#Pandas 的 iterrows() 方法用于遍历 DataFrame 的每一行，返回索引和行数据的序列。
            print("index:"+str(index))
            print("row:"+str(row))

            select_start_date, select_end_date = row['start_time'], row['end_time']
            bandwidth_key = row['bandwidth_key']
            power = row['power']

            # 跳过与当前文件时间范围无关的异常配置
            if (select_end_date < df['date'].iloc[0]):
                continue# 异常结束时间早于数据开始时间
            if (select_start_date > df['date'].iloc[-1]):
                break# 异常开始时间晚于数据结束时间

            # 根据异常生成方法处理数据
            if(row['abnormal_generate_method']=='add'):
                # 添加型异常（直接在原数据上增加功率值）
                if(row['abnormal_class']=='power_abnormal'):

                    value_condition = (df.drop(columns=["date"]) > dataset_parameter["voltage_threshold"]).any(axis=1)
                    date_condition = (df['date'] >= pd.to_datetime(select_start_date)) & (
                            df['date'] <= pd.to_datetime(select_end_date))

                    filtered_df = df[(df['date'] >= pd.to_datetime(row['refer_start_time'])) & (
                            df['date'] <= pd.to_datetime(row['refer_end_time']))]
                    columns_with_values_above_threshold = filtered_df.iloc[:, 1:].columns[
                        (filtered_df.iloc[:, 1:] > dataset_parameter["voltage_threshold"]).any()].tolist()
                    select_bandwidth_key_len=len(bandwidth_key)
                    if select_bandwidth_key_len <= 0 or select_bandwidth_key_len> len(columns_with_values_above_threshold):
                        select_bandwidth_key_len=len(columns_with_values_above_threshold)

                        # 随机选择起始索引
                    start_index = random.randint(0, len(columns_with_values_above_threshold) - select_bandwidth_key_len)
                    new_select_bandwidth_key=columns_with_values_above_threshold[start_index:start_index + select_bandwidth_key_len]
                    df.loc[date_condition&value_condition,new_select_bandwidth_key] += power

                else:
                    df.loc[(df['date'] >= pd.to_datetime(select_start_date)) & (
                                df['date'] <= pd.to_datetime(select_end_date)), bandwidth_key] += power
            else:
                if(row['abnormal_class']=='time_abnormal'):# 替换型异常（用参考数据替换目标数据）

                    power_values=df.loc[(df['date'] >= pd.to_datetime(row['refer_start_time'])) & (
                                    df['date'] <= pd.to_datetime(row['refer_end_time'])), bandwidth_key].values
                    print(power_values)
                    # 从数据框 df 中筛选出日期在 row['refer_start_time'] 和 row['refer_end_time'] 之间的数据，
                    # 并提取对应的 bandwidth_key 列的值，存储到 power_values 中

                    row_num= int((pd.to_datetime(row['end_time'])-pd.to_datetime(row['start_time'])).total_seconds() + 1)# 计算目标时间范围内的行数
                    # 计算从 row['start_time'] 到 row['end_time'] 的时间差（单位：秒），并加 1，得到需要替换的行数

                    # 随机从指定范围内的行进行抽取（此处可能触发错误！）
                    new_power_values =power_values[np.random.choice(power_values.shape[0], size=row_num, replace=True)]
                    # 从 power_values 中随机抽取 row_num 个值，允许重复（replace=True），返回指定大小（row_num）的数组生成新的功率值数组 new_power_values

                    # df.loc[(df['date'] >= pd.to_datetime(select_start_date)) & (
                    #         df['date'] <= pd.to_datetime(select_end_date)), bandwidth_key] = power

                    df.loc[(df['date'] >= pd.to_datetime(select_start_date)) & (
                            df['date'] <= pd.to_datetime(select_end_date)), bandwidth_key] = new_power_values

                elif(row['abnormal_class']=='bandwidth_abnormal'):# 带宽异常（直接替换为固定值）

                    value_condition=(df.drop(columns=["date"]) > dataset_parameter["voltage_threshold"]).any(axis=1)
                    date_condition=(df['date'] >= pd.to_datetime(select_start_date)) & (
                            df['date'] <= pd.to_datetime(select_end_date))
                    df.loc[date_condition&value_condition,bandwidth_key] =power
                #     df.loc[(df['date'] >= pd.to_datetime(select_start_date)) & (
                #             df['date'] <= pd.to_datetime(select_end_date)&((df.drop(columns=["date"])> dataset_parameter["voltage_threshold"]).any(axis=1))), bandwidth_key] =power
                # 行为异常处理
                else:
                    def find_match_files(target_start,target_end,file_list):# 行为异常（从其他文件复制数据）
                        # target_start = datetime.strptime(target_start_str, "%Y/%m/%d %H:%M:%S")
                        # target_end = datetime.strptime(target_end_str, "%Y/%m/%d %H:%M:%S")
                        matching_files = []
                        for file_name in file_list:
                            # 提取文件名中的起始和结束时间戳
                            start_str, end_str, *_ = file_name.split('/')[-1].split('_')[:2]
                            # 将时间戳转换为 datetime 对象
                            start_time = datetime.strptime(start_str[:14], "%Y%m%d%H%M%S")
                            end_time = datetime.strptime(end_str[:14], "%Y%m%d%H%M%S")

                            # 判断文件的时间范围是否与目标时间范围重叠
                            if (start_time <= target_end and end_time >= target_start):
                                matching_files.append(file_name)
                        # 输出结果
                        return matching_files
                        pass

                    matching_files=find_match_files(row['refer_start_time'],row['refer_end_time'],all_file_path)
                    refer_df=None
                    for file in matching_files:
                        if(refer_df is None):
                            refer_df=pd.read_csv(file)
                        else:
                            refer_df=pd.concat([refer_df,pd.read_csv(file)],ignore_index=True)

                    if(refer_df is not None):
                        refer_df['date'] = pd.to_datetime(refer_df['date'])
                        refer_df=refer_df.sort_values(by='date')
                        # A=df.loc[(df['date'] >= pd.to_datetime(select_start_date)) & (
                        #         df['date'] <= pd.to_datetime(select_end_date)), bandwidth_key]
                        # B=refer_df.loc[(refer_df['date'] >= pd.to_datetime(row['refer_start_time'])) & (
                        #             refer_df['date'] <= pd.to_datetime(row['refer_end_time'])), bandwidth_key]
                        df.loc[(df['date'] >= pd.to_datetime(select_start_date)) & (
                                df['date'] <= pd.to_datetime(select_end_date)), bandwidth_key] = \
                            refer_df.loc[(refer_df['date'] >= pd.to_datetime(row['refer_start_time'])) & (
                                    refer_df['date'] <= pd.to_datetime(row['refer_end_time'])), bandwidth_key].values
        df.to_csv(save_file_path, index=False)
        print("    保存:将异常数据写入原始数据文件，生成包含异常信号的测试数据集:"+str(save_file_path))

    #
    #
    #
    # all_abnormal_file_path = get_all_file_list(test_abnormal_data_dir_path)
    # for abnormal_file_path in all_abnormal_file_path:
    #     abnormal_df = pd.read_csv(abnormal_file_path)
    #     abnormal_df['date'] = pd.to_datetime(abnormal_df['date'])
    #     for index, row in abnormal_value_df.iterrows():
    #         bandwidth_key = row['bandwidth_key']
    #         select_start_date, select_end_date = row['start_time'], row['end_time']
    #         if(row['abnormal_class']!='behavior_abnormal'):
    #             continue
    #         if (select_end_date < abnormal_df['date'].iloc[0]):
    #             continue
    #         if (select_start_date > abnormal_df['date'].iloc[-1]):
    #             break
    #         file_path = os.path.join(raw_data_dir, abnormal_file_path.split("/")[-1])
    #         df = pd.read_csv(file_path)
    #         df['date'] = pd.to_datetime(df['date'])
    #         abnormal_df.loc[(abnormal_df['date'] >= pd.to_datetime(select_start_date)) & (
    #                 abnormal_df['date'] <= pd.to_datetime(select_end_date)), bandwidth_key] = df.loc[(df['date'] >= pd.to_datetime(row['refer_start_time'])) & (
    #                         df['date'] <= pd.to_datetime(row['refer_end_time'])), bandwidth_key]
    #     abnormal_df.to_csv(abnormal_file_path, index=False)
    print("----结束将异常数据写入原始数据文件----")
    pass





def signal_normal_feature_value(fine_grained_data_df,dataset_parameter):
    normal_value_list=dataset_parameter["normal_value_list"]
    voltage_threshold=dataset_parameter["voltage_threshold"]
    filtered_df=fine_grained_data_df
    # 找出存在大于阈值的列
    # columns_with_values_above_threshold = filtered_df.iloc[:, 1:].columns[
    #     (filtered_df.iloc[:, 1:] > voltage_threshold).any(axis=0)]

    columns_with_values_above_threshold = filtered_df.iloc[:, 1:].columns[
        (filtered_df.iloc[:, 1:] > voltage_threshold).mean(axis=0)>0.01]

    filtered_df_cols=list(filtered_df.columns)
    filtered_df_cols.remove('date')
    noise_cols=list(set(filtered_df_cols) - set(columns_with_values_above_threshold))
    noise_power_df=filtered_df[noise_cols]
    noise_power_array=noise_power_df.values
    # 计算千分位值
    noise_low, noise_median, noise_high = np.percentile(noise_power_array,
                                                        [value * 100 for value in normal_value_list])
    # 计算噪声的均值
    noise_mean=np.mean(noise_power_array)


    signal_power_df = filtered_df[columns_with_values_above_threshold]

    signal_power_df=signal_power_df [signal_power_df.gt(voltage_threshold).mean(axis=1)>0.1]
    signal_power_array = signal_power_df.values

    # 计算千分位值
    power_low,power_median,power_high = np.percentile(signal_power_array, [value * 100 for value in normal_value_list])


    # 计算均值
    power_mean=np.mean(signal_power_array)
    # 创建一个新列用于存储判断结果

    filtered_df = filtered_df.copy()

    filtered_df = filtered_df.iloc[:, 1:]
    filtered_df = filtered_df[(filtered_df > voltage_threshold).any(axis=1)]

    filtered_df['freq_bandwidth'] = 0
    # filtered_df['Threshold_Exceed_num'] = filtered_df.iloc[:, 1:].gt(voltage_threshold).sum(axis=1)
    filtered_df['freq_bandwidth'] = filtered_df.iloc[:, 1:-2].gt(voltage_threshold).sum(axis=1)

    bandwidth_low,bandwidth_median,bandwidth_high = filtered_df['freq_bandwidth'].quantile(normal_value_list)
    bandwidth_mean=filtered_df['freq_bandwidth'].mean()
    return (power_low,power_median,power_high,power_mean ), ( bandwidth_low,bandwidth_median,bandwidth_high,bandwidth_mean),(noise_low, noise_median, noise_high,noise_mean)


    pass

def normal_value(signal_df, signal_happen_time_df, high_low_perc=[0.95, 0.05]):
    '''
    功能：计算正常信号的特征范围（功率、带宽、时间）。
    输入：
        :param signal_df: （信号数据）
        :param signal_happen_time_df: （信号事件时间记录）
        :param high_low_perc: （高低百分位）
        :return: 功率范围 (power_high, power_low)、带宽范围 (bandwidth_high, bandwidth_low)、时间范围 (time_high, time_low)。
    '''
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



def generate_emi_abnormity(abnormal_num,test_signal_record_and_feature_df,start_and_end_date,df_cols, normal_time_values,normal_power_values, normal_bandwidth_values,normal_noise_values,dataset_parameter):
    start_date, end_date=start_and_end_date[0],start_and_end_date[1]
    time_low, time_high= normal_time_values[0], normal_time_values[2]
    bandwidth_low, bandwidth_median,bandwidth_high=normal_bandwidth_values[0],normal_bandwidth_values[1],normal_bandwidth_values[2]
    normal_power_low,normal_power_median,normal_power_high,normal_power_mean=normal_power_values[0],normal_power_values[1],normal_power_values[2],normal_power_values[3]
    normal_noise_mean=normal_noise_values[3]
    # 生成10个随机时间段并存储在DataFrame中
    time_periods = [generate_random_time_period(start_date, end_date, int(time_high*0.75), int(time_high*2)) for _ in range(abnormal_num)]

    abnormal_emi_list = []


    for time_period in time_periods:


        # 随机确定频段，带宽
        abnormal_bandwidth = random.randint(bandwidth_low, bandwidth_high)
        if abnormal_bandwidth > len(df_cols):
            abnormal_bandwidth = len(df_cols)
            # 随机选择起始位置，确保不会超出列表末尾
        start = random.randint(0, len(df_cols) - abnormal_bandwidth)

        # 获取指定长度的连续子序列
        abnormal_cols = df_cols[start:start + abnormal_bandwidth]

        # abnormal_power=normal_power_high-normal_power_low

        if (dataset_parameter["SIR"] is None):
            abnormal_power = random.uniform(
                max((normal_power_median - normal_power_low), (normal_power_high - normal_power_low) * 0.5),
                (normal_power_high - normal_power_low))

        elif (isinstance(dataset_parameter["SIR"], (int, float))):
            abnormal_power = normal_power_mean - dataset_parameter["SIR"] - normal_noise_mean

        else:
            abnormal_power = normal_power_mean - random.uniform(-10,5) - normal_noise_mean

        abnormal = []
        abnormal.append(time_period[0])
        abnormal.append(time_period[1])
        abnormal.append(time_period[0])
        abnormal.append(time_period[1])
        abnormal.append(abnormal_cols)
        abnormal.append(abnormal_power)
        abnormal.append('add')
        abnormal.append('interfere')

        if (dataset_parameter["adjust_abnormal_time"] is True):
            abnormal_label_start_time = time_period[0] - timedelta(
                seconds=dataset_parameter["left_win_size"] + dataset_parameter["right_win_size"])
            abnormal_label_end_time = time_period[1] + timedelta(
                seconds=dataset_parameter["left_win_size"] + dataset_parameter["right_win_size"])
        else:
            abnormal_label_start_time = time_period[0]
            abnormal_label_end_time = time_period[1]

        abnormal.append(abnormal_label_start_time)
        abnormal.append(abnormal_label_end_time)

        abnormal_emi_list.append(abnormal)
    abnormal_emi_df = pd.DataFrame(abnormal_emi_list,
                                   columns=['start_time', 'end_time','refer_start_time', 'refer_end_time','bandwidth_key', 'power','abnormal_generate_method','abnormal_class','abnormal_label_start_time','abnormal_label_end_time'])
    return abnormal_emi_df
    pass


def generate_transmission_abnormity(abnormal_num,test_signal_record_and_feature_df,start_and_end_date,df_cols, normal_time_values,normal_power_values, normal_bandwidth_values,normal_noise_values,dataset_parameter):
    #功率异常,功率增大
    # 带宽异常，带宽增大
    # 时间异常，时间增大
    print("")
    bandwidth_low, bandwidth_median,bandwidth_high,bandwidth_mean=normal_bandwidth_values[0],normal_bandwidth_values[1],normal_bandwidth_values[2],normal_bandwidth_values[3]
    normal_power_low,normal_power_median,normal_power_high,normal_power_mean=normal_power_values[0],normal_power_values[1],normal_power_values[2],normal_power_values[3]
    normal_noise_mean=normal_noise_values[3]

    abnormal_transmission_list = []
    start_date, end_date = start_and_end_date[0], start_and_end_date[1]

    select_test_signal_record_and_feature_df=test_signal_record_and_feature_df[(test_signal_record_and_feature_df['start_time'] >= pd.to_datetime(start_date)) & (
            test_signal_record_and_feature_df['start_time'] <= pd.to_datetime(end_date))]

    #在采样前检查 abnormal_num 是否合法，并动态调整
    #abnormal_num = min(abnormal_num, len(select_test_signal_record_and_feature_df))

    sampled_df = select_test_signal_record_and_feature_df.sample(n=abnormal_num)
    for index, row in sampled_df.iterrows():
        abnormal = []
        # 功率异常
        if(random.randint(0,2)==0):

            random_duration_seconds = random.randint(max(1,int(0.5*row['duration_time'])), row['duration_time'])

            # random_end_time = random_start_time + timedelta(seconds=random_duration_seconds)

            start_time=row['start_time']
            end_time =start_time+ timedelta(seconds=random_duration_seconds)


            abnormal.append(start_time)
            abnormal.append(end_time)

            abnormal.append(row['start_time'])
            abnormal.append(row['end_time'])

            bandwidth = random.randint(row["freq_bandwidth_min"]//(round(float(df_cols[1])-float(df_cols[0]),6)),
                                       row["freq_bandwidth_max"]//(round(float(df_cols[1])-float(df_cols[0]),6)))
            center = len(df_cols) // 2
            start = max(0, center - bandwidth// 2)
            abnormal_cols=df_cols[start:start + bandwidth]
            abnormal.append(abnormal_cols)

            # power = normal_power_values[2]
            # power = 30

            if (dataset_parameter["normal_signal_change_ratio"] is None):
                power = random.uniform(row["signal_power_max"] * 2, row["signal_power_max"] * 3)
            elif(isinstance(dataset_parameter["normal_signal_change_ratio"], (int, float))):
                power = (normal_power_mean - normal_noise_mean) * dataset_parameter["normal_signal_change_ratio"]
            else:
                power = (normal_power_mean - normal_noise_mean)* random.uniform(0.25,2)
            abnormal.append(power)
            abnormal.append('add')
            abnormal.append('power_abnormal')


            if(dataset_parameter["adjust_abnormal_time"] is True):
                abnormal_label_start_time = start_time - timedelta(
                    seconds=dataset_parameter["left_win_size"] + dataset_parameter["right_win_size"])
                abnormal_label_end_time=end_time+timedelta(seconds=dataset_parameter["left_win_size"]+dataset_parameter["right_win_size"])
            else:
                abnormal_label_start_time=start_time
                abnormal_label_end_time = end_time
            abnormal.append(abnormal_label_start_time)
            abnormal.append(abnormal_label_end_time)

        # 带宽异常
        # if (random.randint(0, 2) == 1):
        elif(random.randint(0,2)==1):
        #
            random_duration_seconds = random.randint(max(1,int(0.5*row['duration_time'])), row['duration_time'])

            start_time=row['start_time']
            end_time =start_time+ timedelta(seconds=random_duration_seconds)

            abnormal.append(start_time)
            abnormal.append(end_time)
            abnormal.append(row['start_time'])
            abnormal.append(row['end_time'])


            signal_bandwidth_max = int(row["freq_bandwidth_min"] // (round(float(df_cols[1]) - float(df_cols[0]), 6)))
            center = len(df_cols) // 2
            start1 = max(0, center - signal_bandwidth_max // 2)
            signal1_cols = df_cols[start1:start1 + signal_bandwidth_max]



            if (dataset_parameter["normal_signal_change_ratio"] is None):
                bandwidth = random.randint(
                    min(int(row["freq_bandwidth_max"] // (round(float(df_cols[1]) - float(df_cols[0]), 6)) * 1.5),
                        int(len(df_cols) * 0.75)),
                    max(int(row["freq_bandwidth_max"] // (round(float(df_cols[1]) - float(df_cols[0]), 6)) * 1.5),
                        int(len(df_cols) * 0.75)))

            elif (isinstance(dataset_parameter["normal_signal_change_ratio"], (int, float))):
                bandwidth = int(
                    row["freq_bandwidth_max"] // (round(float(df_cols[1]) - float(df_cols[0]), 6)) + bandwidth_mean *
                    dataset_parameter["normal_signal_change_ratio"])
            else:
                bandwidth = int(
                    row["freq_bandwidth_max"] // (round(float(df_cols[1]) - float(df_cols[0]), 6)) + bandwidth_mean *
                    random.uniform(0.25, 2))


            start2 = max(0, center - bandwidth // 2)
            signal2_cols = df_cols[start2:start2 + bandwidth]
            abnormal_cols = list(set(signal2_cols) - set(signal1_cols))

            abnormal.append(abnormal_cols)
            # power = normal_power_values[1]
            power = row["signal_power_mean"]
            abnormal.append(power)
            abnormal.append('replace')
            abnormal.append('bandwidth_abnormal')


            if (dataset_parameter["adjust_abnormal_time"] is True):
                abnormal_label_start_time = start_time - timedelta(
                    seconds=dataset_parameter["left_win_size"] + dataset_parameter["right_win_size"])
                abnormal_label_end_time = end_time + timedelta(
                    seconds=dataset_parameter["left_win_size"] + dataset_parameter["right_win_size"])
            else:
                abnormal_label_start_time = start_time
                abnormal_label_end_time = end_time

            abnormal.append(abnormal_label_start_time)
            abnormal.append(abnormal_label_end_time)

        # 时间异常
        # if (random.randint(0, 2) == 2):
        else:
        # # # if(True):
            if (random.random() > 0.5):
                time_high=normal_time_values[2]

                # 定义信号持续时间增长之后的范围

                if (dataset_parameter["normal_signal_change_ratio"] is None):
                    add_time = random.randint(int(max(row['duration_time'] * 2, time_high)),
                                              int(max(row['duration_time'] * 3, time_high * 2)))
                elif (isinstance(dataset_parameter["normal_signal_change_ratio"], (int, float))):
                    add_time = int(normal_time_values[2] * dataset_parameter["normal_signal_change_ratio"])
                else:
                    add_time = int(normal_time_values[2] * random.uniform(0.25, 2))

                start_time=pd.to_datetime(row['start_time']) - pd.to_timedelta(add_time, unit='s')

                abnormal.append(start_time)

                # 修改时间异常更改的范围变为增长的部分
                abnormal.append(row['start_time'])

                abnormal.append(row['start_time'])
                abnormal.append(row['end_time'])

                abnormal_label_start_time = start_time
                abnormal_label_end_time = row['end_time']


            else:

                time_high = normal_time_values[2]

                if (dataset_parameter["normal_signal_change_ratio"] is None):
                    add_time = random.randint(int(max(row['duration_time'] * 2, time_high)),
                                              int(max(row['duration_time'] * 3, time_high * 2)))
                elif (isinstance(dataset_parameter["normal_signal_change_ratio"], (int, float))):
                    add_time = int(normal_time_values[2] * dataset_parameter["normal_signal_change_ratio"])
                else:
                    add_time = int(normal_time_values[2] * random.uniform(0.25, 2))

                end_time = pd.to_datetime(row['end_time']) + pd.to_timedelta(add_time ,unit='s')

                # 修改时间异常更改的范围变为增长的部分
                abnormal.append(row['end_time'])
                abnormal.append(end_time)
                abnormal.append(row['start_time'])
                abnormal.append(row['end_time'])
                abnormal_label_start_time =row['start_time']
                abnormal_label_end_time = end_time


            bandwidth = random.randint(row["freq_bandwidth_min"]//(round(float(df_cols[1])-float(df_cols[0]),6)),
                                       row["freq_bandwidth_max"]//(round(float(df_cols[1])-float(df_cols[0]),6)))


            center = len(df_cols) // 2
            start = max(0, center - bandwidth // 2)
            abnormal.append(df_cols)
            power = normal_power_values[1]
            abnormal.append(power)
            abnormal.append('replace')
            abnormal.append('time_abnormal')

            abnormal.append(abnormal_label_start_time)
            abnormal.append(abnormal_label_end_time)


        abnormal_transmission_list.append(abnormal)
    abnormal_transmission_df = pd.DataFrame(abnormal_transmission_list,
                                            columns=['start_time', 'end_time','refer_start_time', 'refer_end_time',
                                                     'bandwidth_key', 'power','abnormal_generate_method','abnormal_class',
                                                     'abnormal_label_start_time','abnormal_label_end_time'])

    return  abnormal_transmission_df
    pass

def generate_behavior_abnormity(abnormal_num,test_signal_record_and_feature_df,start_and_end_date,df_cols, normal_time_values,normal_power_values, normal_bandwidth_values):

    start_date, end_date = start_and_end_date[0], start_and_end_date[1]
    abnormal_behavior_list = []
    sampled_df = test_signal_record_and_feature_df.sample(n=abnormal_num)
    for index, row in sampled_df.iterrows():
        abnormal = []

        start_time, end_time = generate_random_time_period(start_date, end_date, row['duration_time']-1,
                                                           row['duration_time']-1)
        abnormal.append(start_time)
        abnormal.append(end_time)
        abnormal.append(row['start_time'])
        abnormal.append(row['end_time'])
        abnormal.append(df_cols)
        power = normal_power_values[1]
        abnormal.append(power)
        abnormal.append('replace')
        abnormal.append('behavior_abnormal')


        abnormal_behavior_list.append(abnormal)
    abnormal_behavior_df = pd.DataFrame(abnormal_behavior_list,
                                            columns=['start_time', 'end_time', 'refer_start_time', 'refer_end_time',
                                                     'bandwidth_key', 'power', 'abnormal_generate_method',
                                                     'abnormal_class'])
    return  abnormal_behavior_df
    pass


def generate_behavior_abnormity_new(abnormal_num,test_signal_record_and_feature_df,start_and_end_date,df_cols,dataset_parameter):
    # 在采样前检查 abnormal_num 是否合法，并动态调整
    abnormal_num = min(abnormal_num, len(test_signal_record_and_feature_df))

    sampled_df = test_signal_record_and_feature_df.sample(n=abnormal_num)
    start_date, end_date = start_and_end_date[0][:10], start_and_end_date[1][:10]
    def random_grouping(df, m, n):
        """
        随机分组 DataFrame 的行，每组大小从 m 到 n 不等。
        :param df: 输入的 DataFrame
        :param m: 每组最少的元素数量
        :param n: 每组最多的元素数量
        :return: 包含每组的列表
        """
        if m > n or m <= 0:
            raise ValueError("参数 m 应小于等于 n，且 m 应大于 0。")
        if len(df) < m:
            raise ValueError("DataFrame 的行数不足以分组。")

        # 随机打乱行顺序
        shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        remaining = len(shuffled)
        groups = []

        while remaining > 0:
            # 当前组大小随机选取
            group_size = np.random.randint(m, n + 1)
            group_size = min(group_size, remaining)  # 防止超出剩余行数

            # 取出当前组
            group = shuffled.iloc[:group_size]
            groups.append(group)

            # 更新剩余部分
            shuffled = shuffled.iloc[group_size:]
            remaining -= group_size

        return groups
    def generate_random_times_in_date_range(start_date, end_date, start_time, end_time, num=1):
        """
        在指定日期范围和时间段内生成随机时间值
        :param start_date: 起始日期（字符串格式，YYYY-MM-DD）
        :param end_date: 结束日期（字符串格式，YYYY-MM-DD）
        :param start_time: 每日的开始时间（字符串格式，HH:MM:SS）
        :param end_time: 每日的结束时间（字符串格式，HH:MM:SS）
        :param num: 需要生成的随机时间值数量
        :return: 随机时间值的列表
        """
        # 转换日期和时间
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        if start_date > end_date:
            raise ValueError("结束日期必须晚于或等于开始日期")

        # 计算日期范围天数
        date_range_days = (end_date - start_date).days + 1

        # 转换时间范围为秒数
        start_time_seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(start_time.split(":"))))
        end_time_seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(end_time.split(":"))))
        if start_time_seconds >= end_time_seconds:
            raise ValueError("每日结束时间必须晚于开始时间")

        # 生成随机时间值
        random_times = []
        for _ in range(num):
            # 随机选取日期
            random_date = start_date + timedelta(days=random.randint(0, date_range_days - 1))

            # 随机选取时间
            random_seconds = random.randint(start_time_seconds, end_time_seconds)
            random_time = timedelta(seconds=random_seconds)

            # 组合日期和时间
            full_datetime = random_date + random_time
            random_times.append(full_datetime.strftime("%Y-%m-%d %H:%M:%S"))

        return random_times

    abnormal_behavior_list = []
    m, n =dataset_parameter['m_n'][0],dataset_parameter['m_n'][1]
    groups = random_grouping(sampled_df, m, n)
    times_num=len(groups)
    start_time, end_time=dataset_parameter['abnormal_start_and_end_time'][0],dataset_parameter['abnormal_start_and_end_time'][1]
    random_times = generate_random_times_in_date_range(start_date, end_date, start_time, end_time, times_num)
    for i, group in enumerate(groups, 0):
        abnormal_start_time=pd.to_datetime(random_times[i])
        for index, row in group.iterrows():
            abnormal = []
            abnormal.append(abnormal_start_time)
            abnormal_end_time = pd.to_datetime(abnormal_start_time) + pd.to_timedelta(row['duration_time']-1, unit='s')
            abnormal.append(abnormal_end_time)
            abnormal.append(row['start_time'])
            abnormal.append(row['end_time'])
            abnormal.append(df_cols)
            power = row["signal_power_mean"]
            abnormal.append(power)
            abnormal.append('replace')
            abnormal.append('behavior_abnormal')
            # abnormal_start_time= pd.to_datetime(abnormal_end_time ) + pd.to_timedelta(dataset_parameter["communication_continuous_time_threshold"], unit='s')

            abnormal_label_start_time = abnormal_start_time

            abnormal_start_time= pd.to_datetime(abnormal_end_time ) + pd.to_timedelta(random.randint(1,dataset_parameter["communication_continuous_time_threshold"]*2), unit='s')

            if (dataset_parameter["adjust_behavior_abnormal_time"] is True):
                # abnormal_label_end_time = min(abnormal_start_time,abnormal_end_time+pd.to_timedelta(dataset_parameter["right_win_size"], unit='s'))
                abnormal_label_end_time =abnormal_start_time
            else:
                abnormal_label_end_time=abnormal_end_time

            abnormal.append(abnormal_label_start_time)
            abnormal.append(abnormal_label_end_time)

            abnormal_behavior_list.append(abnormal)

    abnormal_behavior_df = pd.DataFrame(abnormal_behavior_list,
                                            columns=['start_time', 'end_time', 'refer_start_time', 'refer_end_time',
                                                     'bandwidth_key', 'power', 'abnormal_generate_method',
                                                     'abnormal_class','abnormal_label_start_time','abnormal_label_end_time'])
    return  abnormal_behavior_df
    pass






def generate_power_abnormity(start_date, end_date, power_high_and_low, bandwidth_high_and_low, time_high_and_low,
                             abnormal_num):
    '''
    生成功率异常配置（时间段、异常功率值、带宽）
    :param start_date:
    :param end_date:
    :param power_high_and_low:
    :param bandwidth_high_and_low:
    :param time_high_and_low:
    :param abnormal_num:
    :return:包含起止时间、功率值、带宽的 DataFrame。
    '''
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
    '''
    生成带宽异常配置（时间段、带宽异常值）。(本质上就是生成超出正常带宽范围的过宽或过窄带宽值。)
    :param start_date:
    :param end_date:
    :param power_high_and_low:
    :param bandwidth_high_and_low:
    :param time_high_and_low:
    :param abnormal_num:
    :return:包含起止时间、功率值、带宽的 DataFrame。
    '''
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
    '''
    功能：生成持续时间异常配置（时间段过长或过短）。
    逻辑：生成超出正常时间范围的异常持续时间。
    :param start_date:
    :param end_date:
    :param power_high_and_low:
    :param bandwidth_high_and_low:
    :param time_high_and_low:
    :param abnormal_num:
    :return:包含起止时间、功率值、带宽的 DataFrame。
    '''
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