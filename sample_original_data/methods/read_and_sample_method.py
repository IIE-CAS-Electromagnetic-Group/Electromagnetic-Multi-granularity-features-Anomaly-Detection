import os
import time
import struct
import re
from warnings import catch_warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from functools import reduce



# 获取该文件夹中所有的bin文件路径
def get_all_file_list(bin_file_dir="F:\频谱迹线采集\葛洲坝电厂\\11\TraceRecord",select_start_freq=0,select_stop_freq=10):
    print("---------------------get_all_file_list 开始-------------------------")
    print("获取该文件夹中所有的（有效的）bin文件路径:"+bin_file_dir)
    print("get_all_file_list 起始频率:"+str(select_start_freq))
    print("get_all_file_list 终止频率:"+str(select_stop_freq))
    date_dir = os.listdir(bin_file_dir)
    date_dir_list = list(map(lambda x: os.path.join(bin_file_dir, x), date_dir))
    print("子文件夹:"+str(date_dir_list))

    all_file_path = []
    for sub_dir in date_dir_list:
        file_path = os.listdir(sub_dir)
        file_list = list(map(lambda x: os.path.join(sub_dir, x).replace('\\', '/'), file_path))
        #print("file_list:")
        #print(file_list)

        for file in file_list:
            #print("发现文件:"+file)

            # 从文件名中提取起始频率和终止频率
            message_list = file.split('_')
            #print("以'_'分割后的文件信息:"+str(message_list))
            start_freq = float(message_list[-11])
            stop_freq = float(message_list[-10])
            # 检查文件的频率范围是否与目标范围有重叠
            if not (stop_freq < select_start_freq or start_freq > select_stop_freq):
                all_file_path.append(file)
                #print("append file "+file)
            else:
                print("无重叠，被跳过的文件:"+file)



        #all_file_path.extend(file_list)
        # 按文件名排序
    all_file_path = sorted(all_file_path, key=lambda s: (int(s.split('/')[-1].split('_')[0]), int(s.split('/')[-1].split('_')[-1].replace(".bin", ""))))
    print("---------------------get_all_file_list 结束-------------------------\n\n")
    return all_file_path
    pass



def produce_freq_intervals_with_pointnum(bin_file_dir, select_start_freq, select_stop_freq, point_num_of_picture=50000):
    all_file_path = get_all_file_list(bin_file_dir)
    bin_file_path = all_file_path[0]
    message_list = bin_file_path.split('_')
    # 起始终止频率
    start_freq = float(message_list[-11])
    stop_freq = float(message_list[-10])
    # 获取频率列表
    point_num, _ = message_list[-2].split('T')
    point_num = int(point_num)
    show_freq_interval = (stop_freq - start_freq) / point_num * point_num_of_picture

    interval_num = int(
        (min(stop_freq, select_stop_freq) - max(select_start_freq, start_freq)) / show_freq_interval)
    intervals = np.linspace(max(select_start_freq, start_freq), min(stop_freq, select_stop_freq), interval_num + 2,
                            endpoint=True)
    intervals_list = []
    for i in range(len(intervals)):
        if (i == len(intervals) - 1):
            break
        intervals_list.append((intervals[i], intervals[i + 1]))
    return intervals_list
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


def produce_date_intervals(bin_file_dir, start_date, stop_date, minute_of_picture):
    all_file_path = get_all_file_list(bin_file_dir)
    first_bin_file_path = all_file_path[0]
    last_bin_file_path = all_file_path[-1]

    earliest_date = get_file_create_time(first_bin_file_path)

    # 文件中的最晚时间
    last_modified_timestamp = os.path.getmtime(last_bin_file_path)
    last_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_modified_timestamp))
    if (complete_datetime_string(earliest_date) > complete_datetime_string(start_date)):
        earliest_date = complete_datetime_string(earliest_date)
    else:
        earliest_date = complete_datetime_string(start_date)
    if (complete_datetime_string(last_date) > complete_datetime_string(stop_date)):
        last_date = complete_datetime_string(stop_date)
    else:
        last_date = complete_datetime_string(last_date)

    # 使用 pd.date_range 生成时间序列
    time_series = pd.date_range(start=earliest_date, end=last_date, freq=minute_of_picture)

    # 打印生成的时间序列
    print("#打印生成的时间序列:"+str(time_series))


    if(len(time_series)==0 and last_date<earliest_date):
        return []

    date_list = []
    for i in range(len(time_series)):
        if (i == len(time_series) - 1):
            break
        date_list.append((time_series[i], time_series[i + 1]))
    if(len(date_list)==0):
        date_list.append((pd.Timestamp(earliest_date),pd.Timestamp(last_date)))
    else:
        if(date_list[0][0]!=earliest_date):
            date_list.insert(0,(pd.Timestamp(earliest_date),date_list[0][0]))
        if(date_list[-1][-1]!=last_date):
            date_list.append((date_list[-1][-1],pd.Timestamp(last_date)))
    return date_list
    pass


def get_file_create_time(first_bin_file_path):
    # 文件的最早时间
    try:
        with open(first_bin_file_path, "rb") as bin_reader:
            bin_reader.seek(856, 0)
            date_byte = bin_reader.read(23)
            earliest_date = struct.unpack('23s', date_byte)[0].decode("utf-8")
    except struct.error as e:
        print("first_bin_file_path:"+first_bin_file_path)

    return earliest_date


def read_one_bin_file(bin_file_path, filter_freq_list,point_num,read_point_num, start_date, stop_date):
    print(("read_one_bin_file 读取bin文件 " + bin_file_path))

    date_list=[]
    with open(bin_file_path, "rb") as bin_reader:
        bin_reader.seek(856, 0)
        all_value_list = []
        i = 0
        while True:
            try:
                # 解析时间和频点数据
                # date_byte,freq_data=bin_reader.read(23),bin_reader.read(point_num)
                date_byte = bin_reader.read(23)

                bin_reader.seek(filter_freq_list[0], 1)
                freq_data = bin_reader.read(read_point_num)
                # 跳过多余的点
                # bin_reader.seek(point_num - read_point_num, 1)
                bin_reader.seek(point_num - filter_freq_list[-1] - 1, 1)

                date = struct.unpack('23s', date_byte)[0].decode("utf-8")
                if (complete_datetime_string(date) > stop_date):
                    break
                if (complete_datetime_string(date) < start_date):
                    continue
                value_array = np.array(struct.unpack('B' * read_point_num, freq_data)) - 55
                filter_value_list= list(value_array)
                # filter_value_list = [value_list[freq_index] for freq_index in filter_freq_list]
                # filter_value_list = value_list
                del value_array
                # filter_value_list.insert(0, date)
                date_list.append(date)
                all_value_list.append(filter_value_list)
            except:
                break
    return  date_list,all_value_list
    pass


def is_date_string_multiple_formats(date_str, date_formats):
    """
    判断一个字符串是否可以被解析为多种指定格式之一的日期。

    :param date_str: 要判断的字符串
    :param date_formats: 日期格式列表（如 ["%Y-%m-%d", "%d-%m-%Y"]）
    :return: 如果是有效日期格式，返回True；否则返回False。
    """
    for date_format in date_formats:
        try:
            datetime.strptime(date_str, date_format)
            return True
        except ValueError:
            continue
    return False


def read_one_bin_file_center(bin_file_path,all_filter_freq_list,point_num,read_point_num_list, start_date, stop_date,seconds_step,resample_time_method):

    # 初始化参数
    current_block = []

    block_start_date = None
    print(("read_one_bin_file_center reading: " + bin_file_path))
    with open(bin_file_path, "rb") as bin_reader:
        bin_reader.seek(856, 0)
        values_downsampled_data = []
        date_downsampled_data = []
        while True:
            try:
                # 解析时间和频点数据
                date_byte = bin_reader.read(23)
                date = struct.unpack('23s', date_byte)[0].decode("utf-8")

                # if(not (len(date )==23 and is_date_string_multiple_formats(date.split(".")[0], date_formats=["%Y-%m-%d %H:%M:%S"]))):
                #     print(f"再读取 5 个字节后的位置: {bin_reader.tell()}")
                #     bin_reader.seek(point_num, 1)
                #     continue
                #     pass

                if (complete_datetime_string(date) > stop_date):
                    break
                if (complete_datetime_string(date) < start_date):
                    bin_reader.seek(point_num, 1)
                    continue
                if (not block_start_date):
                    block_start_date = date
                current_date = date
                value_list=[]
                pre_right_index=-1
                for i in range(len(all_filter_freq_list)):
                    filter_freq_list=all_filter_freq_list[i]
                    skip=filter_freq_list[0]-pre_right_index-1
                    pre_right_index=filter_freq_list[-1]
                    bin_reader.seek(skip, 1)

                    a=len(filter_freq_list)

                    read_point_num=read_point_num_list[i]

                    # bin_reader.seek(filter_freq_list[0], 1)
                    freq_data = bin_reader.read(read_point_num)
                    # 跳过多余的点
                    # bin_reader.seek(point_num - read_point_num, 1)
                    # bin_reader.seek(point_num - filter_freq_list[-1] - 1, 1)

                    value_array = np.array(struct.unpack('B' * read_point_num, freq_data)) - 55
                    # filter_value_list= list(value_array)

                    value= list(value_array)
                    value_list.extend(value)
                skip=point_num-pre_right_index-1
                bin_reader.seek(skip, 1)


                if type(seconds_step) == str and (seconds_step[-1] == "L" or seconds_step[-1] == "l"):
                    append_flag = calculate_milliseconds_difference(block_start_date, current_date) < int(seconds_step[:-1])
                elif type(seconds_step) == str and (seconds_step[-1] == "S" or seconds_step[-1] == "s"):
                    append_flag = (complete_datetime_string(current_date) - complete_datetime_string(
                        block_start_date)).total_seconds() < int(seconds_step[:-1])
                else:
                    append_flag = (complete_datetime_string(current_date) - complete_datetime_string(
                        block_start_date)).total_seconds() < seconds_step

                if append_flag:
                    current_block.append(value_list)
                else:
                    # 计算当前块的平均值(最大值)并添加到下采样结果中
                    if current_block:

                        if (resample_time_method== "mean"):
                            # current_block = downsample_columns_with_local(current_block, col_step,method)
                            block_result = np.mean(current_block, axis=0).tolist()
                        else:
                            # current_block = downsample_columns_with_local(current_block, col_step, method)
                            block_result = np.max(current_block, axis=0).tolist()
                        # downsampled_data.append([start_date.strftime("%Y-%m-%d %H:%M:%S")] + block_result)
                        if not (type(seconds_step) == str and (
                                seconds_step[-1] == "L" or seconds_step[-1] == "l")) and len(block_start_date) == 23:
                            block_start_date, _ = block_start_date.split(".")
                        date_downsampled_data.append(block_start_date)
                        values_downsampled_data.append(block_result)
                    current_block = [value_list]
                    block_start_date = current_date
            except:
                break
            # 处理最后一个块
        if current_block:
            if (resample_time_method == "mean"):
                # current_block = downsample_columns_with_local(current_block, col_step, method)
                block_result = np.mean(current_block, axis=0).tolist()
            else:
                # current_block = downsample_columns_with_local(current_block, col_step, method)
                block_result = np.max(current_block, axis=0).tolist()
            # downsampled_data.append([start_date.strftime("%Y-%m-%d %H:%M:%S")] + block_result)
            if not (type(seconds_step) == str and (seconds_step[-1] == "L" or seconds_step[-1] == "l")) and len(
                    block_start_date) == 23:
                block_start_date, _ = block_start_date.split(".")
            date_downsampled_data.append(block_start_date)
            values_downsampled_data.append(block_result)
    return  date_downsampled_data,values_downsampled_data
    pass



def read_one_bin_file_new(bin_file_path, filter_freq_list,point_num,read_point_num, start_date, stop_date,seconds_step,resample_time_method):
    """
        从二进制文件中读取特定频段和时间段的数据，并进行时间下采样。

        参数:
        - bin_file_path: 二进制文件的路径。
        - filter_freq_list: 需要筛选的频点列表。
        - point_num: 每个时间点的总频点数。
        - read_point_num: 需要读取的频点数。
        - start_date, stop_date: 选择的时间段。
        - seconds_step: 时间下采样的间隔（秒）。
        - resample_time_method: 下采样的方法（"mean" 或 "max"）。

        返回:
        - date_downsampled_data: 下采样后的时间戳列表。
        - values_downsampled_data: 下采样后的数据值列表。
    """
    # 初始化参数
    current_block = []

    block_start_date = None
    print(("read_one_bin_file_new 开始读取bin文件:" + bin_file_path))
    with open(bin_file_path, "rb") as bin_reader:
        bin_reader.seek(856, 0)
        values_downsampled_data = []
        date_downsampled_data = []
        while True:
            try:
                # 解析时间和频点数据
                # date_byte,freq_data=bin_reader.read(23),bin_reader.read(point_num)
                date_byte = bin_reader.read(23)
                bin_reader.seek(filter_freq_list[0], 1)
                freq_data = bin_reader.read(read_point_num)
                # 跳过多余的点
                # bin_reader.seek(point_num - read_point_num, 1)
                bin_reader.seek(point_num - filter_freq_list[-1] - 1, 1)
                date = struct.unpack('23s', date_byte)[0].decode("utf-8")
                if (complete_datetime_string(date) > stop_date):
                    break
                if (complete_datetime_string(date) < start_date):
                    continue
                if(not block_start_date):
                    block_start_date=date
                current_date=date
                value_array = np.array(struct.unpack('B' * read_point_num, freq_data)) - 55
                # filter_value_list= list(value_array)
                value= list(value_array)

                if type(seconds_step) == str and (seconds_step[-1] == "L" or seconds_step[-1] == "l"):
                    append_flag = calculate_milliseconds_difference(block_start_date, current_date) < int(seconds_step[:-1])
                elif type(seconds_step) == str and (seconds_step[-1] == "S" or seconds_step[-1] == "s"):
                    append_flag = (complete_datetime_string(current_date) - complete_datetime_string(
                        block_start_date)).total_seconds() < int(seconds_step[:-1])
                else:
                    append_flag = (complete_datetime_string(current_date) - complete_datetime_string(
                        block_start_date)).total_seconds() < seconds_step

                if append_flag:
                    current_block.append(value)
                else:
                    # 计算当前块的平均值(最大值)并添加到下采样结果中
                    if current_block:

                        if (resample_time_method== "mean"):
                            # current_block = downsample_columns_with_local(current_block, col_step,method)
                            block_result = np.mean(current_block, axis=0).tolist()
                        else:
                            # current_block = downsample_columns_with_local(current_block, col_step, method)
                            block_result = np.max(current_block, axis=0).tolist()
                        # downsampled_data.append([start_date.strftime("%Y-%m-%d %H:%M:%S")] + block_result)
                        if not (type(seconds_step) == str and (
                                seconds_step[-1] == "L" or seconds_step[-1] == "l")) and len(block_start_date) == 23:
                            block_start_date, _ = block_start_date.split(".")
                        date_downsampled_data.append(block_start_date)
                        values_downsampled_data.append(block_result)
                    current_block = [value]
                    block_start_date = current_date


                # filter_value_list = [value_list[freq_index] for freq_index in filter_freq_list]
                # filter_value_list = value_list
                # del value_array
                # filter_value_list.insert(0, date)
                # date_list.append(date)
                # all_value_list.append(filter_value_list)


            except:
                break
            # 处理最后一个块
        if current_block:
            if (resample_time_method == "mean"):
                # current_block = downsample_columns_with_local(current_block, col_step, method)
                block_result = np.mean(current_block, axis=0).tolist()
            else:
                # current_block = downsample_columns_with_local(current_block, col_step, method)
                block_result = np.max(current_block, axis=0).tolist()
            # downsampled_data.append([start_date.strftime("%Y-%m-%d %H:%M:%S")] + block_result)
            if not (type(seconds_step) == str and (seconds_step[-1] == "L" or seconds_step[-1] == "l")) and len(
                    block_start_date) == 23:
                block_start_date, _ = block_start_date.split(".")
            date_downsampled_data.append(block_start_date)
            values_downsampled_data.append(block_result)



    return  date_downsampled_data,values_downsampled_data
    pass


def get_key_list(bin_file_path,select_start_freq,select_stop_freq):
    """
        从二进制文件路径中提取频率信息，并根据指定的频率范围生成过滤后的频点列表和键列表。

        参数:
        - bin_file_path: 二进制文件的路径。
        - select_start_freq: 用户指定的起始频率。
        - select_stop_freq: 用户指定的终止频率。

        返回:
        - filter_freq_list: 过滤后的频点索引列表。
        - filter_key_list: 过滤后的频率键列表（包含时间戳键 'date'）。
        - point_num: 总频点数。
        - read_point_num: 读取的频点数。
    """
    print("-----get_key_list------")
    print("# bin_file_path: 二进制文件的路径:"+str(bin_file_path))
    print("# select_start_freq: 用户指定的起始频率:"+str(select_start_freq))
    print("# select_stop_freq: 用户指定的终止频率:"+str(select_stop_freq))

    # 解析文件路径，提取起始频率和终止频率
    message_list = bin_file_path.split('_')
    # 起始终止频率
    start_freq = float(message_list[-11])
    stop_freq = float(message_list[-10])
    # 获取频率列表
    point_num, _ = message_list[-2].split('T')
    point_num = int(point_num)
    freq_list = np.linspace(start_freq, stop_freq, point_num, endpoint=True)
    # 生成频点值的列表，并根据设置频点起始和终止点的范围过滤
    freq_list = list(map(lambda x: str(round(x, 6)), freq_list))
    filter_freq_list = list(
        filter(lambda x: (float(freq_list[x]) >= select_start_freq) and (float(freq_list[x]) <= select_stop_freq),
               list(range(0, point_num))))
    read_point_num = filter_freq_list[-1] - filter_freq_list[0] + 1

    # 生成过滤后的频率键列表
    filter_key_list = [freq_list[freq_index] for freq_index in filter_freq_list]
    filter_key_list.insert(0, 'date')
    # 释放不再使用的变量
    del freq_list
    return filter_freq_list, filter_key_list, point_num, read_point_num

def get_key_list_from_center(freq_list,point_num,select_start_freq,select_stop_freq):
    filter_freq_list = list(
        filter(lambda x: (float(freq_list[x]) >= select_start_freq) and (float(freq_list[x]) <= select_stop_freq),
               list(range(0, point_num))))
    read_point_num = filter_freq_list[-1] - filter_freq_list[0] + 1
    filter_key_list = [freq_list[freq_index] for freq_index in filter_freq_list]
    return filter_freq_list, filter_key_list, read_point_num
    pass


def get_all_selected_key(center_freq_list,bandwidth,bin_file_path):
    message_list = bin_file_path.split('_')
    # 起始终止频率
    start_freq = float(message_list[-11])
    stop_freq = float(message_list[-10])
    # 获取频率列表
    point_num, _ = message_list[-2].split('T')
    point_num = int(point_num)
    freq_list = np.linspace(start_freq, stop_freq, point_num, endpoint=True)
    # 生成频点值的列表，并根据设置频点起始和终止点的范围过滤
    freq_list = list(map(lambda x: str(round(x, 6)), freq_list))
    all_filter_freq_list=[]
    all_filter_freq_set=set()
    all_filter_key_list=[]
    read_point_num_list=[]
    for center_freq in center_freq_list:
        select_start_freq=round(center_freq-bandwidth/2,6)
        select_stop_freq=round(center_freq+bandwidth/2,6)
        filter_freq_list, filter_key_list, read_point_num=get_key_list_from_center(freq_list, point_num, select_start_freq, select_stop_freq)


        if(len(all_filter_key_list)>0):
            filter_freq_list= list(set(filter_freq_list) - all_filter_freq_set)
            filter_freq_list=sorted(filter_freq_list)
            filter_key_list=list(set(filter_key_list) - set(all_filter_key_list))
            filter_key_list = sorted(filter_key_list, key=lambda s:float(s))
            read_point_num=len(filter_freq_list)

        all_filter_freq_set.update(filter_freq_list)
        all_filter_freq_list.append(filter_freq_list)
        all_filter_key_list.extend(filter_key_list)
        read_point_num_list.append(read_point_num)


    all_filter_key_list.insert(0, 'date')
    # filter_freq_list, filter_key_list, point_num, read_point_num
    return all_filter_freq_list,all_filter_key_list,point_num,read_point_num_list
    pass





def downsample_by_date_and_column(data, seconds_step, col_step=5):
    # 解析日期并分离出数据部分
    # dates = [datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S") for row in data]
    dates = [complete_datetime_string(row[0]) for row in data]
    values = [row[1:] for row in data]

    # 初始参数：累计的块数据和下采样结果
    initial = ([], [], dates[0])

    def reducer(acc, row):
        current_block, result, start_date = acc
        current_date, value = row
        # 如果当前行的日期与块的起始日期相差小于seconds_step，则添加到当前块
        if (current_date - start_date).total_seconds() < seconds_step:
            current_block.append(value)
        else:
            # 计算当前块的平均值并重置块
            block_mean = np.mean(current_block, axis=0).tolist()
            result.append([start_date.strftime("%Y-%m-%d %H:%M:%S")] + block_mean)
            current_block = [value]
            start_date = current_date
        return current_block, result, start_date

    # 包装数据与日期
    wrapped_data = list(zip(dates, values))

    # 使用reduce处理数据
    final_block, downsampled_data, final_date = reduce(reducer, wrapped_data, initial)

    # 最后一个块处理
    if final_block:
        block_mean = np.mean(final_block, axis=0).tolist()
        downsampled_data.append([final_date.strftime("%Y-%m-%d %H:%M:%S")] + block_mean)

    # # 对下采样后的数据进行列下采样
    # downsampled_matrix = []
    # for row in downsampled_data:
    #     downsampled_matrix.append([row[0]] + row[1:][::col_step])

    # return downsampled_matrix
    return downsampled_data

def downsample_columns_with_local(data, col_step,method):
    # 转置数据以便按列处理
    transposed_data = list(zip(*data))

    downsampled_transposed = []
    for i in range(0, len(transposed_data), col_step):
        block = transposed_data[i:i + col_step]
        # 计算每列的局部最大值
        if(method=='mean'):
            block_max = np.mean(block, axis=0).tolist()
        else:
            block_max = np.max(block, axis=0).tolist()
        downsampled_transposed.append(block_max)
    # 转置回原始形状
    downsampled_data = list(list(row) for row in zip(*downsampled_transposed))
    return downsampled_data

def downsample_columns_with_local_value(data,col_step,method):
    transposed_data = list(zip(*data))
    downsampled_transposed = []
    for i in range(0, len(transposed_data), col_step):
        block = transposed_data[i:i + col_step]

        # 计算每列的局部最大值
        if (method == 'mean'):
            block_value = np.mean(block, axis=0).tolist()
        elif(method=="max"):
            block_value = np.max(block, axis=0).tolist()
        else:
            block_value=block[len(block)//2]

        downsampled_transposed.append(block_value)
    # 转置回原始形状
    downsampled_data = list(list(row) for row in zip(*downsampled_transposed))
    return downsampled_data

    pass

def downsample_by_date_and_column_loop(data, seconds_step, col_step=5,method="mean"):
    # 解析日期并分离出数据部分
    # dates = [datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S") for row in data]
    # dates = [complete_datetime_string(row[0]) for row in data]
    # dates = [complete_datetime_string(row[0]) for row in data[0]]
    dates = data[0]

    # values = [row[1:] for row in data]
    values=data[1]
    # 初始化参数
    downsampled_data = []
    current_block = []
    values_downsampled_data=[]
    date_downsampled_data=[]
    start_date = dates[0]

    for i in range(len(dates)):
        current_date = dates[i]
        value = values[i]

        # 如果当前行的日期与块的起始日期相差小于seconds_step，则添加到当前块
        if (complete_datetime_string(current_date) - complete_datetime_string(start_date)).total_seconds() < seconds_step:
            current_block.append(value)
        else:
            # 计算当前块的平均值(最大值)并添加到下采样结果中
            if current_block:

                if(method=="mean"):
                    # current_block = downsample_columns_with_local(current_block, col_step,method)
                    block_result = np.mean(current_block, axis=0).tolist()

                else:
                    # current_block = downsample_columns_with_local(current_block, col_step, method)
                    block_result = np.max(current_block, axis=0).tolist()
                # downsampled_data.append([start_date.strftime("%Y-%m-%d %H:%M:%S")] + block_result)

                date_downsampled_data.append(complete_datetime_string(start_date))
                values_downsampled_data.append(block_result)
            current_block = [value]
            start_date = current_date

    # 处理最后一个块
    if current_block:
        if (method == "mean"):
            # current_block = downsample_columns_with_local(current_block, col_step, method)
            block_result = np.mean(current_block, axis=0).tolist()
        else:
            # current_block = downsample_columns_with_local(current_block, col_step, method)
            block_result = np.max(current_block, axis=0).tolist()
        # downsampled_data.append([start_date.strftime("%Y-%m-%d %H:%M:%S")] + block_result)
        date_downsampled_data.append(complete_datetime_string(start_date))
        values_downsampled_data.append(block_result)

    # values = [row[1:] for row in downsampled_data]
    values=values_downsampled_data
    # col_downsample=downsample_columns_with_local(values, col_step, method)
    col_downsample=values
    new_result=[]
    for i in range(len(date_downsampled_data)):
        new_result.append([date_downsampled_data[i]]+col_downsample[i])
    downsampled_data=new_result
    return downsampled_data


def downsample_key_list(key_list,col_step=5,method = 'mean'):
    float_key_list=[float(key) for key in key_list[1:]]
    new_key_list=[key_list[0]]
    for i in range(0, len(float_key_list), col_step):
        block = float_key_list[i:i + col_step]
        if (method == 'mean'):
            block_key=str(np.mean(block))
        else:
            block_key=str(block[len(block)//2])
        new_key_list.append(block_key)
    return new_key_list
    pass


def calculate_milliseconds_difference(date_str1, date_str2, date_format='%Y-%m-%d %H:%M:%S.%f'):
    # 将字符串转换为 datetime 对象
    dt1 = datetime.strptime(date_str1, date_format)
    dt2 = datetime.strptime(date_str2, date_format)

    # 计算时间差
    delta = dt2 - dt1

    # 将时间差转换为总毫秒数
    total_milliseconds = delta.total_seconds() * 1000
    return total_milliseconds

def downsample_by_date(data, seconds_step,voltage_threshold,method="mean"):
    dates = data[0]
    values = data[1]
    # 初始化参数
    current_block = []
    values_downsampled_data = []
    date_downsampled_data = []
    start_date = dates[0]
    proportions_mask=[]
    for i in range(len(dates)):
        current_date = dates[i]
        value = values[i]

        if type(seconds_step)==str and (seconds_step[-1]=="L" or  seconds_step[-1]=="l"):
            append_flag=calculate_milliseconds_difference(start_date,current_date)<int(seconds_step[:-1])
        elif type(seconds_step)==str and (seconds_step[-1]=="S" or  seconds_step[-1]=="s"):
            append_flag =(complete_datetime_string(current_date) - complete_datetime_string(
                start_date)).total_seconds() < int(seconds_step[:-1])
        else:
            append_flag = (complete_datetime_string(current_date) - complete_datetime_string(
                start_date)).total_seconds() < seconds_step



        # 如果当前行的日期与块的起始日期相差小于seconds_step，则添加到当前块
        # if (complete_datetime_string(current_date) - complete_datetime_string(
        #         start_date)).total_seconds() < seconds_step:
        #     current_block.append(value)
        if append_flag:
            current_block.append(value)
        else:
            # 计算当前块的平均值(最大值)并添加到下采样结果中
            if current_block:

                if (method == "mean"):
                    # current_block = downsample_columns_with_local(current_block, col_step,method)
                    block_result = np.mean(current_block, axis=0).tolist()
                else:
                    # current_block = downsample_columns_with_local(current_block, col_step, method)
                    block_result = np.max(current_block, axis=0).tolist()
                # downsampled_data.append([start_date.strftime("%Y-%m-%d %H:%M:%S")] + block_result)
                if  not (type(seconds_step) == str and (seconds_step[-1] == "L" or seconds_step[-1] == "l")) and len(start_date) == 23:
                    start_date, _ = start_date.split(".")
                date_downsampled_data.append(start_date)
                values_downsampled_data.append(block_result)

                # 计算每列大于阈值的值所占的比例
                data_array = np.array(current_block)
                proportions = np.mean(data_array > voltage_threshold, axis=0)
                proportions_mask.append(proportions)

            current_block = [value]
            start_date = current_date

    # 处理最后一个块
    if current_block:
        if (method == "mean"):
            # current_block = downsample_columns_with_local(current_block, col_step, method)
            block_result = np.mean(current_block, axis=0).tolist()
        else:
            # current_block = downsample_columns_with_local(current_block, col_step, method)
            block_result = np.max(current_block, axis=0).tolist()
        # downsampled_data.append([start_date.strftime("%Y-%m-%d %H:%M:%S")] + block_result)
        if not (type(seconds_step) == str and (seconds_step[-1] == "L" or seconds_step[-1] == "l")) and len(
                start_date) == 23:
            start_date, _ = start_date.split(".")
        date_downsampled_data.append(start_date)
        values_downsampled_data.append(block_result)

        # 计算每列大于阈值的值所占的比例
        data_array = np.array(current_block)
        proportions = np.mean(data_array > voltage_threshold, axis=0)
        proportions_mask.append(proportions)

    return date_downsampled_data,values_downsampled_data,proportions_mask
    pass


def produce_freq_intervals_with_picturenum(bin_file_dir, select_start_freq, select_stop_freq, picture_num ):
    all_file_path = get_all_file_list(bin_file_dir)
    bin_file_path = all_file_path[0]
    message_list = bin_file_path.split('_')
    # 起始终止频率
    start_freq = float(message_list[-11])
    stop_freq = float(message_list[-10])

    interval_num=picture_num+1
    intervals=np.linspace(max(select_start_freq, start_freq), min(stop_freq, select_stop_freq),interval_num)
    intervals= list(map(lambda x: round(x, 6), intervals))
    intervals_list = []
    for i in range(len(intervals)):
        if (i == len(intervals) - 1):
            break
        intervals_list.append((intervals[i], intervals[i + 1]))
    return intervals_list

    pass

def plot_trace_heatmap(df, jpg_save_path):
    # if (not os.path.exists(output_dir)):
    #     os.makedirs(output_dir)
    #     pass

    # df.set_index('date', inplace=True)

    # _, index_num = df.shape
    # if (index_num > 1000):
    #     step = int(index_num / 1000)
    #     filter_columns = df.columns[::step]
    #     df = df[filter_columns]
    z = df.values
    y = df.index
    x = df.columns.values.tolist()

    # read_start_date = re.sub(r'\D', '', str(y[0]))
    # read_stop_date = re.sub(r'\D', '', str(y[-1]))

    # if (other_message == ""):
    #     all_file = get_all_file_list(output_dir)
    #     order_num = len(all_file)
    #     # html_save_path = os.path.join(output_dir,
    #     #                               read_start_date + "_" + read_stop_date + "_" + str(order_num) + ".html")
    #
    #     jpg_save_path = os.path.join(output_dir,
    #                                   read_start_date + "_" + read_stop_date + "_" + str(order_num) + ".jpg")
    # else:
    #     jpg_save_path = os.path.join(output_dir,
    #                                   read_start_date + "_" + read_stop_date + "_" + other_message + ".jpg")

    # # 创建热图，并隐藏颜色条
    # heatmap = go.Heatmap(z=z,
    #                      x=x,
    #                      y=y,
    #                      colorscale='jet',
    #                      showscale=False,
    #                      zmax=40,
    #                      zmin=-10)
    #
    # # 创建布局，隐藏坐标轴，并使图表填满整个画布
    # layout = go.Layout(
    #     xaxis=dict(
    #         showgrid=False,
    #         zeroline=False,
    #         visible=False
    #     ),
    #     yaxis=dict(
    #         showgrid=False,
    #         zeroline=False,
    #         visible=False,
    #         autorange='reversed'
    #     ),
    #     margin=dict(l=0, r=0, t=0, b=0),
    #     autosize=True
    # )
    #
    # # 创建图表
    # fig = go.Figure(data=[heatmap], layout=layout)

    # 显示图表
    # fig.show()
    # fig.write_image(jpg_save_path, height=600, width=500)



    # 创建热图，并隐藏颜色条
    heatmap = go.Heatmap(z=z,
                         x=x,
                         y=y,
                         colorscale='jet',
                         showscale=True,
                         # zmax=30,
                         # zmin=-50
                         zmax=20,
                         zmin=-40
                         )

    # 创建布局，隐藏坐标轴，并使图表填满整个画布
    layout = go.Layout(
        xaxis=dict(

        nticks=20
        #     showgrid=False,
        #     zeroline=False,
        #     visible=False
        ),
        yaxis=dict(
            # showgrid=False,
            # zeroline=False,
            # visible=False,
            autorange='reversed'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        autosize=True
    )

    # 创建图表
    fig = go.Figure(data=[heatmap], layout=layout)
    html_save_path=jpg_save_path.replace("jpg","html")
    fig.write_html(html_save_path)

    pass


def plot_trace_surface(df, html_save_path):
    # df.set_index('date', inplace=True)
    # _, index_num = df.shape
    # if (index_num > 1000):
    #     step = int(index_num / 1000)
    #     filter_columns = df.columns[::step]
    #     df = df[filter_columns]

    z = df.values
    y = df.index
    x = df.columns.values.tolist()

    read_start_date=re.sub(r'\D', '', str(y[0]))
    read_stop_date=re.sub(r'\D', '', str(y[-1]))

    # if(other_message==""):
    #     all_file = get_all_file_list(output_dir)
    #     order_num = len(all_file)
    #     html_save_path = os.path.join(output_dir, read_start_date+"_"+read_stop_date+"_"+str(order_num) + "_surface.html")
    # else:
    #     html_save_path = os.path.join(output_dir, read_start_date+"_"+read_stop_date+"_"+other_message + "_surface.html")
    # html_save_path=os.path.join(output_dir,"surface.html")

    # 3D曲面图
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='jet',cmin=-10,cmax=70)])
    fig.update_layout(margin=dict(l=0, r=60, b=0, t=20),
                      title='迹线立体图',
                      scene=dict(xaxis=dict(title=dict(
                          text='频点'
                      )),
                          yaxis=dict(title=dict(
                              text='时间'
                          )),
                          zaxis=dict(title=dict(
                              text='电平'
                          ))
                      ))
    # fig.show()
    fig.write_html(html_save_path)
    pass