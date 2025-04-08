from convert_paths.convert_windows_and_linux_path import convert_paths
from sample_original_data.methods.read_and_sample_method import *

def produce_freq_intervals_with_picturenum_from_8090bin(bin_file_dir, select_start_freq, select_stop_freq, picture_num ):
    '''

    :param bin_file_dir:
    :param select_start_freq:
    :param select_stop_freq:
    :param picture_num:
    :return:
    '''
    all_file_path = get_all_file_list(bin_file_dir)
    bin_file_path = all_file_path[0]
    message_list = bin_file_path.split("/")[-1].split('_')

    # message_list = bin_file_path.split("/")[-1].split('_')
    # 起始终止频率
    start_freq = float(message_list[1])
    stop_freq = float(message_list[2])

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

def read_one_8090bin_file(bin_file_path, filter_freq_list,point_num,read_point_num, start_date, stop_date,seconds_step,resample_time_method):
    '''

    :param bin_file_path:
    :param filter_freq_list:
    :param point_num:
    :param read_point_num:
    :param start_date:
    :param stop_date:
    :param seconds_step:
    :param resample_time_method:
    :return:
    '''
    # 初始化参数
    current_block = []

    block_start_date = None
    print(("reading " + bin_file_path))
    with open(bin_file_path, "rb") as bin_reader:
        bin_reader.seek(856, 0)
        values_downsampled_data = []
        date_downsampled_data = []

        while True:
            try:
                # 解析时间和频点数据
                # date_byte,freq_data=bin_reader.read(23),bin_reader.read(point_num)
                date_byte = bin_reader.read(23)
                bin_reader.seek(filter_freq_list[0]*2, 1)
                freq_data = bin_reader.read(read_point_num*2)
                # 跳过多余的点
                # bin_reader.seek(point_num - read_point_num, 1)
                bin_reader.seek((point_num - filter_freq_list[-1] - 1)*2, 1)
                date = struct.unpack('23s', date_byte)[0].decode("utf-8")
                # print(f"文件指针当前的位置: {bin_reader.tell()}")
                if (complete_datetime_string(date) > stop_date):
                    break
                if (complete_datetime_string(date) < start_date):
                    continue
                if(not block_start_date):
                    block_start_date=date
                current_date=date
                value_array = np.array(struct.unpack('h' * read_point_num, freq_data)) /10.0
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


def get_key_list_from_8090bin(bin_file_path,select_start_freq,select_stop_freq):
    '''

    :param bin_file_path:
    :param select_start_freq:
    :param select_stop_freq:
    :return:
    '''
    # message_list = bin_file_path.split('_')
    message_list = bin_file_path.split("/")[-1].split('_')
    # 起始终止频率
    start_freq = float(message_list[1])
    stop_freq = float(message_list[2])
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
    filter_key_list = [freq_list[freq_index] for freq_index in filter_freq_list]
    filter_key_list.insert(0, 'date')
    del freq_list
    return filter_freq_list, filter_key_list, point_num, read_point_num


def generate_one_data_from_8090bin(bin_file_dir, intervals_start_freq, intervals_stop_freq, intervals_start_date,
                         intervals_stop_date,resample_time,resample_point_num,output_dir):




    all_file_path = get_all_file_list(bin_file_dir)

    all_value_list = []
    all_date_list=[]
    key_list = None
    point_num=None
    read_point_num=None
    filter_freq_list=None
    for file_path in all_file_path:
        last_modified_timestamp = os.path.getmtime(file_path)
        last_modified_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_modified_timestamp))
        if (complete_datetime_string(last_modified_date) < intervals_start_date):
            continue
        elif (complete_datetime_string(get_file_create_time(file_path)) > intervals_stop_date):
            break
        else:
            if(key_list == None or filter_freq_list==None or point_num==None or read_point_num==None):
                filter_freq_list, key_list, point_num, read_point_num = get_key_list_from_8090bin(file_path,intervals_start_freq, intervals_stop_freq,)
            # 根据时间进行初步删选，缩小数据规模
            seconds_step = resample_time
            # resample_time_method="max"
            resample_time_method="mean"

            date_list, filter_value_list=read_one_8090bin_file(file_path, filter_freq_list,point_num,read_point_num,intervals_start_date, intervals_stop_date,
                                      seconds_step, resample_time_method)

            if(len(filter_value_list)==0):
                continue

            resample_freq_method = "other"
            col_step = read_point_num // resample_point_num

            if (col_step > 1):
                filter_value_list=downsample_columns_with_local_value(filter_value_list, col_step, resample_freq_method)

        if(len(all_value_list)>0 and len(date_list)>0 and all_date_list[-1]==date_list[0]):
            block=[all_value_list[-1],filter_value_list[0]]
            if (resample_time_method == "mean"):
                block_result = np.mean(block, axis=0).tolist()
            else:
                block_result = np.max(block, axis=0).tolist()
            all_value_list[-1]=block_result
            all_value_list.extend(filter_value_list[1:])
            all_date_list.extend(date_list[1:])

        else:
            all_value_list.extend(filter_value_list)
            all_date_list.extend(date_list)

        del filter_value_list,date_list


    # 缩小数据规模，减少程序运行时间
    # all_value_list=downsample_by_date_and_column(data=all_value_list, seconds_step=1)
    # 下采样，以秒为seconds_step，列元素下采样，以col_step，下采样的方式“mean”或“max”
    # seconds_step = resample_freq
    # point_num=len(key_list)-1
    # col_step=point_num//1000
    # if(col_step<1):
    #     col_step=1
    # all_value_list=downsample_by_date_and_column_loop(data=(all_date_list,all_value_list), seconds_step=seconds_step,col_step=col_step,method="max")

    if(len(all_value_list)==0 or len(all_date_list)==0):
        return 0
    # 对列键值进行相应处理
    if(col_step>1):
        key_list=downsample_key_list(key_list,col_step=col_step,method="other")


    df_value=pd.DataFrame(data=all_value_list, columns=key_list[1:])
    df_date=pd.Series(all_date_list, name=key_list[0])

    df = pd.concat([df_date,df_value], axis=1)
    # pd.DataFrame(data=all_value_list, columns=key_list)


    del all_value_list,key_list
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    # print(df)


    if (not os.path.exists(output_dir)):
        os.makedirs(output_dir)
        pass
    other_message = str(intervals_start_freq) + '_' + str(intervals_stop_freq)

    y = df.index
    read_start_date = re.sub(r'\D', '', str(y[0]))
    read_stop_date = re.sub(r'\D', '', str(y[-1]))
    if (other_message == ""):
        all_file = get_all_file_list(output_dir)
        order_num = len(all_file)
        jpg_save_path = os.path.join(output_dir,
                                      read_start_date + "_" + read_stop_date + "_" + str(order_num) + ".jpg")
    else:
        jpg_save_path = os.path.join(output_dir,
                                      read_start_date + "_" + read_stop_date + "_" + other_message + ".jpg")


    csv_save_path =jpg_save_path.replace("jpg","csv")
    df.to_csv(csv_save_path, index=True)

    # plot_trace_heatmap(df, jpg_save_path)

    # print(all_value_list[0])
    pass



def generate_trace_data_from_8090bin():

    #bin_file_dir = "F:/20240629-第二次出差数据-宜昌（选取部分）/20240629葛电厂厂房机柜-长时设备-电磁数据/TraceRecord"
    bin_file_dir="D:\iie\Data\原始数据\信息工程研究所\\0_30MHz"

    bin_file_dir="E:/葛洲坝/TraceRecord"

    bin_file_dir = convert_paths(bin_file_dir)

    select_start_freq = 0
    select_stop_freq = 30
    start_date = "2024-04-17 00:00"
    stop_date = "2024-04-20 00:00"
    # point_num_of_picture = 50000

    # 设置生成图片的参数
    picture_num = 1  # 图片数量
    minute_of_picture = '10T'  # 每张图片的时间跨度（10分钟）
    resample_time = 1  # 时间重采样参数
    resample_freq_num = 1000  # 频率重采样参数
    # 参数为后期经过探索得出




    file_massages = bin_file_dir.split("/")
    location = file_massages[-2]# 提取倒数第二个目录名作为位置信息
    #prefix_path= "G:/Work/Workspace/Data/预处理数据"# 设置输出目录的前缀路径
    prefix_path = "预处理数据"
    # 构建输出目录路径，包含频率范围和位置信息
    output_dir = os.path.join(prefix_path, location,str(select_start_freq)+"_"+str(select_stop_freq)+"MHz原始信号数据")
    output_dir = convert_paths(output_dir)

    date_list = produce_date_intervals(bin_file_dir, start_date, stop_date, minute_of_picture)
    intervals_list = produce_freq_intervals_with_picturenum_from_8090bin(bin_file_dir, select_start_freq, select_stop_freq,
                                                            picture_num)
    for (intervals_start_date, intervals_stop_date) in date_list:
        for (intervals_start_freq, intervals_stop_freq) in intervals_list:
            print(
                f"generrating the picture of frequency between {intervals_start_freq} and {intervals_stop_freq} , date from {intervals_start_date} to {intervals_stop_date}")
            generate_one_data_from_8090bin(bin_file_dir, intervals_start_freq, intervals_stop_freq, intervals_start_date,
                              intervals_stop_date, resample_time, resample_freq_num, output_dir)
    pass


if __name__=="__main__":
    generate_trace_data_from_8090bin()
    print("处理完成")