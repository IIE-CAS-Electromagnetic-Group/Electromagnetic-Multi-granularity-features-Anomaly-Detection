from sample_original_data.methods.read_and_sample_method import *

def generate_one_data_from_center_list(bin_file_dir,center_freq_list, bandwidth, intervals_start_date,
                         intervals_stop_date,resample_time,output_dir):
    """
        从原始bin文件中提取特定中心频点和带宽的数据，并进行时间下采样。

        参数:
        - bin_file_dir: 存储bin文件的文件夹路径。
        - center_freq_list: 中心频点列表，表示要提取的频段中心频率。
        - bandwidth: 带宽，表示频段的宽度。
        - intervals_start_date, intervals_stop_date: 选择的时间段。
        - resample_time: 时间下采样的间隔（秒）。
        - output_dir: 输出目录，用于保存处理后的数据。
    """
    all_file_path = get_all_file_list(bin_file_dir)
    all_value_list = []
    all_date_list = []
    key_list = None
    point_num = None
    read_point_num_list = None
    all_filter_freq_list = None
    for file_path in all_file_path:
        last_modified_timestamp = os.path.getmtime(file_path)
        last_modified_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_modified_timestamp))
        if (complete_datetime_string(last_modified_date) < intervals_start_date):
            continue
        elif (complete_datetime_string(get_file_create_time(file_path)) > intervals_stop_date):
            break
        else:
            if (key_list == None or all_filter_freq_list == None or point_num == None or read_point_num_list == None):
                all_filter_freq_list, key_list, point_num, read_point_num_list=get_all_selected_key(center_freq_list, bandwidth, file_path)

            # 根据时间进行初步删选，缩小数据规模
            seconds_step = resample_time
            # resample_time_method="max"
            resample_time_method = "mean"


            date_list, filter_value_list=read_one_bin_file_center(file_path, all_filter_freq_list, point_num, read_point_num_list, intervals_start_date, intervals_stop_date,
                                     seconds_step, resample_time_method)
            if (len(filter_value_list) == 0):
                continue

        if (len(all_value_list) > 0 and len(date_list) > 0 and all_date_list[-1] == date_list[0]):
            block = [all_value_list[-1], filter_value_list[0]]
            if (resample_time_method == "mean"):
                block_result = np.mean(block, axis=0).tolist()
            else:
                block_result = np.max(block, axis=0).tolist()
            all_value_list[-1] = block_result
        else:
            all_value_list.extend(filter_value_list)
            all_date_list.extend(date_list)

        del filter_value_list, date_list

    if(len(all_value_list)==0 or len(all_date_list)==0):
        return 0
    df_value = pd.DataFrame(data=all_value_list, columns=key_list[1:])
    df_date = pd.Series(all_date_list, name=key_list[0])
    df = pd.concat([df_date, df_value], axis=1)
    del all_value_list, key_list
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    if (not os.path.exists(output_dir)):
        os.makedirs(output_dir)
        pass
    other_message=''
    for center_freq in center_freq_list:
        other_message=other_message +str(center_freq)+ '_'
    other_message =other_message+str(bandwidth)

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


    csv_save_path = jpg_save_path.replace("jpg", "csv")
    df.to_csv(csv_save_path, index=True)
    print("原始bin文件处理后的文件保存为:"+csv_save_path)
    # plot_trace_heatmap(df, jpg_save_path)
    del df
    pass

def generate_one_data(bin_file_dir, intervals_start_freq, intervals_stop_freq, intervals_start_date,
                         intervals_stop_date,resample_time,resample_point_num,output_dir):
    """
        从原始bin文件中提取特定频段和时间段的数据，并进行频点和时间下采样。

        参数:
        - bin_file_dir: 存储bin文件的文件夹路径。
        - intervals_start_freq, intervals_stop_freq: 选择的频段范围。
        - intervals_start_date, intervals_stop_date: 选择的时间段。
        - resample_time: 时间下采样的间隔（秒）。
        - resample_point_num: 频点下采样的列数。
        - output_dir: 输出目录，用于保存处理后的数据。
    """
    print("------------generate_one_data 开始----------------")
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
        # 根据时间筛选文件
        if (complete_datetime_string(last_modified_date) < intervals_start_date):
            continue# 如果文件修改时间早于起始时间，跳过
        elif (complete_datetime_string(get_file_create_time(file_path)) > intervals_stop_date):
            break# 如果文件创建时间晚于结束时间，停止处理
        else:
            # 初始化key_list、filter_freq_list等变量
            #key_list是存储数据的列名（键），通常是时间戳和频点的组合。它定义了数据的结构
            #filter_freq_list 存储需要提取的频点列表。它决定了从原始数据中提取哪些频段的数据。
            if(key_list == None or filter_freq_list==None or point_num==None or read_point_num==None):
                filter_freq_list, key_list, point_num, read_point_num = get_key_list(file_path,intervals_start_freq, intervals_stop_freq,)


            # 根据时间进行初步删选，缩小数据规模
            seconds_step = resample_time
            # resample_time_method="max"
            resample_time_method="mean"

            date_list, filter_value_list=read_one_bin_file_new(file_path, filter_freq_list,point_num,read_point_num,intervals_start_date, intervals_stop_date,
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
    del all_value_list,key_list
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

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
    html_save_path =jpg_save_path.replace("jpg","csv")
    df.to_csv(csv_save_path, index=True)
    print("原始bin文件处理后的文件保存为:"+csv_save_path)
    print("------------generate_one_data 结束----------------")
    plot_trace_heatmap(df, jpg_save_path)
    # plot_trace_surface(df, jpg_save_path)
    pass