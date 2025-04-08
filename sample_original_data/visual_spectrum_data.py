import os
import pandas as pd
from convert_paths.convert_windows_and_linux_path import convert_paths
from sample_original_data.methods.read_and_sample_method import plot_trace_heatmap


def get_all_file_list(csv_file_dir):
    # 获取该文件夹中所有的bin文件路径
    file_path = os.listdir(csv_file_dir)
    file_list = list(map(lambda x: os.path.join(csv_file_dir, x).replace('\\', '/'), file_path))
    all_file_path = sorted(file_list, key=lambda s: int(s.split('/')[-1].split('_')[0]))
    print("all_file_path:")
    print(all_file_path)
    return all_file_path
    pass

def visual_spectrum_data(csv_file_dir="E:/模型训练/葛洲坝/raw_data/test_raw_data",output_dir = "E:/预处理/visual"):
    # csv_file_dir = "G:/Work/Workspace/Data/预处理数据/信息工程研究所/0_30MHz原始信号数据"
    # output_dir="G:/Work/Workspace/Data/模型输入输出数据或其它处理结果/电磁泄漏信号/"

    # csv_file_dir = "G:/Work/Workspace/Data/预处理数据/信息工程研究所/416.086_0.2MHz原始信号数据"
    # output_dir = "G:/Work/Workspace/Data/模型输入输出数据或其它处理结果/对讲机信号/"
    # csv_file_dir = "G:/Work/Workspace/Data/预处理数据/信息工程研究所/1_2.5MHz原始信号数据"



    # csv_file_dir = "G:/Work/Workspace/Data/预处理数据/信息工程研究所/1.75MHz原始信号数据"
    # output_dir = "G:/Work/Workspace/Data/模型输入输出数据或其它处理结果/电磁泄漏信号/"

    # csv_file_dir = "G:/Work/Workspace/Data/预处理数据/对讲机信号/下采样后的对讲机信号数据"
    # output_dir = "G:/Work/Workspace/Data/模型输入输出数据或其它处理结果/对讲机信号/小论文绘图"


    # 电梯电磁泄漏信号
    # csv_file_dir = "G:/Work/Workspace/Data/模型输入输出数据或其它处理结果/电磁泄漏信号/模型训练数据/EM_leakage/raw_data/test_raw_data"
    # output_dir = "G:/Work/Workspace/Data/模型输入输出数据或其它处理结果/电磁泄漏信号/模型训练数据/EM_leakage/visual"

    #csv_file_dir = "G:/Work/Workspace/Data/模型输入输出数据或其它处理结果/电磁泄漏信号/模型训练数据/EM_leakage/raw_data/test_abnormal_data"
    #output_dir = "G:/Work/Workspace/Data/模型输入输出数据或其它处理结果/电磁泄漏信号/模型训练数据/EM_leakage/visual"


    # 对讲机信号
    # csv_file_dir = "G:/Work/Workspace/Data/模型输入输出数据或其它处理结果/对讲机信号/模型训练数据/interphone_signal/raw_data/test_raw_data"
    # output_dir = "G:/Work/Workspace/Data/模型输入输出数据或其它处理结果/对讲机信号/模型训练数据/interphone_signal/visual"

    # csv_file_dir = "G:/Work/Workspace/Data/模型输入输出数据或其它处理结果/对讲机信号/模型训练数据/interphone_signal/raw_data/test_abnormal_data"
    # output_dir = "G:/Work/Workspace/Data/模型输入输出数据或其它处理结果/对讲机信号/模型训练数据/interphone_signal/visual"

    #csv_file_dir = "E:/预处理/0-15MHz"
    #output_dir = "E:/预处理/visual"

    #csv_file_dir="E:/模型训练/葛洲坝/raw_data/test_raw_data"

    csv_file_dir= convert_paths(csv_file_dir)
    output_dir=convert_paths(output_dir)

    output_file_dir =os.path.join(output_dir,csv_file_dir.split('/')[-1]+"可视化")
    output_file_dir=convert_paths(output_file_dir)
    if (not os.path.exists(output_file_dir)):
        os.makedirs(output_file_dir)
        pass
    all_file_path = get_all_file_list(csv_file_dir)
    # 遍历每个文件，读取数据并生成可视化结果
    for file_path in all_file_path:
        output_file_path=os.path.join(output_file_dir,file_path.split('/')[-1].replace('csv','html'))
        df = pd.read_csv(file_path)
        df.set_index('date', inplace=True)
        plot_trace_heatmap(df, output_file_path)
        print("Save as "+str(output_file_path))


    pass
if __name__=="__main__":
    visual_spectrum_data(csv_file_dir="D:\iie\Python_Workspace\Time-Series-Library-main\数据集\葛洲坝\\raw_data\\test_raw_data",output_dir="E:/可视化/葛洲坝test_raw_data")
    #visual_spectrum_data(csv_file_dir="D:\iie\Python_Workspace\Time-Series-Library-main\数据集\葛洲坝\\raw_data\\test_abnormal_data",output_dir="E:/可视化/葛洲坝test_abnormal_data")