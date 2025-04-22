'''异常可视化'''
import pandas as pd
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import ast

def read_abnormal_label(abnormal_process_file):
    #df=pd.read_csv("./数据集/葛洲坝/raw_data/abnormal_label/abnormal_process.csv")
    df=pd.read_csv(abnormal_process_file)
    abnormal_label_list=[]
    for i in range(df.shape[0]):
        start_time=df.iloc[i,0]
        end_time=df.iloc[i,1]
        bandwidth_key=df.iloc[i,4]
        power=df.iloc[i,5]
        abnormal_generate_method=df.iloc[i,6]
        abnormal_class=df.iloc[i,7]

        abnormal_label=[start_time,end_time,bandwidth_key,abnormal_class]
        abnormal_label_list.append(abnormal_label)
        #print(f"{start_time} {end_time} {bandwidth_key} {power} {abnormal_generate_method} {abnormal_class}")
    return abnormal_label_list



def plot_waterfall_with_annotation(csv_file,abnormal_label_list):
    """
    绘制电磁信号的时频瀑布图，并在指定区域添加标注框

    参数:
    csv_file -- CSV文件路径
    start_time -- 标注区域的开始时间
    end_time -- 标注区域的结束时间
    bandwidth_key -- 标注区域的频段 [低频, 高频]
    """
    #print("abnormal_label_list:"+str(abnormal_label_list))
    # 读取CSV文件
    df = pd.read_csv(csv_file, sep=',', parse_dates=['date'])

    # 提取时间和功率值
    time = df['date'].tolist()
    frequencies = df.columns[1:-1].astype(float).values
    power_values = df.values[:, 1:-1].astype(float)

    # 创建瀑布图
    fig = make_subplots(rows=1, cols=1)

    # 绘制瀑布图
    heatmap = go.Heatmap(
        z=power_values,
        x=frequencies,
        y=time,
        colorscale='Viridis',
        colorbar=dict(title='功率 (dBm)')
    )

    fig.add_trace(heatmap)

    # 设置图表标题和轴标签
    fig.update_layout(
        title='电磁信号时频瀑布图',
        xaxis_title='频率 (GHz)',
        yaxis_title='时间',
        xaxis=dict(
            tickmode='linear',
            dtick=0.001
        ),
        yaxis=dict(
            tickformat='%Y/%m/%d %H:%M:%S'
        )
    )
    for i in range(len(abnormal_label_list)):
        start_time=abnormal_label_list[i][0]
        end_time=abnormal_label_list[i][1]
        bandwidth_key=abnormal_label_list[i][2]
        abnormal_class=abnormal_label_list[i][3]
        #print(abnormal_label_list[i])

        #这里还得加一步判断，万一bandwidth_key是空的呢？
        if isinstance(bandwidth_key, list):
            if len(bandwidth_key)==0:
                continue

        else:
            if len(ast.literal_eval(bandwidth_key)) == 0:
                continue
        # 将字符串时间转换为datetime
        start_time = pd.Timestamp(start_time)
        end_time = pd.Timestamp(end_time)
        # 找到时间范围内的索引
        time_indices = np.where((df['date'] >= start_time) & (df['date'] <= end_time))[0]
        time_range = [time[i] for i in time_indices]

        #print(bandwidth_key)
        #print(type(bandwidth_key))
        # 找到频率范围内的索引
        if isinstance(bandwidth_key, list):
            freq_low = float(bandwidth_key[0])
            freq_high = float(bandwidth_key[-1])
        else:
            freq_low = float(ast.literal_eval(bandwidth_key)[0])
            freq_high = float(ast.literal_eval(bandwidth_key)[-1])
        freq_indices = np.where((frequencies >= freq_low) & (frequencies <= freq_high))[0]
        freq_range = [frequencies[i] for i in freq_indices]

        if len(time_range) > 0 and len(freq_range) > 0:
            # 添加矩形标注
            fig.add_shape(
                type='rect',
                x0=freq_low, y0=start_time,
                x1=freq_high, y1=end_time,
                line=dict(
                    color='Red',
                    width=2
                ),
                fillcolor='Red',
                opacity=0.2
            )

            # 添加文本标注
            fig.add_annotation(
                x=(freq_low + freq_high) / 2,
                y=(start_time + (end_time - start_time) / 2),
                #text=f'标注区域',
                text=abnormal_class,
                showarrow=False,
                font=dict(
                    color='Black',
                    size=12
                )
            )
    # 显示图表
    fig.show()


# 示例使用
if __name__ == "__main__":
    abnormal_label_list=read_abnormal_label("../数据集/葛洲坝/raw_data/abnormal_label/abnormal_process.csv")
    # 调用函数绘制图表
    plot_waterfall_with_annotation("../数据集/葛洲坝/fine_grained_data/test_fine_grained_data.csv",abnormal_label_list)