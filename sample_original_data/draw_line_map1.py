'''专业画折线图
画各种乱七八糟的图'''
'''绘制折线图'''
import pandas as pd
import plotly.express as px
import os
import csv
import matplotlib.pyplot as plt
import plotly.graph_objects as go

'''画一个折线图'''


def draw_line_map(file_path, column_name,file_name):
    '''
    绘制折线图
    :param file_path:
    :param column_name:
    :param file_name:保存的文件名
    :return:
    '''
    # 读取CSV文件
    # file_path = 'D:\\0ProjectFiles\预处理数据\\20240628-0705船闸闸室或阀室数据\\0_30MHz原始信号数据\\20240628113749_20240628173749_0.009_15.0045.csv'
    # column_name = '7.902355'  # 替换为你要读取的列名

    # 使用pandas读取CSV文件
    data = pd.read_csv(file_path)

    # 删除第一行（通常是表头）
    data = data.drop(0)

    # 提取时间列（假设时间信息在第一列）
    time_column = data.iloc[:, 0]  # 第一列是时间信息

    # 提取指定列的数据（强度信息）
    column_data = data[column_name]
    print("提取信息：" + str(column_name))

    # 将列数据转换为数值类型，并忽略非数值数据
    column_data = pd.to_numeric(column_data, errors='coerce')

    # 删除NaN值（非数值数据会被转换为NaN）
    column_data = column_data.dropna()

    # 使用Plotly创建交互式折线图
    fig = px.line(x=time_column, y=column_data, title="", labels={'x': '时间', 'y': '电磁辐射强度'})

    # 设置x轴和y轴的范围（可选）
    fig.update_xaxes(range=[time_column.min(), time_column.max()])  # 设置x轴范围
    fig.update_yaxes(range=[column_data.min(), column_data.max()])  # 设置y轴范围

    # 将图表保存为HTML文件
    fig.write_html(file_name)
    print("图表已保存")
    # 显示图表
    fig.show()


def plot_signal_state(csv_file_path, output_html_path):
    """
    绘制信号状态的交互式折线图并保存为HTML文件。

    参数:
    csv_file_path: str, CSV文件路径，文件应包含两列：'date'和'signal'。
    output_html_path: str, 输出HTML文件的路径。
    """
    # 读取CSV文件
    data = pd.read_csv(csv_file_path)
    data['date'] = pd.to_datetime(data['date'])  # 转换日期列为datetime类型

    # 创建Plotly图表
    fig = go.Figure()

    # 添加折线图数据
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['signal'],
        mode='lines+markers',
        line=dict(shape='hv'),  # 使用水平和垂直线段连接数据点
        hoverinfo='x+y',
        name='Signal State'
    ))

    # 更新图表布局
    fig.update_layout(
        title='Signal State over Time',
        xaxis_title='Time',
        yaxis_title='Signal State',
        yaxis=dict(
            tickvals=[0, 1],  # 设置y轴刻度值
            ticktext=["Signal Not Existing", "Signal Existing"]  # 设置y轴刻度标签
        ),
        hovermode='closest'
    )

    # 显示图表
    fig.show()

    # 保存图表为HTML文件
    fig.write_html(output_html_path)




if __name__=="__main__":
    #CSVfile_merge("E:\预处理\\0-15MHz","tf.csv")
    #draw_line_map("tf.csv","7.902355")
    #draw_line_map("E:\预处理\\0-15MHz\\20240628173749_20240628233749_0.009_15.0045.csv", "7.902355")
    #draw_line_map("E:\预处理\\0-15MHz——7.902355\\20240628173749_20240628233749_0.009_15.0045.csv", "7.902355")
    #draw_line_map("E:\预处理\\0-15MHz——离群点清洗后\\20240630053749_20240630113749_0.009_15.0045.csv", "7.902355","原始信号均值20240630053749_20240630113749_0.009_15.0045.html")
    #draw_line_map("E:\预处理\\0-15MHz——01化\\tf_离群点清洗后.csv", "signal")

    '''draw_line_map("E:\预处理2\葛洲坝\\0_10MHz按天分割后的数据_离群点清洗后\\20240628.csv", "Average",
                  "20240628_origin.html")
    plot_signal_state("E:\预处理2\葛洲坝\\0_10MHz按天分割后的数据_01化\\20240628.csv","信号检测20240628.html")
    '''

    draw_line_map("D:\\0-30MHz电梯\信息工程研究所\\离群点清洗后\\20240925171821_20240925231821_0.009_30.0.csv", "1.673917",
                  "20240925_origin.html")
    #plot_signal_state("E:\预处理2\葛洲坝\\0_10MHz按天分割后的数据_01化\\20240702.csv", "信号检测20240702.html")

