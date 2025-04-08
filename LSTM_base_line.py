import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from jinja2.utils import concat
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import re
from scipy.signal import savgol_filter


def preprocess_data_smooth(df):
    # 获取列的值
    values = df['signal_power_mean_mean'].tolist()

    # 遍历列表，从第二行开始
    for i in range(1, len(values)):
        if values[i] == 0:
            values[i] = values[i - 1]  # 用上一行的值替换

    # 更新DataFrame
    df['signal_power_mean_mean'] = values

    # 获取列的值
    values = df['freq_bandwidth_mean_mean'].tolist()
    # 遍历列表，从第二行开始
    for i in range(1, len(values)):
        if values[i] == 0:
            values[i] = values[i - 1]  # 用上一行的值替换
    # 更新DataFrame
    df['freq_bandwidth_mean_mean'] = values
    return df

def SavitzkyGolay(df):
    # Savitzky-Golay滤波器法
    window_length = 11  # 窗口长度，必须为奇数
    polyorder = 3  # 多项式阶数
    df['signal_power_mean_mean'] = savgol_filter(df['signal_power_mean_mean'], window_length, polyorder)
    df['emission_time_sum'] = savgol_filter(df['emission_time_sum'], window_length, polyorder)
    df['freq_bandwidth_mean_mean'] = savgol_filter(df['freq_bandwidth_mean_mean'], window_length, polyorder)
    return df

# 数据预处理函数
def preprocess_data(df):
    df=preprocess_data_smooth(df)
    df = SavitzkyGolay(df)

    # 创建新的数据列，包含四列数据值
    processed_data = df[['communication_num', 'signal_power_mean_mean', 'emission_time_sum', 'freq_bandwidth_mean_mean']].values.astype(
        np.float32)
    #频次基线，功率基线，发射时长基线，带宽基线

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(processed_data)

    return scaled_data,scaler

    #return processed_data


# 创建时间序列数据集
def create_dataset(data, seq_length=48, pred_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + pred_length, :])  # 取所有列的数据作为目标
    return np.array(X), np.array(y)


# 自定义数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=256, num_layers=4, output_size=4 * 24):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()  # 添加ReLU激活函数

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = out.view(-1, 24, 4)  # 将输出调整为(批次大小, 24, 4)

        # 对freq_bandwidth_mean_mean（第4列，索引为3）应用ReLU
        out[:, :, 3] = self.relu(out[:, :, 3])

        return out


# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs=1000, device='cpu'):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
    print('Training completed!')


def predict_and_invert(model, input_data, scaler, device='cpu'):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_data).float().unsqueeze(0).to(device)
        prediction = model(input_tensor)
        prediction = prediction.cpu().numpy()

    # 还原预测结果到原始尺度
    prediction_inverted = scaler.inverse_transform(prediction.reshape(-1, 4))  # 假设输入数据有5列

    return prediction_inverted


def plot_communication_num(df,df_origin):
    # 确保date列是datetime类型
    df['date'] = pd.to_datetime(df['date'])
    #df["communication_num"] = np.ceil(df["communication_num"])
    # 手动设置三条折线的颜色
    colors = ['blue', 'green', 'red']  # 可以根据需要修改颜色

    # 创建一个Plotly图形对象
    fig = go.Figure()

    df["communication_num_1"]=np.ceil(savgol_filter(df['communication_num']+4,11,3))#+np.random.choice([3, 5], size=len(df))
    df["communication_num_2"]=np.ceil((df_origin['communication_num'] - 4).clip(lower=0))

    # 添加三条折线到图表中
    fig.add_trace(
        go.Scatter(x=df['date'], y=df_origin['communication_num'], mode='lines', name='communication_num',
                   line=dict(color=colors[0], width=4)))
    fig.add_trace(go.Scatter(x=df['date'], y=df['communication_num_1'], mode='lines', name='communication_num_1',
                             line=dict(color=colors[1], width=1)))
    fig.add_trace(go.Scatter(x=df['date'], y=df['communication_num_2'], mode='lines', name='communication_num_2',
                             line=dict(color=colors[2], width=1)))
    # 更新布局，添加标题和轴标签
    fig.update_layout(
        title='频次基线',
        xaxis_title='时间',
        yaxis_title='频次',
        hovermode='x unified',
        template='plotly_white'  # 使用简洁的模板
    )
    fig.update_xaxes(range=[df.iloc[24, 4], df.iloc[84, 4]])
    # 显示图表
    fig.show()

def plot_signal_power_mean_mean(df,df_origin):
    df['date'] = pd.to_datetime(df['date'])
    # 手动设置三条折线的颜色
    colors = ['blue', 'green', 'red']  # 可以根据需要修改颜色

    # 创建一个Plotly图形对象
    fig = go.Figure()

    df["signal_power_mean_mean_1"] = df["signal_power_mean_mean"]+3#+np.random.uniform(1, 1.3, size=len(df))
    df["signal_power_mean_mean_2"] = df["signal_power_mean_mean"]-2#-np.random.uniform(1, 1.3, size=len(df))
    # 添加三条折线到图表中
    fig.add_trace(
        go.Scatter(x=df['date'], y=df_origin['signal_power_mean_mean'], mode='lines', name='signal_power_mean_mean', line=dict(color=colors[0], width=4)))
    fig.add_trace(go.Scatter(x=df['date'], y=df['signal_power_mean_mean_1'], mode='lines', name='signal_power_mean_mean_1', line=dict(color=colors[1], width=1)))
    fig.add_trace(go.Scatter(x=df['date'], y=df['signal_power_mean_mean_2'], mode='lines', name='signal_power_mean_mean_2', line=dict(color=colors[2], width=1)))

    # 更新布局，添加标题和轴标签
    fig.update_layout(
        title='功率基线',
        xaxis_title='时间',
        yaxis_title='功率',
        hovermode='x unified',
        template='plotly_white'  # 使用简洁的模板
    )
    fig.update_xaxes(range=[df.iloc[24,4], df.iloc[84,4]])

    # 显示图表
    fig.show()


def plot_emission_time_sum(df,df_origin):
    df['date'] = pd.to_datetime(df['date'])
    # 手动设置三条折线的颜色
    colors = ['blue', 'green', 'red']  # 可以根据需要修改颜色

    # 创建一个Plotly图形对象
    fig = go.Figure()

    df["emission_time_sum_1"] = savgol_filter(df["emission_time_sum"]*3,11,3)
    df["emission_time_sum_2"] = savgol_filter(df["emission_time_sum"]*0.5,11,3)#*np.random.uniform(0, 0.2, size=len(df))
    # 添加三条折线到图表中
    fig.add_trace(
        go.Scatter(x=df['date'], y=df_origin['emission_time_sum'], mode='lines', name='emission_time_sum', line=dict(color=colors[0], width=4)))
    fig.add_trace(go.Scatter(x=df['date'], y=df['emission_time_sum_1'], mode='lines', name='emission_time_sum_1', line=dict(color=colors[1], width=1)))
    fig.add_trace(go.Scatter(x=df['date'], y=df['emission_time_sum_2'], mode='lines', name='emission_time_sum_2', line=dict(color=colors[2], width=1)))

    # 更新布局，添加标题和轴标签
    fig.update_layout(
        title='发射时长基线',
        xaxis_title='时间',
        yaxis_title='发射时长',
        hovermode='x unified',
        template='plotly_white'  # 使用简洁的模板
    )
    fig.update_xaxes(range=[df.iloc[24,4], df.iloc[84,4]])

    # 显示图表
    fig.show()

def plot_freq_bandwidth_mean_mean(df,df_origin):
    df['date'] = pd.to_datetime(df['date'])
    # 手动设置三条折线的颜色
    colors = ['blue', 'green', 'red']  # 可以根据需要修改颜色

    # 创建一个Plotly图形对象
    fig = go.Figure()

    df["freq_bandwidth_mean_mean_1"] = df["freq_bandwidth_mean_mean"]*1.5#*np.random.uniform(1.1, 1.3, size=len(df))
    df["freq_bandwidth_mean_mean_2"] = df["freq_bandwidth_mean_mean"]*0.5#*np.random.uniform(0.7, 0.9, size=len(df))
    # 添加三条折线到图表中
    fig.add_trace(
        go.Scatter(x=df['date'], y=df_origin['freq_bandwidth_mean_mean'], mode='lines', name='freq_bandwidth_mean_mean', line=dict(color=colors[0], width=4)))
    fig.add_trace(go.Scatter(x=df['date'], y=df['freq_bandwidth_mean_mean_1'], mode='lines', name='freq_bandwidth_mean_mean_1', line=dict(color=colors[1], width=1)))
    fig.add_trace(go.Scatter(x=df['date'], y=df['freq_bandwidth_mean_mean_2'], mode='lines', name='freq_bandwidth_mean_mean_2', line=dict(color=colors[2], width=1)))

    # 更新布局，添加标题和轴标签
    fig.update_layout(
        title='带宽基线',
        xaxis_title='时间',
        yaxis_title='占用带宽',
        hovermode='x unified',
        template='plotly_white'  # 使用简洁的模板
    )
    fig.update_xaxes(range=[df.iloc[24,4], df.iloc[84,4]])

    # 显示图表
    fig.show()



def clean_time_dataframe(df, time_col='date'):
    """
    处理时间序列数据框，删除重复时间记录，按时间排序，并检查时间步长。

    参数:
    df (pd.DataFrame): 输入的数据框，第一列为时间。
    time_col (str): 时间列的列名，默认为'date'。

    返回:
    pd.DataFrame: 处理后的数据框。
    """
    # 确保时间列是datetime类型
    df[time_col] = pd.to_datetime(df[time_col])

    # 删除重复的时间记录


    df = df.drop_duplicates(subset=[time_col])

    # 按时间排序
    df = df.sort_values(by=time_col)

    # 检查时间步长是否为30分钟
    time_diffs = df[time_col].diff().dt.total_seconds() / 60  # 转换为分钟
    expected_diff = 30
    invalid_steps = time_diffs[time_diffs != expected_diff]

    if not invalid_steps.empty:
        #print("发现不符合30分钟步长的时间间隔:")
        for idx, diff in invalid_steps.items():
            #print(f"在索引 {idx} 处，时间间隔为 {diff} 分钟")
            pass
        # 根据需要处理不符合步长的记录，例如删除
        # df = df.drop(invalid_steps.index)
    return df

def plot_all(newdata_file):
    print("Loading--"+newdata_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(input_size=4, hidden_size=256, num_layers=4, output_size=4 * 24).to(device)
    # 加载模型权重
    modelname=re.split(r'[\\/]+',newdata_file)[-3]+"lstm_model_weights.pth"
    model.load_state_dict(torch.load(modelname))
    print('Model weights loaded--'+modelname)
    df_origin = pd.read_csv(newdata_file)
    column_names = ['communication_num', 'signal_power_mean_mean', 'emission_time_sum',
                    'freq_bandwidth_mean_mean']
    # print(df_origin)
    lastprediction = pd.DataFrame(columns=column_names)

    for i in range(int((df_origin.shape[0] - 48) / 24)):
        input_sequence ,scaler= preprocess_data(df_origin.iloc[0 + i * 24:48 + i * 24])
        prediction = predict_and_invert(model, input_sequence,scaler, device=device)
        df_pre = pd.DataFrame(prediction.reshape(24, 4),
                              columns=['communication_num', 'signal_power_mean_mean', 'emission_time_sum',
                                       'freq_bandwidth_mean_mean'])

        df_time = df_origin.loc[:, "date"].iloc[48 + i * 24:72 + i * 24]

        df_time = df_time.reset_index(drop=True)
        df_pre = df_pre.reset_index(drop=True)

        tmp = pd.concat(
            [df_time, df_pre],
            axis=1
        )
        # print(df_time)
        # print(df_pre)

        # print(tmp)

        lastprediction = pd.concat([lastprediction, tmp], axis=0)
    # new_data = preprocess_data(df_origin)  # 替换为你的新数据路径
    # input_sequence = new_data[-48:]  # 获取最后48个时间步的数据
    # prediction = predict(model, input_sequence, device=device)
    # print('Predicted future 24 time steps:', prediction)
    lastprediction = clean_time_dataframe(lastprediction, "date")
    lastprediction.to_csv("lastprediction.csv",index=False)

    df_origin_show = df_origin.iloc[48:48 + int((df_origin.shape[0] - 48) / 24) * 24]

    df_origin_show = preprocess_data_smooth(df_origin_show)
    df_origin_show = SavitzkyGolay(df_origin_show)

    df_origin_show = df_origin_show.reset_index(drop=True)
    lastprediction = lastprediction.reset_index(drop=True)

    plot_communication_num(lastprediction,df_origin_show)
    plot_signal_power_mean_mean(lastprediction,df_origin_show)
    plot_emission_time_sum(lastprediction,df_origin_show)
    plot_freq_bandwidth_mean_mean(lastprediction,df_origin_show)


# 主函数
if __name__ == '__main__':
    '''# 加载数据
    df = pd.read_csv("D:\iie\Python_Workspace\Time-Series-Library-main\数据集\电梯信号\coarse_grained_data\\train_coarse_grained_data.csv")
    df=df.iloc[42:]


    processed_data, scaler = preprocess_data(df)
    X, y = create_dataset(processed_data)

    # 划分训练集和测试集
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 创建数据加载器
    train_dataset = TimeSeriesDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 初始化模型、损失函数和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(input_size=4, hidden_size=256, num_layers=4, output_size=4 * 24).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs=100, device=device)

    # 保存模型权重
    torch.save(model.state_dict(), '电梯信号lstm_model_weights.pth')
    print('Model weights saved!')
    #-----------------------------------------------------------------------------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(input_size=4, hidden_size=256, num_layers=4, output_size=4 * 24).to(device)
    # 加载模型权重
    model.load_state_dict(torch.load('电梯信号lstm_model_weights.pth'))
    print('Model weights loaded!')


    newdata_file="D:\iie\Python_Workspace\Time-Series-Library-main\数据集\葛洲坝\coarse_grained_data\\train_coarse_grained_data.csv"
    # 使用新数据进行预测
    df_origin=pd.read_csv(newdata_file)
    df_origin=clean_time_dataframe(df_origin, "date")

    column_names = ['communication_num', 'signal_power_mean_mean', 'emission_time_sum',
                    'freq_bandwidth_mean_mean']
    #print(df_origin)
    lastprediction = pd.DataFrame(columns=column_names)


    for i in range(int((df_origin.shape[0]-48)/24)):
        input_sequence,scaler=preprocess_data(df_origin.iloc[0+i*24:48+i*24])
        prediction = predict_and_invert(model, input_sequence, scaler,device=device)
        df_pre=pd.DataFrame(prediction.reshape(24, 4),columns=['communication_num', 'signal_power_mean_mean', 'emission_time_sum',
                    'freq_bandwidth_mean_mean'])
        df_time=df_origin.loc[:,"date"].iloc[48 + i * 24:72 + i * 24]
        df_time=df_time.reset_index(drop=True)
        df_pre=df_pre.reset_index(drop=True)

        tmp = pd.concat(
            [df_time,df_pre],
            axis=1
        )


        #print(df_time)
        #print(df_pre)

        #print(tmp)

        lastprediction=pd.concat([lastprediction,tmp],axis=0)
    #new_data = preprocess_data(df_origin)  # 替换为你的新数据路径
    #input_sequence = new_data[-48:]  # 获取最后48个时间步的数据
    #prediction = predict(model, input_sequence, device=device)
    #print('Predicted future 24 time steps:', prediction)

    df_origin_show =df_origin.iloc[48:48 + int((df_origin.shape[0]-48)/24)*24]

    df_origin_show=preprocess_data_smooth(df_origin_show)
    df_origin_show=SavitzkyGolay(df_origin_show)


    #lastprediction=clean_time_dataframe(lastprediction,"date")
    lastprediction.to_csv("lastprediction.csv")

    lastprediction = lastprediction.reset_index(drop=True)
    df_origin_show = df_origin_show.reset_index(drop=True)

    print(lastprediction)
    print(df_origin_show)



    plot_communication_num(lastprediction, df_origin_show)
    plot_signal_power_mean_mean(lastprediction,df_origin_show)
    plot_emission_time_sum(lastprediction,df_origin_show)
    plot_freq_bandwidth_mean_mean(lastprediction,df_origin_show)'''
    plot_all(newdata_file="D:/iie/Python_Workspace/Time-Series-Library-main/数据集/葛洲坝/coarse_grained_data/test_coarse_grained_data.csv")