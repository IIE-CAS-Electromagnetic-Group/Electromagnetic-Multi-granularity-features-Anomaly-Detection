'''该文件的作用就是进行01化处理
采用的方法可能有点复杂'''


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
from scipy.ndimage import binary_dilation, binary_erosion


def extract_features(df, target="7.902355",window_size='60S'):
    """提取滑动窗口特征"""
    rolling = df[target].rolling(window=window_size, min_periods=1)
    features = pd.DataFrame({
        'mean': rolling.mean(),
        'std': rolling.std(),
        'max': rolling.max(),
        'min': rolling.min(),
        'range': rolling.max() - rolling.min()
    })
    # 填充缺失值（如果有）
    features = features.fillna(method='ffill').fillna(method='bfill')
    return features


def cluster_signals(features):
    """使用K-means聚类区分信号和噪声"""
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(features)
    # 确保标签0是噪声，1是信号
    if features[labels == 0]['max'].mean() > features[labels == 1]['max'].mean():
        labels = 1 - labels
    return labels


def post_process(labels, min_duration=60, max_gap=30):
    """后处理确保信号连续性"""
    labels = labels.astype(bool)
    # 形态学处理填充小间隙
    structure = np.ones(max_gap, dtype=bool)
    labels = binary_dilation(labels, structure=structure)
    labels = binary_erosion(labels, structure=structure)
    # 移除短时信号
    structure = np.ones(min_duration, dtype=bool)
    labels = binary_erosion(labels, structure=structure)
    labels = binary_dilation(labels, structure=structure)
    return labels.astype(int)


def process_file(input_path, output_path,target):
    # 读取数据
    df = pd.read_csv(input_path, sep=None, engine='python', parse_dates=['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)

    # 特征提取
    features = extract_features(df,target)

    # 聚类分析
    labels = cluster_signals(features)

    # 后处理
    processed_labels = post_process(labels)

    # 确保长度一致
    if len(features.index) != len(processed_labels):
        raise ValueError(
            f"Length mismatch: features.index ({len(features.index)}) != processed_labels ({len(processed_labels)})")

    # 赋值
    print(len(features.index), len(post_process(labels)))
    df['signal'] = 0
    df.loc[features.index, 'signal'] = processed_labels


    # 保存结果
    df.reset_index()[['date', 'signal']].to_csv(output_path, sep=',', index=False)
    print("保存结果")

def process_all_files(input_dir, output_dir,target):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_file in input_dir.glob('*.csv'):
        print("开始处理:"+str(input_file))
        output_path = output_dir / input_file.name
        process_file(input_file, output_path,target)


# 使用示例
if __name__ == "__main__":
    '''只需要关注三个参数：
    input_folder
    output_folder
    target="Average"，也就是对哪一列进行01化
    '''
    #input_folder = "E:\预处理2\葛洲坝\\0_10MHz按天分割后的数据_离群点清洗后"  # 替换为你的输入文件夹路径
    #output_folder = "E:\预处理2\葛洲坝\\0_10MHz按天分割后的数据_01化"  # 替换为你的输出文件夹路径
    #process_all_files(input_folder, output_folder, target="Average")
    input_folder = "D:\\0-30MHz电梯\信息工程研究所\\离群点清洗后"  # 替换为你的输入文件夹路径
    output_folder = "D:\\0-30MHz电梯\信息工程研究所\\01化后"  # 替换为你的输出文件夹路径

    process_all_files(input_folder, output_folder,target="1.673917")