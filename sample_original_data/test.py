import os

import pandas as pd

from sample_original_data.methods.read_and_sample_method import plot_trace_surface


def filter_trace_data_one(file_path,startFreq,stopFreq,resample_Freq):
    df = pd.read_csv(file_path)
    df = df.drop_duplicates()
    # processed_df = filter_and_resample_data(df, startFreq, stopFreq, resample_Freq)
    return df
    pass

def test():
    # file_path="E:\Work\Workspace\Data\电磁频谱数据\pictures\所里频谱迹线\\20240524000000_20240524235909_0.009_800.0.csv"
    # file_path="E:\Work\Workspace\Data\电磁频谱数据\pictures\所里频谱迹线\\20240525000000_20240525235951_0.009_800.0.csv"
    # file_path="E:\Work\Workspace\Data\电磁频谱数据\pictures\所里频谱迹线\\20240526000000_20240526235947_0.009_800.0.csv"
    file_path="E:\Work\Workspace\Data\电磁频谱数据\pictures\所里频谱迹线\\20240527000000_20240527235949_0.009_800.0.csv"

    messages=file_path.split("\\")

    html_save_path=os.path.join("../pictures",messages[-1].replace("csv","html"))
    df = pd.read_csv(file_path)
    df = df.drop_duplicates()


    plot_trace_surface(df,html_save_path)


    pass
test()