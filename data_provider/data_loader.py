'''
包含多个数据集加载类，如 `Dataset_ETT_hour`、`Dataset_Custom`、`IntercomSegLoader`、`SignalCoarsePreLoader` 等。
这些类负责加载预处理后的数据，并将其转换为模型训练所需的格式（如时间序列窗口、标准化处理、时间特征编码等）。'''
import math
import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from data_provider.intercom import *
from data_provider.signal import read_coarse_grained_data, read_fine_grained_data, \
    read_signal_coarse_continuous_data, read_signal_fine_continuous_data
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe

import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_path, file_list=None, limit_size=None, flag=None):
        self.root_path = root_path
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern='*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        return self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)), \
               torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)

    def __len__(self):
        return len(self.all_IDs)


class IntercomSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1,flag="train",scale=True, timeenc=0, freq='s'):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.root_path=root_path
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.__read_data__()




    def __read_data__(self):
        root_path=self.root_path
        win_size=self.win_size
        # 或者将每个窗口设置为一半
        # left_win_size, right_win_size
        train_raw_df=read_data(root_path,win_size,win_size,flag="train")
        test_raw_df = read_data(root_path, win_size,win_size, flag="test")
        self.scaler = StandardScaler()
        # 分字段进行处理
        train_cols=list(train_raw_df.columns)
        train_cols.remove('date')
        train_data_df= train_raw_df[train_cols]

        test_label_df=test_raw_df[['date','label']]
        test_cols= list(test_raw_df.columns)
        test_cols.remove('label')
        test_raw_df=test_raw_df[test_cols]


        if(self.flag=="train" or self.flag=="val"):
            raw_df=train_raw_df
        else:
            raw_df = test_raw_df

        cols=list(raw_df.columns)
        cols.remove('date')
        data_df = raw_df[cols]
        stamp_df=raw_df[['date']]
        stamp_df['date'] = pd.to_datetime(stamp_df.date)

        if self.scale:
            self.scaler.fit(train_data_df.values)
            data=self.scaler.transform(data_df.values)

        else:
            data = data_df.values
        if self.timeenc == 0:
            stamp_df['month'] = stamp_df.date.apply(lambda row: row.month, 1)
            stamp_df['day'] = stamp_df.date.apply(lambda row: row.day, 1)
            stamp_df['weekday'] = stamp_df.date.apply(lambda row: row.weekday(), 1)
            stamp_df['hour'] = stamp_df.date.apply(lambda row: row.hour, 1)
            stamp_df['minute'] = stamp_df.date.apply(lambda row: row.minute, 1)
            stamp_df['second'] = stamp_df.date.apply(lambda row: row.second, 1)
            data_stamp = stamp_df.drop(['date'], 1).values

        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(stamp_df['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        new_data_df=pd.concat([raw_df[['date']].reset_index(drop=True),pd.DataFrame(data=data, columns=cols).reset_index(drop=True)],axis=1)
        data=read_continuous_data(new_data_df,win_size)

        new_stamp_df=pd.concat([raw_df[['date']].reset_index(drop=True),pd.DataFrame(data=data_stamp , columns=['second','minute','hour','day','weekday','month',]).reset_index(drop=True)],axis=1)
        data_stamp=read_continuous_data(new_stamp_df,win_size)

        if(self.flag=="train"):
            data_len=len(data)
            self.data=data[:(int)(data_len * 0.8)]
            self.data_stamp=data_stamp[:(int)(data_len * 0.8)]
        elif(self.flag=="val"):
            data_len=len(data)
            self.data=data[(int)(data_len * 0.8):]
            self.data_stamp = data_stamp[(int)(data_len * 0.8):]
        else:
            self.data = data
            self.data_stamp = data_stamp
        self.test_labels= read_continuous_data(test_label_df, win_size)
        pass
    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        if self.flag == "train" or self.flag == 'val':
            return np.float32(self.data[index]), np.float32(self.test_labels[0])
        elif (self.flag == 'test'):
            return np.float32(self.data[index]), np.float32(self.test_labels[index])
        else:
            return np.float32(self.data[index ]), np.float32(self.test_labels[index])

class IntercomPreLoader(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val','test_forecast_and_anomaly_detection']
        type_map = {'train': 0, 'val': 1, 'test': 2,'test_forecast_and_anomaly_detection':3}
        self.set_type = type_map[flag]

        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        win_size = self.seq_len + self.pred_len
        left_win_size=self.seq_len
        right_win_size=self.pred_len
        # half_win_size = math.ceil(win_size/2)
        # df_raw=read_data(self.root_path,half_win_size,flag="train")
        train_predict_raw_df=read_data(self.root_path,left_win_size,right_win_size,flag="train")
        abnormal_detection_df=read_data(self.root_path,left_win_size,right_win_size,flag="test")


        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''

        train_predict_cols = list(train_predict_raw_df.columns)
        train_predict_cols.remove(self.target)
        train_predict_cols.remove('date')
        train_predict_raw_df = train_predict_raw_df[['date'] + train_predict_cols + [self.target]]


        test_label_df = abnormal_detection_df[['date', 'label']]
        test_cols = list(abnormal_detection_df.columns)
        test_cols.remove('label')
        test_cols.remove('date')
        test_cols.remove(self.target)
        test_raw_df = abnormal_detection_df[['date'] + test_cols + [self.target]]


        # cols = list(df_raw.columns)
        # cols.remove(self.target)
        # cols.remove('date')
        # df_raw = df_raw[['date'] + cols + [self.target]]

        # num_train = int(len(df_raw) * 0.7)
        # num_test = int(len(df_raw) * 0.2)
        # num_vali = len(df_raw) - num_train - num_test
        # border1s = [0, num_train, len(df_raw) - num_test]
        # border2s = [num_train, num_train + num_vali, len(df_raw)]

        num_train = int(len(train_predict_raw_df) * 0.7)
        num_test = int(len(train_predict_raw_df) * 0.2)
        num_vali = len(train_predict_raw_df) - num_train - num_test
        border1s = [0, num_train, len(train_predict_raw_df) - num_test]
        border2s = [num_train, num_train + num_vali, len(train_predict_raw_df)]

        if(self.flag in ['train', 'test', 'val']):
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            raw_df=train_predict_raw_df.iloc[border1:border2]
        else:
            raw_df=test_raw_df

        if self.features == 'M' or self.features == 'MS':
            data_cols = raw_df.columns[1:]
            df_data = raw_df[data_cols]
            train_data =train_predict_raw_df[data_cols][border1s[0]:border2s[0]]
        elif self.features == 'S':
            df_data = raw_df[[self.target]]
            train_data = train_predict_raw_df[[self.target]][border1s[0]:border2s[0]]

        if self.scale:
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = raw_df[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['second'] = df_stamp.date.apply(lambda row: row.second, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        cols=list(raw_df.columns)
        cols.remove('date')
        if(self.features == 'S'):
            cols=[self.target]
        new_data_df=pd.concat([raw_df[['date']].reset_index(drop=True),pd.DataFrame(data=data, columns=cols).reset_index(drop=True)],axis=1)
        data=read_continuous_data(new_data_df,win_size)

        new_stamp_df=pd.concat([raw_df[['date']].reset_index(drop=True),pd.DataFrame(data=data_stamp , columns=['second','minute','hour','day','weekday','month',]).reset_index(drop=True)],axis=1)
        data_stamp=read_continuous_data(new_stamp_df,win_size)
        test_labels = read_continuous_data(test_label_df, win_size)



        # if(self.flag in ['train', 'test', 'val']):
        #     border1 = border1s[self.set_type]
        #     border2 = border2s[self.set_type]
        #     if self.scale:
        #         train_data = df_data[border1s[0]:border2s[0]]
        #         self.scaler.fit(train_data.values)
        #         data = self.scaler.transform(df_data.values)
        #     else:
        #         data = df_data.values
        #     df_stamp = raw_df[['date']][border1:border2]
        #     df_stamp['date'] = pd.to_datetime(df_stamp.date)
        #     if self.timeenc == 0:
        #         df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        #         df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        #         df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        #         df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        #         df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
        #         df_stamp['second'] = df_stamp.date.apply(lambda row: row.second, 1)
        #         data_stamp = df_stamp.drop(['date'], 1).values
        #     elif self.timeenc == 1:
        #         data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
        #         data_stamp = data_stamp.transpose(1, 0)
        #     data_segments=data[border1:border2]
        #     if self.features == 'M' or self.features == 'MS':
        #         cols = list(raw_df.columns)
        #         cols.remove('date')
        #     elif self.features == 'S':
        #         cols=[self.target]
        #     data_segments_df=pd.DataFrame(data=data_segments, columns=cols)
        #     scale_data_segments_df=pd.concat([df_raw[['date']][border1:border2].reset_index(drop=True),data_segments_df.reset_index(drop=True)],axis=1)
        #     data=read_continuous_data(scale_data_segments_df,win_size)
        #
        #     stamp_segments_df=pd.concat([df_raw[['date']][border1:border2].reset_index(drop=True),
        #                                  pd.DataFrame(data=data_stamp ,columns=['second','minute','hour','day','weekday','month']).reset_index(drop=True)],axis=1)
        #     data_stamp=read_continuous_data(stamp_segments_df,win_size)
        #     test_labels = np.zeros(self.pred_len)
        #
        # else:
        #     test_raw_df=read_data(self.root_path,half_win_size,flag="test")
        #     test_label_df = test_raw_df[['date', 'label']]
        #
        #
        #     test_cols = list(test_raw_df.columns)
        #     test_cols.remove('label')
        #     test_cols.remove('date')
        #     test_cols.remove(self.target)
        #
        #     test_raw_df=test_raw_df[['date'] + test_cols + [self.target]]
        #
        #     # test_stamp_df = test_raw_df[['date']]
        #     # test_stamp_df['date'] = pd.to_datetime(test_stamp_df.date)
        #
        #     if self.features == 'M' or self.features == 'MS':
        #         cols_data = test_raw_df.columns[1:]
        #         test_data_df = test_raw_df[cols_data]
        #     elif self.features == 'S':
        #         test_data_df = test_raw_df[[self.target]]
        #
        #     if self.scale:
        #         train_data = df_data[border1s[0]:border2s[0]]
        #         self.scaler.fit(train_data.values)
        #         data = self.scaler.transform(test_data_df.values)
        #     else:
        #         data = test_data_df.values
        #
        #     df_stamp =  test_raw_df[['date']]
        #     df_stamp['date'] = pd.to_datetime(df_stamp.date)
        #     if self.timeenc == 0:
        #         df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        #         df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        #         df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        #         df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        #         data_stamp = df_stamp.drop(['date'], 1).values
        #     elif self.timeenc == 1:
        #         data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
        #         data_stamp = data_stamp.transpose(1, 0)
        #
        #     data_segments = data
        #     if self.features == 'M' or self.features == 'MS':
        #         cols = list(test_raw_df.columns)
        #         cols.remove('date')
        #     elif self.features == 'S':
        #         cols = [self.target]
        #     data_segments_df = pd.DataFrame(data=data_segments, columns=cols)
        #     scale_data_segments_df = pd.concat(
        #         [test_raw_df[['date']].reset_index(drop=True), data_segments_df.reset_index(drop=True)],
        #         axis=1)
        #     data = read_continuous_data(scale_data_segments_df, win_size)
        #     stamp_segments_df = pd.concat([test_raw_df[['date']].reset_index(drop=True),
        #                                    pd.DataFrame(data=data_stamp,columns=['second', 'minute', 'hour', 'day', 'weekday','month']).reset_index(drop=True)],axis=1)
        #     data_stamp = read_continuous_data(stamp_segments_df, win_size)
        #     test_labels = read_continuous_data(test_label_df, win_size)

        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp
        self.test_labels=test_labels

    def __getitem__(self, index):

        seq_x = np.float32(self.data_x[index][:self.seq_len])
        seq_y = np.float32(self.data_y[index][self.seq_len- self.label_len:self.seq_len+self.pred_len])
        seq_x_mark = np.float32(self.data_stamp[index][:self.seq_len])
        seq_y_mark = np.float32(self.data_stamp[index][self.seq_len- self.label_len:self.seq_len+self.pred_len])

        if (self.flag in ['train', 'test', 'val']):
            seq_label=np.float32(self.test_labels)
        else:
            seq_label = np.float32(self.test_labels[index][self.seq_len:self.seq_len+self.pred_len])

        return seq_x, seq_y, seq_x_mark, seq_y_mark,seq_label
        # return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class SignalCoarsePreLoader(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        '''
        初始化方法的主要作用是设置数据加载器的基本参数，并调用 __read_data__ 方法来加载和预处理数据。
        :param root_path:数据文件的根路径。
        :param flag:数据集的类型，可以是 'train', 'test', 'val', 或 'test_forecast_and_anomaly_detection'。
        :param size:一个包含三个元素的列表 [seq_len, label_len, pred_len]，分别表示序列长度、标签长度和预测长度。
        :param features:特征类型，可以是 'S'（单变量）、'M'（多变量）或 'MS'（多变量，但只预测一个目标）。
        :param data_path:数据文件的具体路径。
        :param target:目标变量的列名。
        :param scale:是否对数据进行标准化。
        :param timeenc:时间特征的编码方式，0 表示使用简单的日期特征，1 表示使用更复杂的时间特征。
        :param freq:时间序列的频率，例如 'h' 表示小时。
        :param seasonal_patterns:季节性模式，用于某些特定的数据集。
        '''
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val','test_forecast_and_anomaly_detection']
        type_map = {'train': 0, 'val': 1, 'test': 2,'test_forecast_and_anomaly_detection':3}
        self.set_type = type_map[flag]

        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        '''
        这个方法的主要作用是加载数据文件，进行预处理，并生成可用于训练和测试的数据结构。
        :return:
        '''
        self.scaler = StandardScaler()
        win_size = max(self.seq_len , self.pred_len)
        left_win_size=self.seq_len
        right_win_size=self.pred_len
        # half_win_size = math.ceil(win_size/2)
        # df_raw=read_data(self.root_path,half_win_size,flag="train")
        train_predict_raw_df,dataset_parameter=read_coarse_grained_data(self.root_path,left_win_size,right_win_size,flag="train")

        # abnormal_detection_df=read_data(self.root_path,left_win_size,right_win_size,flag="test")
        abnormal_detection_df,_=read_coarse_grained_data(self.root_path,left_win_size,right_win_size,flag="test_forecast_and_anomaly_detection")


        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''

        train_predict_cols = list(train_predict_raw_df.columns)
        train_predict_cols.remove(self.target)
        train_predict_cols.remove('date')
        train_predict_cols.remove('rule')
        # train_predict_rule_df= train_predict_raw_df[['date','rule']]
        train_predict_raw_df = train_predict_raw_df[['date','rule'] + train_predict_cols + [self.target]]


        test_label_df = abnormal_detection_df[['date','rule', 'label']]
        # test_rule_df=abnormal_detection_df[['date', 'rule']]
        test_cols = list(abnormal_detection_df.columns)
        test_cols.remove('label')
        test_cols.remove('date')
        test_cols.remove('rule')
        test_cols.remove(self.target)
        test_raw_df = abnormal_detection_df[['date','rule'] + test_cols + [self.target]]

        num_train = int(len(train_predict_raw_df) * 0.7)
        num_test = int(len(train_predict_raw_df) * 0.2)
        num_vali = len(train_predict_raw_df) - num_train - num_test
        border1s = [0, num_train, len(train_predict_raw_df) - num_test]
        border2s = [num_train, num_train + num_vali, len(train_predict_raw_df)]

        if(self.flag in ['train', 'test', 'val']):
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            raw_df=train_predict_raw_df.iloc[border1:border2]

        else:
            raw_df=test_raw_df

        if self.features == 'M' or self.features == 'MS':
            data_cols = raw_df.columns[2:]
            df_data = raw_df[data_cols]
            train_data =train_predict_raw_df[data_cols][border1s[0]:border2s[0]]
        elif self.features == 'S':
            df_data = raw_df[[self.target]]
            train_data = train_predict_raw_df[[self.target]][border1s[0]:border2s[0]]

        if self.scale:
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = raw_df[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            # df_stamp['second'] = df_stamp.date.apply(lambda row: row.second, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        cols=list(raw_df.columns)
        cols.remove('date')
        cols.remove('rule')
        if(self.features == 'S'):
            cols=[self.target]
        new_data_df=pd.concat([raw_df[['date','rule']].reset_index(drop=True),pd.DataFrame(data=data, columns=cols).reset_index(drop=True)],axis=1)

        data,date_data=read_signal_coarse_continuous_data(new_data_df,win_size,dataset_parameter)
        new_stamp_df=pd.concat([raw_df[['date','rule']].reset_index(drop=True),pd.DataFrame(data=data_stamp , columns=['minute','hour','day','weekday','month',]).reset_index(drop=True)],axis=1)
        data_stamp,_=read_signal_coarse_continuous_data(new_stamp_df,win_size,dataset_parameter)

        test_labels ,_= read_signal_coarse_continuous_data(test_label_df, win_size,dataset_parameter)




        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp
        self.date_data=date_data
        self.test_labels=test_labels
        self.start_index = (max(self.seq_len,self.pred_len) - self.pred_len) // 2
        self.end_index = self.start_index + self.pred_len

    def __getitem__(self, index):
        '''
        这个方法的作用是根据索引 index 提取数据，返回一个样本的输入、输出、时间特征和标签。
        :param index:
        :return:
        '''
        # seq_x = np.float32(self.data_x[index][:self.seq_len])
        # seq_y = np.float32(self.data_y[index][self.seq_len- self.label_len:self.seq_len+self.pred_len])
        # seq_x_mark = np.float32(self.data_stamp[index][:self.seq_len])
        # seq_y_mark = np.float32(self.data_stamp[index][self.seq_len- self.label_len:self.seq_len+self.pred_len])



        seq_x = np.float32(self.data_x[index][0][:self.seq_len])
        seq_y = np.float32(np.concatenate((self.data_x[index][0][self.seq_len- self.label_len:],self.data_y[index][1][self.start_index: self.end_index ]), axis=0))

        #print("self.data_stamp[index][0]:")
        #print(self.data_stamp[index][0])

        seq_x_mark = np.float32(self.data_stamp[index][0][:self.seq_len])
        seq_y_mark = np.float32(np.concatenate((self.data_stamp[index][0][self.seq_len- self.label_len:],self.data_stamp[index][1][self.start_index: self.end_index ]), axis=0))
        if (self.flag in ['train', 'test', 'val']):
            seq_label=np.float32(self.test_labels[0][self.start_index: self.end_index ])
            return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_label
        else:
            seq_label = np.float32(self.test_labels[index][1][self.start_index: self.end_index ])
            seq_x_date= self.date_data[index][0][:self.seq_len]
            # print(type(seq_x_date[0,0]))
            seq_y_date= self.date_data[index][1][self.start_index: self.end_index ]
            return seq_x, seq_y, seq_x_mark, seq_y_mark,seq_label,seq_x_date,seq_y_date
        # return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class SignalFineSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1,flag="train",scale=True, timeenc=0, freq='s'):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.root_path=root_path
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.__read_data__()
    def __read_data__(self):
        root_path=self.root_path
        win_size=self.win_size
        # 或者将每个窗口设置为一半
        # left_win_size, right_win_size
        train_raw_df,dataset_parameter=read_fine_grained_data(root_path,win_size//2,win_size//2,flag="train")

        test_raw_df,_ = read_fine_grained_data(root_path, win_size//2,win_size//2, flag="test")
        self.scaler = StandardScaler()
        # 分字段进行处理
        train_cols=list(train_raw_df.columns)
        train_cols.remove('date')
        train_data_df= train_raw_df[train_cols]

        test_label_df=test_raw_df[['date','label']]
        test_cols= list(test_raw_df.columns)
        test_cols.remove('label')
        test_raw_df=test_raw_df[test_cols]


        if(self.flag=="train" or self.flag=="val"):
            raw_df=train_raw_df
        else:
            raw_df = test_raw_df

        cols=list(raw_df.columns)
        cols.remove('date')
        data_df = raw_df[cols]
        stamp_df=raw_df[['date']]
        stamp_df['date'] = pd.to_datetime(stamp_df.date)

        if self.scale:
            self.scaler.fit(train_data_df.values)
            data=self.scaler.transform(data_df.values)

        else:
            data = data_df.values
        if self.timeenc == 0:
            stamp_df['month'] = stamp_df.date.apply(lambda row: row.month, 1)
            stamp_df['day'] = stamp_df.date.apply(lambda row: row.day, 1)
            stamp_df['weekday'] = stamp_df.date.apply(lambda row: row.weekday(), 1)
            stamp_df['hour'] = stamp_df.date.apply(lambda row: row.hour, 1)
            stamp_df['minute'] = stamp_df.date.apply(lambda row: row.minute, 1)
            stamp_df['second'] = stamp_df.date.apply(lambda row: row.second, 1)
            data_stamp = stamp_df.drop(['date'], 1).values

        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(stamp_df['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        new_data_df=pd.concat([raw_df[['date']].reset_index(drop=True),pd.DataFrame(data=data, columns=cols).reset_index(drop=True)],axis=1)
        data,date_data=read_signal_fine_continuous_data(new_data_df,win_size)

        new_stamp_df=pd.concat([raw_df[['date']].reset_index(drop=True),pd.DataFrame(data=data_stamp , columns=['second','minute','hour','day','weekday','month',]).reset_index(drop=True)],axis=1)
        data_stamp,_=read_signal_fine_continuous_data(new_stamp_df,win_size)

        if(self.flag=="train"):
            data_len=len(data)
            self.data=data[:(int)(data_len * 0.8)]
            self.data_stamp=data_stamp[:(int)(data_len * 0.8)]
            self.date_data=date_data[:(int)(data_len * 0.8)]
        elif(self.flag=="val"):
            data_len=len(data)
            self.data=data[(int)(data_len * 0.8):]
            self.data_stamp = data_stamp[(int)(data_len * 0.8):]
            self.date_data=date_data[(int)(data_len * 0.8):]

        else:
            self.data = data
            self.data_stamp = data_stamp
            self.date_data=date_data
        self.test_labels,_= read_signal_fine_continuous_data(test_label_df, win_size)
        pass
    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        if self.flag == "train" or self.flag == 'val':
            return np.float32(self.data[index]), np.float32(self.test_labels[0])
        elif (self.flag == 'test'):
            return np.float32(self.data[index]), np.float32(self.test_labels[index])
        else:
            return np.float32(self.data[index ]), np.float32(self.test_labels[index]),self.date_data[index]

