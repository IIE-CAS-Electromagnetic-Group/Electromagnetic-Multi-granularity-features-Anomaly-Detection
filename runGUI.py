import multiprocessing
import tkinter as tk
from tkinter import ttk, scrolledtext
import argparse
import os
import torch
import random
import numpy as np
import sys
import threading
from exp.exp_forecast_and_anomaly_detection import Exp_Forecast_And_Anomaly_Detection
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
from LSTM_base_line import plot_all


class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, str):
        self.text_widget.insert(tk.END, str)
        self.text_widget.see(tk.END)

    def flush(self):
        pass


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("TimesNet GUI")

        # 创建参数输入框
        self.create_input_frame()

        # 创建运行按钮
        self.create_button_frame()

        # 创建输出框
        self.create_output_frame()

        # 初始化参数
        self.args = None
        self.running = False

    def create_input_frame(self):
        input_frame = ttk.LabelFrame(self.root, text="Parameters")
        input_frame.pack(fill="x", expand="yes", padx=10, pady=10)

        # 定义参数输入框
        parameters = [
            ("task_name", "Task Name", "forecast_and_anomaly_detection"),
            ("is_training", "Is Training", "2"),
            ("root_path", "Root Path", "./数据集/葛洲坝"),
            ("model_id", "Model ID", "GezhoubaPred_3_25"),
            ("model", "Model", "Transformer"),
            ("data", "Data", "SignalCoarsePred"),
            #("target", "Target", "communication_num"),
            #("freq", "Freq", "t"),
            #("features", "Features", "M"),
            #("learning_rate", "Learning Rate", "0.001"),
            #("seq_len", "Seq Len", "10"),
            #("label_len", "Label Len", "1"),
            #("pred_len", "Pred Len", "3"),
            #("e_layers", "E Layers", "2"),
            #("d_layers", "D Layers", "1"),
            #("factor", "Factor", "3"),


            #("enc_in", "Enc In", "86"),
            #("dec_in", "Dec In", "86"),
            #("c_out", "C Out", "86"),

            ("enc_in","采样点数量","86"),

            #("d_model", "D Model", "256"),
            #("d_ff", "D FF", "512"),
            #("des", "Des", "'Exp'"),
            #("itr", "Itr", "1"),
            #("batch_size", "Batch Size", "128"),
            #("anomaly_ratio", "Anomaly Ratio", "1"),
            #("patience", "Patience", "8"),
            ("train_epochs", "Train Epochs", "50")
        ]

        self.entries = {}
        for i, (key, label, default) in enumerate(parameters):
            row = i // 3
            col = i % 3
            lbl = ttk.Label(input_frame, text=label)
            lbl.grid(row=row, column=col * 2, padx=5, pady=5, sticky="w")
            if key == "model_id":
                # 使用Combobox作为选择框
                entry = ttk.Combobox(input_frame, width=28, state="readonly")
                entry['values'] = ("GezhoubaPred_3_25", "GezhoubaSignalSeg_3_25","DiantiPred_3_25","DiantiSignalSeg_3_25")
                entry.current(0)  # 默认选择第一个选项
                entry.grid(row=row, column=col * 2 + 1, padx=5, pady=5)
                entry.bind("<<ComboboxSelected>>", self.update_data_field)
            else:
                entry = ttk.Entry(input_frame, width=30)
                entry.grid(row=row, column=col * 2 + 1, padx=5, pady=5)
                entry.insert(0, default)
            self.entries[key] = entry

        self.update_data_field(None)

    def update_data_field(self, event):
        model_id = self.entries['model_id'].get()
        if model_id == "GezhoubaPred_3_25":
            self.entries['data'].delete(0, tk.END)
            self.entries['data'].insert(0, "SignalCoarsePred")
            self.entries['enc_in'].delete(0, tk.END)
            self.entries['enc_in'].insert(0, "86")
            self.entries['task_name'].delete(0, tk.END)
            self.entries['task_name'].insert(0, "forecast_and_anomaly_detection")
            self.entries['root_path'].delete(0, tk.END)
            self.entries['root_path'].insert(0, "./数据集/葛洲坝")
        elif model_id == "GezhoubaSignalSeg_3_25":
            self.entries['data'].delete(0, tk.END)
            self.entries['data'].insert(0, "SignalFineSeg")
            self.entries['enc_in'].delete(0, tk.END)
            self.entries['enc_in'].insert(0, "27")
            self.entries['task_name'].delete(0, tk.END)
            self.entries['task_name'].insert(0, "anomaly_detection")
            self.entries['root_path'].delete(0, tk.END)
            self.entries['root_path'].insert(0, "./数据集/葛洲坝")
        elif model_id == "DiantiPred_3_25":
            self.entries['data'].delete(0, tk.END)
            self.entries['data'].insert(0, "SignalCoarsePred")
            self.entries['enc_in'].delete(0, tk.END)
            self.entries['enc_in'].insert(0, "86")
            self.entries['task_name'].delete(0, tk.END)
            self.entries['task_name'].insert(0, "forecast_and_anomaly_detection")
            self.entries['root_path'].delete(0, tk.END)
            self.entries['root_path'].insert(0, "./数据集/电梯信号")
        elif model_id == "DiantiSignalSeg_3_25":
            self.entries['data'].delete(0, tk.END)
            self.entries['data'].insert(0, "SignalFineSeg")
            self.entries['enc_in'].delete(0, tk.END)
            self.entries['enc_in'].insert(0, "86")
            self.entries['task_name'].delete(0, tk.END)
            self.entries['task_name'].insert(0, "anomaly_detection")
            self.entries['root_path'].delete(0, tk.END)
            self.entries['root_path'].insert(0, "./数据集/电梯信号")


    def create_button_frame(self):
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill="x", expand="yes", padx=10, pady=10)

        self.run_button = ttk.Button(button_frame, text="Run", command=self.start_experiment)
        self.run_button.pack(side=tk.LEFT, padx=5)

        # 显示基线
        self.jixian_button = ttk.Button(button_frame, text="Base Line", command=self.show_base_line)
        self.jixian_button.pack(side=tk.LEFT, padx=5)

    def create_output_frame(self):
        output_frame = ttk.LabelFrame(self.root, text="Output")
        output_frame.pack(fill="both", expand="yes", padx=10, pady=10)

        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=100, height=20)
        self.output_text.pack(fill="both", expand="yes", padx=5, pady=5)

        # 重定向输出
        self.redirector = RedirectText(self.output_text)
        sys.stdout = self.redirector
        sys.stderr = self.redirector

    def start_experiment(self):
        if not self.running:
            self.running = True
            self.run_button.pack_forget()
            self.jixian_button.pack_forget()
            threading.Thread(target=self.run_experiment).start()

    def show_base_line(self):
        if not self.running:
            self.running = True
            self.run_button.pack_forget()
            self.jixian_button.pack_forget()
            threading.Thread(target=self.run_base_line()).start()

    def run_experiment(self):
        # 获取参数
        self.get_args()

        # 设置随机种子
        fix_seed = 2021
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)

        # 打印参数
        print('Args in experiment:')
        print_args(self.args)

        # 选择实验
        if self.args.task_name == 'long_term_forecast':
            Exp = Exp_Long_Term_Forecast
        elif self.args.task_name == 'short_term_forecast':
            Exp = Exp_Short_Term_Forecast
        elif self.args.task_name == 'imputation':
            Exp = Exp_Imputation
        elif self.args.task_name == 'anomaly_detection':
            Exp = Exp_Anomaly_Detection
        elif self.args.task_name == 'classification':
            Exp = Exp_Classification
        elif self.args.task_name == 'forecast_and_anomaly_detection':
            Exp = Exp_Forecast_And_Anomaly_Detection
            self.args.task_name = 'long_term_forecast'
        else:
            Exp = Exp_Long_Term_Forecast

        # 运行实验
        if self.args.is_training == 1:
            for ii in range(self.args.itr):
                exp = Exp(self.args)
                setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    self.args.task_name,
                    self.args.model_id,
                    self.args.model,
                    self.args.data,
                    self.args.features,
                    self.args.seq_len,
                    self.args.label_len,
                    self.args.pred_len,
                    self.args.d_model,
                    self.args.n_heads,
                    self.args.e_layers,
                    self.args.d_layers,
                    self.args.d_ff,
                    self.args.factor,
                    self.args.embed,
                    self.args.distil,
                    self.args.des, ii)

                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
                torch.cuda.empty_cache()
        elif self.args.is_training == 0:
            ii = 0
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                self.args.task_name,
                self.args.model_id,
                self.args.model,
                self.args.data,
                self.args.features,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.d_model,
                self.args.n_heads,
                self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.factor,
                self.args.embed,
                self.args.distil,
                self.args.des, ii)

            exp = Exp(self.args)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
            torch.cuda.empty_cache()
        else:
            ii = 0
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                self.args.task_name,
                self.args.model_id,
                self.args.model,
                self.args.data,
                self.args.features,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.d_model,
                self.args.n_heads,
                self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.factor,
                self.args.embed,
                self.args.distil,
                self.args.des, ii)

            exp = Exp(self.args)
            print('>>>>>>>testing anomaly detection: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            if (self.args.task_name != 'long_term_forecast' and self.args.task_name != 'anomaly_detection'):
                exp.test(setting, test=1)
            else:
                exp.anomaly_detection(setting, test=1,root_path=self.entries['root_path'].get())
            torch.cuda.empty_cache()

        # 运行结束后恢复按钮
        self.running = False
        self.run_button.pack(side=tk.LEFT, padx=5)
        self.jixian_button.pack(side=tk.LEFT, padx=5)

    def get_args(self):
        # 创建参数解析器
        parser = argparse.ArgumentParser(description='TimesNet')

        # 添加参数
        parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast')
        parser.add_argument('--is_training', type=int, required=True, default=1)
        parser.add_argument('--model_id', type=str, required=True, default='test')
        parser.add_argument('--model', type=str, required=True, default='Autoformer')
        parser.add_argument('--data', type=str, required=True, default='ETTm1')
        parser.add_argument('--root_path', type=str, default='./data/ETT/')
        parser.add_argument('--data_path', type=str, default='ETTh1.csv')
        parser.add_argument('--features', type=str, default='M')
        parser.add_argument('--target', type=str, default='OT')
        parser.add_argument('--freq', type=str, default='h')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
        parser.add_argument('--seq_len', type=int, default=96)
        parser.add_argument('--label_len', type=int, default=48)
        parser.add_argument('--pred_len', type=int, default=96)
        parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
        parser.add_argument('--inverse', action='store_true', default=False)
        parser.add_argument('--mask_rate', type=float, default=0.25)
        parser.add_argument('--anomaly_ratio', type=float, default=0.25)
        parser.add_argument('--top_k', type=int, default=5)
        parser.add_argument('--num_kernels', type=int, default=6)


        parser.add_argument('--enc_in', type=int, default=7)
        parser.add_argument('--dec_in', type=int, default=7)
        parser.add_argument('--c_out', type=int, default=7)

        parser.add_argument('--d_model', type=int, default=512)
        parser.add_argument('--n_heads', type=int, default=8)
        parser.add_argument('--e_layers', type=int, default=2)
        parser.add_argument('--d_layers', type=int, default=1)
        parser.add_argument('--d_ff', type=int, default=2048)
        parser.add_argument('--moving_avg', type=int, default=25)
        parser.add_argument('--factor', type=int, default=1)
        parser.add_argument('--distil', action='store_false', default=True)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--embed', type=str, default='timeF')
        parser.add_argument('--activation', type=str, default='gelu')
        parser.add_argument('--output_attention', action='store_true')
        parser.add_argument('--channel_independence', type=int, default=0)
        parser.add_argument('--num_workers', type=int, default=10)
        parser.add_argument('--itr', type=int, default=1)
        parser.add_argument('--train_epochs', type=int, default=10)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--patience', type=int, default=3)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--des', type=str, default='test')
        parser.add_argument('--loss', type=str, default='MSE')
        parser.add_argument('--lradj', type=str, default='type1')
        parser.add_argument('--use_amp', action='store_true', default=False)
        parser.add_argument('--use_gpu', type=bool, default=True)
        parser.add_argument('--gpu', type=int, default=0)
        parser.add_argument('--use_multi_gpu', action='store_true', default=False)
        parser.add_argument('--devices', type=str, default='0,1,2,3')
        parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128])
        parser.add_argument('--p_hidden_layers', type=int, default=2)

        # 解析参数
        self.args = parser.parse_args([
            '--task_name', self.entries['task_name'].get(),
            '--is_training', self.entries['is_training'].get(),
            '--model_id', self.entries['model_id'].get(),
            '--model', self.entries['model'].get(),
            '--data', self.entries['data'].get(),
            '--root_path', self.entries['root_path'].get(),
            #'--target', self.entries['target'].get(),
            '--target',"communication_num",
            #'--freq', self.entries['freq'].get(),
            '--freq','t',
            #'--features', self.entries['features'].get(),
            '--features','M',
            #'--learning_rate', self.entries['learning_rate'].get(),
            '--learning_rate','0.001',
            #'--seq_len', self.entries['seq_len'].get(),
            '--seq_len','10',
            #'--label_len', self.entries['label_len'].get(),
            '--label_len','1',
            #'--pred_len', self.entries['pred_len'].get(),
            '--pred_len','3',
            #'--e_layers', self.entries['e_layers'].get(),
            '--e_layers','2',
            #'--d_layers', self.entries['d_layers'].get(),
            '--d_layers','1',
            #'--factor', self.entries['factor'].get(),
            '--factor','3',

            '--enc_in', self.entries['enc_in'].get(),
            #'--dec_in', self.entries['dec_in'].get(),
            #'--c_out', self.entries['c_out'].get(),
            '--dec_in', self.entries['enc_in'].get(),
            '--c_out', self.entries['enc_in'].get(),


            #'--d_model', self.entries['d_model'].get(),
            '--d_model',"256",
            #'--d_ff', self.entries['d_ff'].get(),
            '--d_ff','512',
            #'--des', self.entries['des'].get().strip("'"),
            '--des',"Exp",
            #'--itr', self.entries['itr'].get(),
            '--itr','1',
            #'--batch_size', self.entries['batch_size'].get(),
            '--batch_size','128',
            #'--anomaly_ratio', self.entries['anomaly_ratio'].get(),
            '--anomaly_ratio','1',
            #'--patience', self.entries['patience'].get(),
            '--patience','8',

            '--train_epochs', self.entries['train_epochs'].get()
        ])

        # 设置 GPU 参数
        self.args.use_gpu = True if torch.cuda.is_available() and self.args.use_gpu else False
        if self.args.use_gpu and self.args.use_multi_gpu:
            self.args.devices = self.args.devices.replace(' ', '')
            device_ids = self.args.devices.split(',')
            self.args.device_ids = [int(id_) for id_ in device_ids]
            self.args.gpu = self.args.device_ids[0]

    def run_base_line(self):
        plot_all(newdata_file=os.path.join(self.entries['root_path'].get(),"coarse_grained_data/test_coarse_grained_data.csv"))
        # 运行结束后恢复按钮
        self.running = False
        self.run_button.pack(side=tk.LEFT, padx=5)
        self.jixian_button.pack(side=tk.LEFT, padx=5)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = App(root)
    root.mainloop()