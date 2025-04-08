import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import scrolledtext
from sample_original_data.methods.read_and_sample_method import *
from convert_paths.convert_windows_and_linux_path import convert_paths
from sample_original_data.methods.process_one_bin_file import generate_one_data_from_center_list, generate_one_data
import threading
import sys

class RedirectOutput:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()

    def flush(self):
        pass

def generate_trace_data_center(bin_file_dir="G:/Work/Workspace/Data/原始数据/信息工程研究所/410_420MHz",
                               start_date="2024-09-25 00:00",
                               stop_date="2024-10-17 00:00",
                               center_freq_list=[416.086],
                               bandwidth=0.2,
                               minute_of_picture='1D',
                               prefix_path="G:/Work/Workspace/Data/预处理数据"):
    # 从长时设备采集的原始bin文件中读取特点频段（根据中心频点进行选择），特点时间段内的数据，并进行频点下采样，采样时间下采样，目的是精简数据，便宜处理和观察
    # 参数bin_file_dir为存储bin文件的文件夹TraceRecord，一般其中含有子文件夹“2024-05-23，2024-05-24，....”
    # 频段选择select_start_freq ，select_stop_freq
    # 数据时间段选择日期 start_date ，stop_date
    # picture_num将指定频段，指定时间段内的数据分几部分存
    # minute_of_picture一个文件包括多长时间段的数据
    # resample_time在时间上进行下采样
    # resample_freq_num在频点上进行下采样，选取的频点列不超过该值

    bin_file_dir = convert_paths(bin_file_dir)

    picture_num = 1
    resample_time = 1

    file_massages = bin_file_dir.split("/")
    location = file_massages[-2]
    other_message = ''
    for center_freq in center_freq_list:
        other_message = other_message + str(center_freq) + "_"
    other_message = other_message + str(bandwidth)
    output_dir = os.path.join(prefix_path, location, other_message + "MHz原始信号数据")
    output_dir = convert_paths(output_dir)

    date_list = produce_date_intervals(bin_file_dir, start_date, stop_date, minute_of_picture)

    for (intervals_start_date, intervals_stop_date) in date_list:
        print(f"Generating the picture of frequency between , date from {intervals_start_date} to {intervals_stop_date}")
        generate_one_data_from_center_list(bin_file_dir, center_freq_list, bandwidth, intervals_start_date,
                                           intervals_stop_date, resample_time, output_dir)



def browse_folder(entry):
    folder = filedialog.askdirectory()
    if folder:
        entry.delete(0, tk.END)
        entry.insert(0, folder)

def browse_file(entry):
    file = filedialog.askopenfilename()
    if file:
        entry.delete(0, tk.END)
        entry.insert(0, file)

def validate_and_run(output_text, run_button):
    bin_file_dir = bin_file_dir_entry.get()
    start_date = start_date_entry.get()
    stop_date = stop_date_entry.get()
    center_freq_list_str = center_freq_list_entry.get()
    bandwidth_str = bandwidth_entry.get()
    prefix_path = prefix_path_entry.get()
    c_out=c_out_entry.get()

    try:
        center_freq_list = list(map(float, center_freq_list_str.strip('[]').split(',')))
    except ValueError:
        messagebox.showerror("错误", "中心频率列表格式错误，请输入类似 [7.5] 的格式。")
        run_button.grid()
        return

    try:
        bandwidth = float(bandwidth_str)
    except ValueError:
        messagebox.showerror("错误", "带宽需要是一个数字")
        run_button.grid()
        return

    # 重定向输出
    old_stdout = sys.stdout
    sys.stdout = RedirectOutput(output_text)

    generate_trace_data_center(
        bin_file_dir=bin_file_dir,
        start_date=start_date,
        stop_date=stop_date,
        center_freq_list=center_freq_list,
        bandwidth=bandwidth,
        minute_of_picture=minute_of_picture_entry.get(),
        prefix_path=prefix_path
    )

    print("Reshape......")
    for root, dirs, files in os.walk(prefix_path):
        for file in files:
            if file.endswith('.csv'):  # 只处理 .csv 文件
                file_path = os.path.join(root, file)
                print(file_path)
                df = pd.read_csv(file_path)
                df = df.iloc[:, 0:int(c_out) + 1]
                df.to_csv(file_path, index=False)

    messagebox.showinfo("成功", "数据处理已完成")

    # 恢复输出
    sys.stdout = old_stdout

    run_button.grid()

def run_command(output_text, run_button):
    run_button.grid_remove()
    thread = threading.Thread(target=lambda: validate_and_run(output_text, run_button))
    thread.daemon = True
    thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("原始bin文件-->频谱数据")
    root.geometry("1000x800")

    # 创建标签和输入框
    ttk.Label(root, text="bin_file_dir:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    bin_file_dir_entry = ttk.Entry(root, width=70)
    bin_file_dir_entry.grid(row=0, column=1, padx=5, pady=5)
    bin_file_dir_entry.insert(0, "E:/dianci/sample_original_data/葛洲坝船闸迹线/20240628-0705船闸闸室或阀室数据/TraceRecord")
    ttk.Button(root, text="浏览", command=lambda: browse_folder(bin_file_dir_entry)).grid(row=0, column=2, padx=5, pady=5)

    ttk.Label(root, text="start_date:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
    start_date_entry = ttk.Entry(root, width=70)
    start_date_entry.grid(row=1, column=1, padx=5, pady=5)
    start_date_entry.insert(0, "2024-07-02 00:00")

    ttk.Label(root, text="stop_date:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
    stop_date_entry = ttk.Entry(root, width=70)
    stop_date_entry.grid(row=2, column=1, padx=5, pady=5)
    stop_date_entry.insert(0, "2024-07-06 00:00")

    ttk.Label(root, text="center_freq_list:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
    center_freq_list_entry = ttk.Entry(root, width=70)
    center_freq_list_entry.grid(row=3, column=1, padx=5, pady=5)
    center_freq_list_entry.insert(0, "[7.5]")

    ttk.Label(root, text="bandwidth:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
    bandwidth_entry = ttk.Entry(root, width=70)
    bandwidth_entry.grid(row=4, column=1, padx=5, pady=5)
    bandwidth_entry.insert(0, "0.1")

    ttk.Label(root, text="minute_of_picture:").grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
    minute_of_picture_entry = ttk.Entry(root, width=70)
    minute_of_picture_entry.grid(row=5, column=1, padx=5, pady=5)
    minute_of_picture_entry.insert(0, "6H")

    ttk.Label(root, text="prefix_path:").grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)
    prefix_path_entry = ttk.Entry(root, width=70)
    prefix_path_entry.grid(row=6, column=1, padx=5, pady=5)
    prefix_path_entry.insert(0, "D:/葛洲坝船闸预处理")
    ttk.Button(root, text="浏览", command=lambda: browse_folder(prefix_path_entry)).grid(row=6, column=2, padx=5, pady=5)

    ttk.Label(root, text="c_out:").grid(row=7, column=0, padx=5, pady=5, sticky=tk.W)
    c_out_entry = ttk.Entry(root, width=70)
    c_out_entry.grid(row=7, column=1, padx=5, pady=5)
    c_out_entry.insert(0, "27")

    # 创建运行按钮
    run_button = ttk.Button(root, text="运行", command=lambda: run_command(output_text, run_button))
    run_button.grid(row=8, column=0, columnspan=3, pady=20)

    # 创建输出区域
    output_frame = ttk.LabelFrame(root, text="Output")
    output_frame.grid(row=9, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

    output_text = scrolledtext.ScrolledText(output_frame, width=120, height=20)
    output_text.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

    # 设置网格权重
    root.grid_rowconfigure(9, weight=1)
    root.grid_columnconfigure(1, weight=1)
    output_frame.grid_rowconfigure(0, weight=1)
    output_frame.grid_columnconfigure(0, weight=1)

    root.mainloop()