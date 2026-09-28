import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import time

# ================= 配置区域 =================
# 请根据实际情况修改这两个路径
input_folder = r'D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_csv\temp\t1'
output_folder = r'D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_csv\temp\t1\SisFall_dataset_filtered'

# 滤波参数
fs = 200.0       # 采样频率 200Hz
cutoff = 5.0     # 截止频率 5Hz
order = 4        # 四阶巴特沃斯

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# ================= 1. 准备滤波器 =================
nyq = 0.5 * fs
normal_cutoff = cutoff / nyq
b, a = butter(order, normal_cutoff, btype='low', analog=False)

print("开始批量处理...")
start_time = time.time()

count = 0
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_folder, filename)

        # --- 1. 读取数据 (假设无表头) ---
        # 根据你的截图，数据是纯数字，所以 header=None
        df = pd.read_csv(input_path, header=None)

        # --- 2. 修改列名 & 删除列 ---
        # 定义新的列名
        new_columns = ['sn', 'acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'col_8', 'col_9', 'col_10']
        # 如果原文件列数不固定，这里做个保护，只重命名现有的列
        df.columns = new_columns[:len(df.columns)]

        # 删除最后三列，只保留前7列
        df = df.iloc[:, :7]

        # --- 3. 应用滤波 (对除 sn 外的所有列) ---
        # 使用 filtfilt 进行零相位滤波
        # axis=0 表示沿着行（时间轴）进行滤波
        df_filtered_data = filtfilt(b, a, df.iloc[:, 1:], axis=0)

        # 将滤波后的数据放回 DataFrame
        df.iloc[:, 1:] = df_filtered_data

        # --- 4. 保存文件 (添加后缀) ---
        # 分离文件名和扩展名，例如 "D01_SA01_R01.csv" -> ("D01_SA01_R01", ".csv")
        name, ext = os.path.splitext(filename)
        # 拼接新文件名: "D01_SA01_R01_filtered.csv"
        new_filename = f"{name}_filtered{ext}"

        output_path = os.path.join(output_folder, new_filename)

        # 保存，不保存索引，保留6位小数
        df.to_csv(output_path, index=False, float_format='%.6f')

        count += 1

end_time = time.time()
print(f"处理完成！共处理 {count} 个文件。")
print(f"耗时: {end_time - start_time:.2f} 秒")
print(f"文件已保存至: {output_folder}")