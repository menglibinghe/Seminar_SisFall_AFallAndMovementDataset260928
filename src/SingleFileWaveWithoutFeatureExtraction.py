import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ==========================================
# 1. 路径配置与绘图设置
# ==========================================
# 指向原始数据文件夹
# 读取文件
Directory_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\t1\SisFall_dataset_filtered"
csv_file = os.path.join(Directory_path, "F05_SA01_R01.csv")

start_idx = 500
end_idx = 2000


# 设置中文字体，防止绘图乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 辅助函数：巴特沃斯滤波器
# ==========================================
def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# ==========================================
# 3. 传感器参数定义 (与原代码一致)
# ==========================================
# --- 加速度计 (ADXL345) ---
Sensor1_Resolution = 13
Sensor1_Range = 16
g_S1 = (2 * Sensor1_Range) / (2 ** Sensor1_Resolution)

# --- 陀螺仪 (ITG3200) ---
Gyro_Resolution = 16
Gyro_Range = 2000
gyro_factor = (2 * Gyro_Range) / (2 ** Gyro_Resolution)

# ==========================================
# 4. 主处理逻辑 (读取 -> 处理 -> 绘图)
# ==========================================

# 读取数据
# usecols=[1, 2, 3, 4, 5, 6] 对应 Excel 的 B, C, D, E, F, G 列
df = pd.read_csv(csv_file, usecols=[1, 2, 3, 4, 5, 6],
                    names=["Ax", "Ay", "Az", "Rx", "Ry", "Rz"],
                    header=0)       #header=0（默认也是 0）：第一行会被当作列名处理，数据从第二行开始读取

# --- 截取前 5000 行 ---
# 对应 Excel 的第 1-5000 行数据 (pandas索引 0 到 4999)
df = df.iloc[start_idx:end_idx].reset_index(drop=True)

# --- 1. 单位转换 ---
# 加速度计转换 (to g)
# df['Ax'] = g_S1 * df['Ax']
# df['Ay'] = g_S1 * df['Ay']
# df['Az'] = g_S1 * df['Az']

# 陀螺仪转换 (to deg/s)
# df['Rx'] = gyro_factor * df['Rx']
# df['Ry'] = gyro_factor * df['Ry']
# df['Rz'] = gyro_factor * df['Rz']

# --- 2. 滤波与下采样 ---
# 5Hz 低通滤波
for col in ["Ax", "Ay", "Az", "Rx", "Ry", "Rz"]:
    df[col] = butter_lowpass_filter(df[col], cutoff=5, fs=200, order=4)

# 下采样 (200Hz -> 100Hz)
df = df.loc[::2].reset_index(drop=True)

# ==========================================
# 5. 绘图展示
# ==========================================
plt.figure(figsize=(14, 8))

# 绘制加速度计数据 (Ax, Ay, Az) 对应 iloc[:, 0], [:, 1], [:, 2]
plt.plot(df.index, df.iloc[:, 0], label='Acc_x', color='red', linewidth=1)
plt.plot(df.index, df.iloc[:, 1], label='Acc_y', color='green', linewidth=1)
plt.plot(df.index, df.iloc[:, 2], label='Acc_z', color='blue', linewidth=1)

#如果你也想看陀螺仪数据，可以取消下面的注释
# plt.plot(df.index, df.iloc[:, 3], label='Gyro_x', color='cyan', linestyle='--', alpha=0.6)
# plt.plot(df.index, df.iloc[:, 4], label='Gyro_y', color='magenta', linestyle='--', alpha=0.6)
# plt.plot(df.index, df.iloc[:, 5], label='Gyro_z', color='orange', linestyle='--', alpha=0.6)

plt.title(f'Filtered Signal: {csv_file} (First 5000 rows processed & Downsampled)', fontsize=16)
plt.xlabel('Sample Index (Downsampled)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
#plt.tight_layout()
plt.show()