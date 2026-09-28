import os
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from scipy.signal import butter, filtfilt

# ==========================================
# 1. 路径配置
# ==========================================
input_dir = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_csv\temp\t1"
output_dir = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_csv\temp\t1\SisFall_dataset_filtered"

os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "Merged_SisFall_Final_Corrected.csv")

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
# 3. 传感器参数定义
# ==========================================
# --- 加速度计 (ADXL345) ---
Sensor1_Resolution = 13
Sensor1_Range = 16
# 转换为 g
g_S1 = (2 * Sensor1_Range) / (2 ** Sensor1_Resolution)

# --- 陀螺仪 (ITG3200) ---
Gyro_Resolution = 16
Gyro_Range = 2000
# 转换为 deg/s
gyro_factor = (2 * Gyro_Range) / (2 ** Gyro_Resolution)

# ==========================================
# 4. 主处理循环
# ==========================================
all_data = []

# 获取文件列表并排序，确保处理顺序一致
fs = sorted(os.listdir(input_dir))

print("开始处理文件...")
for f in tqdm(fs):
    if not f.endswith('.csv'):
        continue

    file_path = os.path.join(input_dir, f)

    # 读取数据：跳过第1列(A)，读取接下来的6列(B-G)
    # 对应原始数据的: Ax, Ay, Az, Rx, Ry, Rz
    try:
        df = pd.read_csv(file_path, usecols=[1, 2, 3, 4, 5, 6],
                         names=["Ax", "Ay", "Az", "Rx", "Ry", "Rz"],
                         header=None) # 假设原始csv没有表头，如果有表头请去掉 header=None
    except ValueError:
        print(f"跳过文件 {f} (列数不足)")
        continue

    # --- 1. 单位转换 ---
    # 加速度计转换 (to g)
    df['Ax'] = g_S1 * df['Ax']
    df['Ay'] = g_S1 * df['Ay']
    df['Az'] = g_S1 * df['Az']

    # 陀螺仪转换 (to deg/s) - 新增步骤
    df['Rx'] = gyro_factor * df['Rx']
    df['Ry'] = gyro_factor * df['Ry']
    df['Rz'] = gyro_factor * df['Rz']

    # --- 2. 滤波与下采样 ---
    # 5Hz 低通滤波
    for col in ["Ax", "Ay", "Az", "Rx", "Ry", "Rz"]:
        df[col] = butter_lowpass_filter(df[col], cutoff=5, fs=200, order=4)

    # 下采样 (200Hz -> 100Hz)
    df = df.loc[::2].reset_index(drop=True)

    # --- 3. 窗口截取 (基于 SVM) ---
    # 计算 SVM 用于寻找峰值窗口 (使用滤波后的数据)
    df['SVM'] = np.sqrt(df['Ax']**2 + df['Ay']**2 + df['Az']**2)

    max_N = df['SVM'].max()
    max_N_index = df.index[df.SVM == max_N][0]
    len_df = len(df)

    # 截取 300 点窗口 (3秒 @ 100Hz)
    window_size = 300
    half_window = 150

    if max_N_index - half_window < 0:
        df_window = df.iloc[0:window_size].copy()
    elif max_N_index + half_window >= len_df:
        df_window = df.iloc[len_df - window_size:len_df].copy()
    else:
        df_window = df.iloc[max_N_index - half_window : max_N_index + half_window].copy()

    df_window.reset_index(drop=True, inplace=True)

    # --- 4. 特征计算 ---
    features = {}

    # 基础统计函数
    def calc_stats(series, prefix):
        stats = {
            f'min_{prefix}': series.min(),
            f'max_{prefix}': series.max(),
            f'mean_{prefix}': series.mean(),
            f'median_{prefix}': series.median(),
            f'var_{prefix}': series.var(),
            f'std_{prefix}': series.std(),
            f'range_{prefix}': series.max() - series.min(),
            f'kurtosis_{prefix}': series.kurtosis(),
            f'skewness_{prefix}': series.skew()
        }
        return stats

    # A. 加速度特征 (Ax, Ay, Az)
    for col in ['Ax', 'Ay', 'Az']:
        stats = calc_stats(df_window[col], col)
        features.update(stats)

    # B. 陀螺仪特征 (Rx, Ry, Rz)
    for col in ['Rx', 'Ry', 'Rz']:
        stats = calc_stats(df_window[col], col)
        features.update(stats)

    # C. 合成特征 (HOR, VER, Rygo_SVM)
    # 坐标系定义: x向右, y向下, z向前
    # HOR (水平): x-z平面
    df_window['HOR'] = np.sqrt(df_window['Ax']**2 + df_window['Az']**2)
    # VER (垂直): y轴
    df_window['VER'] = df_window['Ay']

    # Rygo_SVM (陀螺仪合成)
    df_window['Rygo_SVM'] = np.sqrt(df_window['Rx']**2 + df_window['Ry']**2 + df_window['Rz']**2)

    # 计算合成特征的统计量
    for col in ['HOR', 'VER', 'Rygo_SVM']:
        stats = calc_stats(df_window[col], col)
        features.update(stats)

    # D. Pitch (C4算法)
    # 公式: atan2( sqrt(Ax^2 + Az^2), -Ay )
    # 注意: atan2(y, x) 在 numpy 中是 atan2(y, x)，但数学公式通常写作 atan2(val1, val2)
    # 对应图片: val1 = sqrt(...), val2 = -Ay
    pitch_vals = []
    for i in range(len(df_window)):
        ax = df_window.loc[i, 'Ax']
        ay = df_window.loc[i, 'Ay']
        az = df_window.loc[i, 'Az']
        # 防止除以零或数值错误，虽然 atan2 处理得很好
        val1 = math.sqrt(ax**2 + az**2)
        val2 = -ay
        pitch_vals.append(math.atan2(val1, val2))

    df_window['C4_Pitch'] = pitch_vals
    # C4 统计特征
    pitch_stats = calc_stats(df_window['C4_Pitch'], 'C4_Pitch')
    features.update(pitch_stats)

    # --- 5. 文件名解析与标签 ---
    # 示例: D05_SA01_R02.csv
    name_parts = f.split('.')[0].split('_')
    move = name_parts[0] if len(name_parts) > 0 else "Unknown"
    sub = name_parts[1] if len(name_parts) > 1 else "Unknown"
    round_val = name_parts[2] if len(name_parts) > 2 else "Unknown"

    label = 0 if f.startswith('D') else 1

    # 添加元数据到特征字典
    features['Move'] = move
    features['Sub'] = sub
    features['Round'] = round_val
    features['Label'] = label

    all_data.append(features)

# ==========================================
# 6. 合并与排序
# ==========================================
print("正在合并与排序...")
final_df = pd.DataFrame(all_data)

# 排序优先级: Label -> Move -> Sub -> Round
# 注意：Round 是字符串 'R01', 'R02'... 字符串排序通常能满足需求，除非超过 R99
final_df.sort_values(by=['Label', 'Move', 'Sub', 'Round'], ascending=True, inplace=True)

# 重置索引
final_df.reset_index(drop=True, inplace=True)

# 保存
final_df.to_csv(output_file, index=False)
print(f"处理完成！文件已保存至: {output_file}")
print(f"总样本数: {len(final_df)}")
print(f"特征列数: {len(final_df.columns)}")