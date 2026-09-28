import pandas as pd
import numpy as np
import os
from scipy import stats
from scipy.fftpack import fft

# ==========================================
# 1. 定义特征提取函数 (对应论文中的 C2 - C13)
# ==========================================
def extract_features(window_data):
    """
    输入: window_data 是一个 DataFrame，包含 acc1_x, acc1_y, acc1_z
    输出: 一个包含特征值的字典
    """
    features = {}

    # 对三个轴分别计算特征
    for axis in ['acc1_x', 'acc1_y', 'acc1_z']:
        data = window_data[axis].values

        # C2: Mean (均值)
        features[f'{axis}_mean'] = np.mean(data)

        # C3: Standard Deviation (标准差)
        features[f'{axis}_std'] = np.std(data)

        # C8: Max (最大值)
        features[f'{axis}_max'] = np.max(data)

        # C9: Min (最小值)
        features[f'{axis}_min'] = np.min(data)

        # C13: Energy / Sum of Squares (能量/平方和) - 这是一个强特征
        features[f'{axis}_energy'] = np.sum(np.square(data))

        # 其他可能的特征 (如果论文里C4-C7是这些的话)
        # C4: Skewness (偏度)
        # features[f'{axis}_skew'] = stats.skew(data)

        # C5: Kurtosis (峰度)
        # features[f'{axis}_kurt'] = stats.kurtosis(data)

    # 合加速度特征 (非常重要！)
    # 计算合加速度向量模长
    window_data['acc_norm'] = np.sqrt(
        window_data['acc1_x']**2 +
        window_data['acc1_y']**2 +
        window_data['acc1_z']**2
    )

    # 对合加速度也计算特征
    features['norm_mean'] = np.mean(window_data['acc_norm'])
    features['norm_std'] = np.std(window_data['acc_norm'])
    features['norm_max'] = np.max(window_data['acc_norm'])
    features['norm_energy'] = np.sum(np.square(window_data['acc_norm']))

    return features

# ==========================================
# 2. 数据预处理与滑动窗口
# ==========================================
def process_dataset(input_path, output_path, window_size=100, step_size=50):
    """
    input_path: 原始数据路径
    output_path: 提取特征后保存的路径
    window_size: 窗口大小 (例如 1秒数据 = 100个点，视采样率而定)
    step_size: 步长 (重叠率)
    """
    # 读取原始数据
    # 注意：根据你的截图，分隔符可能是空格或制表符，这里假设是制表符 \t
    # 如果报错，请尝试 sep='\s+'
    df = pd.read_csv(input_path, sep='\t')

    # 确保列名正确 (根据你的截图调整)
    # 假设列名为: n, acc1_x, acc1_y, acc1_z, subject, Activity, Trial
    # 我们需要根据 'n' 或者 'subject'+'Activity'+'Trial' 来分组
    # 这里我们假设每一行是一个时间点，我们需要按 Trial 进行窗口切片

    all_features = []
    all_labels = []
    all_subjects = []

    # 获取所有唯一的 Trial 组合
    # 注意：SisFall 的数据通常需要按 Trial 分组，或者如果文件已经是分段好的，直接处理
    # 如果你的文件是所有数据连在一起的，我们需要用滑动窗口切

    # 简单处理：直接在整段数据上滑动窗口 (这在实际应用中更常见)
    # 但为了保持标签一致性，我们最好按 Trial 分组
    grouped = df.groupby(['subject', 'Activity', 'Trial'])

    print("正在提取特征...")
    for name, group in grouped:
        subject, activity, trial = name

        # 重置索引以便切片
        group = group.reset_index()

        # 滑动窗口
        for start in range(0, len(group) - window_size + 1, step_size):
            end = start + window_size
            window = group.iloc[start:end]

            # 提取特征
            feats = extract_features(window)

            all_features.append(feats)
            all_labels.append(activity) # 标签是 Activity (D01, D02...)
            all_subjects.append(subject)

    # 转换为 DataFrame
    features_df = pd.DataFrame(all_features)
    features_df['label'] = all_labels
    features_df['subject'] = all_subjects

    # 保存
    features_df.to_csv(output_path, index=False)
    print(f"特征提取完成！已保存至: {output_path}")
    print(features_df.head())

# ==========================================
# 3. 执行
# ==========================================
base_dir = r'D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\csv2txtCombine'
input_file = os.path.join(base_dir, 'reduced_SisFall.txt')
output_file = os.path.join(base_dir, 'SisFall_With_Features.csv')

# 运行处理 (注意：你需要根据你的实际采样率调整 window_size)
# SisFall 采样率通常是 200Hz。1秒窗口就是 200，0.5秒就是 100。
process_dataset(input_file, output_file, window_size=200, step_size=100)