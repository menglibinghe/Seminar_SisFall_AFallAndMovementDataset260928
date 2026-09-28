import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.metrics import balanced_accuracy_score

# ==========================================
# 1. 路径与参数配置
# ==========================================
# 修改为你存放 Merged_SisFall_Final.txt 的实际文件夹路径
base_path = r'D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database'
input_file = os.path.join(base_path, 'reduced_SisFall.txt')

FS = 200        # 采样频率 [cite: 130]
CUTOFF = 5      # 论文指定的 5Hz 截止频率 
WINDOW_SIZE = 200 # 1秒窗口 (200个样本)，适合 C2, C8 等特征 [cite: 226]

# # ==========================================
# 2. 信号处理函数
# ==========================================
def butter_lowpass_filter(data, cutoff, fs, order=4):
    """实现论文要求的 4 阶 Butterworth 滤波器 """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # filtfilt 提供零相位滤波，防止信号延迟 
    return filtfilt(b, a, data, axis=0)

def find_best_t1(y_true, feature_values):
    """搜索使 AC 最大化的 T1 阈值 (AC = (SE + SP) / 2) [cite: 161, 167]"""
    best_ac = 0
    best_t1 = 0
    # 在特征值范围内生成 200 个候选阈值进行遍历
    thresholds = np.linspace(np.min(feature_values), np.max(feature_values), 200)
    
    for t in thresholds:
        y_pred = (feature_values > t).astype(int)
        # balanced_accuracy_score 等价于论文中的 AC 公式 [cite: 167]
        ac = balanced_accuracy_score(y_true, y_pred)
        if ac > best_ac:
            best_ac = ac
            best_t1 = t
    return best_t1, best_ac

# ==========================================
# 3. 主程序逻辑
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(input_file):
        print(f"❌ 找不到文件: {input_file}")
    else:
        print(f"📖 正在加载: {input_file}")
        df = pd.read_csv(input_file, sep='\t')

        # 标注：F 开头为跌倒(1)，D 开头为日常活动(0) [cite: 99, 104]
        df['label'] = df['Activity'].str.startswith('F').astype(int)

        print("⚡ 正在执行 5Hz 低通滤波...")
        cols = ['acc1_x', 'acc1_y', 'acc1_z']
        df[cols] = butter_lowpass_filter(df[cols].values, CUTOFF, FS)

        print("📊 正在计算 Table 4 特征...")
        # C1: Sum vector magnitude 
        df['C1'] = np.sqrt(df['acc1_x']**2 + df['acc1_y']**2 + df['acc1_z']**2)
        
        # C2: 水平面向量模 
        df['C2'] = np.sqrt(df['acc1_x']**2 + df['acc1_z']**2)
        
        # C3: 窗口内最大峰峰值 (Max peak-to-peak) 
        df['C3'] = df['C1'].rolling(WINDOW_SIZE).apply(lambda x: np.max(x) - np.min(x))
        
        # C8: 水平面标准差模 
        std_x = df['acc1_x'].rolling(WINDOW_SIZE).std()
        std_z = df['acc1_z'].rolling(WINDOW_SIZE).std()
        df['C8'] = np.sqrt(std_x**2 + std_z**2)
        
        # C9: 三轴标准差模 
        std_y = df['acc1_y'].rolling(WINDOW_SIZE).std()
        df['C9'] = np.sqrt(std_x**2 + std_y**2 + std_z**2)
        
        # C13: 水平面信号幅值面积 (SMA 积分近似) 
        df['C13'] = df['C2'].rolling(WINDOW_SIZE).mean()

        # 清除因滚动窗口产生的 NaN 值
        df_clean = df.dropna()

        print("\n" + "="*40)
        print(f"{'特征':<10} {'最佳 T1 阈值':<15} {'最大 AC':<10}")
        print("-" * 40)

        for feat in ['C2', 'C3', 'C8', 'C9', 'C13']:
            t1, ac = find_best_t1(df_clean['label'], df_clean[feat])
            print(f"{feat:<10} {t1:<15.4f} {ac:<10.2%}")
        print("="*40)