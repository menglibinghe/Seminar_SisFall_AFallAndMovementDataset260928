import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
import warnings

# 忽略特定的sklearn警告，让输出更清爽
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. 路径与参数配置
# ==========================================
base_path = r'D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database'
input_file = os.path.join(base_path, 'reduced_SisFall_AdultOnly.txt')     #仅验证是否去掉老人样本后AC均值会提升

FS = 200
CUTOFF = 5
WINDOW_SIZE = 200

N_SPLITS = 5
N_REPEATS = 5

# ==========================================
# 2. 信号处理函数
# ==========================================
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

# ==========================================
# 3. 主程序逻辑
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(input_file):
        print(f"❌ 找不到文件: {input_file}")
    else:
        print(f"📖 正在加载: {input_file}")
        df = pd.read_csv(input_file, sep='\t')

        # --- 关键修改 1：标签生成 ---
        # 只要包含 'F' 就是 1 (跌倒)，只要包含 'D' 就是 0 (日常活动)
        # 这样能过滤掉可能存在的其他干扰标签
        df['label'] = df['Activity'].apply(lambda x: 1 if 'F' in x else 0)

        # 检查数据分布
        print(f"数据分布检查 -> 跌倒(1): {df['label'].sum()} / 总数: {len(df)}")

        print("⚡ 正在执行 5Hz 低通滤波...")
        cols = ['acc1_x', 'acc1_y', 'acc1_z']
        df[cols] = butter_lowpass_filter(df[cols].values, CUTOFF, FS)

        print("📊 正在计算特征...")
        # 特征计算保持不变...
        df['C1'] = np.sqrt(df['acc1_x']**2 + df['acc1_y']**2 + df['acc1_z']**2)
        df['C2'] = np.sqrt(df['acc1_x']**2 + df['acc1_z']**2)
        df['C3'] = df['C1'].rolling(WINDOW_SIZE).apply(lambda x: np.max(x) - np.min(x))
        std_x = df['acc1_x'].rolling(WINDOW_SIZE).std()
        std_z = df['acc1_z'].rolling(WINDOW_SIZE).std()
        df['C8'] = np.sqrt(std_x**2 + std_z**2)
        std_y = df['acc1_y'].rolling(WINDOW_SIZE).std()
        df['C9'] = np.sqrt(std_x**2 + std_y**2 + std_z**2)
        df['C13'] = df['C2'].rolling(WINDOW_SIZE).mean()

        df_clean = df.dropna().reset_index(drop=True)

        X = df_clean[['C2', 'C3', 'C8', 'C9', 'C13']]
        y = df_clean['label']

        # --- 关键修改 2：构建组合分组键 ---
        # 将 受试者 和 活动类型 组合，例如 "SA01_F01" 和 "SA01_D01" 被视为不同的组
        # 这能防止某个特定的“人+动作”组合导致测试集标签单一
        df_clean['group_key'] = df_clean['subject'].astype(str) + "_" + df_clean['Activity'].astype(str)
        groups = df_clean['group_key']

        print(f"\n🚀 开始 {N_REPEATS}次 {N_SPLITS}折交叉验证 (修复版)...")

        results = {col: {'thresholds': [], 'accuracies': []} for col in X.columns}

        for repeat in range(N_REPEATS):
            sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=repeat)

            for train_index, test_index in sgkf.split(X, y, groups):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # 检查测试集是否包含两类标签，如果只有单一标签，跳过该折（避免报错）
                if len(np.unique(y_test)) < 2:
                    continue

                for col in X.columns:
                    train_feat = X_train[[col]].values
                    test_feat = X_test[[col]].values

                    clf = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=42)
                    clf.fit(train_feat, y_train)

                    y_pred = clf.predict(test_feat)
                    acc = balanced_accuracy_score(y_test, y_pred)

                    threshold = clf.tree_.threshold[0]

                    results[col]['thresholds'].append(threshold)
                    results[col]['accuracies'].append(acc)

        print("\n" + "="*60)
        print(f"{'特征':<10} {'平均 T1 阈值':<20} {'std':<10} {'平均 AC':<10} {'std':<10}")
        print("-" * 60)

        for col in X.columns:
            # 防止列表为空导致报错
            if len(results[col]['accuracies']) == 0:
                print(f"{col:<10} 数据不足无法计算")
                continue

            mean_t = np.mean(results[col]['thresholds'])
            std_t = np.std(results[col]['thresholds'])
            mean_ac = np.mean(results[col]['accuracies'])
            std_ac = np.std(results[col]['accuracies'])

            print(f"{col:<10} {mean_t:<20.4f} {std_t:<10.4f} {mean_ac:<10.2%} {std_ac:<10.4f}")

        print("="*60)