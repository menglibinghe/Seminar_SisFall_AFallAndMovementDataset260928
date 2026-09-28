import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
# 注意：这里只导入了基础库，不需要 RepeatedStratifiedGroupKFold
from sklearn.model_selection import StratifiedGroupKFold 

# ==========================================
# 1. 路径与参数配置
# ==========================================
base_path = r'D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database'
input_file = os.path.join(base_path, 'reduced_SisFall.txt')   ##仅测试耗时，原为reduced_SisFall.txt

FS = 200        
CUTOFF = 5      
WINDOW_SIZE = 200 

# 交叉验证配置
N_SPLITS = 5
N_REPEATS = 5

# ==========================================
# 2. 信号处理函数 (保持不变)
# ==========================================
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

# ==========================================
# 3. 主程序逻辑 (修改为手动循环)
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(input_file):
        print(f"❌ 找不到文件: {input_file}")
    else:
        print(f"📖 正在加载: {input_file}")
        df = pd.read_csv(input_file, sep='\t')
        df['label'] = df['Activity'].str.startswith('F').astype(int)

        print("⚡ 正在执行 5Hz 低通滤波...")
        cols = ['acc1_x', 'acc1_y', 'acc1_z']
        df[cols] = butter_lowpass_filter(df[cols].values, CUTOFF, FS)

        print("📊 正在计算特征...")
        # C1
        df['C1'] = np.sqrt(df['acc1_x']**2 + df['acc1_y']**2 + df['acc1_z']**2)
        # C2
        df['C2'] = np.sqrt(df['acc1_x']**2 + df['acc1_z']**2)
        # C3
        df['C3'] = df['C1'].rolling(WINDOW_SIZE).apply(lambda x: np.max(x) - np.min(x))
        # C8
        std_x = df['acc1_x'].rolling(WINDOW_SIZE).std()
        std_z = df['acc1_z'].rolling(WINDOW_SIZE).std()
        df['C8'] = np.sqrt(std_x**2 + std_z**2)
        # C9
        std_y = df['acc1_y'].rolling(WINDOW_SIZE).std()
        df['C9'] = np.sqrt(std_x**2 + std_y**2 + std_z**2)
        # C13
        df['C13'] = df['C2'].rolling(WINDOW_SIZE).mean()

        df_clean = df.dropna().reset_index(drop=True)

        # 准备数据
        # 注意：sklearn 需要 numpy 数组
        X = df_clean[['C2', 'C3', 'C8', 'C9', 'C13']]
        y = df_clean['label']
        # 提取 Subject ID 用于分组 (假设 subject 列存在且格式如 SA01)
        # 如果 subject 列是字符串，我们需要将其转换为数字或者确保 GroupKFold 能处理它
        groups = df_clean['subject'] 

        print(f"\n🚀 开始 {N_REPEATS}次 {N_SPLITS}折交叉验证 (手动循环版)...")
        
        # 存储结果
        results = {col: {'thresholds': [], 'accuracies': []} for col in X.columns}

        # --- 手动实现 RepeatedStratifiedGroupKFold ---
        # 外层循环：重复次数
        for repeat in range(N_REPEATS):
            # 内层循环：K折
            # 注意：为了保证每次重复的随机性不同，我们设置不同的 random_state
            # 如果你的 sklearn 版本太老不支持 random_state in StratifiedGroupKFold，可能需要手动打乱索引
            sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=repeat)
            
            for train_index, test_index in sgkf.split(X, y, groups):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # 对每一个特征单独训练一个深度为1的决策树
                for col in X.columns:
                    # 取出单列特征 (需要 reshape 成二维数组)
                    train_feat = X_train[[col]].values
                    test_feat = X_test[[col]].values

                    # 训练决策树桩 (Decision Stump)
                    # max_depth=1 意味着只找一个最佳阈值进行分割
                    clf = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=42)
                    clf.fit(train_feat, y_train)

                    # 预测并计算 Balanced Accuracy
                    y_pred = clf.predict(test_feat)
                    acc = balanced_accuracy_score(y_test, y_pred)

                    # 获取阈值 (Decision Tree 的阈值存储在 tree_.threshold[0] 中)
                    # 因为 max_depth=1，只有一个节点，索引为 0
                    threshold = clf.tree_.threshold[0]

                    # 保存结果
                    results[col]['thresholds'].append(threshold)
                    results[col]['accuracies'].append(acc)

        # --- 输出最终结果 ---
        print("\n" + "="*60)
        print(f"{'特征':<10} {'平均 T1 阈值':<20} {'std':<10} {'平均 AC':<10} {'std':<10}")
        print("-" * 60)

        for col in X.columns:
            mean_t = np.mean(results[col]['thresholds'])
            std_t = np.std(results[col]['thresholds'])
            mean_ac = np.mean(results[col]['accuracies'])
            std_ac = np.std(results[col]['accuracies'])

            print(f"{col:<10} {mean_t:<20.4f} {std_t:<10.4f} {mean_ac:<10.2%} {std_ac:<10.4f}")
        
        print("="*60)