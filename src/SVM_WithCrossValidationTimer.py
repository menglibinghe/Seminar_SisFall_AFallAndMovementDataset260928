import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import os
import time
import psutil

# ==========================================
# 1. 配置路径和参数
# ==========================================
Directory_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_filtered"
#csv_path = os.path.join(Directory_path , "Merged_SisFall_featuresReduced.csv")
csv_path = os.path.join(Directory_path, "Merged_SisFall_featuresReduced - 12.csv")

label_column = 'Label'
random_state = 42

n_splits = 10
n_repeats = 10

# 记录脚本开始时间
script_start_time = time.time()

# ==========================================
# 2. 数据读取与预处理
# ==========================================
print("正在读取数据...")
data_start_time = time.time()

df = pd.read_csv(csv_path)
df[label_column] = df[label_column].astype(int)

# 筛选特征列 (排除非数值列和标签列)
exclude_cols = ['Move', 'Sub', 'Round']
feature_cols = [col for col in df.columns if col not in exclude_cols + [label_column]]

y = df[label_column]
X = df[feature_cols]
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

data_end_time = time.time()

# ==========================================
# 3. 10次10折交叉验证 (Repeated Stratified K-Fold)
# ==========================================
print(f"正在进行 {n_repeats}次 {n_splits}折交叉验证 (SVM)...")

cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

list_se = []
list_sp = []
list_ac = []
list_acc = []

fold_count = 0
total_folds = n_splits * n_repeats

# 记录模型训练开始时间
train_start_time = time.time()

for train_index, test_index in cv.split(X, y):
    fold_count += 1
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 初始化并训练 SVM 模型
    model = SVC(kernel='rbf', C=1.0, random_state=random_state, probability=False)    #tolerance for making a false prediton
    model.fit(X_train_scaled, y_train)

    # 预测
    predictions = model.predict(X_test_scaled)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, predictions)
    
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    # 计算指标
    SE = TP / (TP + FN) if (TP + FN) > 0 else 0
    SP = TN / (TN + FP) if (TN + FP) > 0 else 0
    AC = (SE + SP) / 2
    acc = accuracy_score(y_test, predictions)

    list_se.append(SE)
    list_sp.append(SP)
    list_ac.append(AC)
    list_acc.append(acc)

    if fold_count % 20 == 0:
        print(f"已完成 {fold_count}/{total_folds} 折...")

# 记录模型训练结束时间
train_end_time = time.time()

# ==========================================
# 4. 统一耗时与CPU统计
# ==========================================

# 1. 计算总耗时
total_duration = time.time() - script_start_time

# 2. 计算CPU平均占用率
process = psutil.Process(os.getpid())
cpu_time_seconds = process.cpu_times().user + process.cpu_times().system
cpu_usage_avg = (cpu_time_seconds / total_duration) * 100

# ==========================================
# 5. 最终格式化输出
# ==========================================

# 辅助函数：格式化输出 均值 ± 标准差
def fmt(mean_val, std_val):
    return f"{mean_val:.4f} ± {std_val:.4f}"

print("\n" + "="*50)
print("       SVM 10次10折交叉验证报告")
print(f"Filen name: {os.path.basename(csv_path)}")
print("="*50)

print(f"数据处理耗时: {data_end_time - data_start_time:.2f} 秒")
print(f"总程序运行时长: {total_duration:.2f} 秒")
print(f"CPU平均占用率: {cpu_usage_avg:.1f}%")
print("="*50)

print(f"{'指标':<20} | {'结果 (Mean ± Std)':<20}")
print("-" * 45)
print(f"{'平均准确率 (AC)':<20} | {fmt(np.mean(list_ac), np.std(list_ac))}")
print(f"{'平均灵敏度 (SE)':<20} | {fmt(np.mean(list_se), np.std(list_se))}")
print(f"{'平均特异度 (SP)':<20} | {fmt(np.mean(list_sp), np.std(list_sp))}")
print("="*50)
