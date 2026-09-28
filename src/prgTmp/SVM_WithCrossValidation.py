import pandas as pd
import numpy as np
from sklearn.svm import SVC
# 1. 修改导入：使用 RepeatedStratifiedKFold 替代 StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import os

# ==========================================
# 1. 配置路径和参数
# ==========================================
Directory_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_csv\temp\SisFall_dataset_filtered"
csv_path = os.path.join(Directory_path , "Merged_SisFall_featuresReduced.csv")
label_column = 'Label'
random_state = 42

# 2. 修改参数：10折，重复10次 (总共会训练 10 * 10 = 100 个模型)
n_splits = 10
n_repeats = 10

# ==========================================
# 3. 数据读取与预处理
# ==========================================
print("正在读取数据...")
df = pd.read_csv(csv_path)
df[label_column] = df[label_column].astype(int)

# 筛选特征列 (排除非数值列和标签列)
exclude_cols = ['Move', 'Sub', 'Round']
feature_cols = [col for col in df.columns if col not in exclude_cols + [label_column]]

y = df[label_column]
X = df[feature_cols]
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# ==========================================
# 4. 10次10折交叉验证 (Repeated Stratified K-Fold)
# ==========================================
print(f"正在进行 {n_repeats}次 {n_splits}折交叉验证 (SVM)...")

# 初始化重复分层K折
# Stratified 保证了每一折中 摔倒/正常 的比例与原数据集一致
cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

# 用于存储每次验证的指标
list_se = []
list_sp = []
list_ac = []
list_acc = []

fold_count = 0
total_folds = n_splits * n_repeats

# 循环遍历每一个折
for train_index, test_index in cv.split(X, y):
    fold_count += 1
    
    # 划分数据
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 数据标准化 (SVM必须做标准化)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 初始化并训练 SVM 模型
    # kernel='rbf' 是高斯核，适合非线性分类；C=1.0 是正则化参数
    model = SVC(kernel='rbf', C=1.0, random_state=random_state, probability=False)
    model.fit(X_train_scaled, y_train)

    # 预测
    predictions = model.predict(X_test_scaled)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, predictions)
    
    # 提取 TP, TN, FP, FN
    # 假设 Label 1 是正类 (摔倒), Label 0 是负类 (正常)
    # cm[1,1]是TP, cm[0,0]是TN, cm[0,1]是FP, cm[1,0]是FN
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    # 计算指标
    SE = TP / (TP + FN) if (TP + FN) > 0 else 0
    SP = TN / (TN + FP) if (TN + FP) > 0 else 0
    AC = (SE + SP) / 2
    acc = accuracy_score(y_test, predictions)

    # 存入列表
    list_se.append(SE)
    list_sp.append(SP)
    list_ac.append(AC)
    list_acc.append(acc)

    # 可选：打印进度
    if fold_count % 20 == 0:
        print(f"已完成 {fold_count}/{total_folds} 折...")

# ==========================================
# 5. 输出最终结果
# ==========================================
print("\n" + "="*40)
print("       SVM 最终评估报告 (10x10 CV)")
print("="*40)

# 辅助函数：格式化输出 均值 ± 标准差
def fmt(mean_val, std_val):
    return f"{mean_val:.4f} ± {std_val:.4f}"

print(f"{'指标':<20} | {'结果 (Mean ± Std)':<20}")
print("-" * 45)
print(f"{'平均灵敏度 (SE)':<20} | {fmt(np.mean(list_se), np.std(list_se))}")
print(f"{'平均特异度 (SP)':<20} | {fmt(np.mean(list_sp), np.std(list_sp))}")
print(f"{'平均准确率 (AC)':<20} | {fmt(np.mean(list_ac), np.std(list_ac))}")
print(f"{'标准准确率 (Acc)':<20} | {fmt(np.mean(list_acc), np.std(list_acc))}")
print("="*40)