import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import os

# ==========================================
# 1. 配置路径和参数
# ==========================================
Directory_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_csv\temp\SisFall_dataset_filtered"
csv_path = os.path.join(Directory_path , "Merged_SisFall_featuresReduced.csv")
label_column = 'Label'
random_state = 42

# ==========================================
# 2. 数据读取与预处理 (已修复)
# ==========================================
print("正在读取数据...")
df = pd.read_csv(csv_path)

# 确保标签是整数类型
df[label_column] = df[label_column].astype(int)

# --- 核心修复步骤 ---
# 1. 定义需要剔除的非数值列（元数据）
exclude_cols = ['Move', 'Sub', 'Round']

# 2. 筛选特征列：
# 逻辑：是数字列 + 不是标签列 + 不是元数据列
feature_cols = [col for col in df.columns 
                if col not in exclude_cols + [label_column]]

print(f"检测到 {len(feature_cols)} 个有效数值特征。")
# print(f"剔除的列: {exclude_cols}") # 调试用

# 3. 准备 X 和 y
y = df[label_column]
X = df[feature_cols]

# 强制确保 X 中全是数字 (防止其他潜在脏数据)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. 模型训练 (XGBoost)
# ==========================================
print("正在训练 XGBoost 模型...")
model = xgb.XGBClassifier(
    learning_rate=0.1,
    max_depth=3,
    n_estimators=100,
    random_state=random_state,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train_scaled, y_train)

# ==========================================
# 4. 预测与评估
# ==========================================
print("正在测试集上进行预测...")
predictions = model.predict(X_test_scaled)

# 计算混淆矩阵
cm = confusion_matrix(y_test, predictions)

# 提取 TP, TN, FP, FN
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[1, 1]

# 计算指标
SE = TP / (TP + FN) if (TP + FN) > 0 else 0
SP = TN / (TN + FP) if (TN + FP) > 0 else 0
AC = (SE + SP) / 2
standard_acc = accuracy_score(y_test, predictions)

# ==========================================
# 5. 输出结果
# ==========================================
print("\n" + "="*30)
print("       评估结果报告")
print("="*30)
print(f"混淆矩阵:\n{cm}")
print("-" * 30)
print(f"灵敏度 (SE): {SE:.4f}")
print(f"特异度 (SP): {SP:.4f}")
print(f"平均准确率 (AC): {AC:.4f}")
print(f"标准准确率 (Overall Acc): {standard_acc:.4f}")
print("="*30)