import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score
import warnings
import time
import psutil
import os

warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置路径
# ==========================================
input_file = r'D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\csv2txtCombine\SisFall_With_Features.csv'

# 记录脚本开始时间
script_start_time = time.time()

# ==========================================
# 2. 加载数据
# ==========================================
print("正在加载数据...")
data_start_time = time.time()

df = pd.read_csv(input_file)
print("数据列名:", df.columns.tolist())

# ==========================================
# 3. 数据预处理
# ==========================================
feature_cols = [col for col in df.columns if col not in ['label', 'subject']]

def map_label(act):
    if act.startswith('D01') or act.startswith('D02'):
        return 1
    else:
        return 0

df['target'] = df['label'].apply(map_label)

X = df[feature_cols]
y = df['target']
groups = df['subject']

print(f"数据总量: {len(df)}")
print(f"跌倒样本比例: {y.mean():.2f}")

data_end_time = time.time()

# ==========================================
# 4. 模型训练与验证
# ==========================================
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

cv = GroupKFold(n_splits=10)

def sensitivity_score(y_true, y_pred):
    return recall_score(y_true, y_pred)

def specificity_score(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)

scoring = {
    'Accuracy': 'accuracy',
    'Sensitivity': make_scorer(sensitivity_score),
    'Specificity': make_scorer(specificity_score)
}

print("\n开始 10折交叉验证 (按受试者划分)...")
print("注意：由于使用了 n_jobs=-1，CPU 占用率可能会瞬间飙升...")

train_start_time = time.time()

# 执行交叉验证
scores = cross_validate(model, X, y, groups=groups, cv=cv, scoring=scoring, n_jobs=-1)

train_end_time = time.time()

# ==========================================
# 5. 统一耗时与CPU统计
# ==========================================

# 1. 计算总耗时
total_duration = time.time() - script_start_time

# 2. 计算CPU平均占用率
process = psutil.Process(os.getpid())
cpu_time_seconds = process.cpu_times().user + process.cpu_times().system
cpu_usage_avg = (cpu_time_seconds / total_duration) * 100

# ==========================================
# 6. 输出结果与耗时
# ==========================================
data_duration = data_end_time - data_start_time
train_duration = train_end_time - train_start_time

print("\n" + "="*40)
print("      Random Forest 验证报告")
print("="*40)
print(f"数据处理耗时: {data_duration:.2f} 秒")
print(f"模型训练耗时: {train_duration:.2f} 秒")
print(f"总程序运行时长: {total_duration:.2f} 秒")
print(f"CPU平均占用率: {cpu_usage_avg:.1f}%")
print("-" * 40)
print(f"平均准确率 (Accuracy): {scores['test_Accuracy'].mean():.4f} (+/- {scores['test_Accuracy'].std() * 2:.4f})")
print(f"平均敏感度 (Sensitivity): {scores['test_Sensitivity'].mean():.4f}")
print(f"平均特异度 (Specificity): {scores['test_Specificity'].mean():.4f}")
print("="*40)