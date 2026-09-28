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
Directory_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_csv\temp\SisFall_dataset_filtered"
csv_path = os.path.join(Directory_path, "Merged_SisFall_featuresReduced.csv")

# 记录脚本开始时间
script_start_time = time.time()

# ==========================================
# 2. 加载数据
# ==========================================
print("正在加载数据...")
data_start_time = time.time()

df = pd.read_csv(csv_path)
# print("数据列名:", df.columns.tolist()) # 为了输出整洁，暂时注释掉列名打印

# ==========================================
# 3. 数据预处理
# ==========================================
# 确保排除非特征列，这里假设 'Label' , 'Move','Sub','Round' 是必须排除的
feature_cols = [col for col in df.columns if col not in ['Label', 'Sub', 'Move','Sub','Round']]

# def map_label(act):
#     # 确保 act 是字符串，防止报错
#     act_str = str(act)
#     if act_str.startswith('D'):
#         return 0
#     else:
#         return 1

#df['target'] = df['Label'].apply(map_label)
df['target'] = df['Label']

X = df[feature_cols]
y = df['target']
groups = df['Sub']

print(f"数据总量: {len(df)}")
print(f"跌倒样本比例: {y.mean():.2f}")

data_end_time = time.time()

# ==========================================
# 4. 模型训练与验证
# ==========================================
model = RandomForestClassifier(
    n_estimators=100,                       #the number of decision trees
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
total_duration = time.time() - script_start_time

process = psutil.Process(os.getpid())
cpu_time_seconds = process.cpu_times().user + process.cpu_times().system
cpu_usage_avg = (cpu_time_seconds / total_duration) * 100

# ==========================================
# 6. 输出结果 (格式调整)
# ==========================================

# 1. 定义格式化函数 (与你提供的代码一致)
def format_metric(mean_val, std_val):
    return f"{mean_val:.4f} ± {std_val:.4f}"

# 2. 计算各项指标的均值和标准差
mean_acc = scores['test_Accuracy'].mean()
std_acc = scores['test_Accuracy'].std()

mean_se = scores['test_Sensitivity'].mean()
std_se = scores['test_Sensitivity'].std()

mean_sp = scores['test_Specificity'].mean()
std_sp = scores['test_Specificity'].std()

data_duration = data_end_time - data_start_time
train_duration = train_end_time - train_start_time

# 3. 打印报告
print("\n" + "="*50)
print("      Random Forest 验证报告")
print("="*50)
print(f"数据处理耗时: {data_duration:.2f} 秒")
print(f"模型训练耗时: {train_duration:.2f} 秒")
print(f"总程序运行时长: {total_duration:.2f} 秒")
print(f"CPU平均占用率: {cpu_usage_avg:.1f}%")
print("-" * 50)

# 使用统一的表格格式输出
print(f"{'指标':<20} | {'结果 (均值 ± 波动)':<20}")
print("-" * 50)
print(f"{'平均准确率 (AC)':<20} | {format_metric(mean_acc, std_acc):<20}")
print(f"{'平均灵敏度 (SE)':<20} | {format_metric(mean_se, std_se):<20}")
print(f"{'平均特异度 (SP)':<20} | {format_metric(mean_sp, std_sp):<20}")
print("="*50)