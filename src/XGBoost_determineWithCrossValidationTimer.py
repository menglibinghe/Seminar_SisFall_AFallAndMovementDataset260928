import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold  # 使用分层K折
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import os
import time
import psutil  # 1. 导入 psutil 用于CPU监控

# ==========================================
# 1. 配置路径和参数
# ==========================================
Directory_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_filtered"
csv_path = os.path.join(Directory_path , "Merged_SisFall_featuresReduced - 12.csv")
#csv_path = os.path.join(Directory_path, "Merged_SisFall_FullFeatures.csv")

label_column = 'Label'
random_state = 42
n_splits = 10  # 10折交叉验证

# --- 记录程序开始总时间 ---
script_start_time = time.time()

# ==========================================
# 2. 数据读取与预处理
# ==========================================
print("正在读取数据...")
data_start_time = time.time() # 记录数据处理开始时间

df = pd.read_csv(csv_path)
df[label_column] = df[label_column].astype(int)

exclude_cols = ['Move', 'Sub', 'Round']
feature_cols = [col for col in df.columns if col not in exclude_cols + [label_column]]
X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
y = df[label_column]

data_end_time = time.time() # 记录数据处理结束时间

# ==========================================
# 3. 定义交叉验证过程
# ==========================================
print(f"正在执行 {n_splits}折 交叉验证...")

# 初始化 KFold (分层抽样保证每折中各类别比例一致)
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# 用于存储每一折的指标
cv_ac_list = []
cv_se_list = []
cv_sp_list = []

fold_idx = 1

# --- 记录模型训练开始时间 ---
train_start_time = time.time()

# 循环进行训练和验证
for train_index, test_index in skf.split(X, y):
    print(f"正在训练第 {fold_idx}/{n_splits} 折...", end='\r')
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # 数据标准化 (注意：必须在每折内部fit，不能用全局数据fit)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 模型训练
    model = xgb.XGBClassifier(
        learning_rate=0.1,
        max_depth=3,
        n_estimators=100,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0  # 关闭训练过程的输出
    )
    model.fit(X_train_scaled, y_train)
    
    # 预测
    predictions = model.predict(X_test_scaled)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, predictions)
    
    # 提取 TP, TN, FP, FN
    # 假设标签 0 是负类，1 是正类
    TN, FP, FN, TP = cm.ravel()
    
    # 计算单折指标
    SE = TP / (TP + FN) if (TP + FN) > 0 else 0
    SP = TN / (TN + FP) if (TN + FP) > 0 else 0
    AC = (SE + SP) / 2  # 平均准确率 (平衡准确率)
    
    # 存入列表
    cv_ac_list.append(AC)
    cv_se_list.append(SE)
    cv_sp_list.append(SP)
    
    fold_idx += 1

# --- 记录模型训练结束时间 ---
train_end_time = time.time()
print("\n验证完成。正在计算统计结果...")

# ==========================================
# 4. 输出最终结果 (均值 ± 标准差) 与 耗时
# ==========================================
def format_metric(mean_val, std_val):
    return f"{mean_val:.4f} ± {std_val:.4f}"

mean_ac = np.mean(cv_ac_list)
std_ac = np.std(cv_ac_list)
mean_se = np.mean(cv_se_list)
std_se = np.std(cv_se_list)
mean_sp = np.mean(cv_sp_list)
std_sp = np.std(cv_sp_list)

# 计算耗时
data_duration = data_end_time - data_start_time
train_duration = train_end_time - train_start_time
total_duration = train_end_time - script_start_time

# 获取 CPU 占用率 (与随机森林代码保持一致的统计方式)

# 获取当前进程
process = psutil.Process(os.getpid())

# 获取该进程自启动以来的累计 CPU 时间 (用户态 + 内核态)
# 这是一个非常精确的指标，代表了 CPU 真正为这个程序干活的时间总和
cpu_time_seconds = process.cpu_times().user + process.cpu_times().system

# 计算平均 CPU 占用率
# 公式：(CPU干活总秒数 / 程序运行总秒数) * 100%
# 注意：如果是多核并行计算，这个数值理论上可以超过 100% (例如 400% 代表占满 4 个核)
# 为了和任务管理器 (0-100%) 对应，我们将其限制在 100% 以内，或者显示实际占比
cpu_usage_avg = (cpu_time_seconds / total_duration) * 100

# 如果电脑是多核，且算法并行度高，上面算出来可能超过100%。
# 通常任务管理器显示的是 "总负载 / 总核心数"。
# 假设你的电脑是 8 核，任务管理器显示的 50% 其实相当于 4 个核满载。
# 为了直观对比任务管理器，我们通常看这个比例。



print("\n" + "="*50)
print("       XGBoost 10折交叉验证报告")
print(f"Filen name: {os.path.basename(csv_path)}")
print("="*50)
print(f"数据处理耗时: {data_duration:.2f} 秒")
print(f"总程序耗时: {total_duration:.2f} 秒")
print(f"CPU平均占用率: {cpu_usage_avg:.1f}%")
print("="*50)

print(f"{'指标':<20} | {'结果 (均值 ± 波动)':<20}")
print("-" * 50)
print(f"{'平均准确率 (AC)':<20} | {format_metric(mean_ac, std_ac):<20}")
print(f"{'平均灵敏度 (SE)':<20} | {format_metric(mean_se, std_se):<20}")
print(f"{'平均特异度 (SP)':<20} | {format_metric(mean_sp, std_sp):<20}")
print("="*50)
