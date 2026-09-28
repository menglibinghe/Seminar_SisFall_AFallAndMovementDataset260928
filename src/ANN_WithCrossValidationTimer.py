import pandas as pd
import numpy as np
import time
import os
import warnings
import psutil  # 1. 导入 psutil 用于CPU监控

# 导入神经网络模块
from sklearn.neural_network import MLPClassifier
# 导入交叉验证模块
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# 忽略收敛警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 配置路径和参数
# ==========================================
Directory_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_filtered"
#csv_path = os.path.join(Directory_path , "Merged_SisFall_featuresReduced - 12.csv")
csv_path = os.path.join(Directory_path, "Merged_SisFall_featuresReduced - 12.csv")

label_column = 'Label'
random_state = 42

# 设置 10次 10折交叉验证
n_splits = 10
n_repeats = 10

# ==========================================
# 2. 数据读取与预处理
# ==========================================
print("正在读取数据...")
data_start_time = time.time() # 记录数据处理开始时间
df = pd.read_csv(csv_path)
df[label_column] = df[label_column].astype(int)

# 剔除非数值列
exclude_cols = ['Move', 'Sub', 'Round']
feature_cols = [col for col in df.columns 
                if col not in exclude_cols + [label_column]]

print(f"检测到 {len(feature_cols)} 个有效数值特征。")

# 准备 X 和 y
y = df[label_column]
X = df[feature_cols]
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
data_end_time = time.time() # 记录数据处理结束时间

# ==========================================
# 3. 10次10折交叉验证 (Repeated Stratified K-Fold)
# ==========================================
print(f"正在进行 {n_repeats}次 {n_splits}折交叉验证 (ANN)...")
print("注意：神经网络训练较慢，请耐心等待...")

# 初始化交叉验证器
cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

# 用于存储每折的指标
list_se = []
list_sp = []
list_ac = []
list_acc = []

fold_count = 0
total_folds = n_splits * n_repeats

# --- 记录模型训练开始时间 ---
train_start_time = time.time()

# 循环遍历每一个折
for train_index, test_index in cv.split(X, y):
    fold_count += 1
    
    # 1. 划分数据
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # 2. 数据标准化 (神经网络必须做标准化)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. 构建人工神经网络模型
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),                #first hidden layer is 64 nodes, the second 32.
        activation='relu',                          #activation function type
        solver='adam', 
        alpha=0.0001,                               #L2 normalization parameter
        batch_size='auto', 
        learning_rate='adaptive',
        learning_rate_init=0.001, 
        max_iter=500,                               #maximum iterations
        shuffle=True, 
        random_state=random_state, 
        early_stopping=True,
        validation_fraction=0.1,                    #the proportion of validation set
        verbose=False
    )
    
    # 4. 训练模型
    model.fit(X_train_scaled, y_train)
    
    # 5. 预测
    predictions = model.predict(X_test_scaled)
    
    # 6. 计算混淆矩阵
    cm = confusion_matrix(y_test, predictions)
    
    # 提取 TP, TN, FP, FN
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
    else:
        TN, FP, FN, TP = 0, 0, 0, 0
        
    # 7. 计算指标
    SE = TP / (TP + FN) if (TP + FN) > 0 else 0
    SP = TN / (TN + FP) if (TN + FP) > 0 else 0
    AC = (SE + SP) / 2
    acc = accuracy_score(y_test, predictions)
    
    # 存入列表
    list_se.append(SE)
    list_sp.append(SP)
    list_ac.append(AC)
    list_acc.append(acc)
    
    # 打印进度
    if fold_count % 20 == 0 or fold_count == total_folds:
        print(f"进度: {fold_count}/{total_folds} 折已完成...")

# --- 记录模型训练结束时间 ---
train_end_time = time.time()

# ==========================================
# 4. 输出最终结果与耗时统计
# ==========================================
print("\n" + "="*50)

# 计算耗时
data_duration = data_end_time - data_start_time
train_duration = train_end_time - train_start_time
total_duration = train_end_time - data_start_time

# ==========================================
# 5. 修正后的 CPU 统计逻辑
# ==========================================
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

print(f"       人工神经网络 (ANN) 评估报告")
print(f"       (10次 10折交叉验证)")
print(f"Filen name: {os.path.basename(csv_path)}")
print("="*50)
print(f"数据处理耗时: {data_duration:.2f} 秒")
#print(f"模型训练耗时: {train_duration:.2f} 秒")
print(f"总程序耗时: {total_duration:.2f} 秒")
# 显示修正后的 CPU 占用率
print(f"⚡ CPU平均占用率: {cpu_usage_avg:.1f}%")

# 辅助函数：格式化输出
def fmt(mean_val, std_val):
    return f"{mean_val:.4f} ± {std_val:.4f}"

print(f"{'指标':<20} | {'结果 (均值 ± 波动)':<20}")
print("-" * 50)
print(f"{'平均准确率 (AC)':<20} | {fmt(np.mean(list_ac), np.std(list_ac))}")
print(f"{'平均灵敏度 (SE)':<20} | {fmt(np.mean(list_se), np.std(list_se))}")
print(f"{'平均特异度 (SP)':<20} | {fmt(np.mean(list_sp), np.std(list_sp))}")
print("-" * 50)

print("="*50)