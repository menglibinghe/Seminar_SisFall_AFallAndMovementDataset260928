import numpy as np
import pandas as pd
import os
from sklearn.model_selection import RepeatedStratifiedKFold
import time
import psutil  # 导入 psutil

# ==========================================
# 1. 配置路径和参数
# ==========================================
Directory_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_csv\temp\SisFall_dataset_filtered"
csv_path = os.path.join(Directory_path , "Merged_SisFall_featuresReduced.csv")

features_to_analyze = ['mean_Pitch_C4', 'std_HOR'] 
label_column = 'Label' 
n_splits = 10
n_repeats = 10

# 记录脚本开始时间
script_start_time = time.time()

# ==========================================
# 2. 数据读取与预处理 (计时开始)
# ==========================================
print("正在读取数据...")
data_start_time = time.time()

# 读取数据
df = pd.read_csv(csv_path)
df[label_column] = df[label_column].astype(int)

data_end_time = time.time()

# ==========================================
# 3. 核心优化：使用向量化替代循环 (极速版)
# ==========================================
def fast_threshold_search(train_feat, train_lbl, test_feat, test_lbl):
    """
    使用 NumPy 向量化加速阈值搜索
    """
    # 1. 组合并排序
    combined = pd.DataFrame({'val': train_feat, 'y': train_lbl})
    combined = combined.sort_values('val').reset_index(drop=True)
    
    vals = combined['val'].values
    labels = combined['y'].values
    
    total_pos = labels.sum()
    total_neg = len(labels) - total_pos
    
    if total_pos == 0 or total_neg == 0:
        return 0, 0, 0

    # 2. 计算累积和
    cum_pos = np.cumsum(labels)
    cum_neg = np.cumsum(1 - labels)
    
    # 3. 向量化计算指标
    fns = cum_pos
    tns = cum_neg
    tps = total_pos - cum_pos
    fps = total_neg - cum_neg
    
    ses = np.where((tps + fns) > 0, tps / (tps + fns), 0)
    sps = np.where((tns + fps) > 0, tns / (tns + fps), 0)
    acs = (ses + sps) / 2
    
    # 4. 找到最佳阈值
    best_ac_idx = np.argmax(acs)
    best_threshold = vals[best_ac_idx]
    
    # 5. 测试集验证
    test_preds = (test_feat > best_threshold).astype(int)
    
    TP_test = ((test_preds == 1) & (test_lbl == 1)).sum()
    FN_test = ((test_preds == 0) & (test_lbl == 1)).sum()
    TN_test = ((test_preds == 0) & (test_lbl == 0)).sum()
    FP_test = ((test_preds == 1) & (test_lbl == 0)).sum()
    
    SE_test = TP_test / (TP_test + FN_test) if (TP_test + FN_test) > 0 else 0
    SP_test = TN_test / (TN_test + FP_test) if (TN_test + FP_test) > 0 else 0
    AC_test = (SE_test + SP_test) / 2
    
    return SE_test, SP_test, AC_test

# ==========================================
# 4. 定义交叉验证分析函数
# ==========================================
def analyze_threshold_cv_fast(df, feature_col, label_col, n_splits=10, n_repeats=10):
    # 去掉内部print，保持界面整洁，或者只保留关键进度
    # print(f"\n正在对特征 '{feature_col}' 进行 {n_repeats}次{n_splits}折交叉验证...")
    
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    
    fold_se_final = []
    fold_sp_final = []
    fold_ac_final = []
    
    total_folds = n_splits * n_repeats
    
    # --- 模型训练/验证计时开始 ---
    train_start_time = time.time()
    
    for fold_idx, (train_index, test_index) in enumerate(rkf.split(df, df[label_column])):
        
        # 提取数据
        train_feat = df.iloc[train_index][feature_col].values
        train_lbl = df.iloc[train_index][label_col].values
        test_feat = df.iloc[test_index][feature_col].values
        test_lbl = df.iloc[test_index][label_col].values
        
        # 调用极速函数
        SE_test, SP_test, AC_test = fast_threshold_search(train_feat, train_lbl, test_feat, test_lbl)
        
        fold_se_final.append(SE_test)
        fold_sp_final.append(SP_test)
        fold_ac_final.append(AC_test)

    # --- 模型训练/验证计时结束 ---
    train_end_time = time.time()
    
    # 计算统计值
    final_mean_se = np.mean(fold_se_final)
    final_std_se = np.std(fold_se_final)
    final_mean_sp = np.mean(fold_sp_final)
    final_std_sp = np.std(fold_sp_final)
    final_mean_ac = np.mean(fold_ac_final)
    final_std_ac = np.std(fold_ac_final)
    
    train_duration = train_end_time - train_start_time
    
    return (final_mean_ac, final_std_ac, 
            final_mean_se, final_std_se, 
            final_mean_sp, final_std_sp, 
            train_duration)

# ==========================================
# 5. 执行分析
# ==========================================
all_results = []

for col in features_to_analyze:
    if col in df.columns:
        res = analyze_threshold_cv_fast(df, col, label_column, n_splits=10, n_repeats=10)
        all_results.append((col, res))

# ==========================================
# 6. 统一耗时与CPU统计 (关键修改部分)
# ==========================================

# 1. 计算总耗时
total_duration = time.time() - script_start_time


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


# ==========================================
# 7. 最终格式化输出
# ==========================================
print("\n" + "="*50)
print("       阈值法评估报告 (10x10 CV)")
print("="*50)

def format_metric(mean_val, std_val):
    return f"{mean_val:.4f} ± {std_val:.4f}"


print("="*50)
print(f"数据处理耗时: {data_end_time - data_start_time:.2f} 秒") 
print(f"总程序运行时长: {total_duration:.2f} 秒")
print(f"CPU平均占用率: {cpu_usage_avg:.1f}%") # 统一格式
print("="*50)

# 遍历输出结果
for col_name, res in all_results:
    mean_ac, std_ac, mean_se, std_se, mean_sp, std_sp, train_dur = res
    
    print(f"\n特征: {col_name}")
    print("-" * 50)
    print(f"{'指标':<20} | {'结果 (均值 ± 波动)':<20}")
    print("-" * 50)
    print(f"{'平均准确率 (AC)':<20} | {format_metric(mean_ac, std_ac):<20}")
    print(f"{'平均灵敏度 (SE)':<20} | {format_metric(mean_se, std_se):<20}")
    print(f"{'平均特异度 (SP)':<20} | {format_metric(mean_sp, std_sp):<20}")
    print("-" * 50)
    print(f"⏳ 该特征分析耗时: {train_dur:.2f} 秒")
