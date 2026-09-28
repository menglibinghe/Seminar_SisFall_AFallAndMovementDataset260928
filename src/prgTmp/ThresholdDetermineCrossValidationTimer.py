import numpy as np
import pandas as pd
import os
from sklearn.model_selection import RepeatedStratifiedKFold
import time  # 导入时间模块用于计时

# ==========================================
# 1. 配置路径和参数
# ==========================================
Directory_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_csv\temp\SisFall_dataset_filtered"
csv_path = os.path.join(Directory_path , "Merged_SisFall_featuresReduced.csv")

# 需要分析的特征列
features_to_analyze = ['mean_Pitch_C4', 'std_HOR'] 
label_column = 'Label' 
n_splits = 10
n_repeats = 10

# 读取数据
df = pd.read_csv(csv_path)
df[label_column] = df[label_column].astype(int)

# ==========================================
# 2. 定义交叉验证分析函数 (无绘图版)
# ==========================================
def analyze_threshold_cv(df, feature_col, label_col, n_splits=10, n_repeats=10):
    print(f"\n正在对特征 '{feature_col}' 进行 {n_repeats}次{n_splits}折交叉验证...")
    
    # 初始化交叉验证器
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    
    # 用于存储每一折的最终指标
    fold_se_final = []
    fold_ac_final = []
    
    total_folds = n_splits * n_repeats
    
    # 记录该特征分析的开始时间
    feature_start_time = time.time()
    
    # --- 开始循环 ---
    for fold_idx, (train_index, test_index) in enumerate(rkf.split(df, df[label_column])):
        # 记录当前折的开始时间
        fold_start_time = time.time()
        
        # 1. 划分训练集和测试集
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        
        # 2. 在【训练集】上寻找最佳阈值 (T1)
        train_values = np.sort(train_df[feature_col].unique())
        
        best_ac_train = 0
        T1 = train_values[0] 
        
        # 遍历所有可能的阈值
        for t in train_values:
            preds = (train_df[feature_col] > t).astype(int)
            
            TP = ((preds == 1) & (train_df[label_col] == 1)).sum()
            FN = ((preds == 0) & (train_df[label_col] == 1)).sum()
            TN = ((preds == 0) & (train_df[label_col] == 0)).sum()
            FP = ((preds == 1) & (train_df[label_col] == 0)).sum()
            
            SE = TP / (TP + FN) if (TP + FN) > 0 else 0
            SP = TN / (TN + FP) if (TN + FP) > 0 else 0
            AC = (SE + SP) / 2
            
            if AC > best_ac_train:
                best_ac_train = AC
                T1 = t
        
        # 3. 使用找到的 T1 在【测试集】上评估
        test_preds = (test_df[feature_col] > T1).astype(int)
        
        TP_test = ((test_preds == 1) & (test_df[label_col] == 1)).sum()
        FN_test = ((test_preds == 0) & (test_df[label_col] == 1)).sum()
        TN_test = ((test_preds == 0) & (test_df[label_col] == 0)).sum()
        FP_test = ((test_preds == 1) & (test_df[label_col] == 0)).sum()
        
        SE_test = TP_test / (TP_test + FN_test) if (TP_test + FN_test) > 0 else 0
        SP_test = TN_test / (TN_test + FP_test) if (TN_test + FP_test) > 0 else 0
        AC_test = (SE_test + SP_test) / 2
        
        # 保存结果
        fold_se_final.append(SE_test)
        fold_ac_final.append(AC_test)
        
        # 计算当前折耗时
        fold_duration = time.time() - fold_start_time
        
        # 每20折打印一次进度
        if (fold_idx + 1) % 20 == 0:
            print(f"  进度: {fold_idx + 1}/{total_folds} 折 (本折耗时: {fold_duration:.2f}s)")

    # ==========================================
    # 3. 计算统计结果
    # ==========================================
    # 计算总耗时
    total_duration = time.time() - feature_start_time
    
    # 计算均值和标准差
    final_mean_se = np.mean(fold_se_final)
    final_std_se = np.std(fold_se_final)
    final_mean_ac = np.mean(fold_ac_final)
    final_std_ac = np.std(fold_ac_final)
    
    # ==========================================
    # 4. 输出纯文本报告
    # ==========================================
    print("-" * 60)
    print(f"Treshold determine: {feature_col} 最终评估报告 (10x10 CV)")
    print("-" * 60)
    print(f"平均灵敏度 (SE): {final_mean_se:.4f} ± {final_std_se:.4f}")
    print(f"平均准确率 (AC): {final_mean_ac:.4f} ± {final_std_ac:.4f}")
    print("-" * 60)
    print(f"⏱️  该特征总耗时: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
    print(f"⚡  平均每折耗时: {total_duration/total_folds:.2f} 秒")
    print("=" * 60)

# ==========================================
# 5. 执行分析
# ==========================================
script_start_time = time.time()

for col in features_to_analyze:
    if col in df.columns:
        analyze_threshold_cv(df, col, label_column, n_splits=10, n_repeats=10)
    else:
        print(f"警告: 列 {col} 不存在于数据中！")

print(f"\n🎉 所有特征分析完成！总程序运行时长: {time.time() - script_start_time:.2f} 秒")