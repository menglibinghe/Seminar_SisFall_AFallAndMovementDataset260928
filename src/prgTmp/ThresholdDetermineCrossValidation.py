import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import RepeatedStratifiedKFold

# 设置中文字体，防止画图时中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 配置路径和参数
# ==========================================
Directory_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_csv\temp\SisFall_dataset_filtered"
csv_path = os.path.join(Directory_path , "Merged_SisFall_featuresReduced.csv")

features_to_analyze = ['mean_Pitch_C4', 'std_HOR'] 
label_column = 'Label' 
n_splits = 10
n_repeats = 10

# 读取数据
df = pd.read_csv(csv_path)
df[label_column] = df[label_column].astype(int)

# ==========================================
# 2. 定义交叉验证分析函数
# ==========================================
def analyze_threshold_cv(df, feature_col, label_col, n_splits=10, n_repeats=10):
    print(f"\n正在对特征 '{feature_col}' 进行 {n_repeats}次{n_splits}折交叉验证...")
    
    # 初始化交叉验证器 (分层抽样保证摔倒/正常比例一致)
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    
    # 用于存储每一折的指标，最后求平均曲线
    all_thresholds = []
    all_se_scores = []
    all_sp_scores = []
    all_ac_scores = []
    
    # 用于统计最终的平均性能
    fold_se_final = []
    fold_ac_final = []
    
    total_folds = n_splits * n_repeats
    
    # --- 开始循环 ---
    for fold_idx, (train_index, test_index) in enumerate(rkf.split(df, df[label_column])):
        # 1. 划分训练集和测试集
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        
        # 2. 在【训练集】上寻找最佳阈值 (T1, T2)
        # 注意：这里只能用训练集的数据来定标准，否则就是作弊（数据泄露）
        train_values = np.sort(train_df[feature_col].unique())
        
        best_ac_train = 0
        best_se_train = 0
        T1 = train_values[0] 
        T2 = train_values[0] 
        
        for t in train_values:
            # 在训练集上预测
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
            if SE > best_se_train:
                best_se_train = SE
                T2 = t
        
        # 3. 使用找到的 T1 (最佳准确率阈值) 在【测试集】上评估
        # 我们主要关注 T1 的表现，因为 T2 通常会导致特异度极低
        test_preds = (test_df[feature_col] > T1).astype(int)
        
        TP_test = ((test_preds == 1) & (test_df[label_col] == 1)).sum()
        FN_test = ((test_preds == 0) & (test_df[label_col] == 1)).sum()
        TN_test = ((test_preds == 0) & (test_df[label_col] == 0)).sum()
        FP_test = ((test_preds == 1) & (test_df[label_col] == 0)).sum()
        
        SE_test = TP_test / (TP_test + FN_test) if (TP_test + FN_test) > 0 else 0
        SP_test = TN_test / (TN_test + FP_test) if (TN_test + FP_test) > 0 else 0
        AC_test = (SE_test + SP_test) / 2
        
        # 保存这一折的最终结果
        fold_se_final.append(SE_test)
        fold_ac_final.append(AC_test)
        
        # --- 绘图数据处理 ---
        # 为了画出平滑的平均曲线，我们在当前测试集上计算所有可能阈值的性能
        # 注意：这里为了绘图统一，我们使用整个数据集的唯一值作为X轴，但在Y轴计算时只用当前测试集数据
        # (这是一种近似画法，严谨的做法是只画测试集存在的阈值，但为了曲线连续性，通常取全局阈值)
        
        # 仅在第一次循环时初始化X轴数据（假设所有折的阈值范围差不多）
        if fold_idx == 0:
            global_thresholds = np.sort(df[feature_col].unique())
            # 为了减少计算量和曲线抖动，可以适当稀疏采样，或者保留全部
            # 这里为了演示保留全部，如果数据量大建议采样：global_thresholds = global_thresholds[::10]
            
        current_se = []
        current_sp = []
        current_ac = []
        
        for t in global_thresholds:
            # 在当前测试集上应用阈值 t
            preds = (test_df[feature_col] > t).astype(int)
            
            TP = ((preds == 1) & (test_df[label_col] == 1)).sum()
            FN = ((preds == 0) & (test_df[label_col] == 1)).sum()
            TN = ((preds == 0) & (test_df[label_col] == 0)).sum()
            FP = ((preds == 1) & (test_df[label_col] == 0)).sum()
            
            se = TP / (TP + FN) if (TP + FN) > 0 else 0
            sp = TN / (TN + FP) if (TN + FP) > 0 else 0
            ac = (se + sp) / 2
            
            current_se.append(se)
            current_sp.append(sp)
            current_ac.append(ac)
            
        all_thresholds.append(current_ac) # 暂存，用于后续对齐
        # 这里为了简化，我们直接累积数值，最后除以 fold 数
        if fold_idx == 0:
            sum_se = np.array(current_se)
            sum_sp = np.array(current_sp)
            sum_ac = np.array(current_ac)
        else:
            sum_se += np.array(current_se)
            sum_sp += np.array(current_sp)
            sum_ac += np.array(current_ac)

        if (fold_idx + 1) % 20 == 0:
            print(f"  进度: {fold_idx + 1}/{total_folds} 折...")

    # ==========================================
    # 3. 计算平均值并绘图
    # ==========================================
    # 计算平均曲线
    mean_se = sum_se / total_folds
    mean_sp = sum_sp / total_folds
    mean_ac = sum_ac / total_folds
    
    # 计算最终的统计结果
    final_mean_se = np.mean(fold_se_final)
    final_std_se = np.std(fold_se_final)
    final_mean_ac = np.mean(fold_ac_final)
    final_std_ac = np.std(fold_ac_final)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(global_thresholds, mean_se, label='Avg Sensitivity (SE)', color='red', alpha=0.7)
    plt.plot(global_thresholds, mean_sp, label='Avg Specificity (SP)', color='blue', alpha=0.7)
    plt.plot(global_thresholds, mean_ac, label='Avg Accuracy (AC)', color='green', linewidth=2)
    
    # 标记最佳平均准确率的点
    best_ac_idx = np.argmax(mean_ac)
    best_ac_threshold = global_thresholds[best_ac_idx]
    
    plt.axvline(x=best_ac_threshold, color='green', linestyle='--', alpha=0.5)
    plt.text(best_ac_threshold, 0.2, f'Best AC: {best_ac_threshold:.2f}', color='green', rotation=90, verticalalignment='bottom')

    # 标题处理
    title_name = f"{feature_col}_C8" if 'HOR' in feature_col else feature_col
    plt.title(f'10x10 CV Performance: {title_name}\n(Mean Metrics vs Threshold)')

    plt.xlabel('Threshold Value')
    plt.ylabel('Score')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # ==========================================
    # 4. 输出报告
    # ==========================================
    print("-" * 50)
    print(f"特征: {feature_col} 最终评估报告 (基于测试集)")
    print("-" * 50)
    print(f"平均灵敏度 (SE): {final_mean_se:.4f} ± {final_std_se:.4f}")
    print(f"平均准确率 (AC): {final_mean_ac:.4f} ± {final_std_ac:.4f}")
    print(f"最佳平均阈值: {best_ac_threshold:.4f}")
    print("=" * 50)
    
    return final_mean_ac, final_std_ac

# ==========================================
# 5. 执行分析
# ==========================================
for col in features_to_analyze:
    if col in df.columns:
        analyze_threshold_cv(df, col, label_column, n_splits=10, n_repeats=10)
    else:
        print(f"警告: 列 {col} 不存在于数据中！")