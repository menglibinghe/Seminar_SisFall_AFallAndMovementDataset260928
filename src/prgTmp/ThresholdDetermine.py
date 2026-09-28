import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体，防止画图时中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 配置路径和参数
# ==========================================
csv_path = r"D:\vs_prj\python\seminar\P1_SisFall_AFallAndMovementDataset\database\SisFall_dataset_csv\temp\SisFall_dataset_filtered\Merged_SisFall_Final.csv"

# 需要分析的特征列
features_to_analyze = ['mean_Pitch_C4', 'std_HOR'] 

# 标签列名
label_column = 'Label' 

# 读取数据
df = pd.read_csv(csv_path)
df[label_column] = df[label_column].astype(int)

# ==========================================
# 2. 定义计算与绘图函数
# ==========================================
def analyze_threshold(df, feature_col, label_col):
    print(f"正在分析特征: {feature_col} ...")
    
    thresholds = np.sort(df[feature_col].unique())
    metrics = {'Threshold': [], 'SE': [], 'SP': [], 'AC': []}
    
    best_acc = 0
    best_se = 0
    T1 = 0 
    T2 = 0 

    for t in thresholds:
        predictions = (df[feature_col] > t).astype(int)
        
        TP = ((predictions == 1) & (df[label_col] == 1)).sum()
        TN = ((predictions == 0) & (df[label_col] == 0)).sum()
        FP = ((predictions == 1) & (df[label_col] == 0)).sum()
        FN = ((predictions == 0) & (df[label_col] == 1)).sum()
        
        SE = TP / (TP + FN) if (TP + FN) > 0 else 0
        SP = TN / (TN + FP) if (TN + FP) > 0 else 0
        AC = (SE + SP) / 2
        
        metrics['Threshold'].append(t)
        metrics['SE'].append(SE)
        metrics['SP'].append(SP)
        metrics['AC'].append(AC)
        
        if AC > best_acc:
            best_acc = AC
            T1 = t
            
        if SE > best_se:
            best_se = SE
            T2 = t

    results_df = pd.DataFrame(metrics)
    
    # ==========================================
    # 3. 绘图显示 (修改部分在这里)
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Threshold'], results_df['SE'], label='Sensitivity (SE)', color='red', alpha=0.7)
    plt.plot(results_df['Threshold'], results_df['SP'], label='Specificity (SP)', color='blue', alpha=0.7)
    plt.plot(results_df['Threshold'], results_df['AC'], label='Accuracy (AC)', color='green', linewidth=2)
    
    plt.axvline(x=T1, color='green', linestyle='--', alpha=0.5)
    plt.text(T1, 0.1, f'T1(Acc): {T1:.2f}', color='green', rotation=90, verticalalignment='bottom')
    
    plt.axvline(x=T2, color='red', linestyle='--', alpha=0.5)
    plt.text(T2, 0.9, f'T2(SE): {T2:.2f}', color='red', rotation=90, verticalalignment='top')

    # --- 修改逻辑开始 ---
    # 如果列名包含 'HOR'，则添加 _C8 后缀，否则保持原样
    if 'HOR' in feature_col:
        title_name = f"{feature_col}_C8"
    else:
        title_name = feature_col
        
    plt.title(f'Performance Metrics vs. Threshold for {title_name}')
    # --- 修改逻辑结束 ---

    plt.xlabel('Threshold Value')
    plt.ylabel('Score')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"-> 最佳准确率阈值 (T1): {T1:.4f}, 准确率: {best_acc:.4f}")
    print(f"-> 最大灵敏度阈值 (T2): {T2:.4f}, 灵敏度: {best_se:.4f}")
    print("-" * 30)
    
    return results_df

# ==========================================
# 4. 执行分析
# ==========================================
for col in features_to_analyze:
    if col in df.columns:
        results = analyze_threshold(df, col, label_column)
    else:
        print(f"警告: 列 {col} 不存在于数据中！")